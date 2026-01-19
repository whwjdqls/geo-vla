import collections
import dataclasses
import json
import logging
import math
import pathlib
import time
from typing import Any, Sequence

import imageio
import numpy as np
# from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import image_tools
import tqdm
import tyro


ROBOCASA_DEFAULT_CAMERA_NAMES = [
	"robot0_agentview_left",
	"robot0_agentview_right",
	"robot0_eye_in_hand",
]


@dataclasses.dataclass
class PolicyCheckpoint:
	"""Checkpoint-backed policy settings (loaded in-process, no server)."""

	# Training config name (e.g., "pi05_libero" or your custom finetune config).
	config: str
	# Checkpoint directory (e.g., "/path/to/exp/30000").
	dir: str
	# Optional fallback prompt if prompt is missing.
	default_prompt: str | None = None
	# Only relevant for PyTorch checkpoints (model.safetensors). E.g. "cuda:0" or "cpu".
	pytorch_device: str | None = None


@dataclasses.dataclass
class Args:
	#################################################################################################################
	# Preprocess / rollout parameters
	#################################################################################################################
	resize_size: int = 224
	replan_steps: int = 5

	#################################################################################################################
	# RoboCasa environment-specific parameters
	#################################################################################################################
	env_names: list[str] = dataclasses.field(default_factory=list)
	"""RoboCasa env names to evaluate.
 
['CloseDoubleDoor', 'CloseDrawer', 'CloseSingleDoor', 'CoffeePressButton', 'CoffeeServeMug', 'CoffeeSetupMug', 'OpenDoubleDoor', 
'OpenDrawer', 'OpenSingleDoor', 'PnPCabToCounter', 'PnPCounterToCab', 'PnPCounterToMicrowave', 'PnPCounterToSink', 'PnPCounterToStove', 
'PnPMicrowaveToCounter', 'PnPSinkToCounter', 'PnPStoveToCounter', 'TurnOffMicrowave', 'TurnOffSinkFaucet', 'TurnOffStove', 'TurnOnMicrowave', 
'TurnOnSinkFaucet', 'TurnOnStove', 'TurnSinkSpout']

	If empty, evaluates the default RoboCasa single-stage set (from `robocasa.utils.dataset_registry.SINGLE_STAGE_TASK_DATASETS`),
	excluding `NavigateKitchen` (mirrors the reference evaluator).
	"""

	num_steps_wait: int = 10
	num_episodes_per_env: int = 50
	max_episode_steps: int = 720

	env_img_res: int = 256
	robots: str = "PandaMobile"
	controllers: str = "OSC_POSE"
	obj_instance_split: str = "B"
	randomize_cameras: bool = False
	layout_and_style_ids: tuple[tuple[int, int], ...] = ((1, 1), (2, 2), (4, 4), (6, 9), (7, 10))

	#################################################################################################################
	# Saving / logging
	#################################################################################################################
	out_dir: str = "/scratch2/whwjdqls99/pi/pi_zero/eval_outputs"  # base folder for videos + metrics
	save_videos: bool = True
	video_fps: int = 10
	save_failed_videos: bool = True  # if False, saves only successes
	overwrite: bool = False  # if False, refuses to overwrite existing metrics files

	#################################################################################################################
	# Reproducibility
	#################################################################################################################
	seed: int = 7



def _get_default_env_names() -> list[str]:
	"""Returns RoboCasa env names mirroring `run_robocasa_eval_ref.py` behavior."""
	try:
		from robocasa.utils.dataset_registry import SINGLE_STAGE_TASK_DATASETS  # type: ignore

		names = sorted([name for name in SINGLE_STAGE_TASK_DATASETS if name != "NavigateKitchen"])
		return names
	except Exception as e:
		raise ImportError(
			"Failed to import RoboCasa env registry. "
			"Make sure `robocasa` is installed and importable in this environment."
		) from e


# def _create_policy(policy_args: PolicyCheckpoint) -> _policy.Policy:
def _create_policy(policy_args: PolicyCheckpoint):
	if not policy_args.dir:
		raise ValueError("--policy.dir is required")
	train_cfg = _config.get_config(policy_args.config)
	return _policy_config.create_trained_policy(
		train_cfg,
		policy_args.dir,
		default_prompt=policy_args.default_prompt,
		pytorch_device=policy_args.pytorch_device,
	)


def _create_robocasa_env(env_name: str, *, args: Args, seed: int | None) -> Any:
	"""Create a RoboCasa env with the same config style as `run_robocasa_eval_ref.py`."""
	try:
		import robocasa  # noqa: F401
		import robosuite
		from robosuite.controllers import load_composite_controller_config
	except Exception as e:
		raise ImportError(
			"Failed to import RoboCasa / RoboSuite. "
			"Ensure `robocasa` and `robosuite` are installed in this environment."
		) from e

	# `load_composite_controller_config` expects a *composite* controller name.
	# Many users pass a non-composite controller like "OSC_POSE"; in that case,
	# fall back to the default composite controller config (None) like the ref script.
	try:
		controller_configs = load_composite_controller_config(
			controller=args.controllers,
			robot=args.robots,
		)
	except AssertionError:
		logging.warning(
			"Composite controller '%s' not found; falling back to default composite controller config.",
			args.controllers,
		)
		controller_configs = load_composite_controller_config(
			controller=None,
			robot=args.robots,
		)

	env_kwargs = dict(
		env_name=env_name,
		robots=args.robots,
		controller_configs=controller_configs,
		camera_names=ROBOCASA_DEFAULT_CAMERA_NAMES,
		camera_widths=args.env_img_res,
		camera_heights=args.env_img_res,
		has_renderer=False,
		has_offscreen_renderer=True,
		ignore_done=True,
		use_object_obs=True,
		use_camera_obs=True,
		camera_depths=False,
		seed=seed,
		obj_instance_split=args.obj_instance_split,
		generative_textures=None,
		randomize_cameras=args.randomize_cameras,
		layout_and_style_ids=args.layout_and_style_ids,
		translucent_robot=False,
	)

	return robosuite.make(**env_kwargs)


def _get_task_description(env: Any, fallback_name: str) -> str:
	try:
		ep_meta = env.get_ep_meta() if hasattr(env, "get_ep_meta") else {}
		return str(ep_meta.get("lang", fallback_name))
	except Exception:
		return fallback_name


def _robocasa_success(env: Any, info: dict[str, Any] | None = None) -> bool:
	if hasattr(env, "_check_success"):
		try:
			return bool(env._check_success())
		except Exception:
			pass
	if info and "success" in info:
		try:
			return bool(info["success"])
		except Exception:
			pass
	return False


def _robocasa_dummy_action_for_env(env: Any) -> list[float]:
	# Heuristic: fill zeros, and set final element to -1.0 (common gripper convention).
	action_dim = int(getattr(env, "action_dim", 0) or 0)
	if action_dim <= 0:
		# Fallback to the reference padding behavior: 11D is common for PandaMobile.
		action_dim = 11
	a = [0.0] * action_dim
	if a:
		a[-1] = -1.0
	return a


def _coerce_action_for_env(action: Any, env: Any) -> list[float]:
	a = np.asarray(action, dtype=np.float32).reshape(-1)
	# Mirror the reference evaluator: pad 7D -> 11D with [0,0,0,-1]
	# (common when evaluating arm-only policies on PandaMobile).
	if a.shape[0] == 7:
		pad = np.array([0.0, 0.0, 0.0, -1.0], dtype=a.dtype)
		a = np.concatenate([a, pad], axis=0)

	action_dim = int(getattr(env, "action_dim", a.shape[0]) or a.shape[0])
	if action_dim != a.shape[0]:
		raise ValueError(f"Action dim mismatch: env.action_dim={action_dim}, policy_action={a.shape[0]}")
	return a.tolist()


def eval_robocasa(policy: PolicyCheckpoint, args: Args) -> None:
	np.random.seed(args.seed)

	# Output paths
	out_dir = pathlib.Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	env_names = args.env_names or _get_default_env_names()
	run_tag = f"robocasa_seed{args.seed}_n{len(env_names)}"

	metrics_json_path = out_dir / f"results_{run_tag}.json"
	metrics_csv_path = out_dir / f"results_{run_tag}.csv"
	summary_json_path = out_dir / f"summary_{run_tag}.json"

	# Overwrite protection
	if not args.overwrite:
		for p in [metrics_json_path, metrics_csv_path, summary_json_path]:
			if p.exists():
				raise FileExistsError(f"{p} already exists. Set --overwrite True or change --out-dir.")

	# Videos folder
	videos_dir = out_dir / "videos" / run_tag
	if args.save_videos:
		videos_dir.mkdir(parents=True, exist_ok=True)

	logging.info("RoboCasa envs: %d", len(env_names))
	logging.info("Max steps: %d, wait steps: %d", args.max_episode_steps, args.num_steps_wait)
	logging.info("Episodes per env: %d", args.num_episodes_per_env)
	logging.info("Saving to: %s", str(out_dir))

	# In-process policy
	logging.info("Loading Pi policy in-process (config=%s, dir=%s)", policy.config, policy.dir)
	policy_obj = _create_policy(policy)
	logging.info("Policy metadata: %s", getattr(policy_obj, "metadata", {}))

	# Storage
	episode_rows: list[dict[str, Any]] = []
	per_env_summary: dict[str, dict[str, Any]] = {}

	total_episodes = 0
	total_successes = 0

	for env_idx, env_name in enumerate(tqdm.tqdm(env_names, desc="Envs")):
		logging.info("\n[Env %d/%d] %s", env_idx + 1, len(env_names), env_name)
		env = _create_robocasa_env(env_name, args=args, seed=args.seed)
		env_episodes = 0
		env_successes = 0

		for episode_idx in tqdm.tqdm(range(args.num_episodes_per_env), desc=env_name, leave=False):
			try:
				env.reset()
				if hasattr(env, "_get_observations"):
					obs = env._get_observations(force_update=True)
				else:
					# Some variants return obs from reset.
					obs = getattr(env, "_get_observation", lambda: {})()
			except Exception:
				# Last resort: try reset() output.
				obs = env.reset()

			try:
				policy_obj.reset()
			except Exception:
				pass

			action_plan = collections.deque()
			t = 0
			success = False
			exception_str = ""
			replay_images = []
			start_time = time.time()
			info: dict[str, Any] = {}

			while t < args.max_episode_steps + args.num_steps_wait:
				try:
					if t < args.num_steps_wait:
						dummy_action = _robocasa_dummy_action_for_env(env)
						obs, reward, done, info = env.step(dummy_action)
						t += 1
						continue

					# Observation -> Pi input (match LIBERO evaluator shape/keys)
					left = np.ascontiguousarray(obs["robot0_agentview_left_image"][::-1, ::-1])
					right = np.ascontiguousarray(obs["robot0_agentview_right_image"][::-1, ::-1])
					wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

					img = image_tools.convert_to_uint8(
						image_tools.resize_with_pad(left, args.resize_size, args.resize_size)
					)
					right_img = image_tools.convert_to_uint8(
						image_tools.resize_with_pad(right, args.resize_size, args.resize_size)
					)
					wrist_img = image_tools.convert_to_uint8(
						image_tools.resize_with_pad(wrist, args.resize_size, args.resize_size)
					)

					if args.save_videos:
						replay_images.append(img)

					task_description = _get_task_description(env, env_name)

					if not action_plan:
						element = {
							# "observation/image": img,
							"observation/left_image": img,
							"observation/right_image": right_img,
							"observation/wrist_image": wrist_img,
    #    proprio = np.concatenate([obs["robot0_gripper_qpos"], np.hstack([obs["robot0_eef_pos"], T.quat2axisangle(obs["robot0_eef_quat"])])])
							# "observation/state": np.concatenate(
							# 	(
							# 		obs["robot0_eef_pos"],
							# 		_quat2axisangle(obs["robot0_eef_quat"]),
							# 		obs["robot0_gripper_qpos"],
							# 	)
							# ),
							"observation/state": np.concatenate(
								(
									obs["robot0_gripper_qpos"],
									obs["robot0_eef_pos"],
									_quat2axisangle(obs["robot0_eef_quat"]),
								)
							),
							"prompt": task_description,
						}

						out = policy_obj.infer(element)
						action_chunk = out["actions"]
						if len(action_chunk) < args.replan_steps:
							raise RuntimeError(
								f"replan_steps={args.replan_steps} but policy returned {len(action_chunk)} steps"
							)
						action_plan.extend(action_chunk[: args.replan_steps])

					action = action_plan.popleft()
					action_list = _coerce_action_for_env(action, env)
					obs, reward, done, info = env.step(action_list)
					t += 1

					success = _robocasa_success(env, info)
					if success:
						break
					# RoboSuite often sets done=False with ignore_done=True.
					# We only terminate on success or max steps.
				except Exception as e:
					exception_str = repr(e)
					logging.error(
						"Caught exception (env=%s, episode=%d, t=%d): %s",
						env_name,
						episode_idx,
						t,
						e,
					)
					break

			elapsed = time.time() - start_time

			env_episodes += 1
			total_episodes += 1
			if success:
				env_successes += 1
				total_successes += 1

			if args.save_videos and (success or args.save_failed_videos):
				suffix = "success" if success else "failure"
				task_seg = env_name.replace(" ", "_").replace("/", "_")
				video_path = videos_dir / f"env{env_idx:03d}_{task_seg}_ep{episode_idx:03d}_{suffix}.mp4"
				try:
					imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=args.video_fps)
				except Exception as e:
					logging.error("Failed to write video %s: %s", str(video_path), e)

			episode_rows.append(
				{
					"suite": "robocasa",
					"seed": args.seed,
					"policy_config": policy.config,
					"policy_dir": policy.dir,
					"env_name": env_name,
					"episode_idx": episode_idx,
					"success": bool(success),
					"steps_total": int(t),
					"steps_wait": int(args.num_steps_wait),
					"max_steps": int(args.max_episode_steps),
					"elapsed_sec": float(elapsed),
					"exception": exception_str,
				}
			)

			logging.info(
				"[%s] ep %d | success=%s | steps=%d | elapsed=%.2fs | total_sr=%.2f%%",
				env_name,
				episode_idx,
				success,
				t,
				elapsed,
				(total_successes / total_episodes * 100.0) if total_episodes else 0.0,
			)

		try:
			env.close()
		except Exception:
			pass
		del env

		env_sr = (env_successes / env_episodes) if env_episodes else 0.0
		per_env_summary[env_name] = {
			"env_name": env_name,
			"episodes": env_episodes,
			"successes": env_successes,
			"success_rate": env_sr,
		}
		logging.info("%s summary: %d/%d (%.2f%%)", env_name, env_successes, env_episodes, env_sr * 100.0)

	total_sr = (total_successes / total_episodes) if total_episodes else 0.0

	full = {
		"run": {
			"suite": "robocasa",
			"seed": args.seed,
			"policy": {
				"config": policy.config,
				"dir": policy.dir,
				"default_prompt": policy.default_prompt,
				"pytorch_device": policy.pytorch_device,
			},
			"env_names": env_names,
			"resize_size": args.resize_size,
			"replan_steps": args.replan_steps,
			"num_steps_wait": args.num_steps_wait,
			"num_episodes_per_env": args.num_episodes_per_env,
			"max_episode_steps": args.max_episode_steps,
			"env_img_res": args.env_img_res,
			"save_videos": args.save_videos,
			"videos_dir": str(videos_dir) if args.save_videos else "",
		},
		"summary": {
			"total_episodes": total_episodes,
			"total_successes": total_successes,
			"total_success_rate": total_sr,
		},
		"per_env_summary": per_env_summary,
		"episodes": episode_rows,
	}
	metrics_json_path.write_text(json.dumps(full, indent=2))
	_write_csv(metrics_csv_path, episode_rows)

	summary_json_path.write_text(
		json.dumps(
			{
				"suite": "robocasa",
				"seed": args.seed,
				"policy_config": policy.config,
				"policy_dir": policy.dir,
				"total_episodes": total_episodes,
				"total_successes": total_successes,
				"total_success_rate": total_sr,
				"per_env_summary": per_env_summary,
			},
			indent=2,
		)
	)

	logging.info("DONE.")
	logging.info("Total success rate: %.2f%% (%d/%d)", total_sr * 100.0, total_successes, total_episodes)
	logging.info("Saved:")
	logging.info("  %s", str(metrics_json_path))
	logging.info("  %s", str(metrics_csv_path))
	logging.info("  %s", str(summary_json_path))
	if args.save_videos:
		logging.info("  videos: %s", str(videos_dir))


def _write_csv(path: pathlib.Path, rows: list[dict[str, Any]]) -> None:
	# Simple CSV writer without pandas dependency
	if not rows:
		path.write_text("")
		return
	keys = list(rows[0].keys())
	lines = [",".join(keys)]
	for r in rows:
		vals = []
		for k in keys:
			v = r.get(k, "")
			s = str(v)
			# Escape quotes/commas/newlines
			if any(c in s for c in [",", '"', "\n"]):
				s = '"' + s.replace('"', '""') + '"'
			vals.append(s)
		lines.append(",".join(vals))
	path.write_text("\n".join(lines))


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
	# clip quaternion
	quat = np.asarray(quat).copy()
	if quat[3] > 1.0:
		quat[3] = 1.0
	elif quat[3] < -1.0:
		quat[3] = -1.0

	den = np.sqrt(1.0 - quat[3] * quat[3])
	if math.isclose(float(den), 0.0):
		return np.zeros(3)

	return (quat[:3] * 2.0 * math.acos(float(quat[3]))) / den


if __name__ == "__main__":
	logging.basicConfig(level=logging.INFO)
	tyro.cli(eval_robocasa)
