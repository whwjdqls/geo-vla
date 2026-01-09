import collections
import dataclasses
import json
import logging
import math
import pathlib
import time
from typing import Any

import imageio
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config
from openpi_client import image_tools
import tqdm
import tyro


LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


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
	# LIBERO environment-specific parameters
	#################################################################################################################
	task_suite_name: str = (
		"libero_spatial"  # Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
	)
	num_steps_wait: int = 10
	num_trials_per_task: int = 50

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

	#################################################################################################################
	# Compatibility
	#################################################################################################################
	trust_libero_init_states: bool = True
	"""If True, allowlist NumPy globals needed for LIBERO's init-state `torch.load` on PyTorch 2.6+.

	PyTorch 2.6 changed `torch.load` default `weights_only` to True, which can reject older pickles.
	This allowlist should be safe if you trust the LIBERO init-state files on disk.
	"""


def _configure_torch_load_for_libero(*, trust_init_states: bool) -> None:
	if not trust_init_states:
		return
	try:
		import torch

		# Needed for loading numpy arrays stored in older torch pickles.
		torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])  # type: ignore[attr-defined]
		# These are commonly required alongside _reconstruct.
		torch.serialization.add_safe_globals([np.ndarray, np.dtype])
		try:
			torch.serialization.add_safe_globals([np.core.multiarray.scalar])  # type: ignore[attr-defined]
		except Exception:
			pass
	except Exception:
		# If torch isn't available or API differs, just continue and let the original error surface.
		pass


def _max_steps_for_suite(name: str) -> int:
	if name == "libero_spatial":
		return 220
	if name == "libero_object":
		return 280
	if name == "libero_goal":
		return 300
	if name == "libero_10":
		return 520
	if name == "libero_90":
		return 400
	raise ValueError(f"Unknown task suite: {name}")


def _create_policy(policy_args: PolicyCheckpoint) -> _policy.Policy:
	if not policy_args.dir:
		raise ValueError("--policy.dir is required")
	train_cfg = _config.get_config(policy_args.config)
	return _policy_config.create_trained_policy(
		train_cfg,
		policy_args.dir,
		default_prompt=policy_args.default_prompt,
		pytorch_device=policy_args.pytorch_device,
	)


def eval_libero(policy: PolicyCheckpoint, args: Args) -> None:
	np.random.seed(args.seed)
	_configure_torch_load_for_libero(trust_init_states=args.trust_libero_init_states)

	# Output paths
	out_dir = pathlib.Path(args.out_dir)
	out_dir.mkdir(parents=True, exist_ok=True)

	suite_name = args.task_suite_name
	run_tag = f"{suite_name}_seed{args.seed}"

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

	# Load task suite
	benchmark_dict = benchmark.get_benchmark_dict()
	task_suite = benchmark_dict[suite_name]()
	num_tasks_in_suite = task_suite.n_tasks
	max_steps = _max_steps_for_suite(suite_name)

	logging.info("Task suite: %s (#tasks=%d)", suite_name, num_tasks_in_suite)
	logging.info("Max steps: %d, wait steps: %d", max_steps, args.num_steps_wait)
	logging.info("Trials per task: %d", args.num_trials_per_task)
	logging.info("Saving to: %s", str(out_dir))

	# In-process policy
	logging.info("Loading policy in-process (config=%s, dir=%s)", policy.config, policy.dir)
	policy_obj = _create_policy(policy)
	logging.info("Policy metadata: %s", getattr(policy_obj, "metadata", {}))

	# Storage
	episode_rows: list[dict[str, Any]] = []
	per_task_summary: dict[int, dict[str, Any]] = {}

	total_episodes = 0
	total_successes = 0

	# Evaluate every task in suite
	for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
		task = task_suite.get_task(task_id)
		task_description = str(task.language)

		# Initial states for this task
		initial_states = task_suite.get_task_init_states(task_id)

		# Make env for this task
		env, _ = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

		task_episodes = 0
		task_successes = 0

		for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc=f"Task {task_id}", leave=False):
			logging.info("\n[Task %d] %s", task_id, task_description)

			env.reset()
			try:
				policy_obj.reset()
			except Exception:
				# Not all policies implement stateful reset.
				pass

			action_plan = collections.deque()

			obs = env.set_init_state(initial_states[episode_idx])

			t = 0
			done = False
			exception_str = ""
			replay_images = []

			start_time = time.time()

			while t < max_steps + args.num_steps_wait:
				try:
					if t < args.num_steps_wait:
						obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
						t += 1
						continue

					# Preprocess (match training)
					img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
					wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])

					img = image_tools.convert_to_uint8(
						image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
					)
					wrist_img = image_tools.convert_to_uint8(
						image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
					)

					# Save for replay video
					if args.save_videos:
						replay_images.append(img)

					if not action_plan:
						element = {
							"observation/image": img,
							"observation/wrist_image": wrist_img,
							"observation/state": np.concatenate(
								(
									obs["robot0_eef_pos"],
									_quat2axisangle(obs["robot0_eef_quat"]),
									obs["robot0_gripper_qpos"],
								)
							),
							"prompt": task_description,
						}

						# Query model (in-process)
						out = policy_obj.infer(element)
						action_chunk = out["actions"]

						if len(action_chunk) < args.replan_steps:
							raise RuntimeError(
								f"replan_steps={args.replan_steps} but policy returned {len(action_chunk)} steps"
							)
						action_plan.extend(action_chunk[: args.replan_steps])

					action = action_plan.popleft()
					action_np = np.asarray(action)
					obs, reward, done, info = env.step(action_np.tolist())
					t += 1

					if done:
						break

				except Exception as e:
					exception_str = repr(e)
					logging.error(
						"Caught exception (task_id=%d, episode=%d, t=%d): %s",
						task_id,
						episode_idx,
						t,
						e,
					)
					break

			elapsed = time.time() - start_time

			# Update counters
			task_episodes += 1
			total_episodes += 1
			if done:
				task_successes += 1
				total_successes += 1

			# Save video (unique filename, no overwrite)
			if args.save_videos and (done or args.save_failed_videos):
				suffix = "success" if done else "failure"
				task_seg = task_description.replace(" ", "_").replace("/", "_")
				video_path = videos_dir / f"task{task_id:03d}_ep{episode_idx:03d}_{task_seg}_{suffix}.mp4"
				try:
					imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=args.video_fps)
				except Exception as e:
					logging.error("Failed to write video %s: %s", str(video_path), e)

			# Record episode row
			episode_rows.append(
				{
					"suite": suite_name,
					"seed": args.seed,
					"policy_config": policy.config,
					"policy_dir": policy.dir,
					"task_id": task_id,
					"task_description": task_description,
					"episode_idx": episode_idx,
					"success": bool(done),
					"steps_total": int(t),
					"steps_wait": int(args.num_steps_wait),
					"max_steps": int(max_steps),
					"elapsed_sec": float(elapsed),
					"exception": exception_str,
				}
			)

			logging.info(
				"[Task %d] ep %d | success=%s | steps=%d | elapsed=%.2fs | total_sr=%.2f%%",
				task_id,
				episode_idx,
				done,
				t,
				elapsed,
				(total_successes / total_episodes * 100.0) if total_episodes else 0.0,
			)

		try:
			env.close()
		except Exception:
			pass
		del env

		# Per-task summary
		task_sr = (task_successes / task_episodes) if task_episodes else 0.0
		per_task_summary[task_id] = {
			"task_id": task_id,
			"task_description": task_description,
			"episodes": task_episodes,
			"successes": task_successes,
			"success_rate": task_sr,
		}
		logging.info("Task %d summary: %d/%d (%.2f%%)", task_id, task_successes, task_episodes, task_sr * 100.0)

	total_sr = (total_successes / total_episodes) if total_episodes else 0.0

	# Write JSON (full)
	full = {
		"run": {
			"suite": suite_name,
			"seed": args.seed,
			"policy": {
				"config": policy.config,
				"dir": policy.dir,
				"default_prompt": policy.default_prompt,
				"pytorch_device": policy.pytorch_device,
			},
			"resize_size": args.resize_size,
			"replan_steps": args.replan_steps,
			"num_steps_wait": args.num_steps_wait,
			"num_trials_per_task": args.num_trials_per_task,
			"max_steps": max_steps,
			"save_videos": args.save_videos,
			"videos_dir": str(videos_dir) if args.save_videos else "",
		},
		"summary": {
			"total_episodes": total_episodes,
			"total_successes": total_successes,
			"total_success_rate": total_sr,
		},
		"per_task_summary": per_task_summary,
		"episodes": episode_rows,
	}
	metrics_json_path.write_text(json.dumps(full, indent=2))

	# Write CSV (flat)
	_write_csv(metrics_csv_path, episode_rows)

	# Write summary JSON (small)
	summary_json_path.write_text(
		json.dumps(
			{
				"suite": suite_name,
				"seed": args.seed,
				"policy_config": policy.config,
				"policy_dir": policy.dir,
				"total_episodes": total_episodes,
				"total_successes": total_successes,
				"total_success_rate": total_sr,
				"per_task_summary": per_task_summary,
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


def _get_libero_env(task, resolution: int, seed: int):
	task_description = task.language
	task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
	env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
	env = OffScreenRenderEnv(**env_args)
	env.seed(seed)
	return env, task_description


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
	tyro.cli(eval_libero)
