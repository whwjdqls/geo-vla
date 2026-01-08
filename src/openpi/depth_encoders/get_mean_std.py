#!/usr/bin/env python3

import argparse
import json
import os
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets_.datasets import DepthTemporalPNGDatasetPreload
from models.cnn_models import DepthLatentModel


def parse_args() -> argparse.Namespace:
	# Match inference.py: user provides only checkpoint_path.
	p = argparse.ArgumentParser("Compute mean/std of depth-encoder latent over train split")
	p.add_argument(
		"--checkpoint_path",
		type=str,
		required=True,
		help="checkpoint path to load the model from.",
	)
	return p.parse_args()


def _load_ckpt(checkpoint_path: str) -> dict:
	ckpt = torch.load(checkpoint_path, map_location="cpu")
	if not isinstance(ckpt, dict) or "model" not in ckpt or "args" not in ckpt:
		raise ValueError(
			"Checkpoint must be a dict with keys 'model' and 'args' (as saved by train_AE.py)."
		)
	return ckpt


def _build_model_from_ckpt(ckpt: dict, device: str) -> tuple[DepthLatentModel, dict]:
	model_args = ckpt["args"]
	model = DepthLatentModel(
		z_ch=model_args["z_ch"],
		base_ch=model_args["base_ch"],
		use_vae=model_args["use_vae"],
		z_hw=model_args["z_hw"],
	)
	model.load_state_dict(ckpt["model"], strict=True)
	model.eval()
	model.to(device)
	return model, model_args


@torch.no_grad()
def compute_latent_stats(
	model: DepthLatentModel,
	loader: DataLoader,
	device: str,
	amp: bool,
	latent_kind: str,
	max_batches: int,
) -> dict:
	sum_c = None
	sumsq_c = None
	count_per_channel = 0

	sum_all = 0.0
	sumsq_all = 0.0
	count_all = 0

	n_batches = 0

	for batch in loader:
		# Dataset returns [B, T, 1, H, W] when add_channel_dim=True (T=seq_len)
		# With seq_len=1, it should be [B, 1, 1, H, W].
		if batch.ndim == 4:
			# [B, T, H, W] -> [B, T, 1, H, W]
			batch = batch.unsqueeze(2)
		if batch.ndim != 5:
			raise ValueError(f"Unexpected batch shape: {tuple(batch.shape)}")

		B, T, C, H, W = batch.shape
		if C != 1:
			raise ValueError(f"Expected C=1 depth channel, got {C}")

		x = batch.view(B * T, C, H, W).to(device, non_blocking=True)

		with torch.cuda.amp.autocast(enabled=(amp and device.startswith("cuda"))):
			out = model(x)
			if latent_kind == "z":
				z = out["z"]
			elif latent_kind == "mu":
				if out["mu"] is None:
					raise ValueError("latent-kind='mu' requested but checkpoint is not a VAE (mu is None).")
				z = out["mu"]
			else:
				raise ValueError(f"Unknown latent_kind: {latent_kind}")

		# z: [BT, z_ch, z_hw, z_hw]
		if z.ndim != 4:
			raise ValueError(f"Unexpected latent shape: {tuple(z.shape)}")

		z = z.float()
		bt, z_ch, z_h, z_w = z.shape

		# Accumulate on CPU in float64
		z_sum_c = z.sum(dim=(0, 2, 3)).double().cpu()
		z_sumsq_c = (z * z).sum(dim=(0, 2, 3)).double().cpu()

		if sum_c is None:
			sum_c = z_sum_c
			sumsq_c = z_sumsq_c
		else:
			sum_c += z_sum_c
			sumsq_c += z_sumsq_c

		elems_per_channel = bt * z_h * z_w
		count_per_channel += int(elems_per_channel)

		z_sum_all = float(z.sum().double().cpu())
		z_sumsq_all = float((z * z).sum().double().cpu())
		sum_all += z_sum_all
		sumsq_all += z_sumsq_all
		count_all += int(bt * z_ch * z_h * z_w)

		n_batches += 1
		if max_batches > 0 and n_batches >= max_batches:
			break

	if sum_c is None or sumsq_c is None or count_per_channel == 0:
		raise RuntimeError("No data processed; cannot compute stats.")

	mean_c = (sum_c / count_per_channel).numpy()
	var_c = (sumsq_c / count_per_channel).numpy() - mean_c**2
	var_c = np.maximum(var_c, 0.0)
	std_c = np.sqrt(var_c)

	mean_all = sum_all / max(1, count_all)
	var_all = (sumsq_all / max(1, count_all)) - mean_all**2
	std_all = float(np.sqrt(max(var_all, 0.0)))

	return {
		"count_all": int(count_all),
		"count_per_channel": int(count_per_channel),
		"channel_mean": mean_c.tolist(),
		"channel_std": std_c.tolist(),
		"scalar_mean": float(mean_all),
		"scalar_std": float(std_all),
		"batches": int(n_batches),
	}


def main() -> None:
	args = parse_args()

	ckpt = _load_ckpt(args.checkpoint_path)
	ckpt_args = ckpt["args"]
	device = "cuda" if torch.cuda.is_available() else "cpu"
	model, model_args = _build_model_from_ckpt(ckpt, device=device)

	# Choose latent kind: deterministic for VAE
	latent_kind = "mu" if bool(model_args.get("use_vae", False)) else "z"

	# Build train dataset using checkpoint settings (like inference.py)
	# Important: we use seq_len=1 to cover all frames once.
	ds = DepthTemporalPNGDatasetPreload(
		root=ckpt_args["root"],
		seq_len=1,
		stride=1,
		step=int(ckpt_args.get("step", 1)),
		return_info=False,
		drop_last=False,
		normalize_0_1=bool(ckpt_args.get("normalize_0_1", False)),
		clamp_to_minmax=bool(ckpt_args.get("clamp_to_minmax", True)),
		add_channel_dim=bool(ckpt_args.get("add_channel_dim", True)),
		split="train",
		preload_dtype=torch.float16,
		meta_json_name=ckpt_args.get("meta_json_name", "depth_u16_png_metadata.json"),
	)

	# Loader defaults from ckpt when present
	batch_size = int(ckpt_args.get("batch_size", 256))
	num_workers = int(ckpt_args.get("num_workers", 8))
	pin_memory = bool(ckpt_args.get("pin_memory", True))
	amp = bool(ckpt_args.get("amp", True))

	loader = DataLoader(
		ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=pin_memory and device.startswith("cuda"),
		persistent_workers=(num_workers > 0),
		drop_last=False,
	)

	stats = compute_latent_stats(
		model=model,
		loader=loader,
		device=device,
		amp=amp,
		latent_kind=latent_kind,
		max_batches=-1,
	)

	# Add some metadata
	payload = {
		"checkpoint_path": os.path.abspath(args.checkpoint_path),
		"dataset_root": os.path.abspath(ckpt_args["root"]),
		"split": "train",
		"latent_kind": latent_kind,
		"latent_shape": [int(model_args["z_hw"]), int(model_args["z_hw"]), int(model_args["z_ch"])],
		"stats": stats,
	}

	ckpt_dir = os.path.dirname(os.path.abspath(args.checkpoint_path))
	ckpt_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
	out_path = os.path.join(ckpt_dir, f"latent_stats_{ckpt_name}_train.json")

	os.makedirs(os.path.dirname(out_path), exist_ok=True)
	with open(out_path, "w") as f:
		json.dump(payload, f, indent=2)
	print(f"Wrote latent stats to: {out_path}")
	print(f"latent_kind={latent_kind} latent_shape={payload['latent_shape']}")
	print(
		"channel_mean=",
		np.array(stats["channel_mean"], dtype=np.float64),
		"channel_std=",
		np.array(stats["channel_std"], dtype=np.float64),
	)
	print("scalar_mean=", stats["scalar_mean"], "scalar_std=", stats["scalar_std"])


if __name__ == "__main__":
	main()

