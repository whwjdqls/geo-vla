import glob
import os
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import json
from datasets import Dataset, Image
from PIL import Image as PILImage
import argparse
import json
import os
from dataclasses import asdict
from datasets import Array3D
import numpy as np
import torch
from torch.utils.data import DataLoader
# from 
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




def main() -> None:
    args = parse_args()

    ckpt = _load_ckpt(args.checkpoint_path)
    ckpt_args = ckpt["args"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, model_args = _build_model_from_ckpt(ckpt, device=device)

    data_root = "/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_dataset_depth"
    save_root = "/scratch2/whwjdqls99/libero/whwjdqls99/libero_hdfr_lerobot_dataset_depth_latents"

    # glob 패턴을 사용해 모든 parquet 파일 검색
    files = glob.glob(os.path.join(data_root, "data", "chunk-*", "*.parquet"))
    files.sort()

    print(f"Found {len(files)} episode files.")

    depth_meta_file = "/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_only_depths/depth_u16_png_metadata.json"
    import json
    with open(depth_meta_file, "r") as f:
        meta = json.load(f)
    vmin = float(meta["global_min"])
    vmax = float(meta["global_max"])
    mean_path = meta["original_stats"]["mean_npy"]
    std_path  = meta["original_stats"]["std_npy"]
    mean = torch.from_numpy(np.load(mean_path)).float()  # (H,W) or scalar
    std  = torch.from_numpy(np.load(std_path)).float()

    scale = (vmax - vmin) / 65535.0
    offset = vmin
    
    
    for i, file_path in enumerate(files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(files)}: {file_path}")
        ds = Dataset.from_parquet(file_path)
        dest_file_path = file_path.replace(data_root, save_root)
        print(dest_file_path)
        if os.path.exists(dest_file_path):
            print(f"File already exists at {dest_file_path}, skipping...")
            continue
        depth_images = np.array(ds['depth_image'], dtype=np.float32)  # (Frames, H, W)
        
        depth_images *= scale
        depth_images += offset
        
        depth_images = (depth_images - mean.numpy()) / std.numpy()
        
        depth_images_tensor = torch.from_numpy(depth_images).unsqueeze(1).to(device)  # (Frames, 1, H, W)
        with torch.no_grad():
            out = model(depth_images_tensor)  # Add channel dim if 
        latents = out['z'].cpu().numpy()  # (Frames, z_ch, z_h, z_w)
        print(f"Latents shape: {latents.shape}")
        print(f"latents dtype: {latents.dtype}")
                
        ds = ds.remove_columns("depth_image")

        # 2. Add the new latents as the 'depth_image' column
        # Note: latents is (Frames, 4, 16, 16), so we pass it directly
        # ds = ds.add_column("depth_latent", list(latents))
        ds = ds.add_column("depth_latent", latents.tolist())
        # ds = ds.cast_column("depth_image", Image())
        ds = ds.cast_column("depth_latent", Array3D(shape=(4, 16, 16), dtype='float32'))

        ds.to_parquet(dest_file_path)
        # if i == 1:
        #     break
        # break


if __name__ == "__main__":
    main()