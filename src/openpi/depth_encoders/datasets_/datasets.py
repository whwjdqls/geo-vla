import numpy as np
import torch

import os
import json
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v3 as iio
import glob
import cv2
@dataclass
class DepthDecodeConfig:
    meta_json: str  # path to depth_u16_png_metadata.json
    normalize_0_1: bool = False  # if True, return (u16/65535) as float in [0,1]
    clamp_to_minmax: bool = False  # usually not needed; decode already in range
    add_channel_dim: bool = True  # return (T,1,H,W) instead of (T,H,W)


from typing import List

def _find_depth_png_u16_dirs(root: str) -> List[str]:
    """
    Find episode directories containing depth pngs.
    Uses a cached index to avoid repeated filesystem scans.
    """
    cache_path = os.path.join(root, "depth_png_u16_index.json")

    # 1. Load cache if exists
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            return json.load(f)

    # 2. Slow path: build once
    print("[DepthDataset] Building depth_png_u16 directory index (one-time)...")

    pattern = os.path.join(root, "**", "depth_png_u16", "*.png")
    pngs = glob.glob(pattern, recursive=True)

    dirs = sorted({os.path.dirname(p) for p in pngs})

    # 3. Save cache
    with open(cache_path, "w") as f:
        json.dump(dirs, f)

    print(f"[DepthDataset] Cached {len(dirs)} episode dirs to {cache_path}")

    return dirs

def _sorted_pngs_in_dir(d: str) -> List[str]:
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith(".png")]
    files.sort()
    return files
import os, json
from typing import Optional, List, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
import imageio.v3 as iio

class DepthTemporalPNGDatasetPreload(Dataset):
    def __init__(
        self,
        root: str,
        seq_len: int = 2,
        stride: int = 1,
        step: int = 1,
        return_info: bool = False,
        drop_last: bool = True,
        normalize_0_1: bool = False,
        clamp_to_minmax: bool = True,
        add_channel_dim: bool = True,
        use_cache: bool = False,
        split="train",
        preload_dtype: torch.dtype = torch.float16,  # float16 saves RAM; use float32 if you need precision
        meta_json_name: Optional[str] = "depth_u16_png_metadata.json",
    ):
        assert seq_len >= 1 and stride >= 1 and step >= 1
        self.root = root
        self.seq_len = seq_len
        self.stride = stride
        self.step = step
        self.return_info = return_info
        self.drop_last = drop_last
        self.normalize_0_1 = normalize_0_1
        self.clamp_to_minmax = clamp_to_minmax
        self.add_channel_dim = add_channel_dim
        self.use_cache = use_cache
        self.preload_dtype = preload_dtype
        self.meta_json_name = meta_json_name    
        meta_json = os.path.join(root, meta_json_name)
        with open(meta_json, "r") as f:
            meta = json.load(f)

        self.vmin = float(meta["global_min"])
        self.vmax = float(meta["global_max"])
        if not np.isfinite(self.vmin) or not np.isfinite(self.vmax) or self.vmax <= self.vmin:
            raise ValueError(f"Bad min/max in meta_json: min={self.vmin}, max={self.vmax}")

        mean_path = meta["original_stats"]["mean_npy"]
        std_path  = meta["original_stats"]["std_npy"]
        self.mean = torch.from_numpy(np.load(mean_path)).float()  # (H,W) or scalar
        self.std  = torch.from_numpy(np.load(std_path)).float()


        # if self.use_cache:
        #     cache_path = "/scratch2/whwjdqls99/data_cache/depth_png_dataset_cache.npz"
        #     if os.path.exists(cache_path):

        # discover episodes
        self.episode_dirs = _find_depth_png_u16_dirs(root)
        
        # use the last three episodes for validation
        if split == "train":
            self.episode_dirs = self.episode_dirs[:-3]
        elif split == "val":
            self.episode_dirs = self.episode_dirs[-3:]
        else:
            raise ValueError(f"Unknown split: {split}")
        
        
        if len(self.episode_dirs) == 0:
            raise RuntimeError(f"No depth_png_u16 dirs found under: {root}")

        self.episodes: List[List[str]] = []
        print(f"Found {len(self.episode_dirs)} depth_png_u16 episode dirs.")
        
        print("going through dirs to find pngs...")
        for d in self.episode_dirs:
            pngs = _sorted_pngs_in_dir(d)
            if pngs:
                self.episodes.append(pngs)

        if not self.episodes:
            raise RuntimeError("Found depth_png_u16 dirs but no PNGs.")

        # # PRELOAD: decode all episodes once
        # print("Preloading and decoding all depth PNGs into memory...")
        # self.ep_tensors: List[torch.Tensor] = []
        # for ep_id, frames in enumerate(self.episodes):
        #     if ep_id % 10 == 0:
        #         print(f"Preloading episode {ep_id+1}/{len(self.episodes)} with {len(frames)} frames...")
        #     # decode all frames of episode
        #     decoded = []
        #     for p in frames:
        #         u16 = cv2.imread(p, cv2.IMREAD_UNCHANGED)  # (H,W), uint16
        #         if u16.dtype != np.uint16:
        #             raise ValueError(f"Expected uint16 PNG, got {u16.dtype} at {p}")

        #         if self.normalize_0_1:
        #             x = u16.astype(np.float32) / 65535.0
        #             if self.clamp_to_minmax:
        #                 x = np.clip(x, 0.0, 1.0)
        #         else:
        #             x = (u16.astype(np.float32) / 65535.0) * (self.vmax - self.vmin) + self.vmin
        #             if self.clamp_to_minmax:
        #                 x = np.clip(x, self.vmin, self.vmax)

        #         decoded.append(torch.from_numpy(x))

        #     ep = torch.stack(decoded, dim=0).float()  # (N,H,W)
        #     ep = (ep - self.mean) / self.std          # normalize once
        #     if self.add_channel_dim:
        #         ep = ep.unsqueeze(1)                  # (N,1,H,W)

        #     ep = ep.to(self.preload_dtype)            # shrink RAM if desired
        #     self.ep_tensors.append(ep)

        self.ep_tensors: List[torch.Tensor] = []

        scale = (self.vmax - self.vmin) / 65535.0
        offset = self.vmin

        for ep_id, frames in enumerate(self.episodes):
            if ep_id % 100 == 0:
                print(f"Preloading episode {ep_id+1}/{len(self.episodes)} with {len(frames)} frames...")

            # ---- read first frame to get shape ----
            u0 = cv2.imread(frames[0], cv2.IMREAD_UNCHANGED)
            if u0 is None or u0.dtype != np.uint16:
                raise ValueError(f"Bad uint16 PNG: {frames[0]}")

            H, W = u0.shape
            N = len(frames)

            # ---- preallocate contiguous array ----
            arr = np.empty((N, H, W), dtype=np.float32)

            # ---- decode loop ----
            for i, p in enumerate(frames):
                u16 = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                if u16 is None or u16.dtype != np.uint16:
                    raise ValueError(f"Expected uint16 PNG, got {u16.dtype} at {p}")

                x = u16.astype(np.float32, copy=False)

                if self.normalize_0_1:
                    x *= (1.0 / 65535.0)
                    if self.clamp_to_minmax:
                        np.clip(x, 0.0, 1.0, out=x)
                else:
                    x *= scale
                    x += offset
                    if self.clamp_to_minmax:
                        np.clip(x, self.vmin, self.vmax, out=x)

                arr[i] = x

            # ---- torch + normalize once ----
            ep = torch.from_numpy(arr)                 # (N,H,W)
            ep = (ep - self.mean) / self.std           # normalize once

            if self.add_channel_dim:
                ep = ep.unsqueeze(1)                   # (N,1,H,W)

            ep = ep.to(self.preload_dtype)             # float16 or float32
            self.ep_tensors.append(ep)
        # build index (same as before)
        self.index: List[Tuple[int,int]] = []
        for ep_id, ep in enumerate(self.ep_tensors):
            n = ep.shape[0]
            last_start = n - 1 - (self.seq_len - 1) * self.stride
            if last_start < 0:
                continue
            starts = range(0, last_start + 1, self.step) if self.drop_last else range(0, n, self.step)
            for s in starts:
                self.index.append((ep_id, s))

        if not self.index:
            raise RuntimeError("No valid temporal windows constructed.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int):
        ep_id, start = self.index[idx]
        ep = self.ep_tensors[ep_id]
        n = ep.shape[0]

        # gather indices for window
        js = []
        for k in range(self.seq_len):
            j = start + k * self.stride
            if j >= n:
                j = n - 1
            js.append(j)

        depth = ep[js]  # fancy indexing => (T,1,H,W) or (T,H,W)

        if not self.return_info:
            return depth

        paths = [self.episodes[ep_id][j] for j in js]
        info = {"episode_id": ep_id, "start_index": start, "stride": self.stride, "paths": paths}
        return depth, info


if __name__ == "__main__":
    import time
    import torch
    from torch.utils.data import DataLoader
    dataset = DepthTemporalPNGDatasetPreload(
        root="/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_only_depths",
        seq_len=2,
        stride=1,
        step=1,
        return_info=False,
        drop_last=True,
        normalize_0_1=False,
        clamp_to_minmax=True,
        add_channel_dim=True,
        split="val",
        preload_dtype=torch.float16,
        meta_json_name="depth_u16_png_metadata.json",
    )

    loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=False,
        num_workers=8,       # tune this (4–16 typical)
        pin_memory=True,
        persistent_workers=True,
    )

    # warm-up (important!)
    for i, batch in enumerate(loader):
        if i == 5:
            break

    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.time()
    n_batches = 0
    n_samples = 0

    for batch in loader:
        n_batches += 1
        n_samples += batch.shape[0]

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start

    print(f"Total samples: {n_samples}")
    print(f"Total batches: {n_batches}")
    print(f"Elapsed time: {elapsed:.2f} sec")
    print(f"Samples/sec: {n_samples / elapsed:.1f}")
    print(f"Batches/sec: {n_batches / elapsed:.2f}")