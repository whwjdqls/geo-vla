import os
import json
import glob
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
