#!/usr/bin/env python3
"""
Convert float32 depth .npy files (256,256,1) -> uint16 PNG (256,256) using global min/max.

- Reads global min/max from a JSON stats file (produced by your script).
- Saves 16-bit PNGs (lossless PNG compression, but depth values are quantized to 65536 levels).
- Writes a metadata JSON to the output root so you can invert the mapping later.

Output root:
  /scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_depth
"""

import os
import glob
import json
import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
SRC_GLOB = "/scratch2/jisoo6687/libero/depth_map/*/*/*/agentview/depth_npy/*.npy"
OUT_ROOT = "/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_depth"

# Your previously saved stats JSON (from the code you posted)
STATS_JSON = "/scratch2/whwjdqls99/depth_encoder/datasets/depth_data_stats_.json"

# Compression level for PNG: 0 (fastest, largest) ... 9 (slowest, smallest)
PNG_COMPRESS_LEVEL = 6

# If True, recreate a directory structure under OUT_ROOT:
# depth_png/<task>/<something>/<something>/agentview/depth_png_u16/<file>.png
MIRROR_STRUCTURE = True

# ----------------------------
# Helpers
# ----------------------------
def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def load_global_minmax(stats_json: str):
    with open(stats_json, "r") as f:
        d = json.load(f)
    vmin = float(d["min"])
    vmax = float(d["max"])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        raise ValueError(f"Bad min/max in {stats_json}: min={vmin}, max={vmax}")
    return vmin, vmax, d

def npy_to_float32_2d(path: str) -> np.ndarray:
    x = np.load(path)
    x = np.asarray(x)
    # flip 
    x = x[::-1, ::-1]
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.shape != (256, 256):
        raise ValueError(f"Unexpected shape {x.shape} in {path}")
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return x

def float_to_u16(x: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    # Clip then map to [0, 65535]
    x = np.clip(x, vmin, vmax)
    y = (x - vmin) / (vmax - vmin)  # [0,1]
    u16 = np.round(y * 65535.0).astype(np.uint16)
    return u16

def make_out_path(in_path: str) -> str:
    base = os.path.splitext(os.path.basename(in_path))[0] + ".png"
    if not MIRROR_STRUCTURE:
        return os.path.join(OUT_ROOT, base)

    # Try to mirror: .../depth_map/<A>/<B>/<C>/agentview/depth_npy/<name>.npy
    # into:          OUT_ROOT/<A>/<B>/<C>/agentview/depth_png_u16/<name>.png
    # Find the ".../depth_map/" anchor
    parts = in_path.split(os.sep)
    try:
        idx = parts.index("depth_map")
        rel = parts[idx + 1 : ]  # <A>/<B>/<C>/agentview/depth_npy/<file>
    except ValueError:
        # Fallback: just dump flat
        return os.path.join(OUT_ROOT, base)

    # Replace trailing ".../depth_npy/<file>.npy" with ".../depth_png_u16/<file>.png"
    # rel looks like: [A, B, C, "agentview", "depth_npy", "<file>.npy"]
    if len(rel) < 6 or rel[-2] != "depth_npy":
        # unexpected layout; fallback flat
        return os.path.join(OUT_ROOT, base)

    out_rel_dir = os.path.join(*rel[:-2], "depth_png_u16")  # A/B/C/agentview/depth_png_u16
    return os.path.join(OUT_ROOT, out_rel_dir, base)

# ----------------------------
# Main
# ----------------------------
def main():
    ensure_dir(OUT_ROOT)

    vmin, vmax, stats_obj = load_global_minmax(STATS_JSON)
    print(f"Loaded global min/max from {STATS_JSON}")
    print(f"  min={vmin}, max={vmax}")

    files = sorted(glob.glob(SRC_GLOB))
    print(f"Found {len(files)} .npy files.")
    if not files:
        raise RuntimeError("No .npy files found. Check SRC_GLOB.")

    # Write metadata for decoding later
    meta = {
        "format": "uint16_png_scaled_from_float32",
        "src_glob": SRC_GLOB,
        "out_root": OUT_ROOT,
        "global_min": vmin,
        "global_max": vmax,
        "png_bit_depth": 16,
        "mapping": "u16 = round( clip(x,min,max)-min / (max-min) * 65535 )",
        "inverse": "x = (u16/65535)*(max-min) + min",
        "stats_json_used": STATS_JSON,
        "original_stats": {k: stats_obj.get(k) for k in ("count", "min", "max", "mean_npy", "std_npy")},
    }
    meta_path = os.path.join(OUT_ROOT, "depth_u16_png_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("Wrote:", meta_path)

    # Convert
    num_written = 0
    for i, path in enumerate(tqdm(files, desc="Converting")):
        x = npy_to_float32_2d(path)
        u16 = float_to_u16(x, vmin, vmax)

        out_path = make_out_path(path)
        ensure_dir(os.path.dirname(out_path))

        # imageio writes uint16 PNG correctly
        iio.imwrite(out_path, u16, extension=".png", compress_level=PNG_COMPRESS_LEVEL)
        num_written += 1

        if i % 2000 == 0 and i > 0:
            tqdm.write(f"Progress: {i}/{len(files)}  last={out_path}")

    print("Done.")
    print(f"Wrote {num_written} PNG files under:\n  {OUT_ROOT}")

if __name__ == "__main__":
    main()
