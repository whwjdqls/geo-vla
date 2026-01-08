import json
import numpy as np
import imageio.v3 as iio

META_PATH = "/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_depth/depth_u16_png_metadata.json"

def decode_depth_u16_png(png_path: str, meta_path: str = META_PATH) -> np.ndarray:
    # Load min/max used for encoding
    with open(meta_path, "r") as f:
        meta = json.load(f)
    vmin = float(meta["global_min"])
    vmax = float(meta["global_max"])

    # Read uint16 PNG
    u16 = iio.imread(png_path)  # shape (256,256), dtype=uint16
    if u16.dtype != np.uint16:
        # some pipelines may return uint8 if the PNG wasn't saved as 16-bit
        raise ValueError(f"Expected uint16 PNG, got dtype={u16.dtype}, shape={u16.shape}")

    # Decode back to float32 (approx original)
    x = (u16.astype(np.float32) / 65535.0) * (vmax - vmin) + vmin

    # Return in original (256,256,1) shape if you want
    return x[..., None]  # (256,256,1) float32

# Example
png_path = "/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_depth/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/demo_0/agentview/depth_png_u16/0000.png"
depth = decode_depth_u16_png(png_path)
print(depth.shape, depth.dtype, depth.min(), depth.max())


numpy_path = "/scratch2/jisoo6687/libero/depth_map/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it/demo_0/agentview/depth_npy/0000.npy"
npy_depth = np.load(numpy_path)
print(npy_depth.shape, npy_depth.dtype, npy_depth.min(), npy_depth.max())