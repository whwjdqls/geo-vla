# import numpy as np
# import glob
# # import tmux

# npy_files = glob.glob('/scratch2/jisoo6687/libero/depth_map/*/*/*/agentview/depth_npy/*.npy')
# print(f"Found {len(npy_files)} .npy files.")




# global_min = float('inf')
# global_max = float('-inf')
# data_list = []
# # for file in tmux.tmux(npy_files):
# for i, file in enumerate(npy_files):
#     if i % 1000 == 0:
#         print(f"Processing file {i}/{len(npy_files)}: {file}")
#     data = np.load(file) # (256,256,1)
#     local_min = data.min()
#     local_max = data.max()
    
#     if local_min < global_min:
#         global_min = local_min
#     if local_max > global_max:
#         global_max = local_max
        
#     data_list.append(data)
#     # break
    
    
# data_array = np.stack(data_list, axis=0)  # (N, 256, 256, 1)
# data_mean = data_array.mean(axis=0).squeeze()  # (256, 256)
# data_std = data_array.std(axis=0).squeeze()    # (256, 256)
# print(f"Data mean shape: {data_mean.shape}, std shape: {data_std.shape}")
# print(f"Global min: {global_min}, Global max: {global_max}")

# import json
# stats = {
#     "min": float(global_min),
#     "max": float(global_max),
#     "mean": data_mean.tolist(),
#     "std": data_std.tolist(),
# }
# with open('/scratch2/whwjdqls99/depth_encoder/depth_data_stats.json', 'w') as f:
#     json.dump(stats, f)
    
    
import numpy as np
import glob
import os
import json

npy_files = glob.glob('/scratch2/jisoo6687/libero/depth_map/*/*/*/agentview/depth_npy/*.npy')
print(f"Found {len(npy_files)} .npy files.")

if len(npy_files) == 0:
    raise RuntimeError("No .npy files found. Check your glob path.")

# Initialize streaming stats (per-pixel)
# We'll store in float64 for numeric stability, then cast later if you want.
count = 0
mean = None
M2 = None  # sum of squared deviations

global_min = np.inf
global_max = -np.inf

for i, file in enumerate(npy_files):
    if i % 1000 == 0:
        print(f"Processing {i}/{len(npy_files)}: {file}")

    x = np.load(file)  # (256,256,1) or (256,256)
    x = np.asarray(x)
    if x.ndim == 3 and x.shape[-1] == 1:
        x = x[..., 0]
    if x.shape != (256, 256):
        raise ValueError(f"Unexpected shape {x.shape} in {file}")

    # Update global min/max
    local_min = x.min()
    local_max = x.max()
    if local_min < global_min:
        global_min = float(local_min)
    if local_max > global_max:
        global_max = float(local_max)

    # Welford update
    x = x.astype(np.float64, copy=False)

    if mean is None:
        mean = np.zeros_like(x, dtype=np.float64)
        M2 = np.zeros_like(x, dtype=np.float64)

    count += 1
    delta = x - mean
    mean += delta / count
    delta2 = x - mean
    M2 += delta * delta2

# Finalize
if count < 2:
    std = np.zeros_like(mean)
else:
    var = M2 / (count - 1)   # sample variance
    std = np.sqrt(var)

print("Done.")
print(f"Count: {count}")
print(f"Global min: {global_min}, Global max: {global_max}")
print(f"Mean/std shapes: {mean.shape}, {std.shape}")

out_dir = "/scratch2/whwjdqls99/depth_encoder"
os.makedirs(out_dir, exist_ok=True)

# Save mean/std efficiently
mean_path = os.path.join(out_dir, "depth_mean.npy")
std_path  = os.path.join(out_dir, "depth_std.npy")
np.save(mean_path, mean.astype(np.float32))
np.save(std_path, std.astype(np.float32))

# Save small JSON metadata (recommended)
stats = {
    "count": int(count),
    "min": float(global_min),
    "max": float(global_max),
    "mean_npy": mean_path,
    "std_npy": std_path,
}
json_path = os.path.join(out_dir, "depth_data_stats_.json")
with open(json_path, "w") as f:
    json.dump(stats, f, indent=2)

print(f"Saved:\n- {json_path}\n- {mean_path}\n- {std_path}")
