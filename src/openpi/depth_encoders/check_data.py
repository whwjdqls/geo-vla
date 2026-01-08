import numpy as np
import matplotlib.pyplot as plt

# -----------------------
# Load depth
# -----------------------
a = np.load(
    "/scratch2/jisoo6687/libero/depth_map/libero_goal/open_the_middle_drawer_of_the_cabinet/demo_0/agentview/depth_npy/0000.npy"
)

z = a.squeeze().astype(np.float32)  # (256, 256), z-buffer in [0,1]
z = z[::-1, ::-1]

# -----------------------
# Convert z-buffer → metric depth (meters)
# -----------------------
NEAR = 0.01    # typical MuJoCo default
FAR  = 50.0    # typical MuJoCo default

depth_m = (NEAR * FAR) / (FAR - z * (FAR - NEAR))

# -----------------------
# Visualization (robust)
# -----------------------
valid = np.isfinite(depth_m)
vmin, vmax = np.percentile(depth_m[valid], [1, 99])

plt.figure(figsize=(4, 4))
im = plt.imshow(
    depth_m,
    cmap="turbo",
    vmin=vmin,
    vmax=vmax
)
plt.axis("off")
plt.tight_layout(pad=0)

plt.colorbar(im, fraction=0.046, pad=0.04, label="Depth (meters)")

plt.savefig(
    "depth_metric_colormap.png",
    dpi=300,
    bbox_inches="tight",
    pad_inches=0
)
plt.close()
