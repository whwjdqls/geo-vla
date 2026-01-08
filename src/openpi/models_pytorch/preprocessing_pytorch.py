from collections.abc import Sequence
import logging

import torch
import torch.nn.functional as F

from openpi.shared import image_tools

logger = logging.getLogger("openpi")

# Constants moved from model.py
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    "right_wrist_0_rgb",
)

IMAGE_RESOLUTION = (224, 224)


def preprocess_observation_pytorch(
    observation,
    *,
    train: bool = False,
    image_keys: Sequence[str] = IMAGE_KEYS,
    image_resolution: tuple[int, int] = IMAGE_RESOLUTION,
):
    """Torch.compile-compatible version of preprocess_observation_pytorch with simplified type annotations.

    This function avoids complex type annotations that can cause torch.compile issues.
    """
    if not set(image_keys).issubset(observation.images):
        raise ValueError(f"images dict missing keys: expected {image_keys}, got {list(observation.images)}")

    batch_shape = observation.state.shape[:-1]

    out_images = {}
    for key in image_keys:
        image = observation.images[key]

        # TODO: This is a hack to handle both [B, C, H, W] and [B, H, W, C] formats
        # Handle both [B, C, H, W] and [B, H, W, C] formats
        is_channels_first = image.shape[1] == 3  # Check if channels are in dimension 1

        if is_channels_first:
            # Convert [B, C, H, W] to [B, H, W, C] for processing
            image = image.permute(0, 2, 3, 1)

        if image.shape[1:3] != image_resolution:
            logger.info(f"Resizing image {key} from {image.shape[1:3]} to {image_resolution}")
            image = image_tools.resize_with_pad_torch(image, *image_resolution)

        if train:
            # Convert from [-1, 1] to [0, 1] for PyTorch augmentations
            image = image / 2.0 + 0.5

            # Apply PyTorch-based augmentations
            if "wrist" not in key:
                # Geometric augmentations for non-wrist cameras
                height, width = image.shape[1:3]

                # Random crop and resize
                crop_height = int(height * 0.95)
                crop_width = int(width * 0.95)

                # Random crop
                max_h = height - crop_height
                max_w = width - crop_width
                if max_h > 0 and max_w > 0:
                    # Use tensor operations instead of .item() for torch.compile compatibility
                    start_h = torch.randint(0, max_h + 1, (1,), device=image.device)
                    start_w = torch.randint(0, max_w + 1, (1,), device=image.device)
                    image = image[:, start_h : start_h + crop_height, start_w : start_w + crop_width, :]

                # Resize back to original size
                image = torch.nn.functional.interpolate(
                    image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                    size=(height, width),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

                # Random rotation (small angles)
                # Use tensor operations instead of .item() for torch.compile compatibility
                angle = torch.rand(1, device=image.device) * 10 - 5  # Random angle between -5 and 5 degrees
                if torch.abs(angle) > 0.1:  # Only rotate if angle is significant
                    # Convert to radians
                    angle_rad = angle * torch.pi / 180.0

                    # Create rotation matrix
                    cos_a = torch.cos(angle_rad)
                    sin_a = torch.sin(angle_rad)

                    # Apply rotation using grid_sample
                    grid_x = torch.linspace(-1, 1, width, device=image.device)
                    grid_y = torch.linspace(-1, 1, height, device=image.device)

                    # Create meshgrid
                    grid_y, grid_x = torch.meshgrid(grid_y, grid_x, indexing="ij")

                    # Expand to batch dimension
                    grid_x = grid_x.unsqueeze(0).expand(image.shape[0], -1, -1)
                    grid_y = grid_y.unsqueeze(0).expand(image.shape[0], -1, -1)

                    # Apply rotation transformation
                    grid_x_rot = grid_x * cos_a - grid_y * sin_a
                    grid_y_rot = grid_x * sin_a + grid_y * cos_a

                    # Stack and reshape for grid_sample
                    grid = torch.stack([grid_x_rot, grid_y_rot], dim=-1)

                    image = torch.nn.functional.grid_sample(
                        image.permute(0, 3, 1, 2),  # [b, h, w, c] -> [b, c, h, w]
                        grid,
                        mode="bilinear",
                        padding_mode="zeros",
                        align_corners=False,
                    ).permute(0, 2, 3, 1)  # [b, c, h, w] -> [b, h, w, c]

            # Color augmentations for all cameras
            # Random brightness
            # Use tensor operations instead of .item() for torch.compile compatibility
            brightness_factor = 0.7 + torch.rand(1, device=image.device) * 0.6  # Random factor between 0.7 and 1.3
            image = image * brightness_factor

            # Random contrast
            # Use tensor operations instead of .item() for torch.compile compatibility
            contrast_factor = 0.6 + torch.rand(1, device=image.device) * 0.8  # Random factor between 0.6 and 1.4
            mean = image.mean(dim=[1, 2, 3], keepdim=True)
            image = (image - mean) * contrast_factor + mean

            # Random saturation (convert to HSV, modify S, convert back)
            # For simplicity, we'll just apply a random scaling to the color channels
            # Use tensor operations instead of .item() for torch.compile compatibility
            saturation_factor = 0.5 + torch.rand(1, device=image.device) * 1.0  # Random factor between 0.5 and 1.5
            gray = image.mean(dim=-1, keepdim=True)
            image = gray + (image - gray) * saturation_factor

            # Clamp values to [0, 1]
            image = torch.clamp(image, 0, 1)

            # Back to [-1, 1]
            image = image * 2.0 - 1.0

        # Convert back to [B, C, H, W] format if it was originally channels-first
        if is_channels_first:
            image = image.permute(0, 3, 1, 2)  # [B, H, W, C] -> [B, C, H, W]

        out_images[key] = image

    # obtain mask
    out_masks = {}
    for key in out_images:
        if key not in observation.image_masks:
            # do not mask by default
            out_masks[key] = torch.ones(batch_shape, dtype=torch.bool, device=observation.state.device)
        else:
            out_masks[key] = observation.image_masks[key]

    # Create a simple object with the required attributes instead of using the complex Observation class
    class SimpleProcessedObservation:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

    return SimpleProcessedObservation(
        images=out_images,
        image_masks=out_masks,
        state=observation.state,
        tokenized_prompt=observation.tokenized_prompt,
        tokenized_prompt_mask=observation.tokenized_prompt_mask,
        token_ar_mask=observation.token_ar_mask,
        token_loss_mask=observation.token_loss_mask,
    )


def preprocess_point_cloud_pytorch(point_cloud, stats,*, train: bool = False):
    """
    Docstring for preprocess_point_cloud_pytorch
    
    :param point_cloud: (BS, T, N, D) tensor representing the point cloud data
    :param train: Description
    :type train: bool
    """
    if stats is None:
        raise ValueError("Point cloud stats must be provided for preprocessing.")
    
    mean = torch.tensor(stats["point_cloud"]["mean"], device=point_cloud.device).squeeze()  # (D,)
    std = torch.tensor(stats["point_cloud"]["std"], device=point_cloud.device).squeeze()  # (D,)
    mean = mean.to(point_cloud.dtype)
    std = std.to(point_cloud.dtype)
    point_cloud = (point_cloud - mean) / std
    
    # input point cloud
    input_point_cloud = point_cloud[:, 0, :, :].unsqueeze(1)  # (BS, 1, N, D)
    output_point_delta = point_cloud[:, 1:, :, :] - point_cloud[:, :-1, :, :]  # (BS, T-1, N, D)

    return input_point_cloud, output_point_delta


def _coerce_depth_sequence_to_b_t_1_h_w(depth_image: torch.Tensor) -> torch.Tensor:
    """Coerce depth tensors into a consistent (B, T, 1, H, W) layout.

    Supported common layouts:
    - (B, T, H, W)
    - (B, T, H, W, 1)
    - (B, T, 1, H, W)
    """
    if depth_image.ndim == 4:
        # (B, T, H, W) -> (B, T, 1, H, W)
        return depth_image.unsqueeze(2)
    if depth_image.ndim == 5:
        # (B, T, H, W, 1) -> (B, T, 1, H, W)
        if depth_image.shape[-1] == 1:
            return depth_image.permute(0, 1, 4, 2, 3)
        # Already (B, T, 1, H, W)
        if depth_image.shape[2] == 1:
            return depth_image
    raise ValueError(f"Unsupported depth_image shape: {tuple(depth_image.shape)}")


def _depth_to_latents_placeholder(depth_b_t_1_h_w: torch.Tensor) -> torch.Tensor:
    """Placeholder depth->latent model.

    Assumes input depth is (B, T, 1, 256, 256) and produces latents (B, T, 16, 16, 4).

    Implementation: bilinear downsample to 16x16 and repeat channel to 4.
    """
    bsize, t, c, h, w = depth_b_t_1_h_w.shape
    if c != 1:
        raise ValueError(f"Expected depth channel dim=1, got {c}")

    x = depth_b_t_1_h_w.to(torch.float32)

    # Simple normalization heuristics for integer depth.
    if x.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        x = x.to(torch.float32)
    # If values look like raw uint16 mm-scale, user will replace this placeholder anyway.

    x = x.reshape(bsize * t, 1, h, w)
    x = F.interpolate(x, size=(16, 16), mode="bilinear", align_corners=False)
    x = x.repeat(1, 4, 1, 1)  # (B*T, 4, 16, 16)
    x = x.reshape(bsize, t, 4, 16, 16)
    # (B, T, 4, 16, 16) -> (B, T, 16, 16, 4)
    return x.permute(0, 1, 3, 4, 2)


def preprocess_depth_pytorch(depth_image: torch.Tensor, *, train: bool = False):
    """Preprocess depth sequence into aux tokens.

    Supports two formats:
    1) Precomputed depth latents (fast path):
       - (B, T, 1024) where 1024 = 16*16*4
       - (B, T, 16, 16, 4)
       - (B, T, 4, 16, 16)
    2) Raw depth frames (slow placeholder path):
       - (B, T, 256, 256)
       - (B, T, 256, 256, 1)
       - (B, T, 1, 256, 256)

    Returns:
      input_depth_token: (B, 1, 1024) for the t=0 depth latent
      target_depth_tokens: (B, T-1, 1024) for future depth latents
    """

    # Fast path: already-tokenized latents.
    if depth_image.ndim == 3:
        # (B, T, 1024)
        if depth_image.shape[-1] != 16 * 16 * 4:
            raise ValueError(
                f"Expected depth latents last dim=1024, got {depth_image.shape[-1]} with shape {tuple(depth_image.shape)}"
            )
        tokens = depth_image
    elif depth_image.ndim == 5:
        # (B, T, 16, 16, 4)
        if depth_image.shape[-3:] == (16, 16, 4):
            tokens = depth_image.reshape(depth_image.shape[0], depth_image.shape[1], 16 * 16 * 4)
        # (B, T, 4, 16, 16)
        elif depth_image.shape[-3:] == (4, 16, 16):
            x = depth_image.permute(0, 1, 3, 4, 2)  # -> (B, T, 16, 16, 4)
            tokens = x.reshape(x.shape[0], x.shape[1], 16 * 16 * 4)
        else:
            # Might be raw depth frames (B, T, H, W, 1) or (B, T, 1, H, W)
            depth_b_t_1_h_w = _coerce_depth_sequence_to_b_t_1_h_w(depth_image)
            latents_b_t_16_16_4 = _depth_to_latents_placeholder(depth_b_t_1_h_w)
            tokens = latents_b_t_16_16_4.reshape(latents_b_t_16_16_4.shape[0], latents_b_t_16_16_4.shape[1], 16 * 16 * 4)
    elif depth_image.ndim == 4:
        # Raw frames (B, T, H, W)
        depth_b_t_1_h_w = _coerce_depth_sequence_to_b_t_1_h_w(depth_image)
        latents_b_t_16_16_4 = _depth_to_latents_placeholder(depth_b_t_1_h_w)
        tokens = latents_b_t_16_16_4.reshape(latents_b_t_16_16_4.shape[0], latents_b_t_16_16_4.shape[1], 16 * 16 * 4)
    else:
        raise ValueError(f"Unsupported depth input shape: {tuple(depth_image.shape)}")

    input_depth_token = tokens[:, 0:1, :]
    target_depth_tokens = tokens[:, 1:, :]
    return input_depth_token, target_depth_tokens