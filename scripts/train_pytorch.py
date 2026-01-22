"""
PyTorch training entrypoint for PI0/PI05 with multi-GPU and multi-node (DDP) support.
This script mirrors the behavior of the JAX trainer (`scripts/train.py`) but runs
entirely in PyTorch using the `PI0Pytorch` model and your existing config/data
pipeline from `src/openpi/training/config.py` and `src/openpi/training/data_loader.py`.

Usage
Single GPU:
  python scripts/train_pytorch.py <config_name> --exp_name <run_name> --save_interval <interval>
  Example:
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test
  python scripts/train_pytorch.py debug --exp_name pytorch_ddp_test --resume  # Resume from latest checkpoint
Multi-GPU (single node):
  torchrun --standalone --nnodes=1 --nproc_per_node=<num_gpus> scripts/train_pytorch.py <config_name> --exp_name <run_name>
  Example:
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test
  torchrun --standalone --nnodes=1 --nproc_per_node=2 scripts/train_pytorch.py pi0_aloha_sim --exp_name pytorch_ddp_test --resume
Multi-Node Training:
	torchrun \
    --nnodes=<num_nodes> --nproc_per_node=<gpus_per_node> --node_rank=<rank_of_node> \
    --master_addr=<master_ip> --master_port=<port> \
    scripts/train_pytorch.py <config_name> --exp_name=<run_name> --save_interval <interval>

"""

import dataclasses
import gc
import logging
import os
import platform
import pathlib
import shutil
import time


# Suppress JAX and TensorFlow GPU initialization and logging
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data

import openpi.depth_encoders.load_encoders as _depth_encoders 


def render_point_positions_video(
    positions: np.ndarray,
    out_path: pathlib.Path,
    *,
    fps: int = 10,
    point_size: float = 2.0,
    axis_mode: str = "global",
) -> None:
    """Render a point cloud sequence video.

    Args:
      positions: (T, N, 3) or (T+1, N, 3) absolute positions over time.
      axis_mode:
        - "global": fixed limits from all frames (default)
        - "p0": fixed limits from first frame only
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import imageio.v2 as imageio

    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(f"positions must be (T, N, 3), got {positions.shape}")
    if axis_mode not in {"global", "p0"}:
        raise ValueError(f"axis_mode must be 'global' or 'p0', got {axis_mode!r}")

    ref = positions[0:1] if axis_mode == "p0" else positions
    mins = ref.reshape(-1, 3).min(axis=0)
    maxs = ref.reshape(-1, 3).max(axis=0)
    center = 0.5 * (mins + maxs)
    radius = float(np.max(maxs - mins) * 0.55 + 1e-6)

    fig = plt.figure(figsize=(6, 6), dpi=160)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=20, azim=-60)

    scat = ax.scatter(
        positions[0, :, 0],
        positions[0, :, 1],
        positions[0, :, 2],
        s=point_size,
        c="tab:blue",
        depthshade=False,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(str(out_path), fps=fps) as writer:
        for t in range(positions.shape[0]):
            pts = positions[t]
            scat._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
            ax.set_title(f"t={t:02d}")
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            frame = rgba[:, :, :3].copy()
            writer.append_data(frame)

    plt.close(fig)


def save_aux_videos_and_log_wandb(
    aux_output,
    *,
    out_dir: pathlib.Path,
    global_step: int,
    wandb_enabled: bool,
    fps: int = 10,
    point_size: float = 2.0,
    axis_mode: str = "global",
) -> None:
    if aux_output is None:
        return

    pred = aux_output.get("pred_point_tracks") if isinstance(aux_output, dict) else None
    gt = aux_output.get("gt_point_tracks") if isinstance(aux_output, dict) else None
    if pred is None:
        return

    pred_np = pred.detach().to(device="cpu", dtype=torch.float32).numpy()
    gt_np = None
    if gt is not None:
        gt_np = gt.detach().to(device="cpu", dtype=torch.float32).numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    videos_to_log = {}

    num_to_save = min(3, pred_np.shape[0])
    for i in range(num_to_save):
        pred_path = out_dir / f"pred_{i}.mp4"
        print(f"Saving pred video to {pred_path}")
        render_point_positions_video(pred_np[i], pred_path, fps=fps, point_size=point_size, axis_mode=axis_mode)
        if wandb_enabled:
            # When providing a file path, wandb uses the video's embedded fps.
            # Passing `fps=` triggers a warning and has no effect.
            videos_to_log[f"aux/pred_{i}"] = wandb.Video(str(pred_path), format="mp4")

        if gt_np is not None:
            gt_path = out_dir / f"gt_{i}.mp4"
            render_point_positions_video(gt_np[i], gt_path, fps=fps, point_size=point_size, axis_mode=axis_mode)
            if wandb_enabled:
                videos_to_log[f"aux/gt_{i}"] = wandb.Video(str(gt_path), format="mp4")

    if wandb_enabled and len(videos_to_log) > 0:
        wandb.log(videos_to_log, step=global_step)

def tree_map(fn, *trees):
    """A simple recursive tree_map for nested dictionaries and lists. Supports multiple trees."""
    first = trees[0]
    if first is None:
        return None
    if isinstance(first, dict):
        return {k: tree_map(fn, *(t[k] for t in trees)) for k in first}
    if isinstance(first, list | tuple):
        return type(first)(tree_map(fn, *(t[i] for t in trees)) for i in range(len(first)))
    if dataclasses.is_dataclass(first):
        return type(first)(
            **{f.name: tree_map(fn, *(getattr(t, f.name) for t in trees)) for f in dataclasses.fields(first)}
        )
    return fn(*trees)


def init_logging():
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class CustomFormatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = CustomFormatter(
        fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)
    else:
        logger.handlers[0].setFormatter(formatter)


def init_wandb(config: _config.TrainConfig, *, resuming: bool, enabled: bool = True):
    """Initialize wandb logging."""
    if not enabled:
        wandb.init(mode="disabled")
        return

    ckpt_dir = config.checkpoint_dir
    print("ckpt_dir:", ckpt_dir)
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory {ckpt_dir} does not exist.")

    if resuming:
        run_id = (ckpt_dir / "wandb_id.txt").read_text().strip()
        wandb.init(id=run_id, resume="must", project=config.project_name)
    else:
        wandb.init(
            name=config.exp_name,
            config=dataclasses.asdict(config),
            project=config.project_name,
        )
        (ckpt_dir / "wandb_id.txt").write_text(wandb.run.id)


def init_tensorboard(config: _config.TrainConfig, enabled: bool = True):
    """Initialize tensorboard logging."""
    if not enabled:
        return None
    
    tb_log_dir = config.checkpoint_dir / "tensorboard"
    tb_log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_log_dir))
    logging.info(f"Tensorboard logging to: {tb_log_dir}")
    print(f"[INFO] Tensorboard logging to: {tb_log_dir}")
    return writer


def setup_ddp():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    if use_ddp and not torch.distributed.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        torch.distributed.init_process_group(backend=backend, init_method="env://")

        # Set up debugging environment variables for DDP issues
        if os.environ.get("TORCH_DISTRIBUTED_DEBUG") is None:
            os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp():
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def set_seed(seed: int, local_rank: int):
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()


def get_model_state_dict(model):
    """Get state dict from model, handling DDP wrapper."""
    return (
        model.module.state_dict()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.state_dict()
    )


def load_safetensors_state_dict(model, path, device, *, ignore_key_substrings=None):
    """Load a safetensors state dict with optional key filtering."""
    state_dict = safetensors.torch.load_file(path, device=str(device))
    if ignore_key_substrings:
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if not any(substr in key for substr in ignore_key_substrings)
        }

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if ignore_key_substrings:
        missing = [key for key in missing if not any(substr in key for substr in ignore_key_substrings)]
        unexpected = [key for key in unexpected if not any(substr in key for substr in ignore_key_substrings)]

    if missing:
        preview = ", ".join(missing[:20])
        suffix = " ..." if len(missing) > 20 else ""
        logging.info("Missing %d keys when loading weights: %s%s", len(missing), preview, suffix)
    if unexpected:
        preview = ", ".join(unexpected[:20])
        suffix = " ..." if len(unexpected) > 20 else ""
        logging.info("Unexpected %d keys when loading weights: %s%s", len(unexpected), preview, suffix)


def get_model_parameters(model):
    """Get parameters from model, handling DDP wrapper."""
    return (
        model.module.parameters()
        if isinstance(model, torch.nn.parallel.DistributedDataParallel)
        else model.parameters()
    )


def save_checkpoint(model, optimizer, global_step, config, is_main, data_config, *, aux_output=None):
    """Save a checkpoint with model state, optimizer state, and metadata."""
    if not is_main:
        return

    # Only save if it's time to save or if it's the final step
    if (global_step % config.save_interval == 0 and global_step > 0) or global_step == config.num_train_steps - 1:
        # Create temporary directory for atomic checkpoint saving
        final_ckpt_dir = config.checkpoint_dir / f"{global_step}"
        tmp_ckpt_dir = config.checkpoint_dir / f"tmp_{global_step}"

        # Remove any existing temp directory and create new one
        if tmp_ckpt_dir.exists():
            shutil.rmtree(tmp_ckpt_dir)
        tmp_ckpt_dir.mkdir(parents=True, exist_ok=True)

        # Save model state using safetensors (handle shared tensors)
        model_to_save = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        safetensors.torch.save_model(model_to_save, tmp_ckpt_dir / "model.safetensors")

        # Save optimizer state using PyTorch format
        torch.save(optimizer.state_dict(), tmp_ckpt_dir / "optimizer.pt")

        # Save training metadata (avoid saving full config to prevent JAX/Flax compatibility issues)
        metadata = {
            "global_step": global_step,
            "config": dataclasses.asdict(config),
            "timestamp": time.time(),
        }
        torch.save(metadata, tmp_ckpt_dir / "metadata.pt")

        # save norm stats
        norm_stats = data_config.norm_stats
        if norm_stats is not None and data_config.asset_id is not None:
            _normalize.save(tmp_ckpt_dir / "assets" / data_config.asset_id, norm_stats)

        # Atomically move temp directory to final location
        if final_ckpt_dir.exists():
            shutil.rmtree(final_ckpt_dir)
        tmp_ckpt_dir.rename(final_ckpt_dir)

        logging.info(f"Saved checkpoint at step {global_step} -> {final_ckpt_dir}")

        # Optionally save auxiliary point-track videos and log first 3 to wandb.
        print(f"save_aux: {getattr(config, 'save_aux', False)}")
        if getattr(config, "save_aux", False):
            # try:
            print(f"Saving aux videos to {final_ckpt_dir / 'aux_videos'}")
            save_aux_videos_and_log_wandb(
                aux_output,
                out_dir=final_ckpt_dir / "aux_videos",
                global_step=global_step,
                wandb_enabled=bool(config.wandb_enabled),
            )
            # except Exception:
            #     logging.exception("Failed to save/log aux videos at step %s", global_step)

        # Log checkpoint to wandb
        if config.wandb_enabled:
            wandb.log({"checkpoint_step": global_step}, step=global_step)


def load_checkpoint(model, optimizer, checkpoint_dir, device):
    """Load the latest checkpoint and return the global step."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]

    if not checkpoint_steps:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")

    latest_step = max(checkpoint_steps)
    ckpt_dir = checkpoint_dir / f"{latest_step}"

    # Clear memory before loading checkpoints
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "before_loading_checkpoint")

    try:
        # Load model state with error handling
        logging.info("Loading model state...")
        safetensors_path = ckpt_dir / "model.safetensors"

        if safetensors_path.exists():
            model_to_load = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
            # load_safetensors_state_dict(model_to_load, safetensors_path, device, ignore_key_substrings=("aux",))
            load_safetensors_state_dict(model_to_load, safetensors_path, device)
            logging.info("Loaded model state from safetensors format")
        else:
            raise FileNotFoundError(f"No model checkpoint found at {ckpt_dir}")

        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_model")

        # Load optimizer state with error handling
        logging.info("Loading optimizer state...")
        optimizer_path = ckpt_dir / "optimizer.pt"

        if optimizer_path.exists():
            optimizer_state_dict = torch.load(optimizer_path, map_location=device, weights_only=False)
            logging.info("Loaded optimizer state from pt format")
        else:
            raise FileNotFoundError(f"No optimizer checkpoint found at {ckpt_dir}")

        optimizer.load_state_dict(optimizer_state_dict)
        del optimizer_state_dict
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_optimizer")

        # Load metadata
        logging.info("Loading metadata...")
        metadata = torch.load(ckpt_dir / "metadata.pt", map_location=device, weights_only=False)
        global_step = metadata.get("global_step", latest_step)
        del metadata
        torch.cuda.empty_cache()
        gc.collect()
        log_memory_usage(device, latest_step, "after_loading_metadata")

        logging.info(f"Successfully loaded all checkpoint components from step {latest_step}")
        return global_step

    except RuntimeError as e:
        if "out of memory" in str(e):
            # Clear memory and provide detailed error message
            torch.cuda.empty_cache()
            gc.collect()
            logging.error(f"Out of memory error while loading checkpoint: {e!s}")
            log_memory_usage(device, latest_step, "after_oom_error")
            raise RuntimeError(
                "Out of memory while loading checkpoint. Try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
            ) from e
        raise


def get_latest_checkpoint_step(checkpoint_dir):
    """Get the latest checkpoint step number from a checkpoint directory."""
    checkpoint_steps = [
        int(d.name)
        for d in checkpoint_dir.iterdir()
        if d.is_dir() and d.name.isdigit() and not d.name.startswith("tmp_")
    ]
    return max(checkpoint_steps) if checkpoint_steps else None


def log_memory_usage(device, step, phase="unknown"):
    """Log detailed memory usage information."""
    if not torch.cuda.is_available():
        return

    memory_allocated = torch.cuda.memory_allocated(device) / 1e9
    memory_reserved = torch.cuda.memory_reserved(device) / 1e9
    memory_free = torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    memory_free = memory_free / 1e9

    # Get more detailed memory info
    memory_stats = torch.cuda.memory_stats(device)
    max_memory_allocated = memory_stats.get("allocated_bytes.all.peak", 0) / 1e9
    max_memory_reserved = memory_stats.get("reserved_bytes.all.peak", 0) / 1e9

    # Get DDP info if available
    ddp_info = ""
    if dist.is_initialized():
        ddp_info = f" | DDP: rank={dist.get_rank()}, world_size={dist.get_world_size()}"

    logging.info(
        f"Step {step} ({phase}): GPU memory - allocated: {memory_allocated:.2f}GB, reserved: {memory_reserved:.2f}GB, free: {memory_free:.2f}GB, peak_allocated: {max_memory_allocated:.2f}GB, peak_reserved: {max_memory_reserved:.2f}GB{ddp_info}"
    )


def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)

    # Initialize checkpoint directory and wandb
    resuming = False
    if config.resume:
        # Find checkpoint directory based on experiment name
        exp_checkpoint_dir = config.checkpoint_dir
        if exp_checkpoint_dir.exists():
            # Use validation to find the latest working checkpoint
            latest_step = get_latest_checkpoint_step(exp_checkpoint_dir)
            if latest_step is not None:
                resuming = True
                logging.info(
                    f"Resuming from experiment checkpoint directory: {exp_checkpoint_dir} at step {latest_step}"
                )
            else:
                raise FileNotFoundError(f"No valid checkpoints found in {exp_checkpoint_dir} for resume")
        else:
            raise FileNotFoundError(f"Experiment checkpoint directory {exp_checkpoint_dir} does not exist for resume")
    elif config.overwrite and config.checkpoint_dir.exists():
        shutil.rmtree(config.checkpoint_dir)
        logging.info(f"Overwriting checkpoint directory: {config.checkpoint_dir}")

    # Create checkpoint directory with experiment name
    if not resuming:
        # For new runs, create experiment-specific checkpoint directory
        exp_checkpoint_dir = config.checkpoint_dir
        exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")
    else:
        # For resume, checkpoint_dir is already set to the experiment directory
        logging.info(f"Using existing experiment checkpoint directory: {config.checkpoint_dir}")

    # Initialize wandb and tensorboard (only on main process)
    tb_writer = None
    if is_main:
        init_wandb(config, resuming=resuming, enabled=config.wandb_enabled)
        tb_writer = init_tensorboard(config, enabled=True)

    # Build data loader using the unified data loader
    # Calculate effective batch size per GPU for DDP
    # For N GPUs, each GPU should get batch_size/N samples, so total across all GPUs is batch_size
    world_size = torch.distributed.get_world_size() if use_ddp else 1
    effective_batch_size = config.batch_size // world_size
    logging.info(
        f"Using batch size per GPU: {effective_batch_size} (total batch size across {world_size} GPUs: {config.batch_size})"
    )

    # Pass the original batch size to data loader - it will handle DDP splitting internally
    loader, data_config = build_datasets(config)
    # exit()
    # Log sample images to wandb on first batch
    if is_main and config.wandb_enabled and not resuming:
        # Create a separate data loader for sample batch to avoid consuming the main loader
        sample_data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=False)
        sample_batch = next(iter(sample_data_loader))
        # Convert observation and actions to torch tensors
        observation, actions = sample_batch
        sample_batch = tree_map(lambda x: torch.as_tensor(x) if isinstance(x, np.ndarray | list) else x, observation.to_dict())
        sample_batch["actions"] = torch.as_tensor(actions)

        # Create sample images for wandb
        images_to_log = []
        # Get batch size from the first image tensor
        batch_size = next(iter(sample_batch["image"].values())).shape[0]
        for i in range(min(5, batch_size)):
            # Concatenate all camera views horizontally for this batch item
            # Convert from NCHW to NHWC format for wandb
            img_concatenated = torch.cat([img[i].permute(1, 2, 0) for img in sample_batch["image"].values()], axis=1)
            img_concatenated = img_concatenated.cpu().numpy()
            images_to_log.append(wandb.Image(img_concatenated))

        wandb.log({"camera_views": images_to_log}, step=0)

        # Clear sample batch from memory aggressively
        del sample_batch, observation, actions, images_to_log, img_concatenated
        del sample_data_loader  # Also delete the sample data loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logging.info("Cleared sample batch and data loader from memory")

    # Build model
    if not isinstance(config.model, openpi.models.pi0_config.Pi0Config):
        # Convert dataclass to Pi0Config if needed
        model_cfg = openpi.models.pi0_config.Pi0Config(
            dtype=config.pytorch_training_precision,
            action_dim=config.model.action_dim,
            action_horizon=config.model.action_horizon,
            max_token_len=config.model.max_token_len,
            paligemma_variant=getattr(config.model, "paligemma_variant", "gemma_2b"),
            action_expert_variant=getattr(config.model, "action_expert_variant", "gemma_300m"),
            pi05=getattr(config.model, "pi05", False),
        )
    else:
        model_cfg = config.model
        # Update dtype to match pytorch_training_precision
        object.__setattr__(model_cfg, "dtype", config.pytorch_training_precision)

    model = openpi.models_pytorch.pi0_pytorch.PI0Pytorch(model_cfg).to(device)
    model.get_stats_from_loader(loader)
    
    # load depth encoder if I have to
    if config.model.aux_expert_type == "depth":
        assert hasattr(config.model, "depth_encoder_ckpt"), "Depth encoder checkpoint path must be specified for depth aux expert"
        depth_encoder = _depth_encoders.DepthEncoderWrapper(
            config.model.depth_encoder_ckpt, device=device)
    else:
        depth_encoder = None
        
    if hasattr(model, "gradient_checkpointing_enable"):
        enable_gradient_checkpointing = True
        model.gradient_checkpointing_enable()
        logging.info("Enabled gradient checkpointing for memory optimization")
    else:
        enable_gradient_checkpointing = False
        logging.info("Gradient checkpointing is not supported for this model")

    # Log initial memory usage after model creation
    if is_main and torch.cuda.is_available():
        log_memory_usage(device, 0, "after_model_creation")

    # Enable memory optimizations for large-scale training
    if world_size >= 8:
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Set memory allocation configuration
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,expandable_segments:True"
        logging.info("Enabled memory optimizations for 8+ GPU training")

    if use_ddp:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            find_unused_parameters=True,  # Disable for memory efficiency
            gradient_as_bucket_view=True,  # Enable for memory efficiency
            static_graph=world_size >= 8,  # Enable for 8+ GPUs
        )

    # Load weights from weight_loader if specified (for fine-tuning)
    if config.pytorch_weight_path is not None:
        logging.info(f"Loading weights from: {config.pytorch_weight_path}")

        model_path = os.path.join(config.pytorch_weight_path, "model.safetensors")
        load_safetensors_state_dict(
            (model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model),
            model_path,
            device,
            ignore_key_substrings=("aux",),
        )
        logging.info(f"Loaded PyTorch weights from {config.pytorch_weight_path}")

    # Optimizer + learning rate schedule from config
    warmup_steps = config.lr_schedule.warmup_steps
    peak_lr = config.lr_schedule.peak_lr
    decay_steps = config.lr_schedule.decay_steps
    end_lr = config.lr_schedule.decay_lr

    # Create optimizer with config parameters
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=peak_lr,
        betas=(config.optimizer.b1, config.optimizer.b2),
        eps=config.optimizer.eps,
        weight_decay=config.optimizer.weight_decay,
    )

    # Load checkpoint if d
    global_step = 0
    if resuming:
        global_step = load_checkpoint(model, optim, config.checkpoint_dir, device)
        logging.info(f"Resumed training from step {global_step}")

    def lr_schedule(step: int):
        if step < warmup_steps:
            # Match JAX behavior: start from peak_lr / (warmup_steps + 1)
            init_lr = peak_lr / (warmup_steps + 1)
            return init_lr + (peak_lr - init_lr) * step / warmup_steps
        # cosine decay
        progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
        cos = 0.5 * (1 + np.cos(np.pi * progress))
        return end_lr + (peak_lr - end_lr) * cos

    model.train()
    start_time = time.time()
    infos = []  # Collect stats over log interval
    if is_main:
        logging.info(
            f"Running on: {platform.node()} | world_size={torch.distributed.get_world_size() if use_ddp else 1}"
        )
        logging.info(
            f"Training config: batch_size={config.batch_size}, effective_batch_size={effective_batch_size}, num_train_steps={config.num_train_steps}"
        )
        logging.info(f"Memory optimizations: gradient_checkpointing={enable_gradient_checkpointing}")
        logging.info(
            f"LR schedule: warmup={warmup_steps}, peak_lr={peak_lr:.2e}, decay_steps={decay_steps}, end_lr={end_lr:.2e}"
        )
        logging.info(
            f"Optimizer: {type(config.optimizer).__name__}, weight_decay={config.optimizer.weight_decay}, clip_norm={config.optimizer.clip_gradient_norm}"
        )
        logging.info("EMA is not supported for PyTorch training")
        logging.info(f"Training precision: {model_cfg.dtype}")

    # Training loop - iterate until we reach num_train_steps
    pbar = (
        tqdm.tqdm(total=config.num_train_steps, initial=global_step, desc="Training", disable=not is_main)
        if is_main
        else None
    )

    data_start = time.time()
    print("-------using AUX loss weight:", config.aux_loss_weight)
    while global_step < config.num_train_steps:
        # Set epoch for distributed training
        if use_ddp and hasattr(loader, "set_epoch"):
            loader.set_epoch(global_step // len(loader))

        for observation, actions in loader:
            # actions -> (batch_size, action_horizon, action_dim)
            # obsercations
            # - images["base_0_rgb"]
            # 
            data_time = time.time() - data_start
            # Check if we've reached the target number of steps
            if global_step >= config.num_train_steps:
                break

            compute_start = time.time()
            # The unified data loader returns (observation, actions) tuple
            # observation = tree_map(lambda x: x.to(device, non_blocking=True), observation)  # noqa: PLW2901
            observation = tree_map(lambda x: x.to(device, non_blocking=True) if hasattr(x, "to") else x, observation)  # noqa: PLW2901
            actions = actions.to(torch.float32)  # noqa: PLW2901
            # actions = actions.to(device)  # noqa: PLW2901
            actions = actions.to(device, non_blocking=True)

            if depth_encoder is not None and hasattr(observation, "depth_image") and observation.depth_image is not None:
                # encode depth images
                depth_images = observation.depth_image  # expect shape (B, T, 1, H, W) or (B, T, 4, 16, 16)
                # print("Depth images shape before processing:", depth_images.shape)
                if depth_images.shape[1:] == (config.model.action_horizon+1, 4, 16, 16):
                    # this is already a latent, we just have to noramlize it
                    depth_features = depth_encoder.get_normalized_latent_from_latent(depth_images)
                else:
                    assert depth_images.shape[1:] == (config.model.action_horizon+1, 1, 256, 256), \
                        f"Unexpected depth image shape: {depth_images.shape}"
                    depth_features = depth_encoder.get_normalized_latent(depth_images)
                # observation["depth_features"] = depth_features # (B, T, 16*16*4)
                # print("Depth features shape after encoding:", depth_features.shape)
                observation = observation.replace(depth_image=depth_features)
            # Update LR
            for pg in optim.param_groups:
                pg["lr"] = lr_schedule(global_step)

            # Forward pass
            aux_output = None
            if config.save_aux:
                fm_loss, aux_loss, aux_output = model(observation, actions, return_aux=True)
            else:
                fm_loss, aux_loss = model(observation, actions)
            # Ensure losses is a tensor and handle different return types
            
            losses = (fm_loss, aux_loss)
            # if isinstance(losses, list | tuple):
            #     losses = torch.stack(losses)
            # elif not isinstance(losses, torch.Tensor):
            #     losses = torch.tensor(losses, device=device, dtype=torch.float32)
            loss = fm_loss + aux_loss * config.aux_loss_weight            
            # loss = losses.mean()

            # Backward pass
            loss.backward()

            # Log memory usage after backward pass
            if global_step < 5 and is_main and torch.cuda.is_available():
                log_memory_usage(device, global_step, "after_backward")

            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.optimizer.clip_gradient_norm)

            # Optimizer step
            optim.step()
            optim.zero_grad(set_to_none=True)

            # Clear gradients more aggressively
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.detach_()
                    param.grad = None

            compute_time = time.time() - compute_start

            # Collect stats
            if is_main:
                infos.append(
                    {
                        # "loss": loss.item(),
                        "loss" : fm_loss.item(),
                        "aux_loss": aux_loss.item(),
                        "learning_rate": optim.param_groups[0]["lr"],
                        "grad_norm": float(grad_norm) if isinstance(grad_norm, torch.Tensor) else grad_norm,
                        "data_time": data_time,
                        "compute_time": compute_time,
                    }
                )

            if is_main and (global_step % config.log_interval == 0):
                elapsed = time.time() - start_time

                # Average stats over log interval
                avg_loss = sum(info["loss"] for info in infos) / len(infos)
                avg_aux_loss = sum(info["aux_loss"] for info in infos) / len(infos)
                avg_lr = sum(info["learning_rate"] for info in infos) / len(infos)
                avg_data_time = sum(info["data_time"] for info in infos) / len(infos)
                avg_compute_time = sum(info["compute_time"] for info in infos) / len(infos)

                avg_grad_norm = None
                if any("grad_norm" in info for info in infos):
                    vals = [
                        info["grad_norm"] for info in infos if "grad_norm" in info and info["grad_norm"] is not None
                    ]
                    if len(vals) > 0:
                        avg_grad_norm = sum(vals) / len(vals)
                logging.info(
                    f"step={global_step} loss={avg_loss:.4f} aux_loss={avg_aux_loss:.4f} lr={avg_lr:.2e} grad_norm={avg_grad_norm:.2f} "
                    f"data_time={avg_data_time:.3f}s compute_time={avg_compute_time:.3f}s time={elapsed:.1f}s"
                    if avg_grad_norm is not None
                    else f"step={global_step} loss={avg_loss:.4f} aux_loss={avg_aux_loss:.4f} lr={avg_lr:.2e} "
                    f"data_time={avg_data_time:.3f}s compute_time={avg_compute_time:.3f}s time={elapsed:.1f}s"
                )

                # Log to wandb and tensorboard
                if len(infos) > 0:
                    log_payload = {
                        "loss":     avg_loss,
                        "aux_loss": avg_aux_loss,
                        "learning_rate": avg_lr,
                        "step": global_step,
                        "time_per_step": elapsed / config.log_interval,
                        "data_time": avg_data_time,
                        "compute_time": avg_compute_time,
                    }
                    if avg_grad_norm is not None:
                        log_payload["grad_norm"] = avg_grad_norm
                    
                    # Log to wandb
                    if config.wandb_enabled:
                        wandb.log(log_payload, step=global_step)
                    # Optionally save auxiliary point-track videos and log first 3 to wandb.
                        if getattr(config, "save_aux", False):
                            try:
                                tmp_vis_dir = config.checkpoint_dir / "vis_tmp"
                                tmp_vis_dir.mkdir(parents=True, exist_ok=True)

                                save_aux_videos_and_log_wandb(
                                    aux_output,
                                    out_dir=tmp_vis_dir / "aux_videos",
                                    global_step=global_step,
                                    wandb_enabled=bool(config.wandb_enabled),
                                )
                            except Exception:
                                logging.exception("Failed to save/log aux videos at step %s", global_step)

                    # Log to tensorboard
                    if tb_writer is not None:
                        tb_writer.add_scalar("train/loss", avg_loss, global_step)
                        tb_writer.add_scalar("train/aux_loss", avg_aux_loss, global_step)
                        tb_writer.add_scalar("train/learning_rate", avg_lr, global_step)
                        tb_writer.add_scalar("train/time_per_step", elapsed / config.log_interval, global_step)
                        tb_writer.add_scalar("train/data_time", avg_data_time, global_step)
                        tb_writer.add_scalar("train/compute_time", avg_compute_time, global_step)
                        if avg_grad_norm is not None:
                            tb_writer.add_scalar("train/grad_norm", avg_grad_norm, global_step)

                start_time = time.time()
                infos = []  # Reset stats collection

            global_step += 1
            # Save checkpoint using the new mechanism
            save_checkpoint(model, optim, global_step, config, is_main, data_config, aux_output=aux_output if config.save_aux else None)

            # Update progress bar
            if pbar is not None:
                pbar.update(1)
                pbar.set_postfix(
                    {"loss": f"{fm_loss.item():.4f}", "aux_loss": f"{aux_loss.item():.4f}", "lr": f"{optim.param_groups[0]['lr']:.2e}", "step": global_step}
                )
            data_start = time.time()

    # Close progress bar
    if pbar is not None:
        pbar.close()

    # Finish wandb run and close tensorboard
    if is_main:
        if config.wandb_enabled:
            wandb.finish()
        if tb_writer is not None:
            tb_writer.close()
            logging.info("Closed tensorboard writer")

    cleanup_ddp()


def main():
    init_logging()
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
