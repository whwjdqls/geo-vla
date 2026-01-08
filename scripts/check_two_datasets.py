
import dataclasses
import gc
import logging
import os

# Suppress JAX and TensorFlow GPU initialization and logging
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import platform
import shutil
import time

import numpy as np
import safetensors.torch
import torch
import torch.distributed as dist
import torch.nn.parallel
import tqdm
import wandb

import openpi.models.pi0_config
import openpi.models_pytorch.pi0_pytorch
import openpi.shared.normalize as _normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data


def build_datasets(config: _config.TrainConfig):
    # Use the unified data loader with PyTorch framework
    data_loader = _data.create_data_loader(config, framework="pytorch", shuffle=True)
    return data_loader, data_loader.data_config()

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




def train_loop(config: _config.TrainConfig):
    use_ddp, local_rank, device = setup_ddp()
    is_main = (not use_ddp) or (dist.get_rank() == 0)
    set_seed(config.seed, local_rank)


        # For new runs, create experiment-specific checkpoint directory
    exp_checkpoint_dir = config.checkpoint_dir
    exp_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created experiment checkpoint directory: {exp_checkpoint_dir}")

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

    for observation, actions in loader:
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


    # Close progress bar
    if pbar is not None:
        pbar.close()

    cleanup_ddp()


def main():
    config = _config.cli()
    train_loop(config)


if __name__ == "__main__":
    main()
