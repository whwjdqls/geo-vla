
import os
import time
import argparse
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

# W&B
import wandb

# Your files
from datasets_.datasets import DepthTemporalPNGDatasetPreload
from models.cnn_models import DepthLatentModel

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="checkpoint path to load the model from.",
    )
    args = parser.parse_args()
    return args


    # def save_ckpt(tag: str, extra: dict | None = None, loss: float | None = None):
    #     if loss is not None:
    #         tag += f"_loss{loss:.5f}"
    #     path = os.path.join(args.outdir, f"ckpt_{tag}.pt")
    #     payload = {
    #         "args": vars(args),
    #         "model": model.state_dict(),
    #         "opt": opt.state_dict(),
    #         "scaler": scaler.state_dict() if scaler.is_enabled() else None,
    #         "global_step": global_step,
    #         "best_val": best_val,
    #     }
    #     if extra:
    #         payload.update(extra)
    #     torch.save(payload, path)
    #     print(f"[ckpt] saved: {path}")
    #     return path
def load_ckpt_and_prepare_model(checkpoint_path: str) -> DepthLatentModel:
    # load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model_args = ckpt["args"]

    # build model
    model = DepthLatentModel(
        z_ch=model_args["z_ch"],
        base_ch=model_args["base_ch"],
        use_vae=model_args["use_vae"],
        z_hw=model_args["z_hw"]
    )
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.cuda()

    print("Model loaded and ready for inference.")
    return model

@torch.no_grad()
def inference(model, loader, device, args):
    model.eval()

    inputs = []
    reconstructions = []
    for batch in loader:
        # batch: [B, T, 1, H, W] if add_channel_dim True, else [B,T,H,W]
        # if args.temp_loss_weight > 0.0:
        #     assert batch.ndim == 5, "Temporal loss requires channel dim in data"
        # else:
        # if batch.ndim == 5:
        #     batch = batch[:, 0]  # [B,1,H,W]

        # elif batch.ndim == 4:
        #     batch = batch[:, 0].unsqueeze(1)  # [B,1,H,W]
        # else:
        #     raise ValueError(f"Unexpected batch shape: {batch.shape}")

        if batch.ndim == 4:
            batch = batch.unsqueeze(2)  # [B,T,1,H,W]
            
        assert batch.ndim == 5, "Temporal loss requires temporal dim and channel dim"
        assert batch.size(1) >= 2, "Temporal loss requires at least 2 frames"
        assert batch.size(2) == 1, "Temporal loss requires channel dim of size 1"
        temp_dim = int(batch.size(1))
        B,T,C,H,W = batch.shape
        batch = batch.view(B*T, C, H, W)
        
        batch = batch.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
            if batch.ndim == 5:
                assert temp_dim >= 2, "Temporal loss requires temporal dim and channel dim"
            out = model(batch)

        batch = batch.view(B, T, C, H, W)
        out_recon = out["recon"].view(B, T, C, H, W)
        
        batch = batch.cpu().numpy().astype(np.float32)
        out_recon = out_recon.cpu().numpy().astype(np.float32)
        
        inputs.append(batch)
        reconstructions.append(out_recon)
    
    inputs = np.concatenate(inputs, axis=0)  # [N,T,1,H,W]
    reconstructions = np.concatenate(reconstructions, axis=0)  # [N,T,1,H,W]
    
    mean = loader.dataset.mean.cpu().numpy().astype(np.float32)
    std = loader.dataset.std.cpu().numpy().astype(np.float32)
    
    inputs = inputs * std + mean
    reconstructions = reconstructions * std + mean
    
    return inputs, reconstructions

def save_as_figure(
    input_seq,
    recon_seq,
    save_path,
    near=0.01,
    far=50.0,
    cmap="turbo",
):
    """
    input_seq, recon_seq: [T, 1, H, W]  (MuJoCo z-buffer depth in [0,1])
    """

    # -----------------------
    # Shape checks
    # -----------------------
    T, C, H, W = input_seq.shape
    assert C == 1, "Expected channel dim of size 1"
    assert recon_seq.shape == input_seq.shape

    # -----------------------
    # z-buffer -> metric depth
    # -----------------------
    def zbuffer_to_depth(z):
        return (near * far) / (far - z * (far - near))

    input_depth = zbuffer_to_depth(input_seq[:, 0])   # [T, H, W]
    recon_depth = zbuffer_to_depth(recon_seq[:, 0])   # [T, H, W]

    # -----------------------
    # Robust visualization range (shared!)
    # -----------------------
    all_depth = np.concatenate(
        [input_depth.reshape(-1), recon_depth.reshape(-1)]
    )

    valid = np.isfinite(all_depth)
    vmin, vmax = np.percentile(all_depth[valid], [1, 99])

    # -----------------------
    # Plot
    # -----------------------
    fig, axes = plt.subplots(
        2, T,
        figsize=(T * 3, 6),
        squeeze=False
    )

    for t in range(T):
        im0 = axes[0, t].imshow(
            input_depth[t],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[0, t].set_title(f"Input t={t}")
        axes[0, t].axis("off")

        im1 = axes[1, t].imshow(
            recon_depth[t],
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        axes[1, t].set_title(f"Recon t={t}")
        axes[1, t].axis("off")

    # -----------------------
    # One global colorbar
    # -----------------------
    cbar = fig.colorbar(
        im1,
        ax=axes,
        fraction=0.025,
        pad=0.02
    )
    cbar.set_label("Depth (meters)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    model = load_ckpt_and_prepare_model(args.checkpoint_path)
    
    args_dict = torch.load(args.checkpoint_path, map_location="cpu")["args"] # dict
    # args = argparse.Namespace(**args_dict)
    # live the original one and add args_dict
    for k, v in args_dict.items():
        setattr(args, k, v)

    val_ds = DepthTemporalPNGDatasetPreload(
        root=args.root,
        seq_len=5,
        stride=5,
        step=args.step,
        return_info=False,
        drop_last=args.drop_last,
        normalize_0_1=args.normalize_0_1,
        clamp_to_minmax=args.clamp_to_minmax,
        add_channel_dim=args.add_channel_dim,
        split="val",
        preload_dtype=torch.float32,
        meta_json_name=getattr(args, "meta_json_name", "depth_u16_png_metadata_old.json")
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=128//5,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.pin_memory and (device == "cuda"),
        persistent_workers=(max(1, args.num_workers // 2) > 0),
        drop_last=False,
    )
    
    
    inputs, reconstructions = inference(model, val_loader, device, args)
    
    # save_dir = os.path.join(os.path.dirname(args.checkpoint_path), "inference_results")
    # add checkpoint name to the dir
    ckpt_name = os.path.splitext(os.path.basename(args.checkpoint_path))[0]
    save_dir = os.path.join(os.path.dirname(args.checkpoint_path), f"inference_results_{ckpt_name}")
    os.makedirs(save_dir, exist_ok=True)
    
    batch_size = inputs.shape[0]
    for i in range(batch_size):
        save_as_figure(
            inputs[i],
            reconstructions[i],
            os.path.join(save_dir, f"sample_{i:04d}.png"),
        )