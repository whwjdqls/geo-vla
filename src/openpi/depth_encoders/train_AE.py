#!/usr/bin/env python3
import os
import time
import argparse
import re
import glob
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

# W&B
import wandb

# Your files
from datasets_.datasets import DepthTemporalPNGDatasetPreload
from models.cnn_models import DepthLatentModel


def parse_args():
    p = argparse.ArgumentParser("Train Depth AE/VAE on uint16 depth PNG windows")

    # data
    p.add_argument("--root", type=str, default="/scratch2/whwjdqls99/libero/libero_hdfr_lerobot_datasets_only_depths", help="Dataset root containing depth_u16_png_metadata.json")
    p.add_argument("--seq-len", type=int, default=2)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--drop-last", action="store_true", default=True)
    p.add_argument("--no-drop-last", action="store_false", dest="drop_last")

    # dataset decode settings (match how you preloaded)
    p.add_argument("--normalize-0-1", action="store_true", default=False)
    p.add_argument("--clamp-to-minmax", action="store_true", default=True)
    p.add_argument("--no-clamp-to-minmax", action="store_false", dest="clamp_to_minmax")
    p.add_argument("--add-channel-dim", action="store_true", default=True)
    p.add_argument("--no-add-channel-dim", action="store_false", dest="add_channel_dim")
    p.add_argument("--meta-json-name", type=str, default="depth_u16_png_metadata.json", help="Name of the metadata JSON file in the dataset root")
    # split + performance
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--pin-memory", action="store_true", default=True)
    p.add_argument("--no-pin-memory", action="store_false", dest="pin_memory")

    # model
    p.add_argument("--z-ch", type=int, default=8)
    p.add_argument("--z-hw", type=int, default=32)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--use-vae", action="store_true", default=False)
    p.add_argument("--beta-kl", type=float, default=1e-4)
    p.add_argument("--huber-delta", type=float, default=1.0)

    # training
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--no-amp", action="store_false", dest="amp")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--temp_loss_weight", type=float, default=0.0)

    # checkpoints
    p.add_argument("--outdir", type=str, default="./runs/depth_ae")
    p.add_argument("--save-every-epochs", type=int, default=4)

    # resume
    p.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint .pt to resume from (restores model/opt/scaler/steps).",
    )
    p.add_argument(
        "--resume-auto",
        action="store_true",
        default=False,
        help="Resume from latest ckpt_*.pt in outdir (prefers ckpt_last.pt if present).",
    )

    # W&B
    p.add_argument("--wandb", action="store_true", default=True, help="Enable wandb logging")
    p.add_argument("--no-wandb", action="store_false", dest="wandb")
    p.add_argument("--wandb-project", type=str, default="depth-encoder")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-name", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, nargs="*", default=None)
    p.add_argument("--wandb-notes", type=str, default=None)
    p.add_argument("--wandb-mode", type=str, default=None, choices=[None, "online", "offline", "disabled"])
    p.add_argument("--wandb-log-best-artifact", action="store_true", default=False)

    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, args):
    model.eval()
    total_loss = 0.0
    total_recon = 0.0
    total_kl = 0.0
    total_temp_loss = 0.0
    n = 0

    for batch in loader:
        # batch: [B, T, 1, H, W] if add_channel_dim True, else [B,T,H,W]
        # if args.temp_loss_weight > 0.0:
        #     assert batch.ndim == 5, "Temporal loss requires channel dim in data"
        # else:
        if batch.ndim == 5:
            if args.temp_loss_weight <= 0.0:
                batch = batch[:, 0]  # [B,1,H,W]

        elif batch.ndim == 4:
            if args.temp_loss_weight <= 0.0:
                batch = batch[:, 0].unsqueeze(1)  # [B,1,H,W]
        else:
            raise ValueError(f"Unexpected batch shape: {batch.shape}")

        temp_dim = 1
        if args.temp_loss_weight > 0.0:
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
            loss_dict = model.loss(
                depth=batch,
                recon=out["recon"],
                mu=out["mu"],
                logvar=out["logvar"],
                valid_mask=None,
                beta_kl=args.beta_kl,
                huber_delta=args.huber_delta,
                temp_loss_weight=args.temp_loss_weight,
                temp_dim=temp_dim,
            )

        bs = batch.shape[0]
        
        total_loss += float(loss_dict["total"]) * bs 
        total_temp_loss += float(loss_dict.get("vel", 0.0)) * bs
        total_recon += float(loss_dict["recon"]) * bs
        total_kl += float(loss_dict["kl"]) * bs
        n += bs

    return {
        "loss": total_loss / max(1, n),
        "recon": total_recon / max(1, n),
        "kl": total_kl / max(1, n),
        "vel": total_temp_loss / max(1, n), 
    }


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # -----------------------
    # W&B init
    # -----------------------
    run = None
    if args.wandb:
        if args.wandb_mode is not None:
            os.environ["WANDB_MODE"] = args.wandb_mode

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_name,
            tags=args.wandb_tags,
            notes=args.wandb_notes,
            config=vars(args),
        )

    # -----------------------
    # Dataset / DataLoaders
    # -----------------------
    train_ds = DepthTemporalPNGDatasetPreload(
        root=args.root,
        seq_len=args.seq_len,
        stride=args.stride,
        step=args.step,
        return_info=False,
        drop_last=args.drop_last,
        normalize_0_1=args.normalize_0_1,
        clamp_to_minmax=args.clamp_to_minmax,
        add_channel_dim=args.add_channel_dim,
        split="train",
        preload_dtype=torch.float16,  # keep RAM lower
        meta_json_name=args.meta_json_name,
    )
    val_ds = DepthTemporalPNGDatasetPreload(
        root=args.root,
        seq_len=args.seq_len,
        stride=args.stride,
        step=args.step,
        return_info=False,
        drop_last=args.drop_last,
        normalize_0_1=args.normalize_0_1,
        clamp_to_minmax=args.clamp_to_minmax,
        add_channel_dim=args.add_channel_dim,
        split="val",
        preload_dtype=torch.float16,
        meta_json_name=args.meta_json_name,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and (device == "cuda"),
        persistent_workers=(args.num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=args.pin_memory and (device == "cuda"),
        persistent_workers=(max(1, args.num_workers // 2) > 0),
        drop_last=False,
    )

    # -----------------------
    # Model / Optim
    # -----------------------
    model = DepthLatentModel(z_ch=args.z_ch, base_ch=args.base_ch, z_hw=args.z_hw, use_vae=args.use_vae).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == "cuda"))

    def _optimizer_to(optimizer: torch.optim.Optimizer, dev: str):
        if dev == "cpu":
            return
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(dev)

    def _auto_resume_path(outdir: str) -> str | None:
        last = os.path.join(outdir, "ckpt_last.pt")
        if os.path.isfile(last):
            return last
        cands = []
        for fn in os.listdir(outdir):
            if fn.startswith("ckpt_") and fn.endswith(".pt"):
                cands.append(os.path.join(outdir, fn))
        if not cands:
            return None
        # newest mtime wins
        cands.sort(key=lambda p: os.path.getmtime(p))
        return cands[-1]

    def _infer_epoch_from_path(path: str) -> int | None:
        m = re.search(r"epoch(\d+)", os.path.basename(path))
        if not m:
            return None
        try:
            return int(m.group(1))
        except ValueError:
            return None

    # -----------------------
    # Train loop
    # -----------------------
    global_step = 0
    best_val = float("inf")
    start_epoch = 1

    # -----------------------
    # Resume from checkpoint
    # -----------------------
    resume_path = args.resume
    if args.resume_auto and resume_path is None:
        resume_path = _auto_resume_path(args.outdir)

    if resume_path is not None:
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        print(f"[resume] loading: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"], strict=True)
        if "opt" in ckpt and ckpt["opt"] is not None:
            opt.load_state_dict(ckpt["opt"])
            _optimizer_to(opt, device)
        if scaler.is_enabled() and ("scaler" in ckpt) and (ckpt["scaler"] is not None):
            scaler.load_state_dict(ckpt["scaler"])

        global_step = int(ckpt.get("global_step", global_step))
        best_val = float(ckpt.get("best_val", best_val))

        ckpt_epoch = ckpt.get("epoch", None)
        if ckpt_epoch is None:
            ckpt_epoch = _infer_epoch_from_path(resume_path)
        if ckpt_epoch is not None:
            start_epoch = int(ckpt_epoch) + 1

        print(
            f"[resume] start_epoch={start_epoch} global_step={global_step} best_val={best_val}"
        )

    def save_ckpt(tag: str, extra: dict | None = None, loss: float | None = None, epoch: int | None = None):
        # Use stable filenames for best/last so they don't stack.
        # (Epoch checkpoints may include loss in the name.)
        if tag in {"best", "last"}:
            path = os.path.join(args.outdir, f"ckpt_{tag}.pt")
        else:
            if loss is not None:
                tag += f"_loss{loss:.5f}"
            path = os.path.join(args.outdir, f"ckpt_{tag}.pt")

        if os.path.basename(path).startswith("ckpt_best"):
            # remove older stacked best checkpoints (e.g., ckpt_best_loss*.pt)
            for p in glob.glob(os.path.join(args.outdir, "ckpt_best*.pt")):
                if os.path.abspath(p) != os.path.abspath(path):
                    try:
                        os.remove(p)
                    except FileNotFoundError:
                        pass
        payload = {
            "args": vars(args),
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler.is_enabled() else None,
            "global_step": global_step,
            "best_val": best_val,
            "epoch": epoch,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, path)
        print(f"[ckpt] saved: {path}")
        return path

    # quick sanity
    print(f"Device: {device}")
    print(f"Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    first_batch = next(iter(train_loader))
    print(f"Example batch shape: {first_batch.shape}")

    if run is not None:
        wandb.summary["device"] = device
        wandb.summary["train_samples"] = len(train_ds)
        wandb.summary["val_samples"] = len(val_ds)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_t0 = time.time()

        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "vel": 0.0}
        seen = 0
        temp_dim = 1

        for it, batch in enumerate(train_loader, start=1):
            iter_t0 = time.time()

            if batch.ndim == 5:
                if args.temp_loss_weight <= 0.0:
                    batch = batch[:, 0]  # [B,1,H,W]
            elif batch.ndim == 4:
                if args.temp_loss_weight <= 0.0:
                    batch = batch[:, 0].unsqueeze(1)  # [B,1,H,W]
            else:
                raise ValueError(f"Unexpected batch shape: {batch.shape}")
            
            if args.temp_loss_weight > 0.0:
                assert batch.ndim == 5, "Temporal loss requires temporal dim and channel dim"
                assert batch.size(1) >= 2, "Temporal loss requires at least 2 frames"
                assert batch.size(2) == 1, "Temporal loss requires channel dim of size 1"
                temp_dim = int(batch.size(1))
                B,T,C,H,W = batch.shape
                batch = batch.view(B*T, C, H, W)
                
            batch = batch.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(args.amp and device == "cuda")):
                out = model(batch)
                loss_dict = model.loss(
                    depth=batch,
                    recon=out["recon"],
                    mu=out["mu"],
                    logvar=out["logvar"],
                    valid_mask=None,
                    beta_kl=args.beta_kl,
                    huber_delta=args.huber_delta,
                    temp_loss_weight=args.temp_loss_weight,
                    temp_dim=temp_dim,
                )
                loss = loss_dict["total"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

            bs = batch.size(0)
            seen += bs
            running["loss"] += float(loss_dict["total"]) * bs
            running["recon"] += float(loss_dict["recon"]) * bs
            running["vel"] += float(loss_dict.get("vel", 0.0)) * bs
            running["kl"] += float(loss_dict["kl"]) * bs
            global_step += 1

            # metrics
            iter_dt = time.time() - iter_t0
            samples_per_sec = bs / max(1e-9, iter_dt)
            lr_now = opt.param_groups[0]["lr"]

            if (it % args.log_every) == 0:
                dt = time.time() - epoch_t0
                msg = (
                    f"[ep {epoch:03d} | it {it:05d}] "
                    f"loss={running['loss']/seen:.5f} "
                    f"recon={running['recon']/seen:.5f} "
                    f"vel={running['vel']/seen:.5f} "
                    f"kl={running['kl']/seen:.5f} "
                    f"seen={seen} "
                    f"sec={dt:.1f}"
                )
                print(msg)

            if run is not None:
                wandb.log(
                    {
                        "train/loss": float(loss_dict["total"]),
                        "train/recon": float(loss_dict["recon"]),
                        "train/kl": float(loss_dict["kl"]),
                        "train/vel": float(loss_dict.get("vel", 0.0)),
                        "train/lr": lr_now,
                        "train/samples_per_sec": samples_per_sec,
                        "train/iter_time_sec": iter_dt,
                        "epoch": epoch,
                    },
                    step=global_step,
                )

        # validate
        val = evaluate(model, val_loader, device, args)
        epoch_time = time.time() - epoch_t0
        print(
            f"[ep {epoch:03d}] "
            f"train_loss={running['loss']/seen:.5f} "
            f"| val_loss={val['loss']:.5f} val_recon={val['recon']:.5f} val_kl={val['kl']:.5f} val_vel={val.get('vel', 0.0):.5f} "
            f"| time={epoch_time:.1f}s"
        )

        if run is not None:
            wandb.log(
                {
                    "val/loss": val["loss"],
                    "val/recon": val["recon"],
                    "val/kl": val["kl"],
                    "val/vel": val.get("vel", 0.0),
                    "epoch_time_sec": epoch_time,
                    "epoch": epoch,
                },
                step=global_step,
            )

        # save
        if (epoch % args.save_every_epochs) == 0:
            ckpt_path = save_ckpt(f"epoch{epoch:03d}", extra={"val": val}, loss=val["loss"], epoch=epoch)
            if run is not None:
                wandb.save(ckpt_path, base_path=args.outdir)

        if val["loss"] < best_val:
            best_val = val["loss"]
            best_path = save_ckpt("best", extra={"val": val}, loss=best_val, epoch=epoch)
            if run is not None:
                wandb.summary["best_val_loss"] = best_val
                wandb.save(best_path, base_path=args.outdir)

                if args.wandb_log_best_artifact:
                    art = wandb.Artifact(
                        name=f"{wandb.run.name}-best",
                        type="model",
                        metadata={"best_val_loss": best_val},
                    )
                    art.add_file(best_path)
                    wandb.log_artifact(art)

        # always keep a "last" checkpoint for easy resume (overwrites each epoch)
        save_ckpt("last", extra={"val": val}, epoch=epoch)

    last_path = os.path.join(args.outdir, "ckpt_last.pt")
    if run is not None:
        wandb.save(last_path, base_path=args.outdir)
        wandb.finish()
    print("Done.") 


if __name__ == "__main__":
    main()
