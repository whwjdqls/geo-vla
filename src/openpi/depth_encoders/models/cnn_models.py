import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Utils
# -----------------------------
def _assert_shape(x: torch.Tensor, nd: int, name: str):
    if x.ndim != nd:
        raise ValueError(f"{name} must have {nd} dims, got {x.shape}")



def huber(x: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    return F.smooth_l1_loss(x, torch.zeros_like(x), beta=delta, reduction="none")


# -----------------------------
# Depth (V)AE: 256x256 -> latent grid (e.g., 32x32)
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
            nn.GroupNorm(num_groups=min(32, out_ch), num_channels=out_ch),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DepthEncoder(nn.Module):
    """
    Input:  depth  [B, 1, 256, 256]
    Output: latent [B, z_ch, z_h, z_w] (e.g., z_h=z_w=32)
    If use_vae: also outputs mu/logvar per spatial location.
    """
    def __init__(
        self,
        in_ch: int = 1,
        base_ch: int = 64,
        z_ch: int = 8,
        z_hw: int = 32,
        use_vae: bool = False,
    ):
        super().__init__()
        self.use_vae = use_vae
        # 256 -> 128 -> 64 -> 32
        assert z_hw in [32, 16, 8], "z_hw must be one of [32,16,8]"
        self.down1 = nn.Sequential(ConvBlock(in_ch, base_ch, s=2), ConvBlock(base_ch, base_ch))
        self.down2 = nn.Sequential(ConvBlock(base_ch, base_ch * 2, s=2), ConvBlock(base_ch * 2, base_ch * 2))
        self.down3 = nn.Sequential(ConvBlock(base_ch * 2, base_ch * 4, s=2), ConvBlock(base_ch * 4, base_ch * 4))
        if z_hw in [16, 8]:
            self.down4 = nn.Sequential(ConvBlock(base_ch * 4, base_ch * 8, s=2), ConvBlock(base_ch * 8, base_ch * 8))
        else:
            self.down4 = nn.Identity()
        if z_hw == 8:
            self.down5 = nn.Sequential(ConvBlock(base_ch * 8, base_ch * 16, s=2), ConvBlock(base_ch * 16, base_ch * 16))
        else:
            self.down5 = nn.Identity()
        
        last_ch = base_ch * 4
        if z_hw == 16:
            last_ch = base_ch * 8
        elif z_hw == 8:
            last_ch = base_ch * 16
        # Project to latent channels
        if use_vae:
            self.to_mu = nn.Conv2d(last_ch, z_ch, kernel_size=1)
            self.to_logvar = nn.Conv2d(last_ch, z_ch, kernel_size=1)
        else:
            self.to_z = nn.Conv2d(last_ch, z_ch, kernel_size=1)

        self.z_hw = z_hw

    def forward(self, depth: torch.Tensor):
        _assert_shape(depth, 4, "depth")
        x = self.down1(depth)
        x = self.down2(x)
        x = self.down3(x)  # [B, 4*base, 32, 32] if input 256
        x = self.down4(x)
        x = self.down5(x)
        if self.use_vae:
            mu = self.to_mu(x)
            logvar = self.to_logvar(x).clamp(-20.0, 10.0)
            return mu, logvar
        else:
            z = self.to_z(x)
            return z


class DepthDecoder(nn.Module):
    """
    Input:  latent [B, z_ch, 32, 32]
    Output: depth  [B, 1, 256, 256]
    """
    def __init__(
        self,
        out_ch: int = 1,
        base_ch: int = 64,
        z_ch: int = 8,
        z_hw: int = 32,
    ):
        super().__init__()
        
        assert z_hw in [32, 16, 8], "z_hw must be one of [32,16,8]"
        last_ch = base_ch * 4
        if z_hw == 16:
            last_ch = base_ch * 8
        elif z_hw == 8:
            last_ch = base_ch * 16
            
        self.up1 = nn.Sequential(
            ConvBlock(z_ch, last_ch),
            ConvBlock(last_ch, last_ch),
        )
        # 32 -> 64 -> 128 -> 256
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(last_ch, last_ch // 2),
            ConvBlock(last_ch // 2, last_ch // 2),
        )
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            ConvBlock(last_ch // 2, last_ch // 4),
            ConvBlock(last_ch // 4, last_ch // 4),
        )
        
        if z_hw in [16, 8]:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvBlock(last_ch//4, last_ch//8),
                ConvBlock(last_ch//8, last_ch//8),
            )
        else:
            self.up4 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvBlock(last_ch//4, base_ch),
            )
            
        if z_hw == 16:
            self.up5 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvBlock(last_ch//8, base_ch),
            )
        elif z_hw == 8:
            self.up5 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvBlock(last_ch//8, last_ch//16),
                ConvBlock(last_ch//16, base_ch),
            )
        else:
            self.up5 = nn.Identity()
        
        if z_hw == 8:
            self.up6 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                ConvBlock(base_ch, base_ch),
            )
        else:
            self.up6 = nn.Identity()
            
        self.to_depth = nn.Conv2d(base_ch, out_ch, kernel_size=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        _assert_shape(z, 4, "z")
        x = self.up1(z)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        x = self.up6(x)
        depth = self.to_depth(x)
        return depth


class DepthLatentModel(nn.Module):
    """
    Wrapper that can be:
      - Deterministic AE (use_vae=False)
      - Spatial VAE (use_vae=True), where mu/logvar are per-latent-pixel.

    depth: [B, 1, 256, 256]
    """
    def __init__(
        self,
        z_ch: int = 8,
        z_hw: int = 32,
        base_ch: int = 64,
        use_vae: bool = True,
    ):
        super().__init__()
        self.use_vae = use_vae
        self.enc = DepthEncoder(in_ch=1, base_ch=base_ch, z_ch=z_ch, z_hw=z_hw, use_vae=use_vae)
        self.dec = DepthDecoder(out_ch=1, base_ch=base_ch, z_ch=z_ch, z_hw=z_hw)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def encode(self, depth: torch.Tensor):
        if self.use_vae:
            mu, logvar = self.enc(depth)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar
        else:
            z = self.enc(depth)
            return z, None, None

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.dec(z)

    def forward(self, depth: torch.Tensor):
        z, mu, logvar = self.encode(depth)
        recon = self.decode(z)
        return {"z": z, "mu": mu, "logvar": logvar, "recon": recon}

    def loss(
        self,
        depth: torch.Tensor,
        recon: torch.Tensor,
        mu: torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
        valid_mask: torch.Tensor | None = None,   # [B,1,256,256] where depth valid
        beta_kl: float = 1e-4,
        huber_delta: float = 1.0,
        temp_loss_weight: float = 0.0,
        temp_dim: int = 1,
    ):
        # reconstruction loss (masked)
        if valid_mask is None:
            valid_mask = torch.ones_like(depth)

        # Huber per-pixel
        recon_err = huber(recon - depth, delta=huber_delta)  # [B,1,H,W]
        recon_loss = (recon_err * valid_mask).sum() / (valid_mask.sum().clamp_min(1.0))

        # temporal loss
        vel_loss = torch.tensor(0.0, device=depth.device)
        if temp_loss_weight > 0.0:
            BT, C, H, W = depth.shape
            assert BT % temp_dim == 0, "Batch size must be multiple of temp_dim"
            B = BT // temp_dim
            
            # Reshape to [B, T, C, H, W] to compute temporal differences
            d_seq = depth.view(B, temp_dim, C, H, W)
            r_seq = recon.view(B, temp_dim, C, H, W)
            m_seq = valid_mask.view(B, temp_dim, C, H, W)

            assert C == 1, "Expected channel dim of size 1 for depth"
            recon_vel = r_seq[:, 1:] - r_seq[:, :-1]  # [B, T-1, 1, H, W]
            depth_vel = d_seq[:, 1:] - d_seq[:, :-1]  # [B, T-1, 1, H, W]
            vel_err = huber(recon_vel - depth_vel, delta=huber_delta)  
            vel_loss = (vel_err * m_seq[:, 1:]).sum() / (m_seq[:, 1:].sum().clamp_min(1.0))
            
        kl_loss = torch.tensor(0.0, device=depth.device)
        if self.use_vae and (mu is not None) and (logvar is not None):
            # spatial KL averaged over B,C,H,W
            kl = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp())
            kl_loss = kl.mean()

        total = recon_loss + beta_kl * kl_loss + temp_loss_weight * vel_loss
        return {"total": total, "recon": recon_loss, "kl": kl_loss, "vel": vel_loss}




# -----------------------------
# Example instantiation
# -----------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    B = 2
    L = 256         # VLA token length (example)
    d_vla = 1024
    K = 5           # short horizon
    act_dim = 7     # EE delta dims (example)


    depth_model = DepthLatentModel(z_ch=4, base_ch=64, z_hw=16, use_vae=False).to(device)
    dummy_depth = torch.randn(B, 1, 256, 256).to(device)
    out = depth_model(dummy_depth)
    print("DepthLatentModel output keys:", out.keys())
    print("Latent z shape:", out["z"].shape)
    print("Reconstructed depth shape:", out["recon"].shape)