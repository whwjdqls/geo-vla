from openpi.depth_encoders.models.cnn_models import DepthLatentModel
import torch
import numpy as np
import os
import json


def load_ckpt_and_prepare_model(checkpoint_path: str) -> DepthLatentModel:
    # load checkpoint
    print(f"Loading DepthLatentModel checkpoint from {checkpoint_path}...")
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

    # make gradient false
    for param in model.parameters():
        param.requires_grad = False

    return model, model_args


# make a depth_encoder_wrapper
class DepthEncoderWrapper:
    def __init__(self,
                 checkpoint_path: str,
                 device: str = "cpu"):
        self.model, self.model_args = load_ckpt_and_prepare_model(checkpoint_path)
        self.model = self.model.to(device)
        self.device = device
        ckpt_name = os.path.basename(checkpoint_path).split('.')[0]
        base_dir = os.path.dirname(checkpoint_path)
        latent_stats_path = os.path.join(base_dir, "latent_stats_" + ckpt_name + "_train.json")
        with open(latent_stats_path, "r") as f:
            latent_stats = json.load(f)
            
        self.latent_mean = torch.tensor(latent_stats["stats"]["channel_mean"])[:, None, None].to(device)
        self.latent_std = torch.tensor(latent_stats["stats"]["channel_std"])[:,None,None].to(device)
        # (4,1,1)
        meta_json_name = self.model_args["meta_json_name"]
        meta_json = os.path.join(self.model_args["root"], meta_json_name)
        with open(meta_json, "r") as f:
            meta = json.load(f)

        self.vmin = float(meta["global_min"])
        self.vmax = float(meta["global_max"])
        if not np.isfinite(self.vmin) or not np.isfinite(self.vmax) or self.vmax <= self.vmin:
            raise ValueError(f"Bad min/max in meta_json: min={self.vmin}, max={self.vmax}")

        mean_path = meta["original_stats"]["mean_npy"]
        std_path  = meta["original_stats"]["std_npy"]
        # self.mean = torch.from_numpy(np.load(mean_path)).float()  # (H,W) or scalar
        # self.std  = torch.from_numpy(np.load(std_path)).float()
        self.mean = float(np.load(mean_path)) # make it scalar 
        self.std = float(np.load(std_path)) # make it scalar
        
        self.scale = (self.vmax - self.vmin) / 65535.0
        self.offset = self.vmin
        
        
    def uint_to_depth(self, depth_uint: torch.Tensor) -> torch.Tensor:
        """
        Convert uint16 depth to original depth range.
        Args: float32
            depth_uint: torch.Tensor of shape [B, T, 1, H, W], values in [0, 65535].
        Returns:
            depth tensor in original range [vmin, vmax].
        """
        depth = depth_uint * self.scale + self.offset
        return depth
    
    
    
    def preprocess_depth(self, depth_image: torch.Tensor) -> torch.Tensor:
        """
        Preprocess depth image before encoding.
        Args:
            depth_image: torch.Tensor of shape [B, T, 1, H, W], values in original depth range.
        Returns:
            preprocessed depth image tensor.
        """
        # Clamp and scale to [0, 65535]
        depth_image = self.uint_to_depth(depth_image)
        
        # Normalize
        depth_image = (depth_image - self.mean) / self.std
        return depth_image
        
    def encode(self, depth_image: torch.Tensor) -> torch.Tensor:
        depth_image = self.preprocess_depth(depth_image)
        with torch.no_grad():
            z,  mu, logvar = self.model.encode(depth_image)
        return z, mu, logvar

    def get_normalized_latent(self, depth_image) -> torch.Tensor:
        if depth_image.ndim != 5:
            raise ValueError(f"depth_image must have 5 dims [B,T,C,H,W], got {depth_image.shape}")
        BS_, T_, C_, H_, W_ = depth_image.shape
        ndim = 5
        depth_image = depth_image.view(BS_ * T_, C_, H_, W_)

        z, mu, logvar = self.encode(depth_image)
        z_norm = (z - self.latent_mean) / self.latent_std
        
        BT, C, H, W = z_norm.shape
        z_norm = z_norm.view(BS_, T_, C * H * W)
        return z_norm
    
    def get_normalized_latent_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 5:
            raise ValueError(f"z must have 5 dims [B,T,C,H,W], got {z.shape}")
        BS_, T_, C_, H_, W_ = z.shape
        z = z.view(BS_ * T_, C_, H_, W_)

        z_norm = (z - self.latent_mean) / self.latent_std
        
        BT, C, H, W = z_norm.shape
        z_norm = z_norm.view(BS_, T_, C * H * W)
        return z_norm
    
    def decode_from_normalized_latent(self, z_norm: torch.Tensor) -> torch.Tensor:
        B, T, D = z_norm.shape
        C = self.model_args["z_ch"]
        H = self.model_args["z_hw"]
        W = self.model_args["z_hw"]
        z_norm = z_norm.view(B, T, C, H, W)
        z = z_norm * self.latent_std + self.latent_mean
        with torch.no_grad():
            depth_recon = self.model.decode(z)
        return depth_recon
    
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            depth_recon = self.model.decode(z)
        return depth_recon