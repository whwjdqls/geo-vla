import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks

def enforce_aux_attention_masks(att_2d_masks, aux_pad_masks, aux_start, aux_end, prefix_len, suffix_len, allow_aux_to_attend_suffix=False):
    """Enforce auxiliary attention masks in the overall attention masks.

    Args:
      att_2d_masks: bool[B, S, S] original 2D attention masks.
      aux_pad_masks: bool[B, A] padding masks for auxiliary inputs.
      aux_start: int, start index of auxiliary inputs in the overall sequence.
      aux_end: int, end index of auxiliary inputs in the overall sequence.
        prefix_len: int, length of the prefix inputs.
        suffix_len: int, length of the suffix inputs.
    """
    B, N, N = att_2d_masks.shape #(Bs, 968+10+10, 968+10+10)

    # no query can attend to aux inputs
    att_2d_masks[:,:, aux_start:aux_end] = False
 
    # initialize aux query rows
    att_2d_masks[:, aux_start:aux_end, :] = False

    # aux can attend to prefix
    att_2d_masks[:, aux_start:aux_end, :prefix_len] = att_2d_masks[:,0:1, :prefix_len].expand(B, aux_end - aux_start, prefix_len)
    
    # aux can attend to itself
    att_2d_masks[:, aux_start:aux_end, aux_start:aux_end] = True

    if allow_aux_to_attend_suffix:
        # aux can attend to suffix
        att_2d_masks[:, aux_start:aux_end, prefix_len:prefix_len+suffix_len] = True
        
    return att_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        
        # for auxiliary expert
        if config.aux_expert_type == "point":
            aux_expert_config = _gemma.get_config(config.point_expert_variant)
        elif config.aux_expert_type == "depth":
            aux_expert_config = _gemma.get_config(config.depth_expert_variant)
        else:
            aux_expert_config = None
        self.aux_dim = aux_expert_config.aux_dim if aux_expert_config is not None else 0
        
        self.condition_aux_on_timestep = config.condition_aux_on_timestep
        # if condition_aux_on_timestep is true, aux expert CANNOT USE FLOWMATCHING
        # this is because we only condition on regression models
        assert not (self.condition_aux_on_timestep and config.use_flow_matching), "condition_aux_on_timestep and use_flow_matching cannot be both true."
        
        self.use_new_head = config.use_new_head 
        assert not (self.use_new_head and config.aux_expert_type == "none"), "use_new_head is only for auxiliary experts."
        # we cannot use self.use_new_head with allow_aux_to_attend_suffix
        assert not (self.use_new_head and config.allow_aux_to_attend_suffix), "use_new_head cannot be used with allow_aux_to_attend_suffix."
        
        use_adarms = [False, True] if self.pi05 else [False, False]
        if config.aux_expert_type in ["point", "depth"]:
            if self.config.use_flow_matching:
                use_adarms.append(True)  # use adaRMSNorm for flow matching aux expert
            else:
                use_adarms.append(False) # no adaRMSNorm for regression aux expert
                
        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            aux_expert_config, 
            aux_expert_type=config.aux_expert_type, # pass aux_expert_type to the model
            # use_adarms=[False, True] if self.pi05 else [False, False],
            use_adarms=use_adarms,
            precision=config.dtype,
            use_new_head=self.use_new_head,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        self.use_flow_matching = config.use_flow_matching # this is for the aux head
        if config.aux_expert_type in ["point", "depth"]:
            self.aux_in_proj = nn.Linear(self.aux_dim, aux_expert_config.width)
            self.aux_out_proj = nn.Linear(aux_expert_config.width, self.aux_dim)
            if not config.use_flow_matching:
                # for regression tasks, we use learnable query tokens
                self.aux_query_tokens = nn.Parameter(torch.zeros(1, self.config.action_horizon, aux_expert_config.width))
                if self.condition_aux_on_timestep:
                    # project adarms_cond to use it for aux expert
                    self.adarms_cond_to_aux = nn.Linear(action_expert_config.width, aux_expert_config.width)
            else:
                self.aux_query_tokens = None
            # else: # we use the same time_embeding after MLP as the main action head
            #     self.aux_time_mlp_in = nn.Linear(aux_expert_config.width, aux_expert_config.width)
            #     self.aux_time_mlp_out = nn.Linear(aux_expert_config.width, aux_expert_config.width)
                
                
        # # Optional training-only auxiliary head.
        # # This is intentionally lightweight and can be disabled by setting aux_loss_weight=0.
        # self.aux_loss_weight = float(getattr(config, "aux_loss_weight", 0.0))
        # self.aux_action_out_proj = (
        #     nn.Linear(action_expert_config.width, 32) if self.aux_loss_weight > 0.0 else None
        # )

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
            if self.use_flow_matching and config.aux_expert_type in ["point", "depth"]:
                # even if we use pi0 with flow matching, we use adaRMSNorm for aux expert
                self.time_mlp_in = nn.Linear(aux_expert_config.width, aux_expert_config.width)
                self.time_mlp_out = nn.Linear(aux_expert_config.width, aux_expert_config.width)

        torch.set_float32_matmul_precision("high")
        # NOTE: Do not eagerly torch.compile() here.
        # The model is typically moved to the target device (e.g. CUDA) after construction.
        # Compiling while parameters are still on CPU can lead to Dynamo device propagation
        # errors later when calling the compiled function on CUDA.
        self._sample_actions_compiled = False

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None
        
        self.stats = None

    def compile_sample_actions(self, *, mode: str = "max-autotune") -> None:
        """Optionally compile `sample_actions` after the model is on the target device."""
        if self._sample_actions_compiled:
            return
        # Only compile if torch.compile exists (PyTorch 2.x) and Dynamo is available.
        if not hasattr(torch, "compile"):
            return
        self.sample_actions = torch.compile(self.sample_actions, mode=mode)
        self._sample_actions_compiled = True

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True
        if self.paligemma_with_expert.aux_expert is not None:
            aux = self.paligemma_with_expert.aux_expert
            # Old aux expert is a HF GemmaForCausalLM with `.model.gradient_checkpointing`.
            if hasattr(aux, "model") and hasattr(aux.model, "gradient_checkpointing"):
                aux.model.gradient_checkpointing = True
            # New aux head is a plain nn.Module.
            elif hasattr(aux, "gradient_checkpointing"):
                aux.gradient_checkpointing = True
            elif hasattr(aux, "gradient_checkpointing_enable"):
                aux.gradient_checkpointing_enable()

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False
        if self.paligemma_with_expert.aux_expert is not None:
            aux = self.paligemma_with_expert.aux_expert
            if hasattr(aux, "model") and hasattr(aux.model, "gradient_checkpointing"):
                aux.model.gradient_checkpointing = False
            elif hasattr(aux, "gradient_checkpointing"):
                aux.gradient_checkpointing = False
            elif hasattr(aux, "gradient_checkpointing_disable"):
                aux.gradient_checkpointing_disable()

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )
    def get_stats_from_loader(self, loader):
        """Helper method to get point cloud stats from dataset."""
        if hasattr(loader._data_loader._data_loader.dataset, "meta"):
            self.stats = loader._data_loader._data_loader.dataset.meta.stats
        elif hasattr(loader._data_loader._data_loader.dataset._dataset, "meta"):
            self.stats = loader._data_loader._data_loader.dataset._dataset.meta.stats
        elif hasattr(loader._data_loader._data_loader.dataset._dataset._dataset, "meta"):
            self.stats = loader._data_loader._data_loader.dataset._dataset._dataset.meta.stats

        else:
            raise ValueError("Cannot find dataset meta to get stats.")
    
    def _preprocess_point_cloud(self, point_cloud, *, train=True):
        """Helper method to preprocess point cloud."""
        input_point_cloud, output_point_delta = _preprocessing.preprocess_point_cloud_pytorch(point_cloud, self.stats, train=train)
        return input_point_cloud, output_point_delta

    def _preprocess_depth(self, depth_image, *, train=True):
        """Helper method to preprocess depth sequence into aux tokens."""
        input_depth_token, target_depth_tokens = _preprocessing.preprocess_depth_pytorch(depth_image, train=train)
        return input_depth_token, target_depth_tokens
    
    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs)) # (Bs, num_img_embs)
            # num_img_embs = 256, as there is 3 images, and the last one is padded, it is
            # 1s for 512, then 0s for [512:768]
            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs # 0 means attend, 1 meand mask

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        # att_masks is a (Bs, Seq_len) tensor with all 0 values which means
        # full attention for lang and img
        # pad_mask is a (Bs, num_image_embs*3 + max_token_len) tensor
        # (Bs, 256 *3 + 200)
        
        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05: # if pi0
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
            if self.use_flow_matching:
                time_emb = time_emb[:, 0, :]  # (BS, D)
                
                def time_mlp_func(time_emb):
                    x = self.time_mlp_in(time_emb)
                    x = F.silu(x)  # swish == silu
                    x = self.time_mlp_out(x)
                    return F.silu(x)
                time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
                adarms_cond = time_emb
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb) # (BS, T, D)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # for pi05
        # pad masks: (Bs, action_horizon) -> all Trues
        # (Bs, 10) 
        # att_masks (Bs, action_horizon) -> [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        # (Bs, 10)
        return embs, pad_masks, att_masks, adarms_cond
    
    def embed_aux(self, aux_input, aux_target, BS, device, aux_x_t=None, adarms_cond=None, t_aux=None): 
        
        if aux_input is None or aux_target is None:
            embs = None
            aux_pad_masks = torch.empty(BS, 0, dtype=torch.bool, device=device)
            aux_att_masks = torch.empty(BS, 0, dtype=torch.bool, device=device)
            return embs, aux_pad_masks, aux_att_masks
        
        embs = []
        pad_masks = []
        att_masks = []
        
        if self.config.aux_expert_type == "point":
            # point cloud auxiliary expert
            # aux_input: (BS, 1, N, D)
            # aux_target: (BS, T, N, D)
            bsize, T, D = aux_target.shape

            # project input point cloud
            def aux_in_proj_func(aux_input):
                return self.aux_in_proj(aux_input)

            aux_input_proj = self._apply_checkpoint(aux_in_proj_func, aux_input) # (BS, 1, width)
            embs.append(aux_input_proj)

            # New head: add a simple aux-time token (independent of action timestep).
            if self.use_new_head:
                if t_aux is None:
                    raise ValueError("t_aux must be provided when use_new_head=True")
                aux_time_emb = create_sinusoidal_pos_embedding(
                    t_aux,
                    aux_input_proj.shape[-1],
                    min_period=4e-3,
                    max_period=4.0,
                    device=t_aux.device,
                ).to(dtype=aux_input_proj.dtype)
                embs.append(aux_time_emb[:, None, :])
            
            # use learnable query tokens for point cloud regression
            if self.aux_query_tokens is not None:
                
                if self.condition_aux_on_timestep and (not self.use_new_head): # condition aux expert on action expert timestep
                    assert adarms_cond is not None, "adarms_cond must be provided when conditioning aux on timestep."
                    # project adarms_cond to use it for aux expert
                    def adarms_cond_to_aux(adarms_cond):
                        return self.adarms_cond_to_aux(adarms_cond)
                    adarms_cond_proj = self._apply_checkpoint(adarms_cond_to_aux, adarms_cond) # (BS, width)
                    embs.append(adarms_cond_proj[:, None, :]) # (BS, 1, width)
                    
                aux_query_tokens = self.aux_query_tokens.expand(bsize, self.config.action_horizon, -1) # (BS, 1, width)
                embs.append(aux_query_tokens)
            else: # aux_query_tokens is NONE
                assert aux_x_t is not None, "aux_x_t must be provided when using flow matching."
                # project aux_x_t to get query tokens
                aux_x_t_proj = self._apply_checkpoint(aux_in_proj_func, aux_x_t) # (BS, T, width)
                embs.append(aux_x_t_proj)
            
            # # all auxilary inputs are valid
            # aux_pad_masks = torch.ones(aux_target.shape[0], 1 + aux_target.shape[1], dtype=torch.bool, device=aux_target.device)
            # pad_masks.append(aux_pad_masks)
            
            # # 2d attention will be updated anyways so this is a place holder for the sake of compatibility
            # aux_att_masks = torch.zeros(aux_target.shape[0], 1 + aux_target.shape[1], dtype=torch.bool, device=aux_target.device)
            # att_masks.append(aux_att_masks)
            
        elif self.config.aux_expert_type == "depth":
            # depth auxiliary expert
            # aux_input: (BS, 1, aux_dim=1024)
            # aux_target: (BS, T, aux_dim=1024)

            bsize = aux_target.shape[0]

            def aux_in_proj_func(aux_input):
                return self.aux_in_proj(aux_input)

            aux_input_proj = self._apply_checkpoint(aux_in_proj_func, aux_input)  # (BS, 1, width)
            embs.append(aux_input_proj)

            # New head: add a simple aux-time token (independent of action timestep).
            if self.use_new_head:
                if t_aux is None:
                    raise ValueError("t_aux must be provided when use_new_head=True")
                aux_time_emb = create_sinusoidal_pos_embedding(
                    t_aux,
                    aux_input_proj.shape[-1],
                    min_period=4e-3,
                    max_period=4.0,
                    device=t_aux.device,
                ).to(dtype=aux_input_proj.dtype)
                embs.append(aux_time_emb[:, None, :])

            if self.aux_query_tokens is not None:
                
                if self.condition_aux_on_timestep and (not self.use_new_head): # condition aux expert on action expert timestep
                    assert adarms_cond is not None, "adarms_cond must be provided when conditioning aux on timestep."
                    # project adarms_cond to use it for aux expert
                    def adarms_cond_to_aux(adarms_cond):
                        return self.adarms_cond_to_aux(adarms_cond)
                    adarms_cond_proj = self._apply_checkpoint(adarms_cond_to_aux, adarms_cond) # (BS, width)
                    embs.append(adarms_cond_proj[:, None, :]) # (BS, 1, width)
                    
                    
                # Learnable query tokens for future depth forecasting.
                aux_query_tokens = self.aux_query_tokens.expand(bsize, self.config.action_horizon, -1)
                embs.append(aux_query_tokens)
            else:
                assert aux_x_t is not None, "aux_x_t must be provided when using flow matching."
                # project aux_x_t to get query tokens
                aux_x_t_proj = self._apply_checkpoint(aux_in_proj_func, aux_x_t) # (BS, T, width)
                embs.append(aux_x_t_proj)

            # aux_pad_masks = torch.ones(
            #     bsize,
            #     1 + self.config.action_horizon,
            #     dtype=torch.bool,
            #     device=aux_target.device,
            # )
            # pad_masks.append(aux_pad_masks)

            # aux_att_masks = torch.zeros(
            #     bsize,
            #     1 + self.config.action_horizon,
            #     dtype=torch.bool,
            #     device=aux_target.device,
            # )
            # att_masks.append(aux_att_masks)
        else:
            raise ValueError(f"Unknown auxiliary expert type: {self.config.aux_expert_type}")
        
        embs = torch.cat(embs, dim=1)
        # pad_masks = torch.cat(pad_masks, dim=1)
        # att_masks = torch.cat(att_masks, dim=1)
        pad_masks = torch.ones(embs.shape[0], embs.shape[1], dtype=torch.bool, device=embs.device)
        att_masks = torch.zeros(embs.shape[0], embs.shape[1], dtype=torch.bool, device=embs.device)
        
        return embs, pad_masks, att_masks
    
    def forward(self, observation, actions, noise=None, time=None, aux_noise=None, return_aux=False) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        # observation.state (B, state_dim)
        # observation.images
        #   - "base_0_rgb"
        #   - "left_wrist_0_rgb"
        #   - "right_wrist_0_rgb"
        # point_cloud - (B,N, 1024, 3)
        # depth_features - (B, T, 16*16*4)
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        
        point_cloud = None
        # check if they have the point_cloud attribute
        if hasattr(observation, "point_cloud") and observation.point_cloud is not None and self.config.aux_expert_type == "point":
            input_point_cloud, output_point_delta = self._preprocess_point_cloud(observation.point_cloud, train=True)
            assert output_point_delta.shape[1] == self.config.action_horizon, f"output_point_delta.shape[1] ({output_point_delta.shape[1]}) must be equal to action_horizon ({self.config.action_horizon})"
            BS, _, original_N, D = input_point_cloud.shape
            sample_stride = original_N // (self.aux_dim//3) # 1024 // (512*3 // 3) = 2
            input_point_cloud = input_point_cloud[:,:,::sample_stride,:] # input_point_cloud: (BS, 1, N, D)  -> (BS, 1, N//sample_stride, D)
            output_point_delta = output_point_delta[:,:,::sample_stride,:] # output_point_delta: (BS, T, N, D) -> (BS, T, N//sample_stride, D)

            aux_input = input_point_cloud.reshape(input_point_cloud.shape[0], input_point_cloud.shape[1], -1) # input_point_cloud: (BS, 1, N, D) -> aux_input (BS, 1, N*D)
            aux_target = output_point_delta.reshape(output_point_delta.shape[0], output_point_delta.shape[1], -1) # output_point_delta: (BS, T, N, D) -> aux_target (BS, T, N*D)

            # aux_embs, aux_pad_masks, aux_att_masks = self.embed_aux(aux_input, aux_target)

        elif hasattr(observation, "depth_image") and observation.depth_image is not None and self.config.aux_expert_type == "depth":
            input_depth_token, target_depth_tokens = self._preprocess_depth(observation.depth_image, train=True)
            assert target_depth_tokens.shape[1] == self.config.action_horizon, (
                f"target_depth_tokens.shape[1] ({target_depth_tokens.shape[1]}) must be equal to action_horizon ({self.config.action_horizon})"
            )
            aux_input = input_depth_token  # (BS, 1, 1024)
            aux_target = target_depth_tokens  # (BS, T, 1024)
            # aux_embs, aux_pad_masks, aux_att_masks = self.embed_aux(aux_input, aux_target)
        else:
            # print("hasattr(observation, 'point_cloud'):", hasattr(observation, "point_cloud"))
            # print("observation.point_cloud is None:", observation.point_cloud is None if hasattr(observation, "point_cloud") else "N/A")
            # print("No auxiliary expert input found. aux_embs is NONE")
            # aux_embs = None
            aux_input = None
            aux_target = None
            # aux_pad_masks = torch.empty(state.shape[0], 0, dtype=torch.bool, device=state.device)
            # aux_att_masks = torch.empty(state.shape[0], 0, dtype=torch.bool, device=state.device)
        
        
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
            
        if aux_noise is None:
            aux_noise = self.sample_noise(aux_target.shape, aux_target.device) if aux_target is not None else None

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        
        
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # Aux head time for the new head (kept independent from action diffusion time).
        t_aux = None
        if self.use_new_head and aux_target is not None:
            if self.config.use_flow_matching:
                t_aux = self.sample_time(actions.shape[0], actions.device)
            else:
                t_aux = torch.zeros(actions.shape[0], dtype=torch.float32, device=actions.device)

        if self.config.use_flow_matching:
            aux_time_expanded = time_expanded if (not self.use_new_head) else (t_aux[:, None, None] if t_aux is not None else time_expanded)
            aux_x_t = aux_time_expanded * aux_noise + (1 - aux_time_expanded) * aux_target if aux_target is not None else None
            aux_u_t = aux_noise - aux_target if aux_target is not None else None
            aux_target = aux_u_t  # for flow matching, the target is the velocity field
        else:
            aux_x_t = None
        
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        adarms_cond_for_aux = None if self.use_new_head else (adarms_cond if self.condition_aux_on_timestep else None)
        aux_embs, aux_pad_masks, aux_att_masks = self.embed_aux(
            aux_input,
            aux_target,
            state.shape[0],
            state.device,
            aux_x_t=aux_x_t,
            adarms_cond=adarms_cond_for_aux,
            t_aux=t_aux,
        )
        
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
            aux_embs = aux_embs.to(dtype=torch.bfloat16) if aux_embs is not None else None
            # aux_target = aux_target.to(dtype=torch.bfloat16) if aux_target is not None else None

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks, aux_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks, aux_att_masks], dim=1)
        
        ## for debugging
        # print("prefix_pad_masks:", prefix_pad_masks[0])1
        # import numpy as np
        # np.save("prefix_pad_masks.npy", prefix_pad_masks.cpu().numpy())
        # np.save("suffix_pad_masks.npy", suffix_pad_masks.cpu().numpy())
        # np.save("prefix_att_masks.npy", prefix_att_masks.cpu().numpy())
        # np.save("suffix_att_masks.npy", suffix_att_masks.cpu().numpy())
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

        
        if aux_embs is not None and aux_target is not None:
            # when using auxiliary expert, we have to change the attention map
            prefix_len = prefix_embs.shape[1]
            suffix_len = suffix_embs.shape[1]
            aux_start = prefix_len + suffix_len
            aux_end = aux_start + aux_embs.shape[1]
            att_2d_masks = enforce_aux_attention_masks(att_2d_masks, aux_pad_masks, aux_start, aux_end, prefix_len, \
                suffix_len, allow_aux_to_attend_suffix=self.config.allow_aux_to_attend_suffix)
            # import numpy as np
            # from PIL import Image  
            # # np.save("att_2d_masks_new.npy", att_2d_masks.cpu().numpy())
            # att_2d_masks_cpu = att_2d_masks[0].cpu().numpy().astype(np.uint8) * 255
            # Image.fromarray(att_2d_masks_cpu).save("att_2d_masks_new_debug.png")
        
        position_ids = torch.cumsum(pad_masks, dim=1) - 1


        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)
        # np.save("att_2d_masks_4d.npy", att_2d_masks_4d.cpu().numpy())
        # Apply gradient checkpointing if enabled
        # print("aux_embs shape:", aux_embs.shape if aux_embs is not None else "None" )
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out, aux_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs, aux_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond, adarms_cond] if self.use_flow_matching else [None, adarms_cond, None],
            )
            return suffix_out, aux_out

        suffix_out, aux_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        if aux_out is not None:
            # compute auxiliary loss
            aux_suffix_out = aux_out[:, -aux_target.shape[1]:, :]  # (BS, T, D)
            aux_suffix_out = aux_suffix_out.to(dtype=torch.float32)
            def aux_out_proj_func(aux_suffix_out):
                return self.aux_out_proj(aux_suffix_out)
            aux_pred = self._apply_checkpoint(aux_out_proj_func, aux_suffix_out)  # (BS, T, aux_dim)
            
            # aux_loss = F.mse_loss(aux_pred, aux_target, reduction="none").mean(dim=-1)  # (BS, T)   
            aux_loss = F.mse_loss(aux_pred, aux_target, reduction="mean")  # scalar
            # aux_loss = aux_loss * self.config.aux_loss_weight
        else:
            aux_loss = torch.tensor(0.0, device=state.device) # for compatibility
            
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        # loss = F.mse_loss(u_t, v_t, reduction="none")
        loss = F.mse_loss(u_t, v_t, reduction="mean")

        # Training-only auxiliary loss. Kept fully optional so inference behavior is unchanged.
        # if self.training and self.aux_action_out_proj is not None and self.aux_loss_weight > 0.0:
        #     aux_v_t = self._apply_checkpoint(self.aux_action_out_proj, suffix_out)
        #     aux_loss = F.mse_loss(u_t, aux_v_t, reduction="none")
        #     loss = loss + (self.aux_loss_weight * aux_loss)

        return loss, aux_loss

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)


    @torch.no_grad()
    def predict_actions_and_point_tracks(
        self,
        device: torch.device,
        observation,
        *,
        num_action_steps: int = 10,
        num_aux_steps: int = 10,
        norm_bug: bool = False,
    ):
        """Predict actions and point tracks (as per-step point deltas) for visualization.

        Returns:
          actions_pred: (B, action_horizon, action_dim)
          p0: (B, N, 3) raw point cloud at t=0 (sampled to match aux_dim)
          delta_pred: (B, action_horizon, N, 3) predicted per-step deltas in raw units
        """
        if self.config.aux_expert_type != "point":
            raise ValueError(
                f"predict_actions_and_point_tracks only supports aux_expert_type='point' (got {self.config.aux_expert_type!r})"
            )
        if self.stats is None:
            raise ValueError("Point cloud stats are not set. Call get_stats_from_loader(...) before inference.")
        if not hasattr(observation, "point_cloud") or observation.point_cloud is None:
            raise ValueError("Observation has no point_cloud for point-track visualization.")

        self.eval()

        # 1) Predict actions using the model's existing diffusion sampler.
        actions_pred = self.sample_actions(device, observation, num_steps=num_action_steps)

        # 2) Prepare prefix/suffix embeddings for aux prediction.
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Build point-cloud aux input (sample points so that N*3 == aux_dim).
        point_cloud = observation.point_cloud
        if point_cloud.ndim != 4 or point_cloud.shape[-1] != 3:
            raise ValueError(f"Expected point_cloud shape (B, T, N, 3), got {tuple(point_cloud.shape)}")

        p0_full = point_cloud[:, 0, :, :]  # (B, N, 3) in raw units
        original_n = p0_full.shape[1]
        expected_n = self.aux_dim // 3
        if expected_n <= 0:
            raise ValueError(f"Invalid aux_dim={self.aux_dim} for point tracks")
        if original_n % expected_n != 0:
            raise ValueError(
                f"Point count {original_n} is not divisible by expected aux points {expected_n} (aux_dim={self.aux_dim})."
            )
        sample_stride = original_n // expected_n
        p0 = p0_full[:, ::sample_stride, :]  # (B, expected_n, 3)

        # Normalize input point cloud using dataset stats (matches preprocess_point_cloud_pytorch behavior).
        mean = torch.tensor(self.stats["point_cloud"]["mean"], device=p0.device).squeeze().to(p0.dtype)
        std = torch.tensor(self.stats["point_cloud"]["std"], device=p0.device).squeeze().to(p0.dtype)
        p0_norm = (p0 - mean) / std
        aux_input = p0_norm.reshape(p0_norm.shape[0], 1, -1)  # (B, 1, aux_dim)

        bsize = aux_input.shape[0]

        def denorm_deltas(delta_norm: torch.Tensor) -> torch.Tensor:
            delta_mean = torch.tensor(self.stats["point_cloud"]["delta_mean"], device=delta_norm.device).squeeze().to(
                delta_norm.dtype
            )
            delta_std = torch.tensor(self.stats["point_cloud"]["delta_std"], device=delta_norm.device).squeeze().to(
                delta_norm.dtype
            )
            return (delta_norm * delta_std) + delta_mean

        # 3) Predict normalized deltas (aux head).
        if not self.use_flow_matching:
            # Regression aux head: use learnable query tokens; optionally condition on timestep.
            time = torch.zeros(bsize, dtype=torch.float32, device=device)
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, actions_pred, time)

            dummy_aux_target = torch.zeros(
                bsize,
                self.config.action_horizon,
                self.aux_dim,
                dtype=aux_input.dtype,
                device=device,
            )
            aux_embs, aux_pad_masks, aux_att_masks = self.embed_aux(
                aux_input,
                dummy_aux_target,
                bsize,
                device,
                aux_x_t=None,
                adarms_cond=None if self.use_new_head else (adarms_cond if self.condition_aux_on_timestep else None),
                t_aux=torch.zeros(bsize, dtype=torch.float32, device=device) if self.use_new_head else None,
            )

            # Cast to bfloat16 if the backbone runs in bf16.
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                prefix_embs = prefix_embs.to(dtype=torch.bfloat16)
                suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                aux_embs = aux_embs.to(dtype=torch.bfloat16) if aux_embs is not None else None

            pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks, aux_pad_masks], dim=1)
            att_masks = torch.cat([prefix_att_masks, suffix_att_masks, aux_att_masks], dim=1)
            att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

            prefix_len = prefix_embs.shape[1]
            suffix_len = suffix_embs.shape[1]
            aux_start = prefix_len + suffix_len
            aux_end = aux_start + aux_embs.shape[1]
            att_2d_masks = enforce_aux_attention_masks(
                att_2d_masks,
                aux_pad_masks,
                aux_start,
                aux_end,
                prefix_len,
                suffix_len,
                allow_aux_to_attend_suffix=self.config.allow_aux_to_attend_suffix,
            )

            position_ids = torch.cumsum(pad_masks, dim=1) - 1
            att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

            (outputs_embeds, _) = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs, aux_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond, None],
            )
            aux_out = outputs_embeds[2]
            aux_suffix_out = aux_out[:, -self.config.action_horizon :, :].to(dtype=torch.float32)
            aux_pred_norm = self.aux_out_proj(aux_suffix_out)  # (B, T, aux_dim)

        else:
            # Flow matching aux head: Euler-integrate aux deltas while holding actions fixed.
            # This is meant for visualization/debug and is not optimized.
            dt = -1.0 / float(num_aux_steps)
            dt = torch.tensor(dt, dtype=torch.float32, device=device)

            aux_x_t = self.sample_noise((bsize, self.config.action_horizon, self.aux_dim), device)

            time_scalar = torch.tensor(1.0, dtype=torch.float32, device=device)

            # Cache prefix embeds/masks since they don't change across steps.
            prefix_embs_cached = prefix_embs
            prefix_pad_masks_cached = prefix_pad_masks
            prefix_att_masks_cached = prefix_att_masks

            while time_scalar >= -dt / 2:
                time = time_scalar.expand(bsize)
                suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, actions_pred, time)

                dummy_aux_target = torch.zeros(
                    bsize,
                    self.config.action_horizon,
                    self.aux_dim,
                    dtype=aux_x_t.dtype,
                    device=device,
                )
                aux_embs, aux_pad_masks, aux_att_masks = self.embed_aux(
                    aux_input,
                    dummy_aux_target,
                    bsize,
                    device,
                    aux_x_t=aux_x_t,
                    adarms_cond=None if self.use_new_head else (adarms_cond if self.condition_aux_on_timestep else None),
                    t_aux=time if self.use_new_head else None,
                )

                if (
                    self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                    == torch.bfloat16
                ):
                    prefix_embs_step = prefix_embs_cached.to(dtype=torch.bfloat16)
                    suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
                    aux_embs = aux_embs.to(dtype=torch.bfloat16) if aux_embs is not None else None
                else:
                    prefix_embs_step = prefix_embs_cached

                pad_masks = torch.cat([prefix_pad_masks_cached, suffix_pad_masks, aux_pad_masks], dim=1)
                att_masks = torch.cat([prefix_att_masks_cached, suffix_att_masks, aux_att_masks], dim=1)
                att_2d_masks = make_att_2d_masks(pad_masks, att_masks)

                prefix_len = prefix_embs_step.shape[1]
                suffix_len = suffix_embs.shape[1]
                aux_start = prefix_len + suffix_len
                aux_end = aux_start + aux_embs.shape[1]
                att_2d_masks = enforce_aux_attention_masks(
                    att_2d_masks,
                    aux_pad_masks,
                    aux_start,
                    aux_end,
                    prefix_len,
                    suffix_len,
                    allow_aux_to_attend_suffix=self.config.allow_aux_to_attend_suffix,
                )

                position_ids = torch.cumsum(pad_masks, dim=1) - 1
                att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

                (outputs_embeds, _) = self.paligemma_with_expert.forward(
                    attention_mask=att_2d_masks_4d,
                    position_ids=position_ids,
                    past_key_values=None,
                    inputs_embeds=[prefix_embs_step, suffix_embs, aux_embs],
                    use_cache=False,
                    adarms_cond=[None, adarms_cond, adarms_cond],
                )
                aux_out = outputs_embeds[2]
                aux_suffix_out = aux_out[:, -self.config.action_horizon :, :].to(dtype=torch.float32)
                v_aux = self.aux_out_proj(aux_suffix_out)  # velocity in normalized space

                aux_x_t = aux_x_t + dt * v_aux
                time_scalar = time_scalar + dt

            aux_pred_norm = aux_x_t

        # 4) Convert to per-point deltas in raw units.
        delta_pred_norm = aux_pred_norm.reshape(bsize, self.config.action_horizon, expected_n, 3)
        
        if not norm_bug: # this is the way it is supposed to be
            delta_pred = denorm_deltas(delta_pred_norm)
            point_tracks = p0[:, None, :, :] + torch.cumsum(delta_pred, dim=1)  # (B, T, N, 3)
        else: # this is when there was a norm bug, where we first get the pts and then denorm
            point_tracks_norm = p0_norm[:, None, :, :] + torch.cumsum(delta_pred_norm, dim=1)  # (B, T, N, 3)
            point_tracks = point_tracks_norm * std[None, None, :, :] + mean[None, None, :, :]

        # concat along time dimension
        point_tracks = torch.cat([p0[:, None, :, :], point_tracks], dim=1)  # (B, T+1, N, 3)
        
        # return actions_pred, p0, delta_pred
        return actions_pred, point_tracks
