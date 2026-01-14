from typing import Literal

import pytest
import torch
from torch import nn
from transformers import GemmaForCausalLM
from transformers import PaliGemmaForConditionalGeneration
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.gemma import modeling_gemma


class PaliGemmaWithExpertModel(nn.Module):
    def __init__(
        self,
        vlm_config,
        action_expert_config,
        aux_expert_config=None,
        aux_expert_type="none",
        use_adarms=None,
        precision: Literal["bfloat16", "float32"] = "bfloat16",
    ):
        if use_adarms is None:
            use_adarms = [False, False, False]
        super().__init__()

        vlm_config_hf = CONFIG_MAPPING["paligemma"]()
        vlm_config_hf._vocab_size = 257152  # noqa: SLF001
        vlm_config_hf.image_token_index = 257152
        vlm_config_hf.text_config.hidden_size = vlm_config.width
        vlm_config_hf.text_config.intermediate_size = vlm_config.mlp_dim
        vlm_config_hf.text_config.num_attention_heads = vlm_config.num_heads
        vlm_config_hf.text_config.head_dim = vlm_config.head_dim
        vlm_config_hf.text_config.num_hidden_layers = vlm_config.depth
        vlm_config_hf.text_config.num_key_value_heads = vlm_config.num_kv_heads
        vlm_config_hf.text_config.hidden_activation = "gelu_pytorch_tanh"
        vlm_config_hf.text_config.torch_dtype = "float32"
        vlm_config_hf.text_config.vocab_size = 257152
        vlm_config_hf.text_config.use_adarms = use_adarms[0]
        vlm_config_hf.text_config.adarms_cond_dim = vlm_config.width if use_adarms[0] else None
        vlm_config_hf.vision_config.intermediate_size = 4304
        vlm_config_hf.vision_config.projection_dim = 2048
        vlm_config_hf.vision_config.projector_hidden_act = "gelu_fast"
        vlm_config_hf.vision_config.torch_dtype = "float32"

        action_expert_config_hf = CONFIG_MAPPING["gemma"](
            head_dim=action_expert_config.head_dim,
            hidden_size=action_expert_config.width,
            intermediate_size=action_expert_config.mlp_dim,
            num_attention_heads=action_expert_config.num_heads,
            num_hidden_layers=action_expert_config.depth,
            num_key_value_heads=action_expert_config.num_kv_heads,
            vocab_size=257152,
            hidden_activation="gelu_pytorch_tanh",
            torch_dtype="float32",
            use_adarms=use_adarms[1],
            adarms_cond_dim=action_expert_config.width if use_adarms[1] else None,
        )
        ###############################################
        if aux_expert_config is not None:
            aux_expert_config_hf = CONFIG_MAPPING["gemma"](
                head_dim=aux_expert_config.head_dim,
                hidden_size=aux_expert_config.width,
                intermediate_size=aux_expert_config.mlp_dim,
                num_attention_heads=aux_expert_config.num_heads,
                num_hidden_layers=aux_expert_config.depth,
                num_key_value_heads=aux_expert_config.num_kv_heads,
                vocab_size=257152,
                hidden_activation="gelu_pytorch_tanh",
                torch_dtype="float32",
                # use_adarms=False,
                # adarms_cond_dim=None,
                use_adarms=use_adarms[2],
                adarms_cond_dim=aux_expert_config.width if use_adarms[2] else None,
            )
            self.aux_expert = GemmaForCausalLM(config=aux_expert_config_hf)
            self.aux_expert.model.embed_tokens = None
        else:
            self.aux_expert = None
        self.aux_expert_type = aux_expert_type
        ###############################################
        
        self.paligemma = PaliGemmaForConditionalGeneration(config=vlm_config_hf)
        self.gemma_expert = GemmaForCausalLM(config=action_expert_config_hf)
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_for_selected_params(precision)

    def to_bfloat16_for_selected_params(self, precision: Literal["bfloat16", "float32"] = "bfloat16"):
        if precision == "bfloat16":
            self.to(dtype=torch.bfloat16)
        elif precision == "float32":
            self.to(dtype=torch.float32)
            return
        else:
            raise ValueError(f"Invalid precision: {precision}")

        params_to_keep_float32 = [
            "vision_tower.vision_model.embeddings.patch_embedding.weight",
            "vision_tower.vision_model.embeddings.patch_embedding.bias",
            "vision_tower.vision_model.embeddings.position_embedding.weight",
            "input_layernorm",
            "post_attention_layernorm",
            "model.norm",
        ]

        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_keep_float32):
                param.data = param.data.to(dtype=torch.float32)

    def embed_image(self, image: torch.Tensor):
        return self.paligemma.model.get_image_features(image)

    def embed_language_tokens(self, tokens: torch.Tensor):
        return self.paligemma.language_model.embed_tokens(tokens)

    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | pytest.Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] | None = None,
        use_cache: bool | None = None,
        adarms_cond: list[torch.Tensor] | None = None,
    ):
        if adarms_cond is None:
            adarms_cond = [None, None, None]
            
        ### forward only the language model
        if inputs_embeds[1] is None: # only prefix
            prefix_output = self.paligemma.language_model.forward(
                inputs_embeds=inputs_embeds[0],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[0] if adarms_cond is not None else None,
            )
            prefix_past_key_values = prefix_output.past_key_values
            prefix_output = prefix_output.last_hidden_state
            suffix_output = None
            aux_output = None # added for aux
            
        elif inputs_embeds[0] is None: # only suffix
            suffix_output = self.gemma_expert.model.forward(
                inputs_embeds=inputs_embeds[1],
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                adarms_cond=adarms_cond[1] if adarms_cond is not None else None,
            )
            suffix_output = suffix_output.last_hidden_state
            prefix_output = None
            prefix_past_key_values = None
            aux_output = None # added for aux
            
        # this is when only the prefix and suffix are used (no aux expert)
        elif inputs_embeds[0] is not None and inputs_embeds[1] is not None and inputs_embeds[2] is None:
            inputs_embeds = [inputs_embeds[0], inputs_embeds[1]]
            
            models = [self.paligemma.language_model, self.gemma_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers

            # Check if gradient checkpointing is enabled for any of the models
            use_gradient_checkpointing = (
                hasattr(self.gemma_expert.model, "gradient_checkpointing")
                and self.gemma_expert.model.gradient_checkpointing
                and self.training
            ) or (hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training)

            # Force enable gradient checkpointing if we're in training mode and the model supports it
            if self.training and hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                if not self.gemma_expert.model.gradient_checkpointing:
                    print("Forcing gradient checkpointing to be enabled for Gemma expert model")
                    self.gemma_expert.model.gradient_checkpointing = True
                use_gradient_checkpointing = True

            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gemma expert model gradient checkpointing: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                print(
                    f"Gemma expert model has gradient_checkpointing attr: {hasattr(self.gemma_expert.model, 'gradient_checkpointing')}"
                )
                if hasattr(self.gemma_expert.model, "gradient_checkpointing"):
                    print(
                        f"Gemma expert model gradient_checkpointing value: {self.gemma_expert.model.gradient_checkpointing}"
                    )
                self._debug_gc_printed = True

            # Define the complete layer computation function for gradient checkpointing
            def compute_layer_complete(layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond):
                models = [self.paligemma.language_model, self.gemma_expert.model]

                query_states = []
                key_states = []
                value_states = []
                gates = []
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[i])  # noqa: PLW2901
                    gates.append(gate)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(query_state)
                    key_states.append(key_state)
                    value_states.append(value_state)

                # Concatenate and process attention
                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy_tensor = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy_tensor, position_ids)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                # Attention computation
                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    scaling,
                )
                # Get head_dim from the current layer, not from the model
                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim)

                # Process layer outputs
                outputs_embeds = []
                start_pos = 0
                for i, hidden_states in enumerate(inputs_embeds):
                    layer = models[i].layers[layer_idx]
                    end_pos = start_pos + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start_pos:end_pos])

                    # first residual
                    out_emb = modeling_gemma._gated_residual(hidden_states, out_emb, gates[i])  # noqa: SLF001
                    after_first_residual = out_emb.clone()
                    out_emb, gate = layer.post_attention_layernorm(out_emb, cond=adarms_cond[i])
                    # Convert to bfloat16 if the next layer (mlp) uses bfloat16
                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out_emb = out_emb.to(dtype=torch.bfloat16)

                    out_emb = layer.mlp(out_emb)
                    # second residual
                    out_emb = modeling_gemma._gated_residual(after_first_residual, out_emb, gate)  # noqa: SLF001
                    outputs_embeds.append(out_emb)
                    start_pos = end_pos

                return outputs_embeds

            # Process all layers with gradient checkpointing if enabled
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer_complete,
                        layer_idx,
                        inputs_embeds,
                        attention_mask,
                        position_ids,
                        adarms_cond,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer_complete(
                        layer_idx, inputs_embeds, attention_mask, position_ids, adarms_cond
                    )

                # Old code removed - now using compute_layer_complete function above

            # final norm
            # Define final norm computation function for gradient checkpointing
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds

            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            prefix_past_key_values = None
            aux_output = None # added for aux
            
        ## ADDED for aux expert
        elif inputs_embeds[0] is not None and inputs_embeds[1] is not None and inputs_embeds[2] is not None:
            # when aux expert is used
            assert self.aux_expert is not None, "Aux expert model is not initialized."
            assert adarms_cond is not None and len(adarms_cond) == 3, "adarms_cond must be provided for all three models."
            
            models_for_norm = [self.paligemma.language_model, self.gemma_expert.model, self.aux_expert.model]
            num_layers = self.paligemma.config.text_config.num_hidden_layers
            aux_num_layers = self.aux_expert.config.num_hidden_layers
            assert num_layers % aux_num_layers == 0, "Aux expert layers must divide evenly into total layers."
            stride = num_layers // aux_num_layers # e.g., 2 => aux participates every 2 layers
            
            
            ############################## this is with gradient checkpointing ##############################
            def maybe_enable_gradient_checkpointing(model, label):
                if not hasattr(model, "gradient_checkpointing"):
                    return False
                if self.training and not model.gradient_checkpointing:
                    print(f"Forcing gradient checkpointing to be enabled for {label} model")
                    model.gradient_checkpointing = True
                return self.training and model.gradient_checkpointing

            gc_flags = [
                maybe_enable_gradient_checkpointing(self.gemma_expert.model, "Gemma expert"),
                maybe_enable_gradient_checkpointing(self.aux_expert.model, "Aux expert"),
            ]
            use_gradient_checkpointing = any(gc_flags) or (
                hasattr(self, "gradient_checkpointing") and self.gradient_checkpointing and self.training
            )
                
            # Debug gradient checkpointing status
            if hasattr(self, "_debug_gc_printed") and not self._debug_gc_printed:
                print(f"Gradient checkpointing enabled: {use_gradient_checkpointing}")
                print(f"Model training mode: {self.training}")
                for label, model in (
                    ("Gemma expert", self.gemma_expert.model),
                    ("Aux expert", self.aux_expert.model),
                ):
                    print(f"{label} model has gradient_checkpointing attr: {hasattr(model, 'gradient_checkpointing')}")
                    if hasattr(model, "gradient_checkpointing"):
                        print(f"{label} model gradient_checkpointing value: {model.gradient_checkpointing}")
                self._debug_gc_printed = True
            ########################################################################################################################
            
            prefix_len = inputs_embeds[0].shape[1]
            suffix_len = inputs_embeds[1].shape[1]
            aux_len = inputs_embeds[2].shape[1]
            total_len_2 = prefix_len + suffix_len
            total_len_3 = prefix_len + suffix_len + aux_len
            
            def _slice_attn_mask(mask: torch.Tensor | None, total_len: int) -> torch.Tensor | None:
                if mask is None:
                    return None
                # expected [B, 1, Q, KV] or [B, Q] etc; only support the 4D path used by eager_attention_forward
                if mask.ndim == 4:
                    return mask[:, :, :total_len, :total_len]
                if mask.ndim == 2:
                    return mask[:, :total_len]
                # fallback: return as-is (may break if shapes mismatch)
                return mask

            def _slice_pos_ids(pos: torch.LongTensor | None, total_len: int) -> torch.LongTensor | None:
                if pos is None:
                    return None
                return pos[:, :total_len]

            
            
            # NOTE: This keeps aux lightweight by only running its block on every `stride` layers.
            # On other layers, aux hidden state is carried forward unchanged.
            def compute_layer(layer_idx: int, cur_inputs: list[torch.Tensor]):
                include_aux = (layer_idx % stride == 0)
                aux_layer_idx = layer_idx // stride  # only valid when include_aux=True

                # decide which streams are active in attention this layer
                if include_aux:
                    active = [0, 1, 2]
                    models = [self.paligemma.language_model, self.gemma_expert.model, self.aux_expert.model]
                    layer_indices = [layer_idx, layer_idx, aux_layer_idx]
                    total_len = total_len_3
                else:
                    active = [0, 1]
                    models = [self.paligemma.language_model, self.gemma_expert.model]
                    layer_indices = [layer_idx, layer_idx]
                    total_len = total_len_2
                    
                am = _slice_attn_mask(attention_mask, total_len)
                pid = _slice_pos_ids(position_ids, total_len)
                
                # ---- build q/k/v over active streams only ----
                query_states = []
                key_states = []
                value_states = []
                gates = []
                hiddens_active = []

                for j, stream_idx in enumerate(active):
                    hidden_states = cur_inputs[stream_idx]
                    layer = models[j].layers[layer_indices[j]] # this is different for only aux

                    hidden_states, gate = layer.input_layernorm(hidden_states, cond=adarms_cond[stream_idx])
                    gates.append(gate)
                    hiddens_active.append(hidden_states)

                    input_shape = hidden_states.shape[:-1]
                    hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)
                    q = layer.self_attn.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    k = layer.self_attn.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
                    v = layer.self_attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                    query_states.append(q)
                    key_states.append(k)
                    value_states.append(v)

                query_states = torch.cat(query_states, dim=2)
                key_states = torch.cat(key_states, dim=2)
                value_states = torch.cat(value_states, dim=2)

                dummy = torch.zeros(
                    query_states.shape[0],
                    query_states.shape[2],  # total_len
                    query_states.shape[-1],
                    device=query_states.device,
                    dtype=query_states.dtype,
                )
                cos, sin = self.paligemma.model.language_model.rotary_emb(dummy, pid)
                query_states, key_states = modeling_gemma.apply_rotary_pos_emb(
                    query_states, key_states, cos, sin, unsqueeze_dim=1
                )

                batch_size = query_states.shape[0]
                scaling = self.paligemma.language_model.layers[layer_idx].self_attn.scaling

                att_output, _ = modeling_gemma.eager_attention_forward(
                    self.paligemma.language_model.layers[layer_idx].self_attn,
                    query_states,
                    key_states,
                    value_states,
                    am,
                    scaling,
                )

                head_dim = self.paligemma.language_model.layers[layer_idx].self_attn.head_dim
                att_output = att_output.reshape(batch_size, -1, 1 * 8 * head_dim) # just match it to 8

                # ---- split back and run per-stream o_proj + mlp ----
                new_inputs = list(cur_inputs)  # copy (so we can keep aux identity on skipped layers)
                start = 0

                for j, stream_idx in enumerate(active):
                    layer = models[j].layers[layer_indices[j]]
                    hidden_states = cur_inputs[stream_idx]
                    end = start + hidden_states.shape[1]

                    att_slice = att_output[:, start:end]
                    if att_slice.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_slice = att_slice.to(layer.self_attn.o_proj.weight.dtype)

                    out = layer.self_attn.o_proj(att_slice)
                    
                    # first residual
                    out = modeling_gemma._gated_residual(hidden_states, out, gates[j])  # noqa: SLF001
                    after_first = out.clone()
                    out, gate2 = layer.post_attention_layernorm(out, cond=adarms_cond[stream_idx])

                    if layer.mlp.up_proj.weight.dtype == torch.bfloat16:
                        out = out.to(dtype=torch.bfloat16)

                    out = layer.mlp(out)
                    out = modeling_gemma._gated_residual(after_first, out, gate2)  # noqa: SLF001
                    new_inputs[stream_idx] = out
                    start = end

                return new_inputs
            
            # run all layers
            for layer_idx in range(num_layers):
                if use_gradient_checkpointing:
                    inputs_embeds = torch.utils.checkpoint.checkpoint(
                        compute_layer,
                        layer_idx,
                        inputs_embeds,
                        use_reentrant=False,
                        preserve_rng_state=False,
                    )
                else:
                    inputs_embeds = compute_layer(layer_idx, inputs_embeds)
                    
                    
            def compute_final_norms(inputs_embeds, adarms_cond):
                outputs_embeds = []
                for i, hidden_states in enumerate(inputs_embeds):
                    out_emb, _ = models_for_norm[i].norm(hidden_states, cond=adarms_cond[i])
                    outputs_embeds.append(out_emb)
                return outputs_embeds
            
            # Apply gradient checkpointing to final norm if enabled
            if use_gradient_checkpointing:
                outputs_embeds = torch.utils.checkpoint.checkpoint(
                    compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
                )
            else:
                outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            # # final norm
            # # Define final norm computation function for gradient checkpointing
            # def compute_final_norms(inputs_embeds, adarms_cond):
            #     outputs_embeds = []
            #     for i, hidden_states in enumerate(inputs_embeds):
            #         out_emb, _ = models_for_norm[i].norm(hidden_states, cond=adarms_cond[i] if i < 2 else None)
            #         outputs_embeds.append(out_emb)
            #     return outputs_embeds

            # # Apply gradient checkpointing to final norm if enabled
            # if use_gradient_checkpointing:
            #     outputs_embeds = torch.utils.checkpoint.checkpoint(
            #         compute_final_norms, inputs_embeds, adarms_cond, use_reentrant=False, preserve_rng_state=False
            #     )
            # else:
            #     outputs_embeds = compute_final_norms(inputs_embeds, adarms_cond)

            prefix_output = outputs_embeds[0]
            suffix_output = outputs_embeds[1]
            aux_output = outputs_embeds[2]
            prefix_past_key_values = None
        else:
            raise ValueError("Invalid inputs_embeds configuration.")

        # return shape: keep backward compatibility (2 outputs) unless aux is provided
        # if len(inputs_embeds) >= 3 and inputs_embeds[2] is not None:
        #     return [prefix_output, suffix_output, aux_output], prefix_past_key_values
        return [prefix_output, suffix_output, aux_output], prefix_past_key_values            
        # return [prefix_output, suffix_output], prefix_past_key_values
