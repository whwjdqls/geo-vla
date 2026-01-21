from __future__ import annotations

import math

import torch
from torch import nn


class _AuxBlock(nn.Module):
	def __init__(
		self,
		*,
		aux_dim: int,
		prefix_dim: int,
		num_heads: int,
		mlp_dim: int,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		self.aux_dim = aux_dim

		self.ln_self = nn.LayerNorm(aux_dim)
		self.self_attn = nn.MultiheadAttention(
			embed_dim=aux_dim,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=True,
		)

		self.ln_cross = nn.LayerNorm(aux_dim)
		# If dims match, keep it identity to avoid extra parameters.
		self.kv_proj = nn.Identity() if prefix_dim == aux_dim else nn.Linear(prefix_dim, aux_dim, bias=False)
		self.cross_attn = nn.MultiheadAttention(
			embed_dim=aux_dim,
			num_heads=num_heads,
			dropout=dropout,
			batch_first=True,
		)

		self.ln_mlp = nn.LayerNorm(aux_dim)
		self.mlp = nn.Sequential(
			nn.Linear(aux_dim, mlp_dim),
			nn.GELU(),
			nn.Linear(mlp_dim, aux_dim),
		)
		self.dropout = nn.Dropout(dropout)

	def forward(
		self,
		aux_tokens: torch.Tensor,
		*,
		prefix_tokens: torch.Tensor,
		prefix_key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		# aux_tokens: (B, L_aux, D_aux)
		# prefix_tokens: (B, L_prefix, D_prefix)

		x = aux_tokens

		# Self-attn over aux tokens
		h = self.ln_self(x)
		h, _ = self.self_attn(
			query=h,
			key=h,
			value=h,
			need_weights=False,
		)
		x = x + self.dropout(h)

		# Cross-attn: aux queries attend to prefix keys/values
		h = self.ln_cross(x)
		kv = self.kv_proj(prefix_tokens)
		h, _ = self.cross_attn(
			query=h,
			key=kv,
			value=kv,
			key_padding_mask=prefix_key_padding_mask,
			need_weights=False,
		)
		x = x + self.dropout(h)

		# MLP
		h = self.ln_mlp(x)
		h = self.mlp(h)
		x = x + self.dropout(h)

		return x


class NewAuxHead(nn.Module):
	"""Aux head that predicts aux token outputs by cross-attending to prefix outputs.

	Contract:
	- Input `aux_tokens` is the same sequence used for the aux loss.
	- Output preserves token order; training can slice the last T tokens.
	"""

	def __init__(
		self,
		*,
		aux_dim: int,
		prefix_dim: int,
		depth: int,
		num_heads: int,
		mlp_dim: int,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		if aux_dim % num_heads != 0:
			raise ValueError(f"aux_dim ({aux_dim}) must be divisible by num_heads ({num_heads})")
		self.aux_dim = aux_dim
		self.prefix_dim = prefix_dim

		self.blocks = nn.ModuleList(
			[
				_AuxBlock(
					aux_dim=aux_dim,
					prefix_dim=prefix_dim,
					num_heads=num_heads,
					mlp_dim=mlp_dim,
					dropout=dropout,
				)
				for _ in range(depth)
			]
		)
		self.final_norm = nn.LayerNorm(aux_dim)

	def forward(
		self,
		*,
		prefix_tokens: torch.Tensor,
		aux_tokens: torch.Tensor,
		prefix_key_padding_mask: torch.Tensor | None = None,
	) -> torch.Tensor:
		x = aux_tokens
		for block in self.blocks:
			x = block(x, prefix_tokens=prefix_tokens, prefix_key_padding_mask=prefix_key_padding_mask)
		return self.final_norm(x)

