from functools import partial

import einops
import torch
import torch.nn.functional as F
from kappamodules.attention import DotProductAttention1d
from kappamodules.functional.pos_embed import interpolate_sincos
from kappamodules.transformer import PrenormBlock
from kappamodules.vit import VitPatchEmbed, VitPosEmbed2d, VitClassTokens
from torch import nn

from ksuit.models import SingleModel
from ksuit.optim.param_group_modifiers import WeightDecayByNameModifier
from ksuit.utils.formatting_utils import list_to_string
from ksuit.utils.param_checking import to_ntuple


class Vit(SingleModel):
    def __init__(
            self,
            patch_size,
            dim,
            depth,
            num_attn_heads,
            drop_path_rate=0.,
            drop_path_decay=True,
            num_cls_tokens=1,
            layerscale=1.0,
            use_last_norm=True,
            init_weights="truncnormal002",
            eps=1e-6,
            **kwargs,
    ):
        super().__init__(**kwargs)
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.static_ctx["patch_size"] = self.patch_size
        self.dim = dim
        self.depth = depth
        self.num_attn_heads = num_attn_heads
        self.drop_path_rate = drop_path_rate
        self.eps = eps
        self.use_last_norm = use_last_norm

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
            init_weights=init_weights,
        )
        self.static_ctx["sequence_lengths"] = self.patch_embed.seqlens

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim, is_learnable=True)
        self.logger.info(f"pos_embed.is_learnable={self.pos_embed.is_learnable}")

        # 0, 1 or more cls tokens
        self.cls_tokens = VitClassTokens(dim=dim, num_tokens=num_cls_tokens)
        self.static_ctx["num_aux_tokens"] = num_cls_tokens

        # stochastic depth
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
            self.logger.info(f"using drop_path_decay: {list_to_string(dpr)}")
        else:
            dpr = [drop_path_rate] * depth
            self.logger.info(f"drop_path_rate: {drop_path_rate}")

        # blocks
        self.blocks = nn.ModuleList([
            PrenormBlock(
                dim=dim,
                num_heads=num_attn_heads,
                drop_path=dpr[i],
                layerscale=layerscale,
                eps=eps,
                init_weights=init_weights,
                attn_ctor=partial(DotProductAttention1d, rel_pos_bias="learnable", seqlens=self.patch_embed.seqlens),
            )
            for i in range(depth)
        ])
        if use_last_norm:
            self.norm = nn.LayerNorm(dim, eps=eps)
        else:
            self.norm = nn.Identity()

        # output_shape
        self.output_shape = (self.cls_tokens.num_tokens + self.patch_embed.num_patches, dim)

    def load_state_dict(self, state_dict, strict=True):
        # interpolate for different resolution
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            self.logger.info(f"interpolate pos_embed: {old_pos_embed.shape} -> {self.pos_embed.embed.shape}")
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        # remove last norm
        if not self.use_last_norm:
            state_dict.pop("norm.weight", None)
            state_dict.pop("norm.bias", None)
        # deit3-tiny has merged layernorm into head
        if "head.1.weight" in state_dict:
            state_dict["norm.weight"] = state_dict.pop("head.0.weight")
            state_dict["norm.bias"] = state_dict.pop("head.0.bias")
            state_dict.pop("head.1.weight")
            state_dict.pop("head.1.bias")
        missing_keys, unexpected_keys = super().load_state_dict(state_dict=state_dict, strict=False)
        # rel_pos is added after pretraining
        missing_keys = [key for key in missing_keys if ".rel_pos_" not in key]
        # layerscale is added after pretraining
        missing_keys = [key for key in missing_keys if ".gamma" not in key]
        if strict:
            assert len(missing_keys) == 0, missing_keys
            assert len(unexpected_keys) == 0, unexpected_keys
        return missing_keys, unexpected_keys

    def get_param_group_modifiers(self):
        modifiers = [WeightDecayByNameModifier(name="pos_embed.embed", value=0.0)]
        for i in range(self.depth):
            modifiers.append(WeightDecayByNameModifier(name=f"blocks.{i}.attn.rel_pos_bias_table", value=0.0))
        if self.cls_tokens.num_tokens > 0:
            modifiers.append(WeightDecayByNameModifier(name="cls_tokens.tokens", value=0.0))
        return modifiers

    def forward(self, x, mask_generator=None, idx=None):
        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)

        if mask_generator is not None:
            # generate mask -> apply mask
            x, mask, ids_restore, ids_shuffle = mask_generator.get_mask(x, idx=idx)
            outputs["mask"] = mask
            outputs["ids_restore"] = ids_restore
            outputs["ids_shuffle"] = ids_shuffle
        else:
            # no mask -> flatten to 1d
            x = einops.rearrange(x, "b ... d -> b (...) d")

        # add cls token
        x = self.cls_tokens(x)

        # apply blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

