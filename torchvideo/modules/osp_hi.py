# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math
import warnings
import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from torch.distributed.tensor import Shard, Replicate
from torchvideo.utils.utils import is_npu_available

from .attention import flash_attention, attention
from .want2v import (
    WanAttentionBlock,
    WanSelfAttention,
    WanT2VCrossAttention,
    sinusoidal_embedding_1d,
    rope_params,
    rope_apply,
    WanModel,
    WanLayerNorm,
    WanRMSNorm,
)

from torchvideo.distributed.redistribution import Redistribution

T5_CONTEXT_TOKEN_NUMBER = 512

class RearrangeType:

    Identity = "identity"

    Skiparse1DSingle = "skiparse_1d_single"
    Skiparse1DSingleReverse = "skiparse_1d_single_reverse"
    Skiparse1DGroup = "skiparse_1d_group"
    Skiparse1DGroupReverse = "skiparse_1d_group_reverse"
    Skiparse1DSingle2Group = "skiparse_1d_single_to_group"
    Skiparse1DGroup2Single = "skiparse_1d_group_to_single"

    Skiparse2DSingle = "skiparse_2d_single"
    Skiparse2DSingleReverse = "skiparse_2d_single_reverse"
    Skiparse2DGroup = "skiparse_2d_group"
    Skiparse2DGroupReverse = "skiparse_2d_group_reverse"
    Skiparse2DSingle2Group = "skiparse_2d_single_to_group"
    Skiparse2DGroup2Single = "skiparse_2d_group_to_single"

class SkiparseRearrange:
    def __init__(self, sparse_ratio=4, rearrange_type=RearrangeType.Identity):

        self.sparse_ratio = sparse_ratio
        self.rearrange_type = rearrange_type

        self.skiparse_1d = "skiparse_1d" in self.rearrange_type
        self.skiparse_2d = "skiparse_2d" in self.rearrange_type

        if self.skiparse_1d and self.skiparse_2d:
            raise ValueError(f"We can only use skiparse 1d or skiparse 2d, not both at the same time!")
        if (not self.skiparse_1d and not self.skiparse_2d) and self.sparse_ratio > 1:
            warnings.warn("When skiparse_1d = skiparse_2d = False, sparse ratio should be 1, we instead use full attention.")
            self.sparse_ratio = 1

        rearrange_func = f"_{rearrange_type}"
        if not hasattr(self, rearrange_func):
            raise ValueError(f"Unsupported rearrange operation: {rearrange_func}")
        self.rearrange_func = getattr(self, rearrange_func)

    def _identity(self, x, grid_sizes=None):
        return x

    def _skiparse_1d_single(self, x, grid_sizes=None):
        return rearrange(x, 'b (n p) c -> (b p) n c', p=self.sparse_ratio)
    
    def _skiparse_1d_single_reverse(self, x, grid_sizes=None):
        return rearrange(x, '(b p) n c -> b (n p) c', p=self.sparse_ratio)

    def _skiparse_1d_group(self, x, grid_sizes=None):
        return rearrange(x, 'b (n p q) c -> (b p) (n q) c', p=self.sparse_ratio, q=self.sparse_ratio)
    
    def _skiparse_1d_group_reverse(self, x, grid_sizes=None):
        return rearrange(x, '(b p) (n q) c -> b (n p q) c', p=self.sparse_ratio, q=self.sparse_ratio)
    
    def _skiparse_1d_single_to_group(self, x, grid_sizes=None):
        k = int(self.sparse_ratio ** 0.5)
        return rearrange(x, '(b p q) (n r s) c -> (b r s) (n p q) c', p=k, q=k, r=k, s=k)

    def _skiparse_1d_group_to_single(self, x, grid_sizes=None):
        k = int(self.sparse_ratio ** 0.5)
        return rearrange(x, '(b r s) (n p q) c -> (b p q) (n r s) c', p=k, q=k, r=k, s=k)

    def _skiparse_2d_single(self, x, grid_sizes):
        T, H, W = grid_sizes
        return rearrange(x, 'b (t h p w q) c -> (b p q) (t h w) c', p=self.sparse_ratio, q=self.sparse_ratio, h=H // self.sparse_ratio, w=W // self.sparse_ratio)

    def _skiparse_2d_single_reverse(self, x, grid_sizes):
        T, H, W = grid_sizes
        return rearrange(x, '(b p q) (t h w) c -> b (t h p w q) c', p=self.sparse_ratio, q=self.sparse_ratio, h=H // self.sparse_ratio, w=W // self.sparse_ratio)
    
    def _skiparse_2d_group(self, x, grid_sizes):
        T, H, W = grid_sizes
        return rearrange(
            x, 'b (t h p1 p2 w q1 q2) c -> (b p1 q1) (t h p2 w q2) c',
            p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h=H // (self.sparse_ratio ** 2), w=W // (self.sparse_ratio ** 2)
        )

    def _skiparse_2d_group_reverse(self, x, grid_sizes):
        T, H, W = grid_sizes
        return rearrange(
            x, '(b p1 q1) (t h p2 w q2) c -> b (t h p1 p2 w q1 q2) c', 
            p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h=H // (self.sparse_ratio ** 2), w=W // (self.sparse_ratio ** 2)
        )

    def _skiparse_2d_single_to_group(self, x, grid_sizes):
        T, H, W = grid_sizes
        return rearrange(
            x, '(b p2 q2) (t h_p1 p1 w_q1 q1) c -> (b p1 q1) (t h_p1 p2 w_q1 q2) c',
            p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h_p1=H // (self.sparse_ratio ** 2), w_q1=W // (self.sparse_ratio ** 2)
        )

    def _skiparse_2d_group_to_single(self, x, grid_sizes):
        T, H, W = grid_sizes
        return rearrange(
            x, '(b p1 q1) (t h_p1 p2 w_q1 q2) c -> (b p2 q2) (t h_p1 p1 w_q1 q1) c',
            p1=self.sparse_ratio, q1=self.sparse_ratio, p2=self.sparse_ratio, q2=self.sparse_ratio, h_p1=H // (self.sparse_ratio ** 2), w_q1=W // (self.sparse_ratio ** 2)
        )
        
    def __call__(self, x, grid_sizes=None):
        return self.rearrange_func(x, grid_sizes)

class SkiparseChecker:
    pass


class SkiparseAttentionBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        sparse_ratio=4,
        skiparse_1d=True,
        skiparse_2d=False,
        is_first_sparse_block=False,
        is_last_sparse_block=False,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()

        self.skiparse_single_attention_block = WanAttentionBlock(
            dim,
            ffn_dim,
            num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )

        self.skiparse_group_attention_block = WanAttentionBlock(
            dim,
            ffn_dim,
            num_heads,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
        )

        self.skiparse_1d = skiparse_1d
        self.skiparse_2d = skiparse_2d
        self.sparse_ratio = sparse_ratio
        self.is_first_sparse_block = is_first_sparse_block
        self.is_last_sparse_block = is_last_sparse_block

        self.rearrange_skiparse_mid_sparse_block = SkiparseRearrange(
            sparse_ratio=self.sparse_ratio, 
            rearrange_type=RearrangeType.Skiparse1DSingle2Group if self.skiparse_1d else RearrangeType.Skiparse2DSingle2Group
        )

        self.cp_skiparse_before_rerrange_mid_sparse_block = nn.Identity()
        self.cp_skiparse_after_rerrange_mid_sparse_block = nn.Identity()

        if self.is_first_sparse_block:
            self.rearrange_skiparse_in_first_sparse_block = SkiparseRearrange(
                sparse_ratio=self.sparse_ratio, 
                rearrange_type=RearrangeType.Skiparse1DSingle if self.skiparse_1d else RearrangeType.Skiparse2DSingle
            )
            self.cp_skiparse_in_first_sparse_block = nn.Identity()

        if self.is_last_sparse_block:
            self.rearrange_skiparse_out_last_sparse_block = SkiparseRearrange(
                sparse_ratio=self.sparse_ratio, 
                rearrange_type=RearrangeType.Skiparse1DGroupReverse if self.skiparse_1d else RearrangeType.Skiparse2DGroupReverse
            )
            self.cp_skiparse_out_last_sparse_block = nn.Identity()
        else:
            self.rearrange_skiparse_out_sparse_block = SkiparseRearrange(
                sparse_ratio=self.sparse_ratio, 
                rearrange_type=RearrangeType.Skiparse1DGroup2Single if self.skiparse_1d else RearrangeType.Skiparse2DGroup2Single
            )
            self.cp_skiparse_before_rerrange_out_sparse_block = nn.Identity()
            self.cp_skiparse_after_rerrange_out_sparse_block = nn.Identity()
        

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        # full attn
        if not self.skiparse_1d and not self.skiparse_2d:
            return self.skiparse_group_attention_block(
                self.skiparse_single_attention_block(
                    x,
                    e,
                    seq_lens,
                    grid_sizes,
                    freqs,
                    context,
                    context_lens,
                )
            )

        # full attn -> skiparse attn, call rerrange single
        if self.is_first_sparse_block:
            x = self.rearrange_skiparse_in_first_sparse_block(x, grid_sizes=grid_sizes)
            x = self.cp_skiparse_in_first_sparse_block(x)

        # single attn
        x = self.skiparse_single_attention_block(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        )

        # single to group, cp gather -> rerrange -> cp shard
        x = self.cp_skiparse_before_rerrange_mid_sparse_block(x)
        x = self.rearrange_skiparse_mid_sparse_block(x)
        x = self.cp_skiparse_after_rerrange_mid_sparse_block(x)

        # group attn
        x = self.skiparse_group_attention_block(
            x,
            e,
            seq_lens,
            grid_sizes,
            freqs,
            context,
            context_lens,
        )

        # last sparse block, cp gather -> rerrange to get full sequence
        if self.is_last_sparse_block:
            x = self.cp_skiparse_out_last_sparse_block(x)
            x = self.rearrange_skiparse_out_last_sparse_block(x)
        # group to single, cp gather -> rerrange -> cp shard
        else:
            x = self.cp_skiparse_before_rerrange_out_sparse_block(x)
            x = self.rearrange_skiparse_out_sparse_block(x)
            x = self.cp_skiparse_after_rerrange_out_sparse_block(x)

        return x

models = {
    "wan_t2v": WanModel
}

models_main_block = {
    "wan_t2v": WanAttentionBlock
}

models_blocks_to_float = {
    "wan_t2v": [WanLayerNorm, WanRMSNorm]
}

models_blocks_to_output_float = {
    "wan_t2v": None
}

cp_plans = {
    "wan_t2v": {
        WanModel:{
            "cp_input_layer": Redistribution(
                original_layouts=(Replicate(),),
                target_layouts=(Shard(1),), # split on sequence dim, (B, N, C) -> (B, N / cp_size, C)
            ),
            "cp_output_layer": Redistribution(
                original_layouts=(Shard(1),),
                target_layouts=(Replicate(),), # gather on sequence dim, (B, N / cp_size, C) -> (B, N, C)
            ),
        },
        WanSelfAttention: {
            "cp_self_attn_before_attention_layer": Redistribution(
                original_layouts=(Shard(1),), 
                target_layouts=(Shard(2),), # all to all, (B, N / cp_size, H, D) -> (B, N, H / cp_size, D)
            ),
            "cp_self_attn_after_attention_layer": Redistribution(
                original_layouts=(Shard(2),),
                target_layouts=(Shard(1),), # all to all, (B, N, H / cp_size, D) -> (B, N / cp_size, H, D)
            ),
        },
        WanT2VCrossAttention: {
            "cp_cross_attn_before_attention_layer": Redistribution(
                original_layouts=(Shard(1),), 
                target_layouts=(Shard(2),), # all to all, (B, N / cp_size, H, D) -> (B, N, H / cp_size, D)
            ),
            "cp_cross_attn_after_attention_layer": Redistribution(
                original_layouts=(Shard(2),),
                target_layouts=(Shard(1),), # all to all, (B, N, H / cp_size, D) -> (B, N / cp_size, H, D)
            ),
        }
    }
}


if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.bfloat16
    model = WanModel().to(device=device, dtype=dtype)
    model.set_gradient_checkpointing(True)
    x = torch.randn(2, 16, 21, 60, 104, device=device, dtype=dtype)
    t = torch.randint(0, 1000, (2,), device=device)
    context = torch.randn(2, 512, 4096, device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype):
        y = model(x, t, context)
    print(y.shape)