import torch
from torch import nn
import torch_npu
import time
import os
from enum import Enum
from einops import rearrange
import math
from torch import einsum
import math
from .quant.base.QType import QType
from .quant.base.QTensor import quant_dequant_float

from .dump import dump


DUMP_STEP_LIST = [i for i in range(0, 1001, 20)]

class PoolType(Enum):
    AVG = 'avg'
    MAX = 'max'
    NEAREST = 'nearest'

Q_SCALE = {}
K_SCALE = {}
V_SCALE = {}
dO_SCALE = {}

AMAX_FA = {}

FWD_FA_IDX = 0
BWD_FA_IDX = 0

FWD_FA_CHANGE_INTERVAL = 10

G_POOL_SIZE = 128
G_POOL_TYPE = PoolType.AVG
GRAD_NAN_INF_FLAG = False


QTYPE = QType('hif8')

USE_DUMP = False
HiF8_QUANT = False

@torch.no_grad()  # 正确使用装饰器
def quant(x, hif8_max=15, scale=None):
 
    if scale is None:
        scale = torch.tensor(hif8_max, dtype=torch.float32) / (torch.max(torch.abs(x)) + 1e-10)
    
    x = x.float() * scale
    qx = quant_dequant_float(x.float(), QTYPE, force_py=False, force_fp32=False)
    qx = (qx / scale).to(torch.bfloat16)
    return qx

@torch.no_grad()
def static_quant(x, scale):
    x = x.float() * scale
    qx = quant_dequant_float(x.float(), QTYPE, force_py=False, force_fp32=False)
    qx = (qx / scale).to(torch.bfloat16)
    return qx


# flash_attn_varlen_func
def quant_flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
):
    # # current scaling
    # num_heads = q.shape[1]
    # q = q.unsqueeze(0).transpose(1, 2)
    # k = k.unsqueeze(0).transpose(1, 2)
    # v = v.unsqueeze(0).transpose(1, 2)

    # attn = AscendCurrentScalingFlashAttentionFunction.apply(
    #     q,
    #     k,
    #     v)

    # delayed scaling
    num_heads = q.shape[1]
    q = q.unsqueeze(0).transpose(1, 2)
    k = k.unsqueeze(0).transpose(1, 2)
    v = v.unsqueeze(0).transpose(1, 2)

    attn = AscendDelayedScalingFlashAttentionFunction.apply(
        q,
        k,
        v)

        # 检查是否包含 NaN 或 Inf
    # debug0413===========
    # q_is = torch.isnan(q).any() or torch.isinf(q).any()
    # k_is = torch.isnan(k).any() or torch.isinf(k).any()
    # v_is = torch.isnan(v).any() or torch.isinf(v).any()
    # attn_is = torch.isnan(attn).any() or torch.isinf(attn).any()
    # if not q_is and not k_is and not v_is and attn_is:
    #     print("q shape:", q.shape)
    #     print("k shape:", k.shape)
    #     print("v shape:", v.shape)
    # debug0413 end===========

    # # pytorch fa
    # attn = FlashAttentionFunction.apply(q.permute(1, 0, 2),
    #                                     k.permute(1, 0, 2), 
    #                                     v.permute(1, 0, 2), 
    #                                     None, 
    #                                     causal, 
    #                                     512, 
    #                                     512).permute(1, 0, 2)
    return attn


class AscendCurrentScalingFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v):
        device = q.device
        num_heads = q.shape[1]
        if HiF8_QUANT:
            q = quant(q, 15)
            k = quant(k, 15)
            v = quant(v, 15)
            
        # print("319 enter ascendfa")
        res = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="BNSD",
            keep_prob=1,
            scale=q.shape[-1] ** -0.5,
        )
        attn = res[0].squeeze(0).transpose(0, 1)
        ctx.head_num = num_heads
        ctx.save_for_backward(q, k, v, res[0], res[1], res[2])
        return attn

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        q, k, v, attn_in, softmax_max, softmax_sum = ctx.saved_tensors
        num_heads = ctx.head_num

        do = do.unsqueeze(0).transpose(1, 2)
        if HiF8_QUANT:
            do = quant(do, 224)

        res = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            do,
            head_num=num_heads,
            input_layout="BNSD",
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attn_in,
            scale_value=q.shape[-1] ** -0.5,
        )
        dq = res[0]
        dk = res[1]
        dv = res[2]

        return dq, dk, dv


class AscendDelayedScalingFlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def check_nan_inf(x):
            if isinstance(x, torch.Tensor):
                return torch.isnan(x).any() or torch.isinf(x).any()
            
            elif isinstance(x, (list, tuple, set)):
                return any(check_nan_inf(i) for i in x)
            
            elif isinstance(x, dict):
                return any(check_nan_inf(v) for v in x.values())
            
            elif isinstance(x, (float, int)):
                return isinstance(x, float) and (math.isnan(x) or math.isinf(x))
            
            else:
                return False  # 其他类型直接忽略
    def forward(ctx, q, k, v):
        device = q.device
        num_heads = q.shape[1]
        old_q = q
        old_k = k
        old_v = v

        global FWD_FA_IDX
        global Q_SCALE
        global K_SCALE
        global V_SCALE

        FWD_FA_IDX += 1
        
        # print("FWD_FA_IDX:", FWD_FA_IDX)

        # cur_iter = int(os.environ["iter"], 0)
        cur_iter = int(os.environ.get("iter", "0"), 0)
        ctx.cur_iter = cur_iter

        if HiF8_QUANT:
            if cur_iter == 0:
                Q_SCALE[FWD_FA_IDX] = torch.tensor(15, dtype=torch.float32) / (torch.max(torch.abs(q)) + 1e-10)
                K_SCALE[FWD_FA_IDX] = torch.tensor(15, dtype=torch.float32) / (torch.max(torch.abs(k)) + 1e-10)
                V_SCALE[FWD_FA_IDX] = torch.tensor(15, dtype=torch.float32) / (torch.max(torch.abs(v)) + 1e-10)

            q = quant(q, scale=Q_SCALE[FWD_FA_IDX])
            k = quant(k, scale=K_SCALE[FWD_FA_IDX])
            v = quant(v, scale=V_SCALE[FWD_FA_IDX])

            if (cur_iter + 1) % FWD_FA_CHANGE_INTERVAL == 0:
                if not check_nan_inf(q):
                    Q_SCALE[FWD_FA_IDX] = torch.tensor(15, dtype=torch.float32) / (torch.max(torch.abs(q)) + 1e-10)
                if not check_nan_inf(k):
                    K_SCALE[FWD_FA_IDX] = torch.tensor(15, dtype=torch.float32) / (torch.max(torch.abs(k)) + 1e-10) 
                if not check_nan_inf(v):
                    V_SCALE[FWD_FA_IDX] = torch.tensor(15, dtype=torch.float32) / (torch.max(torch.abs(v)) + 1e-10)
        
        # if cur_iter > 240:
        #     if check_nan_inf(q):
        #         print(f"matmul cur_iter {cur_iter}, q , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(k):
        #         print(f"matmul cur_iter {cur_iter}, k , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(v):
        #         print(f"matmul cur_iter {cur_iter}, v , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(old_q):
        #         print(f"matmul cur_iter {cur_iter}, old_q , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(old_k):  
        #         print(f"matmul cur_iter {cur_iter}, old_k , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(old_v):
        #         print(f"matmul cur_iter {cur_iter}, old_v , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        res = torch_npu.npu_fusion_attention(
            q,
            k,
            v,
            head_num=num_heads,
            input_layout="BNSD",
            keep_prob=1,
            scale=q.shape[-1] ** -0.5,
        )
        attn = res[0].squeeze(0).transpose(0, 1)
        ctx.head_num = num_heads
        ctx.save_for_backward(q, k, v, res[0], res[1], res[2])

        return attn

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        q, k, v, attn_in, softmax_max, softmax_sum = ctx.saved_tensors
        num_heads = ctx.head_num

        global BWD_FA_IDX
        global dO_SCALE
        global AMAX_FA
        cur_iter = ctx.cur_iter
        def check_nan_inf(x):
            if isinstance(x, torch.Tensor):
                return torch.isnan(x).any() or torch.isinf(x).any()
            
            elif isinstance(x, (list, tuple, set)):
                return any(check_nan_inf(i) for i in x)
            
            elif isinstance(x, dict):
                return any(check_nan_inf(v) for v in x.values())
            
            elif isinstance(x, (float, int)):
                return isinstance(x, float) and (math.isnan(x) or math.isinf(x))
            
            else:
                return False  # 其他类型直接忽略
        if cur_iter > 240:
            if check_nan_inf(AMAX_FA):      #  debug0413:  加个判断， current》300之后才做这个if
                print(f"fa cur_iter {cur_iter}, AMAX_FA {AMAX_FA}, BWD_FA_IDX {BWD_FA_IDX}")
            if check_nan_inf(do):      #  debug0413:  加个判断， current》300之后才做这个if
                print(f"fa cur_iter {cur_iter}, do , BWD_FA_IDX {BWD_FA_IDX}")
            if check_nan_inf(q):      #  debug0413:  加个判断， current》300之后才做这个if
                print(f"fa cur_iter {cur_iter}, q , BWD_FA_IDX {BWD_FA_IDX}")
            if check_nan_inf(k):      #  debug0413:  加个判断， current》300之后才做这个if
                print(f"fa cur_iter {cur_iter}, k , BWD_FA_IDX {BWD_FA_IDX}")
            if check_nan_inf(v):      #  debug0413:  加个判断， current》300之后才做这个if
                print(f"fa cur_iter {cur_iter}, v , BWD_FA_IDX {BWD_FA_IDX}")
        BWD_FA_IDX +=1 

        # print("BWD_FA_IDX:", BWD_FA_IDX)
        
        do = do.unsqueeze(0).transpose(1, 2)
        if HiF8_QUANT:
            if cur_iter > G_POOL_SIZE-1:
                if G_POOL_TYPE == PoolType.AVG:
                    avg_amax = torch.mean(torch.tensor(AMAX_FA[BWD_FA_IDX]))
                elif G_POOL_TYPE == PoolType.MAX:
                    avg_amax = torch.max(torch.tensor(AMAX_FA[BWD_FA_IDX]))
                elif G_POOL_TYPE == PoolType.NEAREST:
                    avg_amax = AMAX_FA[BWD_FA_IDX][-1]
                else:
                    raise ValueError(f"Unknown pool type: {G_POOL_TYPE}")
        
                dO_SCALE[BWD_FA_IDX] = torch.tensor(224, dtype=torch.float32) / (avg_amax + 1e-10)

                do = quant(do, scale=dO_SCALE[BWD_FA_IDX])
    
                amax = torch.max(torch.abs(do))
               
                amax_flag = True
                if torch.isnan(amax) or torch.isinf(amax):
                    amax_flag = False

                if amax_flag:
                    AMAX_FA[BWD_FA_IDX].pop(0)
                    AMAX_FA[BWD_FA_IDX].append(amax)

                if torch_npu.npu.current_device() == 0 and BWD_FA_IDX == 1:
                    print(f"iter {cur_iter} in list")

            else:
                if cur_iter == 0:
                    AMAX_FA[BWD_FA_IDX] = []
                amax = torch.max(torch.abs(do))
                AMAX_FA[BWD_FA_IDX].append(amax)

                dO_SCALE[BWD_FA_IDX] = torch.tensor(224, dtype=torch.float32) / (amax + 1e-10)

                do = quant(do, scale=dO_SCALE[BWD_FA_IDX])
        res = torch_npu.npu_fusion_attention_grad(
            q,
            k,
            v,
            do,
            head_num=num_heads,
            input_layout="BNSD",
            softmax_max=softmax_max,
            softmax_sum=softmax_sum,
            attention_in=attn_in,
            scale_value=q.shape[-1] ** -0.5,
        )
        dq = res[0]
        dk = res[1]
        dv = res[2]

        return dq, dk, dv, None


# constants
EPSILON = 1e-10


# helper functions
def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


# flash attention forwards and backwards
# flash attention v1 - https://arxiv.org/abs/2205.14135
# flash attention v2 - https://tridao.me/publications/flash2/flash2.pdf
class FlashAttentionFunction(torch.autograd.Function):
    @staticmethod
    @torch.no_grad()
    def forward(ctx, q, k, v, mask, causal, q_bucket_size, k_bucket_size):
        """ Algorithm 1 in the v2 paper """
        # quantize q, k, v
        if HiF8_QUANT:
            q = quant(q, 15)
            k = quant(k, 15)
            v = quant(v, 15)

        device = q.device
        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        o = torch.zeros_like(q)
        all_row_sums = torch.zeros((*q.shape[:-1], 1), device = device)
        all_row_maxes = torch.full((*q.shape[:-1], 1), max_neg_value, device = device)

        scale = (q.shape[-1] ** -0.5)

        num_row_tiles = math.ceil(q.shape[-2] / q_bucket_size)
        num_col_tiles = math.ceil(k.shape[-2] / k_bucket_size)

        if exists(mask) and mask.ndim == 2:
            mask = rearrange(mask, 'b n -> b 1 1 n')

        if not exists(mask):
            col_masks = (None,) * num_col_tiles
            mask = (col_masks,) * num_row_tiles 
        else:
            mask = ((mask,) * num_row_tiles) if mask.shape[-2] == 1 else mask.split(q_bucket_size, dim = -2)
            mask = tuple(((row_mask,) * num_col_tiles) if row_mask.shape[-1] == 1 else row_mask.split(k_bucket_size, dim = -1) for row_mask in mask)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            mask,
            all_row_sums.split(q_bucket_size, dim = -2),
            all_row_maxes.split(q_bucket_size, dim = -2),
        )

        for ind, (qc, oc, row_mask, row_sums, row_maxes) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                row_mask
            )

            for k_ind, (kc, vc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if exists(col_mask):
                    attn_weights.masked_fill_(~col_mask, max_neg_value)

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]),
                    dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                block_row_maxes = attn_weights.amax(dim = -1, keepdims = True)
                new_row_maxes = torch.maximum(block_row_maxes, row_maxes)

                exp_weights = torch.exp(attn_weights - new_row_maxes)

                if exists(col_mask):
                    exp_weights.masked_fill_(~col_mask, 0.)

                block_row_sums = exp_weights.sum(dim = -1, keepdims = True).clamp(min = EPSILON)

                # static quantize P
                if HiF8_QUANT:
                    exp_weights = static_quant(exp_weights, 15)

                exp_values = einsum('... i j, ... j d -> ... i d', exp_weights, vc)

                exp_row_max_diff = torch.exp(row_maxes - new_row_maxes)

                new_row_sums = exp_row_max_diff * row_sums + block_row_sums

                oc.mul_(exp_row_max_diff).add_(exp_values)

                row_maxes.copy_(new_row_maxes)
                row_sums.copy_(new_row_sums)

            oc.div_(row_sums)

        lse = all_row_sums.log() + all_row_maxes

        ctx.args = (causal, scale, mask, q_bucket_size, k_bucket_size)
        ctx.save_for_backward(q, k, v, o, lse)

        return o

    @staticmethod
    @torch.no_grad()
    def backward(ctx, do):
        """ Algorithm 2 in the v2 paper """

        causal, scale, mask, q_bucket_size, k_bucket_size = ctx.args
        q, k, v, o, lse = ctx.saved_tensors

        device = q.device

        max_neg_value = -torch.finfo(q.dtype).max
        qk_len_diff = max(k.shape[-2] - q.shape[-2], 0)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        row_splits = zip(
            q.split(q_bucket_size, dim = -2),
            o.split(q_bucket_size, dim = -2),
            do.split(q_bucket_size, dim = -2),
            mask,
            lse.split(q_bucket_size, dim = -2),
            dq.split(q_bucket_size, dim = -2)
        )

        for ind, (qc, oc, doc, row_mask, lsec, dqc) in enumerate(row_splits):
            q_start_index = ind * q_bucket_size - qk_len_diff

            col_splits = zip(
                k.split(k_bucket_size, dim = -2),
                v.split(k_bucket_size, dim = -2),
                dk.split(k_bucket_size, dim = -2),
                dv.split(k_bucket_size, dim = -2),
                row_mask
            )

            for k_ind, (kc, vc, dkc, dvc, col_mask) in enumerate(col_splits):
                k_start_index = k_ind * k_bucket_size

                attn_weights = einsum('... i d, ... j d -> ... i j', qc, kc) * scale

                if causal and q_start_index < (k_start_index + k_bucket_size - 1):
                    causal_mask = torch.ones((qc.shape[-2], kc.shape[-2]), 
                    dtype = torch.bool, device = device).triu(q_start_index - k_start_index + 1)
                    attn_weights.masked_fill_(causal_mask, max_neg_value)

                p = torch.exp(attn_weights - lsec)

                # static quantize P
                if HiF8_QUANT:
                    p = static_quant(p, 15)

                if exists(col_mask):
                    p.masked_fill_(~col_mask, 0.)

                D = (doc * oc).sum(dim = -1, keepdims = True)

                # static quantize dO
                if HiF8_QUANT:
                    doc = static_quant(doc, 224)
                
                dv_chunk = einsum('... i j, ... i d -> ... j d', p, doc)
                dp = einsum('... i d, ... j d -> ... i j', doc, vc)

                # static quantize dS
                ds = p * scale * (dp - D)

                if HiF8_QUANT:
                    ds = static_quant(ds, 224)

                dq_chunk = einsum('... i j, ... j d -> ... i d', ds, kc)
                dk_chunk = einsum('... i j, ... i d -> ... j d', ds, qc)

                dqc.add_(dq_chunk)
                dkc.add_(dk_chunk)
                dvc.add_(dv_chunk)

        return dq, dk, dv, None, None, None, None