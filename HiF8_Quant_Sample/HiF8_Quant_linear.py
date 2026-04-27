import torch
from torch import nn
import torch_npu
from enum import Enum
from collections import Counter

import os
import math

from .quant.base.QType import QType
from .quant.base.QTensor import quant_dequant_float
from .dump import dump

USE_DUMP = False          # dump开关
VERBOSE = False
USE_QUANTIZATION = False # 量化开关
DUMP_STEP_LIST = [0,10,50,100,500]
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 49, 99, 149, 199, 249, 299, 349, 399, 449, 499]


class HiF8CurrectScalingLinear(nn.Linear):
    def __init__(self, in_features, out_features, name=None, bias=True):
        self.name = name
        super(HiF8CurrectScalingLinear, self).__init__(in_features, out_features, bias)
       
    def forward(self, x, scale_max=15):
        return HiF8CurrectScalingQuantMatmul.apply(x, self.weight, self.bias, self.name, scale_max)

class HiF8DelayedScalingLinear(nn.Linear):
    def __init__(self, in_features, out_features, name=None, bias=True):
        self.name = name
        super(HiF8DelayedScalingLinear, self).__init__(in_features, out_features, bias)
       
    def forward(self, x, scale_max=15):
        return HiF8DelayedScalingQuantMatmul.apply(x, self.weight, self.bias, self.name, scale_max)

@torch.no_grad()
def quantize(tensor, scale=None, scale_max=15):
    qtype = QType('hif8') 
    """统一量化函数，返回量化后的值和使用的尺度"""
    if scale is None:
        scale = torch.tensor(scale_max, dtype=torch.float32) / (torch.max(torch.abs(tensor)) + 1e-10)
    scaled = tensor.float() * scale
    quantized = quant_dequant_float(scaled, qtype, force_py=False, force_fp32=True)
    return (quantized / scale).to(tensor.dtype), scale


@torch.no_grad()
def block_quantize(x, scale=None, scale_max = 224):
    qtype = QType('hif8')

    t = torch.tensor(x, dtype=torch.float32)
    length = t.numel()

    # 计算 padding 数量
    pad_len = (128 - length % 128) % 128

    # padding：使用 -inf，不影响最大值
    if pad_len > 0:
        pad_tensor = torch.full((pad_len,), float('-inf'))
        t = torch.cat([t, pad_tensor])

    # reshape
    t = t.reshape(-1, 128)

    amax_val = torch.max(t, dim=1)
    if scale is None:
        scale =  torch.tensor(scale_max, dtype=torch.float32) / (amax_val + 1e-10)
    scaled = t2 * amax_val
    quantized = quant_dequant_float(scaled, qtype, force_py=False, force_fp32=True) / amax_val
    quantized = quantized.reshape(-1)[:length]
    quantized = quantized.reshape(x.shpae)

    return quantized.to(x.dtype), scale

class HiF8CurrectScalingQuantMatmul(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias=None, name=None, scale_max=15):
        ctx.save_for_backward(x, weight, bias)
        ctx.name = name
        
        # 保存输入维度信息用于反向传播
        ctx.x_dim = x.dim()  # 记录输入是2D还是3D
        
        # cur_iter = int(os.environ["iter"], 0)
        cur_iter = int(os.environ.get("iter", "0"), 0)
        ctx.cur_iter = cur_iter

        # 始终进行前向dump
        if USE_DUMP and cur_iter in DUMP_STEP_LIST and torch_npu.npu.current_device() == 0:
            if VERBOSE:
                print(f'*************[Step {os.environ["iter"]}] dump forward_{ctx.name} check *********', flush=True)

            dump(x, f"forward_{ctx.name}_activation")
            dump(weight, f"forward_{ctx.name}_weight")
            if bias is not None:
                dump(bias, f"forward_{ctx.name}_bias")
        
        # 根据量化开关决定是否启用量化
        qx, xscale = x, None
        qweight, wscale = weight, None
        
        if USE_QUANTIZATION:
            qx, xscale = quantize(x, scale_max=15)
            qweight, wscale = quantize(weight, scale_max=15)

        # 保存量化尺度用于反向传播
        ctx.xscale = xscale
        ctx.wscale = wscale

        # 矩阵乘法
        output = torch.matmul(qx, qweight.t())  
        if bias is not None:
            output += bias
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        name = f"backward_{ctx.name}"
        xscale, wscale = ctx.xscale, ctx.wscale
        x_dim = ctx.x_dim  # 获取输入维度信息
        
        # 根据量化开关决定是否启用量化
        qx, qgrad_output, qweight = x, grad_output, weight
        
        if USE_QUANTIZATION:
            qgrad_output, _ = quantize(grad_output, scale_max=224) #224) #
            qx, _ = quantize(x, xscale)  # 使用前向相同的尺度
            qweight, _ = quantize(weight, wscale)  # 使用前向相同的尺度

        # 计算输入梯度
        grad_input = torch.matmul(qgrad_output, qweight)

        # 计算权重梯度 - 正确处理2D和3D情况
        if x_dim == 3:  # 3D输入 [batch, seq_len, features]
            # 维度信息:
            # 调整形状以在批次维度上进行矩阵乘法
            batch_size, seq_len, out_features = qgrad_output.shape
            in_features = qx.shape[-1]
            
            # 重塑为 [batch, seq_len, out_features] -> [batch*seq_len, out_features]
            qgrad_output_reshaped = qgrad_output.reshape(-1, out_features)
            # 重塑为 [batch, seq_len, in_features] -> [batch*seq_len, in_features]
            qx_reshaped = qx.reshape(-1, in_features)
            
            # 计算每个样本的权重梯度并在批次维度求和
            # 结果形状: [out_features, in_features]
            grad_weight = torch.matmul(qgrad_output_reshaped.t(), qx_reshaped)
            
        else:  # 2D输入 [batch, features]
            # 常规矩阵乘法计算权重梯度
            grad_weight = torch.matmul(qgrad_output.t(), qx)
        
        # 计算bias梯度
        grad_bias = None
        if bias is not None:
            # 对所有批次和序列维度求和
            sum_dims = tuple(range(qgrad_output.dim() - 1))  # 除了最后一个维度外都求和
            grad_bias = torch.sum(qgrad_output, dim=sum_dims)

        if USE_DUMP and int(os.environ["iter"]) in DUMP_STEP_LIST and torch_npu.npu.current_device() == 0:
            if VERBOSE:
                print(f'*************[Step {os.environ["iter"]}] dump backward_{ctx.name} check *********', flush=True)
            dump(grad_input, f"backward_{ctx.name}_grad_input")
            dump(grad_weight, f"backward_{ctx.name}_grad_weight")

        return grad_input, grad_weight, grad_bias, None, None


class PoolType(Enum):
    AVG = 'avg'
    MAX = 'max'
    NEAREST = 'nearest'

A_SCALE = {}
W_SCALE = {}
G_SCALE = {}

AMAX = {}

SCALE_MAX_LIST = [15, 28, 56, 112, 224, 384, 768]
# FWD_FA_IDX= 0
# BWD_FA_IDX = 0

FWD_LINEAR_IDX = 0
BWD_LINEAR_IDX = 0
FWD_CHANGE_INTERVAL = 10
BWD_CHANGE_INTERVAL = 10 # 每10个iter更新一次grad scale, 数值为1代表每次都更新
#AVG_POOL_SIZE = 100  NOTE: each_steps_update方案与avg_pool方案冲突, 不能同时使用
GRAD_NAN_INF_FLAG = False
G_POOL_SIZE = 128
G_POOL_TYPE = PoolType.AVG

a_scale_max = 15
w_scale_max = 15
cross_grad_scale_max = 15
self_grad_scale_max = 224


MARGIN_ENABLE=False ## 如果要开margin search 改成true
MARGIN_SEARCH_INTERVAL = 100
print(G_POOL_SIZE,G_POOL_TYPE, "MARGIN_ENABLE",MARGIN_ENABLE)
A_SCALE_MAX  = {}
W_SCALE_MAX  = {}
CG_SCALE_MAX = {}
SG_SCALE_MAX = {}


def master_device():
    return torch_npu.npu.current_device() == 0


def forward_margin_search(x, weight, index):
    original_res = x @ weight.t()
    best_mse = float('inf')
    if master_device():
        old_a_max = A_SCALE_MAX[index]
        old_w_max = W_SCALE_MAX[index]
    for _, cur_a_max in enumerate(SCALE_MAX_LIST):
        for _, cur_w_max in enumerate(SCALE_MAX_LIST):
            qx, _ = quantize(x, scale_max=cur_a_max)
            qw, _ = quantize(weight, scale_max=cur_w_max)
            res = qx @ qw.t()
            mse = torch.mean((res - original_res) ** 2)
            if mse < best_mse:
                best_mse = mse
                A_SCALE_MAX[index] = cur_a_max
                W_SCALE_MAX[index] = cur_w_max
    # if master_device():
    #     if old_a_max != A_SCALE_MAX[index]:
    #         print(f"forward index: {index}, old_a_max: {old_a_max}, new_a_max: {A_SCALE_MAX[index]}")
    #     if old_w_max != W_SCALE_MAX[index]:
    #         print(f"forward index: {index}, old_w_max: {old_w_max}, new_w_max: {W_SCALE_MAX[index]}")


def backward_margin_search(grad_output, x, weight, index, name):
    if x.dim() == 3:  # 3D输入 [batch, seq_len, features]
        # 维度信息:
        # qgrad_output: [batch, seq_len, out_features]
        # qx: [batch, seq_len, in_features]
        
        # 调整形状以在批次维度上进行矩阵乘法
        batch_size, seq_len, out_features = grad_output.shape
        in_features = x.shape[-1]
        
        # 重塑为 [batch, seq_len, out_features] -> [batch*seq_len, out_features]
        grad_output = grad_output.reshape(-1, out_features)
        # 重塑为 [batch, seq_len, in_features] -> [batch*seq_len, in_features]
        x =x.reshape(-1, in_features)

    original_grad_input = grad_output @ weight
    original_grad_weight = grad_output.t() @ x
    best_mse = float('inf')

    # # 未定义
    # is_cross = "cross" in name
    if master_device():
        old_amax = CG_SCALE_MAX[index] if is_cross else SG_SCALE_MAX[index]
    for _, cur_max in enumerate(SCALE_MAX_LIST):
        qgrad_output, _ = quantize(grad_output, scale_max=cur_max)
        res1 = qgrad_output @ weight
        mse1 = torch.mean((res1 - original_grad_input) ** 2)
        res2 = qgrad_output.t() @ x
        mse2 = torch.mean((res2 - original_grad_weight) ** 2)
        mse = mse1 + mse2
        if mse < best_mse:
            best_mse = mse
            if is_cross:
                CG_SCALE_MAX[index] = cur_max
            else:
                SG_SCALE_MAX[index] = cur_max
            CG_SCALE_MAX[index] = cur_max
            SG_SCALE_MAX[index] = cur_max
    # if master_device():
    #     if old_amax != CG_SCALE_MAX[index] if is_cross else SG_SCALE_MAX[index]:
    #         print(f"backward index: {index}, {'cross' if is_cross else 'self'}, old_amax: {old_amax}, new_amax: {CG_SCALE_MAX[index] if is_cross else SG_SCALE_MAX[index]}")



class HiF8DelayedScalingQuantMatmul(torch.autograd.Function):

    @staticmethod
    
    def forward(ctx, x, weight, bias=None, name=None, scale_max = 15):
        ctx.save_for_backward(x, weight, bias)
        ctx.name = name
        
        # 保存输入维度信息用于反向传播
        ctx.x_dim = x.dim()  # 记录输入是2D还是3D
        
        global A_SCALE
        global W_SCALE
        global FWD_LINEAR_IDX
        global A_SCALE_MAX
        global W_SCALE_MAX

        global a_scale_max
        global w_scale_max

        FWD_LINEAR_IDX += 1
        # print("FWD_LINEAR_IDX:", FWD_LINEAR_IDX)

        # cur_iter = int(os.environ["iter"], 0)
        cur_iter = int(os.environ.get("iter", "0"), 0)
        # print("cur_iter:", cur_iter)

        # import pdb; pdb.set_trace()
        # breakpoint()

        ctx.cur_iter = cur_iter
        
        # 始终进行前向dump
        if USE_DUMP and cur_iter in DUMP_STEP_LIST and torch_npu.npu.current_device() == 0:
            if VERBOSE:
                print(f'*************[Step {os.environ["iter"]}] dump forward_{ctx.name} check *********', flush=True)
            if 'cross' not in f"{ctx.name}" and 'layer0' in f"{ctx.name}":
                dump(x, f"forward_{ctx.name}_activation")
                dump(weight, f"forward_{ctx.name}_weight")
                if bias is not None:
                    dump(bias, f"forward_{ctx.name}_bias")
        
        # 根据量化开关决定是否启用量化
        qx, xscale = x, None
        qweight, wscale = weight, None
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
        if USE_QUANTIZATION:
            if cur_iter == 0:
                A_SCALE[FWD_LINEAR_IDX] = torch.tensor(a_scale_max, dtype=torch.float32) / (torch.max(torch.abs(x)) + 1e-10)
                W_SCALE[FWD_LINEAR_IDX] = torch.tensor(w_scale_max, dtype=torch.float32) / (torch.max(torch.abs(weight)) + 1e-10)

            qx, xscale = quantize(x, scale=A_SCALE[FWD_LINEAR_IDX]) #224) #
            qweight, wscale = quantize(weight, scale=W_SCALE[FWD_LINEAR_IDX]) #56) #8.0)

            
            if (cur_iter + 1) % FWD_CHANGE_INTERVAL == 0:
                if not check_nan_inf(x):
                    A_SCALE[FWD_LINEAR_IDX] = torch.tensor(a_scale_max, dtype=torch.float32) / (torch.max(torch.abs(x)) + 1e-10)
                if not check_nan_inf(weight):
                    W_SCALE[FWD_LINEAR_IDX] = torch.tensor(w_scale_max, dtype=torch.float32) / (torch.max(torch.abs(weight)) + 1e-10)

        # def check_nan_inf(x):
        #     if isinstance(x, torch.Tensor):
        #         return torch.isnan(x).any() or torch.isinf(x).any()
            
        #     elif isinstance(x, (list, tuple, set)):
        #         return any(check_nan_inf(i) for i in x)
            
        #     elif isinstance(x, dict):
        #         return any(check_nan_inf(v) for v in x.values())
            
        #     elif isinstance(x, (float, int)):
        #         return isinstance(x, float) and (math.isnan(x) or math.isinf(x))
            
        #     else:
        #         return False  # 其他类型直接忽略
        # if cur_iter > 240:
        #     if check_nan_inf(weight):
        #         print(f"matmul cur_iter {cur_iter}, weight , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(qweight):
        #         print(f"matmul cur_iter {cur_iter}, qweight , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(x):
        #         print(f"matmul cur_iter {cur_iter}, x , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        #     if check_nan_inf(qx):
        #         print(f"matmul cur_iter {cur_iter}, qx , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        # 保存量化尺度用于反向传播
        ctx.xscale = xscale
        ctx.wscale = wscale
        if cur_iter == 0:
            A_SCALE_MAX[FWD_LINEAR_IDX] = a_scale_max
            W_SCALE_MAX[FWD_LINEAR_IDX] = w_scale_max 

        if MARGIN_ENABLE and cur_iter != 0 and cur_iter % MARGIN_SEARCH_INTERVAL == 0:
            forward_margin_search(x, weight, FWD_LINEAR_IDX)

            if FWD_LINEAR_IDX == len(A_SCALE) - 1:
                a_counter = Counter(A_SCALE_MAX.values())
                w_counter = Counter(W_SCALE_MAX.values())

                a_scale_max = a_counter.most_common(1)[0][0]
                w_scale_max = w_counter.most_common(1)[0][0]
                #print(f"iter {cur_iter} a_hif8_amax {a_hif8_amax} w_hif8_amax {w_hif8_amax}")

        # 矩阵乘法
        output = torch.matmul(qx, qweight.t())  
        if bias is not None:
            output += bias
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        name = f"backward_{ctx.name}"
        xscale, wscale = ctx.xscale, ctx.wscale
        x_dim = ctx.x_dim  # 获取输入维度信息
        
        # 根据量化开关决定是否启用量化
        ###############0402 test #####
        # grad_output = torch.full_like(grad_output, float('inf'))

        qx, qgrad_output, qweight = x, grad_output, weight

        cur_iter = ctx.cur_iter
        
        global G_SCALE
        global BWD_LINEAR_IDX
        global GRAD_NAN_INF_FLAG
        global CG_SCALE_MAX
        global SG_SCALE_MAX
        global G_POOL_SIZE
        global G_POOL_TYPE
        global self_grad_scale_max
        global cross_grad_scale_max
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
            if check_nan_inf(AMAX):
                print(f"matmul cur_iter {cur_iter}, AMAX {AMAX}, BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
            if check_nan_inf(weight):
                print(f"matmul cur_iter {cur_iter}, weight , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
            if check_nan_inf(qweight):
                print(f"matmul cur_iter {cur_iter}, qweight , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
            if check_nan_inf(grad_output):
                print(f"matmul cur_iter {cur_iter}, grad_output , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
            if check_nan_inf(qgrad_output):
                print(f"matmul cur_iter {cur_iter}, qgrad_output , BWD_LINEAR_IDX {BWD_LINEAR_IDX}")
        BWD_LINEAR_IDX += 1
        if USE_QUANTIZATION:
            if cur_iter > G_POOL_SIZE-1:
                if G_POOL_TYPE == PoolType.AVG:
                    avg_amax = torch.mean(torch.tensor(AMAX[BWD_LINEAR_IDX]))
                elif G_POOL_TYPE == PoolType.MAX:
                    avg_amax = torch.max(torch.tensor(AMAX[BWD_LINEAR_IDX]))
                elif G_POOL_TYPE == PoolType.NEAREST:
                    avg_amax = AMAX[BWD_LINEAR_IDX][-1]
                else:
                    raise ValueError(f"Unknown pool type: {G_POOL_TYPE}")
                if not ('cross' in name):         
                    G_SCALE[BWD_LINEAR_IDX] = torch.tensor(self_grad_scale_max, dtype=torch.float32) / (avg_amax + 1e-10)
                else:
                    G_SCALE[BWD_LINEAR_IDX] = torch.tensor(cross_grad_scale_max, dtype=torch.float32) / (avg_amax + 1e-10)

                qgrad_output, _ = quantize(grad_output, scale=G_SCALE[BWD_LINEAR_IDX])
    
                amax = torch.max(torch.abs(qgrad_output))
               
                amax_flag = True
                if torch.isnan(amax) or torch.isinf(amax):
                    amax_flag = False

                if amax_flag:
                    AMAX[BWD_LINEAR_IDX].pop(0)
                    AMAX[BWD_LINEAR_IDX].append(amax)
                    
                if torch_npu.npu.current_device() == 0 and BWD_LINEAR_IDX == 1:
                    print(f"iter {cur_iter} in list")

            else:
                if cur_iter == 0:
                    AMAX[BWD_LINEAR_IDX] = []
                amax = torch.max(torch.abs(qgrad_output))
                AMAX[BWD_LINEAR_IDX].append(amax)

                if 'cross' in name:
                    G_SCALE[BWD_LINEAR_IDX] = torch.tensor(cross_grad_scale_max, dtype=torch.float32) / (torch.max(torch.abs(grad_output)) + 1e-10)
                else:
                    G_SCALE[BWD_LINEAR_IDX] = torch.tensor(self_grad_scale_max, dtype=torch.float32) / (torch.max(torch.abs(grad_output)) + 1e-10)

                qgrad_output, _ = quantize(grad_output, scale=G_SCALE[BWD_LINEAR_IDX])

            
            qx, _ = quantize(x, xscale)  # 使用前向相同的尺度
            qweight, _ = quantize(weight, wscale)  # 使用前向相同的尺度

        # 计算输入梯度
        grad_input = torch.matmul(qgrad_output, qweight)

        # 计算权重梯度 - 正确处理2D和3D情况
        if x_dim == 3:  # 3D输入 [batch, seq_len, features]
            # 维度信息:
            # qgrad_output: [batch, seq_len, out_features]
            # qx: [batch, seq_len, in_features]
            
            # 调整形状以在批次维度上进行矩阵乘法
            batch_size, seq_len, out_features = qgrad_output.shape
            in_features = qx.shape[-1]
            
            # 重塑为 [batch, seq_len, out_features] -> [batch*seq_len, out_features]
            qgrad_output_reshaped = qgrad_output.reshape(-1, out_features)
            # 重塑为 [batch, seq_len, in_features] -> [batch*seq_len, in_features]
            qx_reshaped = qx.reshape(-1, in_features)
            
            # 计算每个样本的权重梯度并在批次维度求和
            # 结果形状: [out_features, in_features]
            grad_weight = torch.matmul(qgrad_output_reshaped.t(), qx_reshaped)
            
        else:  # 2D输入 [batch, features]
            # 常规矩阵乘法计算权重梯度
            grad_weight = torch.matmul(qgrad_output.t(), qx)
        
        ## 通信
        # if USE_QUANTIZATION:
        #     grad_weight, _ = quantize(grad_weight, scale_max=224)
        # 计算bias梯度
        
        grad_bias = None
        if bias is not None:
            # 对所有批次和序列维度求和
            sum_dims = tuple(range(qgrad_output.dim() - 1))  # 除了最后一个维度外都求和
            grad_bias = torch.sum(qgrad_output, dim=sum_dims)

        if USE_DUMP and int(os.environ["iter"]) in DUMP_STEP_LIST and torch_npu.npu.current_device() == 0:
            if VERBOSE:
                print(f'*************[Step {os.environ["iter"]}] dump backward_{ctx.name} check *********', flush=True)
            if 'cross' not in f"{ctx.name}" and 'layer0' in f"{ctx.name}":
                # dump(grad_output, f"backward_{ctx.name}_grad_output")
                dump(grad_input, f"backward_{ctx.name}_grad_input")
                dump(grad_weight, f"backward_{ctx.name}_grad_weight")
                # if bias is not None:
                #     dump(grad_bias, f"backward_{ctx.name}_grad_bias")

        if cur_iter == 0:
            if "cross" in f"{ctx.name}":
                CG_SCALE_MAX[BWD_LINEAR_IDX] = cross_grad_scale_max
            else:
                SG_SCALE_MAX[BWD_LINEAR_IDX] = self_grad_scale_max


        if MARGIN_ENABLE and cur_iter != 0 and cur_iter % MARGIN_SEARCH_INTERVAL == 0:
            backward_margin_search(grad_output, x, weight, BWD_LINEAR_IDX, ctx.name)

            if BWD_LINEAR_IDX == len(A_SCALE) - 1:
                cg_counter = Counter(CG_SCALE_MAX.values())
                sg_counter = Counter(SG_SCALE_MAX.values())

                cross_grad_scale_max = cg_counter.most_common(1)[0][0]
                self_grad_scale_max = sg_counter.most_common(1)[0][0]
                #print(f"[iter {cur_iter}] backward margin search {ctx.name}, self_grad_scale_max={self_grad_scale_max}, cross_grad_scale_max={cross_grad_scale_max}")

        return grad_input, grad_weight, grad_bias, None, None


class NormHiF8Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        qx, scale = block_quantize(x, scale_max=224)
        return qx
    
    @staticmethod
    def backward(ctx, grad_output):
        qgrad_output, _ = block_quantize(grad_output, scale_max=224)
        return qgrad_output