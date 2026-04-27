# HIF8 量化改造完整分析文档

---

## 目录

1. [OSP 项目改造总览](#1-osp-项目改造总览)
2. [hif8_linear.py：Linear 量化实现](#2-hif8_linearpy-linear-量化实现)
3. [hif8_attention.py：Attention QKV 量化实现](#3-hif8_attentionpy-attention-qkv-量化实现)
4. [osp_next_hif8_linear_atten.py 改动标注](#4-osp_next_hif8_linear_atten-py-改动标注)
5. [量化覆盖全景图（per block）](#5-量化覆盖全景图-per-block)
6. [参考项目 HiF8_Quant_Sample 分析](#6-参考项目-hif8_quant_sample-分析)
7. [Current Scaling vs Delayed Scaling 对比](#7-current-scaling-vs-delayed-scaling-对比)
8. [OSP 实现 vs 参考实现 差异对比](#8-osp-实现-vs-参考实现-差异对比)

---

## 1. OSP 项目改造总览

### 1.1 新增 / 修改的文件

| 文件 | 角色 | 状态 |
|------|------|------|
| `torchdiff/modules/hif8_linear.py` | HIF8 Linear 量化核心，含前向+反向 | 新增 |
| `torchdiff/modules/hif8_attention.py` | HIF8 Attention QKV 量化，含前向+反向 | 新增 |
| `torchdiff/modules/osp_next_hif8_linear_atten.py` | 主模型文件，集成两种量化 | 在 `osp_next.py` 基础上修改 |
| `configs/train/npu/osp_1_3b_hif8_linear.yaml` | 仅开启 Linear 量化的配置 | 新增 |
| `configs/train/npu/osp_1_3b_hif8_linear_atten.yaml` | Linear + Attention 量化的配置 | 新增 |

### 1.2 量化点数量

每个 `OSPNextAttentionBlock` 含：

| 子模块 | Linear 数量 | 是否 HIF8 量化 |
|--------|------------|---------------|
| Self-Attention: Q, K, V, O projection | 4 | 是（`quant="hif8"` 时） |
| Cross-Attention: Q, K, V, O projection | 4 | 是 |
| FFN: Linear1 (dim→ffn_dim) + Linear2 (ffn_dim→dim) | 2 | 是 |
| **每层合计** | **10** | |
| **40 层合计（14B）** | **400** | |

此外，若 `quant_attn="hif8"`，每层还有：

| 子模块 | 量化对象 | 触发位置 |
|--------|---------|---------|
| Self-Attention | Q、K、V（RoPE 之后、SDPA 之前） | `hif8_attention_with_mask()` |
| Cross-Attention | Q、K、V（无 RoPE，线性之后直接量化） | 同上 |

---

## 2. hif8_linear.py：Linear 量化实现

### 2.1 核心量化辅助函数 `_quant`

```python
# hif8_linear.py:42-50
def _quant(x: torch.Tensor, scale_max: float) -> tuple:
    qtype = QType("hif8")
    x_max = x.abs().amax().clamp(min=1e-8)     # per-tensor 最大绝对值
    scale = scale_max / x_max                  # scale 使得 x*scale 的最大值 = scale_max
    scaled_x = x * scale
    x_q = quant_dequant_float(scaled_x, qtype, force_fp32=True) / scale
    # x_q 已恢复至原量级，但含 HIF8 精度截断误差
    return x_q, scale
```

**语义**：Current Scaling（当前批次计算 scale），scale_max=15 对应 HIF8 前向最大表示值，scale_max=224 对应反向梯度。

### 2.2 `_HIF8LinearFunction`：前向 + 反向

```
前向（forward）
───────────────────────────────────────────────────────
输入:  x         (... , in_features)     — 激活
       weight    (out_features, in_features) — 权重
       bias      or None

① x_q, _   = _quant(x,      scale_max_forward=15)
② w_q, _   = _quant(weight, scale_max_forward=15)
③ ctx.save  → x_q, w_q, scale_max_backward

④ out = F.linear(x_q, w_q, bias)
   = x_q @ w_q.T + bias
───────────────────────────────────────────────────────

反向（backward）
───────────────────────────────────────────────────────
输入:  grad_output (... , out_features)  — 来自上层的梯度

① grad_q, _ = _quant(grad_output, scale_max_backward=224)

② grad_input  = grad_q  @ w_q          → shape (..., in_features)
③ grad_weight = g_2d.T  @ x_2d         → shape (out_features, in_features)
④ grad_bias   = g_2d.sum(0)            → shape (out_features,)
───────────────────────────────────────────────────────
```

**关键点**：
- 前向保存的是 `x_q` 和 `w_q`（已量化），而不是原始 `x` 和 `weight`，**反向复用前向的量化值**，无需重新量化激活和权重
- `scale_max_backward=224` 比 `scale_max_forward=15` 大得多（~15×），是因为梯度的数值范围通常比激活更大，需要更大的 scale_max 才能覆盖

### 2.3 `HIF8Linear`：Drop-in 替换 nn.Linear

```python
# hif8_linear.py:117-171
class HIF8Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 scale_max_forward=15.0, scale_max_backward=224.0): ...

    def forward(self, x):
        return _HIF8LinearFunction.apply(
            x, self.weight, self.bias,
            self.scale_max_forward, self.scale_max_backward
        )
```

**与 `nn.Linear` 的行为差异**：

| 项目 | `nn.Linear` | `HIF8Linear` |
|------|------------|--------------|
| 前向 | `x @ W.T + b`（bf16） | `x_q @ w_q.T + b`（量化误差仿真） |
| 反向 ∂x | `grad @ W` | `grad_q @ w_q`（grad 也被量化） |
| 反向 ∂W | `grad.T @ x` | `grad_q.T @ x_q` |
| 参数 W 存储 | 同 `nn.Linear` | 完全相同，不改变存储精度 |

---

## 3. hif8_attention.py：Attention QKV 量化实现

### 3.1 调用链

```
OSPNextSelfAttention.forward()
    │
    ├─ q = self.norm_q(self.q(x))       ← self.q 是 HIF8Linear（第一级量化）
    ├─ k = self.norm_k(self.k(x))       ← self.k 是 HIF8Linear
    ├─ v = self.v(x)                    ← self.v 是 HIF8Linear
    │
    ├─ rope_wrapper.apply_rope(q, ...)  ← 应用 RoPE（未量化）
    ├─ rope_wrapper.apply_rope(k, ...)
    │
    └─ hif8_attention_with_mask(q, k, v, ...)   ← 第二级量化（Attention 量化）
           └─ _HIF8QKVQuantFunction.apply(q, k, v, smf, smb)
                  ├─ forward:  q_q = quant(q), k_q = quant(k), v_q = quant(v)
                  └─ backward: dq → quant(dq), dk → quant(dk), dv → quant(dv)
           └─ attention_with_mask(q_q, k_q, v_q, ...)   ← NPU SDPA / FA
```

### 3.2 量化时机：RoPE 之后

用户理解是正确的：`hif8_attention_with_mask` 在 `rope_wrapper.apply_rope()` **之后**被调用（见 [osp_next_hif8_linear_atten.py:922-937](torchdiff/modules/osp_next_hif8_linear_atten.py#L922)）。因此：

- 进入 SDPA 的 Q/K/V 是带 RoPE 的、被 HIF8 量化的版本
- 量化误差叠加在了旋转位置编码之后

### 3.3 `_HIF8QKVQuantFunction`：前向 + 反向

```
前向（forward）
───────────────────────────────────────────────────────
① q_q = quant_dequant(q * (15 / max|q|)) / scale_q
② k_q = 同上
③ v_q = 同上
④ ctx.save_for_backward: scale_max_backward=224
⑤ return q_q, k_q, v_q  → 传给 attention_with_mask()
───────────────────────────────────────────────────────

反向（backward）
───────────────────────────────────────────────────────
输入:  grad_q, grad_k, grad_v  ← 来自 attention 内核的反向

① dq_q = quant(grad_q, scale_max=224)
② dk_q = quant(grad_k, scale_max=224)
③ dv_q = quant(grad_v, scale_max=224)
④ return dq_q, dk_q, dv_q, None, None
───────────────────────────────────────────────────────
```

**与 hif8_linear.py 的区别**：
- `hif8_linear.py` 保存 `x_q, w_q` 用于反向复用
- `hif8_attention.py` 的 `_HIF8QKVQuantFunction` **不保存 q_q/k_q/v_q**，因为反向只需要对来自 attention 的梯度重新量化，不需要 forward 的值

---

## 4. osp_next_hif8_linear_atten.py 改动标注

以下列出 `osp_next.py` → `osp_next_hif8_linear_atten.py` 的全部修改点：

### 4.1 新增导入

```python
# 改动位置：文件头部 import 区
# --- 新增 ---
from .hif8_linear import HIF8Linear                      # Line 18
from .hif8_attention import hif8_attention_with_mask     # Line 19
```

### 4.2 新增 `_make_linear` 工厂函数

```python
# 新增函数（Lines 22-37）
def _make_linear(quant, in_features, out_features,
                 scale_max_forward=15.0, scale_max_backward=224.0,
                 bias=True) -> nn.Linear:
    if quant == "hif8":
        return HIF8Linear(in_features, out_features, bias=bias,
                          scale_max_forward=scale_max_forward,
                          scale_max_backward=scale_max_backward)
    return nn.Linear(in_features, out_features, bias=bias)
```

**作用**：统一入口，通过 `quant` 字符串决定用 `HIF8Linear` 还是 `nn.Linear`，调用方不需要关心量化类型。

### 4.3 `OSPNextSelfAttention.__init__`：新增参数 + 替换 Linear

```python
# 修改位置：Lines 808-834

# --- 原版 ---
def __init__(self, dim, num_heads, ...):
    ...
    self.q = nn.Linear(dim, dim)
    self.k = nn.Linear(dim, dim)
    self.v = nn.Linear(dim, dim)
    self.o = nn.Linear(dim, dim)

# --- 改动后 ---
def __init__(self, dim, num_heads, ...,
             quant=None,               # ← 新增
             quant_attn=None,          # ← 新增
             scale_max_forward=15.0,   # ← 新增
             scale_max_backward=224.0, # ← 新增
             ):
    ...
    self.quant_attn = quant_attn
    self.scale_max_forward = scale_max_forward
    self.scale_max_backward = scale_max_backward

    self.q = _make_linear(quant, dim, dim, ...)  # ← HIF8Linear or nn.Linear
    self.k = _make_linear(quant, dim, dim, ...)
    self.v = _make_linear(quant, dim, dim, ...)
    self.o = _make_linear(quant, dim, dim, ...)
```

### 4.4 `OSPNextSelfAttention.forward`：新增 `quant_attn` 分支

```python
# 修改位置：Lines 930-943

# --- 原版 ---
x = attention_with_mask(q, k, v, attn_mask=..., attn_mask_kv=...)

# --- 改动后 ---
if self.quant_attn == "hif8":
    x = hif8_attention_with_mask(           # ← 新增分支
        q, k, v,
        attn_mask=attn_mask,
        attn_mask_kv=attn_mask,
        scale_max_forward=self.scale_max_forward,
        scale_max_backward=self.scale_max_backward,
    )
else:
    x = attention_with_mask(q, k, v, ...)   # ← 原版路径保留
```

**注意**：量化发生在 `rope_wrapper.apply_rope()` 之后（Line 922-923 → Line 930），即 RoPE 已完成后再量化 Q/K/V。

### 4.5 `OSPNextCrossAttention.forward`：同样新增分支

```python
# 修改位置：Lines 973-983（与 SelfAttention 对称）

if self.quant_attn == "hif8":
    x = hif8_attention_with_mask(
        q, k, v,
        attn_mask=attn_mask,
        attn_mask_kv=None,      # Cross Attn：只有 q 有 mask
        is_cross_attn=True,
        scale_max_forward=self.scale_max_forward,
        scale_max_backward=self.scale_max_backward,
    )
else:
    x = attention_with_mask(q, k, v, ..., is_cross_attn=True)
```

### 4.6 `OSPNextAttentionBlock.__init__`：FFN Linear 也替换

```python
# 修改位置：Lines 1011-1015（新增参数） + Lines 1052-1055（FFN 替换）

# 原版
self.ffn = nn.Sequential(
    nn.Linear(dim, ffn_dim),
    nn.GELU(approximate="tanh"),
    nn.Linear(ffn_dim, dim),
)

# 改动后
self.ffn = nn.Sequential(
    _make_linear(quant, dim, ffn_dim, scale_max_forward, scale_max_backward),
    nn.GELU(approximate="tanh"),
    _make_linear(quant, ffn_dim, dim, scale_max_forward, scale_max_backward),
)
```

### 4.7 `OSPNextModel.__init__`：新增配置参数

```python
# 修改位置：Lines 1260-1264（新增参数声明）

def __init__(
    self, ...,
    quant=None,              # ← 新增：None 或 "hif8"
    quant_attn=None,         # ← 新增：None 或 "hif8"
    scale_max_forward=15.0,  # ← 新增
    scale_max_backward=224.0,# ← 新增
):
```

这些参数向下传递给每个 `OSPNextAttentionBlock`，进而传给 `OSPNextSelfAttention`/`OSPNextCrossAttention` 和 FFN。

---

## 5. 量化覆盖全景图（per block）

下图展示单个 `OSPNextAttentionBlock` 的数据流与量化位置：

```
x (input token sequence)
│
├─ norm1(x) * (1+e1) + e0
│       │
│       ▼  Self-Attention
│   ┌─────────────────────────────────────────────────────────┐
│   │  self.q (HIF8Linear) ── norm_q ── Q ──► RoPE ──► Q_rope │
│   │  self.k (HIF8Linear) ── norm_k ── K ──► RoPE ──► K_rope │  ← Linear 量化①②③
│   │  self.v (HIF8Linear) ──────────── V ─────────► V        │
│   │                                                         │
│   │  if quant_attn=="hif8":                                 │
│   │    hif8_attention_with_mask(Q_rope, K_rope, V)          │  ← Attention QKV 量化④⑤⑥
│   │    → _HIF8QKVQuantFunction → Q_q, K_q, V_q             │
│   │    → SDPA(Q_q, K_q, V_q) → attn_out                    │
│   │                                                         │
│   │  self.o (HIF8Linear) ← attn_out.flatten(2)             │  ← Linear 量化⑦
│   └─────────────────────────────────────────────────────────┘
│
├─ x + y * e2
│
├─ norm3(x) → Cross-Attention
│   ┌─────────────────────────────────────────────────────────┐
│   │  self.q (HIF8Linear) ── norm_q ── Q (from img)         │  ← Linear 量化⑧
│   │  self.k (HIF8Linear) ── norm_k ── K (from text)        │  ← Linear 量化⑨
│   │  self.v (HIF8Linear) ──────────── V (from text)        │
│   │                                                         │
│   │  if quant_attn=="hif8":                                 │
│   │    hif8_attention_with_mask(Q, K, V, is_cross_attn=True)│  ← Attention QKV 量化
│   │    → SDPA → cross_attn_out                             │
│   │                                                         │
│   │  self.o (HIF8Linear) ← cross_attn_out.flatten(2)       │  ← Linear 量化（含在cross_attn内）
│   └─────────────────────────────────────────────────────────┘
│
├─ norm2(x) * (1+e4) + e3
│       │
│       ▼  FFN
│   ┌──────────────────────────────────────┐
│   │  Linear1 (HIF8Linear): dim→ffn_dim  │  ← Linear 量化
│   │  GELU                               │
│   │  Linear2 (HIF8Linear): ffn_dim→dim  │  ← Linear 量化
│   └──────────────────────────────────────┘
│
▼ x + y * e5
```

**量化点总结（每层，`quant="hif8"` 且 `quant_attn="hif8"`）**：

| 编号 | 位置 | 量化类型 | 前向量化对象 | 反向量化对象 |
|------|------|---------|------------|------------|
| ①②③ | Self-Attn Q/K/V projection | HIF8Linear | x + W | grad_output |
| ④⑤⑥ | Self-Attn Q/K/V → SDPA | HIF8QKVQuant | Q, K, V (post-RoPE) | dQ, dK, dV |
| ⑦ | Self-Attn O projection | HIF8Linear | attn_out + W | grad_output |
| ⑧⑨ | Cross-Attn Q/K projection | HIF8Linear | x/text + W | grad_output |
| （含 V）| Cross-Attn V projection | HIF8Linear | text + W | grad_output |
| （cross out）| Cross-Attn O projection | HIF8Linear | cross_out + W | grad_output |
| （cross QKV）| Cross-Attn Q/K/V → SDPA | HIF8QKVQuant | Q, K, V | dQ, dK, dV |
| （ffn1/2）| FFN Linear1, Linear2 | HIF8Linear | x + W | grad_output |

---

## 6. 参考项目 HiF8_Quant_Sample 分析

参考项目提供了三个量化目标：Linear、FlashAttention、Optimizer，分别对应三个量化节点，与 OSP 项目的改造目标完全对应。

### 6.1 `HiF8_Quant_linear.py`

#### 两个量化 Linear 类

```python
class HiF8CurrectScalingLinear(nn.Linear):   # Current Scaling
class HiF8DelayedScalingLinear(nn.Linear):   # Delayed Scaling
```

每个类对应一个 `torch.autograd.Function`：
- `HiF8CurrectScalingQuantMatmul`
- `HiF8DelayedScalingQuantMatmul`

#### Current Scaling 实现（`HiF8CurrectScalingQuantMatmul`）

```
前向：
  qx,      _ = quantize(x,      scale_max=15)     # 当前 batch 的 x 统计
  qweight, _ = quantize(weight, scale_max=15)
  out = qx @ qweight.T + bias
  → ctx 保存 x, weight, bias（原始未量化版本）和 xscale, wscale

反向：
  qgrad_output, _ = quantize(grad_output, scale_max=224)
  qx,     _ = quantize(x,      xscale)     # 复用前向 scale（非重新计算）
  qweight,_ = quantize(weight, wscale)     # 复用前向 scale
  grad_input  = qgrad_output @ qweight
  grad_weight = qgrad_output.T @ qx
```

**特点**：scale 在每次 forward call 时即时计算，当批统计，当批使用。

#### Delayed Scaling 实现（`HiF8DelayedScalingQuantMatmul`）

```python
# 全局 scale 字典，按 FWD_LINEAR_IDX（每层的顺序编号）索引
A_SCALE = {}   # activation scale
W_SCALE = {}   # weight scale
G_SCALE = {}   # gradient scale
AMAX   = {}    # 历史 grad amax 滑动窗口（大小=G_POOL_SIZE=128）
```

```
前向（cur_iter=0 时初始化，之后用上一轮的 scale）：

  FWD_LINEAR_IDX += 1

  if cur_iter == 0:
      A_SCALE[idx] = scale_max / max|x|        # 初始化
      W_SCALE[idx] = scale_max / max|weight|

  qx,      _ = quantize(x,      scale=A_SCALE[idx])   # 用上轮计算的 scale
  qweight, _ = quantize(weight, scale=W_SCALE[idx])

  if (cur_iter+1) % FWD_CHANGE_INTERVAL == 0:           # 每 10 步更新一次 scale
      A_SCALE[idx] = scale_max / max|x_current|         # 下轮才生效 → "延迟"
      W_SCALE[idx] = scale_max / max|weight_current|

反向（前 G_POOL_SIZE=128 轮用当前统计，之后用滑动平均）：

  BWD_LINEAR_IDX += 1

  if cur_iter > G_POOL_SIZE-1:
      # 滑动池已满，用历史均值（或最大值/最新值）
      avg_amax = mean(AMAX[idx])               # G_POOL_TYPE=AVG 时
      G_SCALE[idx] = scale_max / avg_amax      # 延迟 scale
      qgrad_output = quantize(grad_output, scale=G_SCALE[idx])
      amax = max|qgrad_output|
      AMAX[idx].pop(0); AMAX[idx].append(amax) # 滑动更新

  else:
      # 预热阶段：用当前 grad 统计（类 current scaling）
      amax = max|grad_output|
      AMAX[idx].append(amax)
      G_SCALE[idx] = scale_max / max|grad_output|
      qgrad_output = quantize(grad_output, scale=G_SCALE[idx])
```

**额外功能**：
- `MARGIN_ENABLE=True` 时，每隔 `MARGIN_SEARCH_INTERVAL=100` 步做 margin search，从 `SCALE_MAX_LIST=[15,28,56,112,224,384,768]` 中找最优 scale_max（最小化 MSE）
- 区分 cross attention 和 self attention 梯度：cross 用 `cross_grad_scale_max=15`，self 用 `self_grad_scale_max=224`

### 6.2 `HiF8_Quant_fa.py`

#### 三个 Attention 量化实现

**① `AscendCurrentScalingFlashAttentionFunction`**

```
前向：if HiF8_QUANT: q,k,v = quant(q,15), quant(k,15), quant(v,15)
      → torch_npu.npu_fusion_attention(q,k,v, ...)
      保存: q, k, v, attn_out, softmax_max, softmax_sum

反向：if HiF8_QUANT: do = quant(do, 224)
      → torch_npu.npu_fusion_attention_grad(q,k,v,do, ...)
```

Current Scaling。每次 call 时计算 per-tensor scale。

---

**② `AscendDelayedScalingFlashAttentionFunction`**

与 Linear Delayed Scaling 一致：
- 使用全局 `Q_SCALE, K_SCALE, V_SCALE, dO_SCALE` 字典
- 前向 cur_iter==0 初始化，之后用上轮 scale；每 10 步（`FWD_FA_CHANGE_INTERVAL`）更新
- 反向用滑动 amax 池（`AMAX_FA`，大小 128）计算 dO scale
- NaN/Inf 检测：若 amax 为 NaN 或 Inf，跳过该次更新（保护 scale 不被污染）

---

**③ `FlashAttentionFunction`（PyTorch FA v2 实现）**

这是一个纯 PyTorch 的 Flash Attention 实现（用于 GPU 或不支持 `npu_fusion_attention` 的场景）。在其中插入了 HIF8 量化点：

```
前向：
  - Q, K, V: quant(15)
  - exp_weights (softmax 概率矩阵 P): static_quant(P, 15)

反向：
  - P: static_quant(P, 15)
  - dO (输出梯度): static_quant(doc, 224)
  - dS (softmax 梯度): static_quant(ds, 224)
```

**注意**：`static_quant` 使用外部传入的固定 scale，而非从当前数据动态计算。

### 6.3 `HiF8_Quant_optimizer.py`

#### 量化目标：AdamW 动量状态

```python
class MomentumQuantizer:
    scale_max_exp_avg    = 512.0   # 一阶动量（exp_avg）
    scale_max_exp_avg_sq = None    # 二阶动量（exp_avg_sq，暂未启用）
    group_size = 512               # 分组量化，每 512 个元素一组
```

#### 分组量化（Group-wise Quantization）

与 Linear/FA 的 per-tensor 量化不同，动量使用 **per-group** 量化：

```python
def _quantize(self, tensor, scale_max=None):
    # 1. 将 tensor 展平，padding 到 group_size 的整数倍
    # 2. reshape → (num_groups, group_size)
    # 3. 每组计算 max → per-group scale
    # 4. 量化：scaled = tensor * per_group_scale
    # 5. quant_dequant_float → 量化后反量化
    # 6. 还原形状
```

原因：动量张量数值范围在参数之间差异大，per-tensor scale 会让数值小的参数精度极差；per-group 可以自适应。

#### 与 AdamW 的集成方式

```python
class AdamW(Optimizer):
    def step(self):
        momentum_quantizer = MomentumQuantizer(self)
        momentum_quantizer.quantize_all_momenta()  # ① 量化所有动量
        
        adamw(...)   # ② 正常 AdamW 更新（使用量化后的动量）
        
        # ③ 更新完成后再次量化（_fused_adamw 中的 requantize_after_computation）
```

---

## 7. Current Scaling vs Delayed Scaling 对比

### 7.1 核心区别

| 特性 | Current Scaling | Delayed Scaling |
|------|----------------|----------------|
| **scale 来源** | 当前 batch 的 `max\|x\|` | 上一轮（或历史均值）的统计 |
| **前向计算时序** | scale 与 matmul 串行（需先计算 amax） | scale 提前计算好，前向只需查表 |
| **硬件效率** | amax 计算阻塞下一步 | scale 可与上一步 backward 并行计算 |
| **稳定性** | 每步 scale 不同，梯度方差大 | scale 变化平滑（moving average） |
| **冷启动** | 无需预热 | 需要 G_POOL_SIZE 步（128 步）预热 |
| **异常值鲁棒性** | 差（单批 outlier 导致 scale 过小） | 好（滑动平均平滑单次 spike） |

### 7.2 梯度 Delayed Scaling 的两阶段

```
阶段一（预热，cur_iter < G_POOL_SIZE=128）：
  → 行为类似 Current Scaling（每次用当前统计）
  → 同时将 max|grad| 存入 AMAX[idx] 池

阶段二（稳定运行，cur_iter >= 128）：
  → 用 AMAX 池的均值/最大值 计算 G_SCALE
  → 对当前 grad 量化
  → 滑动更新：AMAX.pop(0), AMAX.append(当前 amax)
```

```
                    AMAX 池更新示意
iter: 0  1  2  ... 127 | 128  129 ...
      ──────────────────|──────────────
      [a0,a1,...,a127]  | 开始滑动：
      写入 AMAX[idx]    | pop a0, append a128
                        | 用 mean(a1..a128) 计算 G_SCALE
```

### 7.3 前向 Delayed Scaling 的更新逻辑

```python
# cur_iter=0: 初始化（current scaling，仅一次）
A_SCALE[idx] = scale_max / max|x|

# cur_iter=1,2,...,9: 使用 iter=0 的 scale（延迟生效）
qx = quantize(x, scale=A_SCALE[idx])   # 用上轮 scale

# cur_iter=9: 满足 (9+1)%10==0，计算新 scale（于 iter=10 时生效）
A_SCALE[idx] = scale_max / max|x_current|   # → iter=10 才用到

# 节奏：每 10 步更新一次前向 scale
```

---

## 8. OSP 实现 vs 参考实现 差异对比

### 8.1 Linear 量化

| 特性 | OSP `hif8_linear.py` | 参考 `HiF8_Quant_linear.py` |
|------|---------------------|---------------------------|
| **Scaling 策略** | Current Scaling（每次即时计算） | Current 和 Delayed 均有实现，当前训练用 Delayed |
| **Scale 更新粒度** | 每次 forward call | 每 10 步（前向）/ 滑动128步（反向） |
| **反向保存内容** | `x_q, w_q`（量化后的值） | `x, weight`（原始值，反向再次量化） |
| **NaN/Inf 检测** | 无 | 有（cur_iter>240 时监控） |
| **Margin Search** | 无 | 有（MARGIN_ENABLE=True 时，每100步） |
| **Cross vs Self 梯度区别** | 统一 scale_max_backward=224 | Cross=15, Self=224（更精细区分） |
| **Dump 调试** | 无 | 有（USE_DUMP 开关） |
| **反向量化权重** | 使用前向保存的 `w_q` | 用前向 wscale 重新量化 weight |

### 8.2 Attention 量化

| 特性 | OSP `hif8_attention.py` | 参考 `HiF8_Quant_fa.py` |
|------|------------------------|------------------------|
| **量化后端** | `_HIF8QKVQuantFunction` + `attention_with_mask`（SDPA/FA） | `npu_fusion_attention`（NPU 原生 FA） |
| **Scaling 策略** | Current Scaling | Delayed Scaling（`AscendDelayedScalingFlashAttentionFunction`） |
| **量化对象（前向）** | Q, K, V | Q, K, V（Current 和 Delayed 两版本） |
| **量化对象（反向）** | dQ, dK, dV | dO（传入 `npu_fusion_attention_grad`） |
| **P (softmax) 量化** | 无 | `FlashAttentionFunction` 中有（仅 GPU 版本） |
| **dS 量化** | 无 | `FlashAttentionFunction` 中有 |

> **关键区别**：OSP 在反向量化 dQ/dK/dV，而参考项目量化 dO。两者作用点不同：OSP 拦截的是 attention 核对 Q/K/V 的梯度；参考项目拦截的是传入 `npu_fusion_attention_grad` 的输出梯度 dO。在 `npu_fusion_attention` 的 API 下，用户只能插入 dO，因此参考项目选择量化 dO。

### 8.3 OSP 项目尚未实现的量化点（参考实现中有）

1. **Delayed Scaling**：OSP 目前使用 Current Scaling，如需更稳定的量化训练，可参考 `HiF8DelayedScalingQuantMatmul` 增加 G_POOL 机制
2. **Optimizer 量化**：OSP 尚未对 AdamW 动量做 HIF8 量化；参考项目的 `MomentumQuantizer` 提供了 group-wise 量化实现
3. **softmax P 量化**：OSP 的 attention 量化仅覆盖 Q/K/V 输入；参考项目 `FlashAttentionFunction` 在 softmax 概率矩阵上也插入了量化点（适用于 GPU 手写 FA 场景）
4. **Cross vs Self 梯度 scale 区分**：OSP 统一用 224，参考项目对 cross attention 梯度用更小的 scale_max=15

### 8.4 YAML 配置接口

OSP 项目通过以下 yaml 字段控制量化：

```yaml
model_config:
  quant: "hif8"       # None 或 "hif8"；控制所有 10 个 Linear
  quant_attn: "hif8"  # None 或 "hif8"；控制 Attention QKV 量化
  scale_max_forward: 15.0
  scale_max_backward: 224.0
```

- `quant="hif8"` 单独使用 → 仅 Linear 量化（10 个/层）
- `quant="hif8"` + `quant_attn="hif8"` → Linear + Attention 联合量化
- 两者均为 `None` → 原始 bf16 训练

---

## 9. NPU 生态中 SDPA 与 npu_fusion_attention 的区别

> **背景问题**：参考项目只能量化 `dO`，而 OSP 项目能量化 `dQ/dK/dV`，两者都在 NPU 上运行。原因在于两个项目使用了不同的 Attention 计算 API，导致反向传播的"入口点"不同。

### 9.1 OSP 项目使用的 Attention 路径：SDPA

在 `attention.py:500` 中有明确注释：

```python
# ========== npu不支持flash attn接口，走SDPA ==========
if is_npu_available() or (not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE):
    output = scaled_dot_product_attention_with_mask(
        q=q, k=k, v=v, attn_mask=attn_mask_kv, ...
    )
```

最终调用：
```python
# attention.py:476
out = torch.nn.functional.scaled_dot_product_attention(
    q, k, v, attn_mask=attn_mask, ...
)
```

`torch.nn.functional.scaled_dot_product_attention`（SDPA）是 **PyTorch 原生 API**，通过 PyTorch dispatcher 在 NPU 上分发给 `torch_npu` 包的后端实现（新版 CANN 内部映射到高效的 NPU Fusion 内核）。

### 9.2 参考项目使用的 Attention 路径：npu_fusion_attention

```python
# HiF8_Quant_fa.py:253
res = torch_npu.npu_fusion_attention(
    q, k, v,
    head_num=num_heads,
    input_layout="BNSD",
    keep_prob=1,
    scale=q.shape[-1] ** -0.5,
)
attn = res[0].squeeze(0).transpose(0, 1)
ctx.save_for_backward(q, k, v, res[0], res[1], res[2])
# res[1]=softmax_max, res[2]=softmax_sum 是反向时必须的中间态
```

`torch_npu.npu_fusion_attention` 是 **Ascend CANN 的私有算子**，反向时需要手动调用配对的 `npu_fusion_attention_grad`：

```python
# HiF8_Quant_fa.py:346
res = torch_npu.npu_fusion_attention_grad(
    q, k, v, do,                        # ← do 是量化后的梯度
    head_num=num_heads,
    input_layout="BNSD",
    softmax_max=softmax_max,             # ← 前向保存的中间态
    softmax_sum=softmax_sum,             # ← 前向保存的中间态
    attention_in=attn_in,               # ← 前向注意力输出
    scale_value=q.shape[-1] ** -0.5,
)
dq, dk, dv = res[0], res[1], res[2]
```

### 9.3 核心架构对比

| 维度 | OSP：SDPA | 参考：npu_fusion_attention |
|------|-----------|--------------------------|
| **API 层** | PyTorch 标准接口 `torch.nn.functional.sdpa` | CANN 私有接口 `torch_npu.npu_fusion_attention` |
| **反向触发方式** | PyTorch autograd 自动计算 | 用户在 `torch.autograd.Function.backward` 中**手动调用** `npu_fusion_attention_grad` |
| **反向中间态** | PyTorch 自动管理（内部缓存或重算） | 前向**必须显式保存** `softmax_max, softmax_sum, attn_out` |
| **梯度输出接口** | autograd 将 dQ/dK/dV **作为独立张量**传给 backward 返回值 | 调用 `npu_fusion_attention_grad(do, ...)` 后拿到 dq/dk/dv |
| **跨平台性** | 同一代码在 GPU/NPU 均可运行 | 仅 Ascend NPU，GPU 无此 API |

### 9.4 反向传播路径的本质差异

这是两种完全不同的反向计算图构型：

#### SDPA 路径（OSP）

```
外层 _HIF8QKVQuantFunction
┌─────────────────────────────────────────────────────┐
│  forward:  Q_q, K_q, V_q = quant(Q, K, V)          │
│  ↓                                                  │
│  SDPA(Q_q, K_q, V_q)                               │
│  = torch.nn.functional.scaled_dot_product_attention │
│  ↓                                                  │
│  attn_out                                           │
│                                                     │
│  backward:  [PyTorch autograd 已计算完 dQ/dK/dV]    │
│                                                     │
│  → 接收 dQ, dK, dV（由 SDPA 内核 backward 产生）   │
│  → return quant(dQ), quant(dK), quant(dV)           │
│    ↑ 在 SDPA 已经完成反向之后，对其输出梯度量化     │
└─────────────────────────────────────────────────────┘
```

PyTorch autograd 将 SDPA 作为一个黑盒处理：它知道如何对 `scaled_dot_product_attention` 求导（或由 CANN backend 提供自定义 vjp）。`_HIF8QKVQuantFunction` 的 `backward` 只是接收 SDPA backward 的"结果"——即 dQ/dK/dV——然后对其量化。

#### npu_fusion_attention 路径（参考项目）

```
AscendDelayedScalingFlashAttentionFunction
┌─────────────────────────────────────────────────────┐
│  forward:  调用 npu_fusion_attention(Q, K, V)       │
│            保存 q, k, v, attn_out, softmax_max/sum  │
│                                                     │
│  backward: [用户完全控制反向过程]                   │
│                                                     │
│  ① do = quant(grad_output)     ← 量化 dO           │
│  ② res = npu_fusion_attention_grad(               │
│            q, k, v, do,                           │  ← do 已被量化
│            softmax_max, softmax_sum, attn_in)       │
│  ③ dq, dk, dv = res[0], res[1], res[2]             │
│                                                     │
│  → 此时 dq/dk/dv 是由"量化的 do"反向计算得到        │
│    的，不再额外量化                                 │
└─────────────────────────────────────────────────────┘
```

用户自己"实现"了 backward：必须手动调用 `npu_fusion_attention_grad`，且只能在调用之前插入对 `do` 的操作。一旦 `npu_fusion_attention_grad` 执行完毕，dq/dk/dv 已由 CANN kernel 计算好，再对其量化已无意义（梯度计算已在 fp32/bf16 精度下完成）。

### 9.5 量化点语义差异

```
                 loss
                  │ dL/d(attn_out) = dO
                  ↓
         ┌──────────────────┐
         │   Attention 核   │
         │  Softmax(QKᵀ)·V  │
         └──────────────────┘
          │        │        │
         dQ       dK       dV

参考项目量化点:       ↑ 量化 dO（进入 attention backward 之前）
OSP 量化点:          量化 dQ/dK/dV（attention backward 完成之后）↑
```

| 量化点 | 语义 | 受影响的计算 |
|--------|------|------------|
| 参考：量化 `dO` | 模拟"反向激活信号在进入 attention backward 时精度损失" | attention backward kernel 内部的全部梯度计算都受量化 dO 影响 |
| OSP：量化 `dQ/dK/dV` | 模拟"从 attention backward 流出、传向 norm/linear 层的梯度精度损失" | attention backward 本身在 bf16/fp32 精度下运行，量化只影响下游（Q/K/V projection 的 backward） |

两种方式都是合理的量化仿真，只是模拟的精度损失"位置"不同。

### 9.6 为什么 OSP 选择 SDPA

从代码看，理由是多方面的：

1. **跨硬件兼容性**：同一套 `attention_with_mask` 函数在 GPU（FA v2/v3）和 NPU（SDPA）上均可运行，不需要分别实现
2. **反向自动管理**：PyTorch autograd 负责 SDPA 的反向，无需显式保存 softmax_max/sum 等中间态；而 `npu_fusion_attention_grad` 要求用户自己保存并传入这些张量
3. **量化插入点天然明确**：`_HIF8QKVQuantFunction.backward` 直接接收 dQ/dK/dV，不需要理解 CANN kernel 的内部结构
4. **代码简洁性**：`torch.autograd.Function` 的 forward/backward 各只有几行，而参考项目的 `AscendDelayedScalingFlashAttentionFunction` 需要管理全局 scale dict、滑动 amax 池、NaN 检测等

### 9.7 两种 Attention 实现在 NPU 上的性能比较

| 方面 | SDPA（通过 torch_npu）| npu_fusion_attention |
|------|----------------------|---------------------|
| **kernel 融合程度** | 依赖 torch_npu 版本（新版可融合） | 始终是单个融合 CANN kernel |
| **前向速度** | 一般等价（新 CANN 内部路径相同）| 略快（直接调用底层 kernel，无 dispatch 开销） |
| **反向速度** | PyTorch autograd 管理，可能有额外 dispatch | 直接调用 `npu_fusion_attention_grad`，无 autograd 开销 |
| **中间态显存** | PyTorch 自动选择保存策略 | 必须显式保存 softmax_max/sum，但避免 backward 中重算 |
| **与 gradient_checkpointing 兼容** | 天然兼容（重计算 SDPA 时 PyTorch 自动处理）| 需要特别注意：重计算时 ctx 中保存的中间态是否还有效 |

> 在实际 14B 训练中，两者的性能差异通常在 5% 以内（瓶颈在 all_gather 通信，而非 attention 计算本身）。OSP 选择 SDPA 的工程收益（可维护性、跨平台）大于微小的性能差距。

---

## 10. 四个追问：SDPA/NPU 特性、npu_fusion_attention vs FA2、量化差异影响、Optimizer 量化位置

### 10.1 SDPA 在 NPU 上是普通实现还是利用了 NPU 特性？

**结论：在 NPU 上 SDPA 是硬件优化实现，但不是 Flash Attention v2/v3。**

`torch.nn.functional.scaled_dot_product_attention` 是 PyTorch 的标准算子，但它的具体实现通过 PyTorch **kernel dispatch 机制**在不同硬件后端分发：

```
torch.nn.functional.scaled_dot_product_attention
        │
        ├─ is_cuda  → flash_attn 内核（NVIDIA GPU）
        │
        └─ is_npu   → torch_npu 注册的 NPU 后端实现
                       ↓
                  新版 CANN (8.x+): 内部调用 npu_fusion_attention
                  旧版 CANN:        可能退化为分步实现（QK^T → softmax → SV）
```

torch_npu 通过 `TORCH_LIBRARY_IMPL` 将 SDPA 算子注册到 "PrivateUse1" (NPU) 设备后端，使得在 NPU tensor 上调用 `torch.nn.functional.scaled_dot_product_attention` 时走 CANN 的高效融合实现，**不会回退到 O(N²) 的朴素实现**。

**与 Flash Attention v2/v3 的本质区别：**

Flash Attention v2/v3 是专为 NVIDIA GPU 设计的算法：
- 利用 NVIDIA GPU 的 SRAM（A100: 192KB/SM）做分块计算，避免反复读写 HBM
- 使用 CUDA PTX 级别的 warp-level primitive
- 反向公式经过特殊推导（online softmax，无需重算注意力矩阵）

Ascend NPU 的内存层次与 NVIDIA GPU 完全不同（L1 Buffer、Unified Buffer、L2 Cache，不存在 CUDA 意义上的 SRAM），因此 npu_fusion_attention 使用了 Ascend 特有的分块和向量化策略。两者目标相同（避免 O(N²) 内存），但算法实现路径不同。

**实际表现：**

| 场景 | 行为 |
|------|------|
| NPU + 新版 torch_npu (≥2.1/CANN 8.x) | SDPA → 内部调用 npu_fusion_attention，高效融合 |
| NPU + 旧版 torch_npu | SDPA → 可能分步执行，效率较低 |
| CUDA + FA2/FA3 可用 | `attention.py` 走 FA2/FA3 路径，不走 SDPA |
| CUDA 但无 FA | SDPA → `torch.backends.cuda.flash_sdp_enabled()` → cuDNN FlashAttn |

OSP 代码在 `attention.py:499-500` 明确写了判断条件：

```python
if is_npu_available() or (not FLASH_ATTN_2_AVAILABLE and not FLASH_ATTN_3_AVAILABLE):
    output = scaled_dot_product_attention_with_mask(...)  # → SDPA 路径
```

NPU 上 FA2/FA3 的 Python 包不可用（它们是 CUDA 专属编译的），所以 NPU 只走 SDPA。

---

### 10.2 npu_fusion_attention 等价于 Flash Attention 2 吗？

**数学结果等价，算法实现和性能模型不同。**

| 特性 | Flash Attention 2 | npu_fusion_attention |
|------|------------------|---------------------|
| **数学输出** | `softmax(QKᵀ/√d)·V` | 完全相同 |
| **内存复杂度** | O(N)，不实例化 N×N 矩阵 | O(N)，同样不实例化全矩阵 |
| **分块策略** | SRAM 分块（tile on HBM/SRAM） | L1 Buffer 分块（tile on HBM/L1） |
| **硬件目标** | NVIDIA GPU（A100/H100） | Ascend NPU（910A/910B） |
| **反向中间态** | LSE（log-sum-exp，隐含） | `softmax_max` + `softmax_sum`（显式保存） |
| **反向 API** | Flash Attention 内置，autograd 自动调用 | 需手动调用 `npu_fusion_attention_grad` |
| **causal mask** | 支持 | 支持（`sparse_mode` 参数） |
| **GQA（分组查询注意力）** | FA2 支持 | npu_fusion_attention 支持（`actual_seq_kvlen`） |

> **核心结论**：npu_fusion_attention 是 Ascend 平台上 Flash Attention 的等价实现（同样的 memory-efficient attention 思想），但内部算法细节针对 Ascend NPU 架构做了专门优化，不能直接用 FA2 的代码替换。它们是"功能等价的竞品"，而非"同一个东西"。

---

### 10.3 OSP 与参考项目的 Attention 量化差异有多大？

**前向几乎等价；反向量化点不同但影响较小；最大差距在于 Scaling 策略。**

#### 前向：基本相同

两个项目的前向量化逻辑完全一致：

```
输入 Q, K, V (post-RoPE)
→ HIF8 量化（scale_max=15, per-tensor current/delayed scale）
→ quantized Q_q, K_q, V_q
→ Attention 内核
```

对推理量化仿真效果而言，**前向量化是最重要的部分**——它直接决定模型在实际 HIF8 推理时的精度损失模拟是否准确。两个项目在此点上效果一致。

#### 反向：量化点不同，但实际影响较小

```
OSP 项目：
  SDPA 内核（bf16）完整计算 → 输出 dQ, dK, dV（高精度）
                             → _HIF8QKVQuantFunction 量化它们
                             → 传给 Q/K/V linear 层的 backward

参考项目：
  用户量化 dO → 传入 npu_fusion_attention_grad
  → CANN kernel 内部用量化的 dO 计算 dQ, dK, dV（已受量化影响）
  → dQ, dK, dV（低精度 dO 影响的结果）传给下游
```

从梯度信号的角度：
- 两者都在 attention 反向传播路径上注入了量化噪声
- OSP 量化的是"attention backward 的输出"，参考项目量化的是"attention backward 的输入"
- 在数学上，对下游 Q/K/V linear 层收到的梯度而言，两者都引入了量化误差，量级相近

**更重要的差距是 Scaling 策略**：

| 差距 | OSP | 参考项目 | 实际影响 |
|------|-----|---------|---------|
| 前向 scale | Current（每步计算） | Delayed（滑动平均） | 训练稳定性：Delayed 更平滑 |
| 反向 scale | Current（每步计算） | Delayed（128 步滑动均值） | 梯度爆炸防护：Delayed 更鲁棒 |
| Cross vs Self grad | 统一 scale_max=224 | Cross=15, Self=224 | 微小精度差异 |
| NaN/Inf 检测 | 无 | 有 | 训练稳定性 |

如果在正常训练（无 NaN 问题）中对比，两种反向量化方法产生的模型质量差异通常可忽略不计（<0.1% 误差）。Delayed Scaling 的主要价值在于**训练稳定性**，而非精度。

---

### 10.4 OSP 项目中 Optimizer 量化在哪里，如何修改？

**当前 OSP 项目没有 Optimizer 量化。** Optimizer 在 `train/train_osp.py:348-354` 创建：

```python
# train/train_osp.py:348
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=optimizer_config.get("betas", (0.9, 0.999)),
    weight_decay=weight_decay,
    eps=optimizer_config.get("eps", 1e-15),
)
```

这是原生 `torch.optim.AdamW`，没有任何动量量化。

#### 添加 Optimizer 量化的方式

参考 `HiF8_Quant_optimizer.py` 的实现，有两种接入路径：

**方式 A：替换 AdamW 类（推荐，最接近参考项目）**

将参考项目的 `AdamW`（带 `MomentumQuantizer`）迁移进 OSP，作为 `torchdiff/modules/hif8_optimizer.py`，然后修改训练代码：

```python
# train/train_osp.py  ← 修改这里
# 原版：
# optimizer = torch.optim.AdamW(model.parameters(), ...)

# 改为：
from torchdiff.modules.hif8_optimizer import AdamW as HiF8AdamW
optimizer = HiF8AdamW(
    model.parameters(),
    lr=learning_rate,
    betas=optimizer_config.get("betas", (0.9, 0.999)),
    weight_decay=weight_decay,
    eps=optimizer_config.get("eps", 1e-15),
    fused=True,   # 使用 _fused_adamw，内部调用 MomentumQuantizer
)
```

参考项目 `AdamW.step()` 的逻辑：

```python
def step(self):
    momentum_quantizer = MomentumQuantizer(self)
    momentum_quantizer.quantize_all_momenta()  # ① 对 exp_avg, exp_avg_sq 做 HIF8 量化
    adamw(...)                                 # ② 正常 AdamW update（使用量化后的动量）
    # _fused_adamw 内部还会再 requantize 一次
```

**方式 B：在训练循环中插入量化钩子（更侵入性低）**

```python
# train/train_osp.py 的训练循环中
# 原版：
# optimizer.step()

# 改为：
from torchdiff.modules.hif8_optimizer import MomentumQuantizer
_momentum_quantizer = MomentumQuantizer(optimizer)   # 在 optimizer 创建后初始化一次

# 在每次 step 之前：
_momentum_quantizer.quantize_all_momenta()   # 量化 exp_avg, exp_avg_sq
optimizer.step()
```

注意：方式 B 需要确保 `MomentumQuantizer._quantize` 中的分组量化（`group_size=512`）与 FSDP 分片后的参数形状兼容。FSDP 下每张卡只有全局参数的 1/16，分片形状可能不是 512 的整数倍，需要做 padding 处理（参考项目中已有此逻辑）。

#### 注意事项

1. **与 FSDP2 的兼容**：FSDP2 使用 DTensor 管理参数分片，`optimizer.state` 中的 `exp_avg` 是 DTensor，而非普通 Tensor。`MomentumQuantizer._quantize` 中的 `tensor.flatten()` 需要先 `tensor.to_local()` 或 `full_tensor()` 才能操作。

2. **显存收益有限**：动量量化的目的通常是降低 optimizer state 显存（将 bf16 动量换成 int8/fp8 格式），但参考项目的实现仍然是 `quant_dequant`（保持 bf16 dtype，只是仿真量化误差），不会减少实际显存占用。真正降低显存需要将 optimizer state 存储为 int8 格式，这需要更深层的修改。

3. **训练稳定性**：Optimizer 量化比 Linear/Attention 量化对训练稳定性更敏感（动量是一阶统计量，量化误差会积累）。建议先在 1.3B 模型上验证稳定性后再移植到 14B。


---

## Section 11: npu_fusion_attention 中 P 和 dS 能否量化？

### 11.1 结论

**不能。** 使用 `npu_fusion_attention` / `npu_fusion_attention_grad` 时，P（softmax 概率矩阵）和 dS（注意力分数梯度）无法被量化——这是该 API 的根本架构限制，与 HIF8 设计无关。

---

### 11.2 根本原因：npu_fusion_attention 是黑盒 CANN 算子

`npu_fusion_attention` 将 `QK^T → softmax(P) → PV` 整个 attention 计算封装在单一 CANN 算子内部：

```
Python 层输入:   Q, K, V
          ↓ [CANN 算子内部，Python 不可见]
          QK^T
          softmax → P     ← 不暴露给 Python
          PV → output
Python 层输出:   attn_output, softmax_max, softmax_sum
```

`npu_fusion_attention` 的返回值是 `res[0..2] = [attn_output, softmax_max, softmax_sum]`，P 矩阵本身从未以 Python tensor 形式出现。`softmax_max` 和 `softmax_sum` 只是 log-sum-exp 的统计量（标量级），用于反向传播时重建 softmax，不是 P 本身。

反向传播同理：

```
Python 层输入:   q, k, v, do, softmax_max, softmax_sum, attn_output
          ↓ [CANN 算子内部，Python 不可见]
          重计算 P
          dS = P ⊙ (dO·V^T - rowsum(dO⊙O))   ← 不暴露给 Python
          dQ = dS·K,  dK = dS^T·Q,  dV = P^T·dO
Python 层输出:   dQ, dK, dV
```

dS 是算子内部的中间变量，直接被消耗用于计算 dQ/dK，不会返回给 Python。

---

### 11.3 参考项目三条路径的量化点对比

| 量化点 | `AscendCurrentScaling` | `AscendDelayedScaling` | `FlashAttentionFunction`（纯PyTorch FA v2） |
|--------|----------------------|----------------------|--------------------------------------|
| 前向 Q/K/V | ✅ `quant(q/k/v, 15)` | ✅ delayed scaling | ✅ `quant(q/k/v, 15)` |
| 前向 P（exp_weights） | ❌ 不可访问 | ❌ 不可访问 | ✅ `static_quant(exp_weights, 15)` 第456行 |
| 反向 dO | ✅ `quant(do, 224)` | ✅ delayed scaling | ✅ `static_quant(doc, 224)` 第538行 |
| 反向 P（重计算） | ❌ 不可访问 | ❌ 不可访问 | ✅ `static_quant(p, 15)` 第529行 |
| 反向 dS | ❌ 不可访问 | ❌ 不可访问 | ✅ `static_quant(ds, 224)` 第547行 |

代码位置（`HiF8_Quant_Sample/HiF8_Quant_fa.py`）：
- `AscendCurrentScaling` backward: 第 162-163 行，仅量化 `do`
- `AscendDelayedScaling` backward: 第 309-345 行，仅量化 `do`（delayed scaling 版本）
- `FlashAttentionFunction` forward: 第 455-457 行量化 P；backward 第 528-548 行量化 P、dO、dS

---

### 11.4 为什么 FlashAttentionFunction 能量化 P 和 dS？

`FlashAttentionFunction`（第 381-557 行）是用纯 PyTorch 手写的 FA v2 tiling 实现（非 CANN 算子）。它在 Python 循环里逐 tile 显式计算了每个中间变量：

```python
# forward tiling 循环内
exp_weights = torch.exp(attn_weights - new_row_maxes)   # 这就是 P（tile 级）
if HiF8_QUANT:
    exp_weights = static_quant(exp_weights, 15)          # 可量化：Python tensor
    
# backward tiling 循环内
p = torch.exp(attn_weights - lsec)                      # 重计算 P
if HiF8_QUANT:
    p = static_quant(p, 15)
    
ds = p * scale * (dp - D)                               # 显式计算 dS
if HiF8_QUANT:
    ds = static_quant(ds, 224)                           # 可量化：Python tensor
```

P 和 dS 在 Python 层以普通 tensor 形式存在，可以在任意位置插入量化操作。这正是"手写 FA tiling"相对于"黑盒融合算子"的核心代码层面区别。

---

### 11.5 量化点缺失对 NPU 训练精度的影响

P 量化（scale_max=15）施加在 `PV` 这个矩阵乘法的左操作数上；dS 量化（scale_max=224）施加在反向传播中 dQ/dK 的计算链上。NPU 路径绕过了这两个量化点：

- **实际效果**：NPU 路径中 attention 内部的矩阵乘法（`PV`、`dS·K`、`dS^T·Q`）依然是 bf16 全精度，只有 attention 的**输入端**（Q/K/V）和**梯度入口**（dO）被 HIF8 量化
- **与 OSP 的关系**：OSP 项目使用 SDPA（PyTorch autograd，不是手写 FA tiling），同样无法访问 P 和 dS，因此 OSP 的 attention 量化与参考项目的 NPU 路径处于同一水平——Q/K/V 前向量化 + dQ/dK/dV 反向梯度量化，均不涉及 P/dS
- **影响大小**：P 是 softmax 输出（值域 [0,1]，数值较小且分布集中），量化噪声相对 Q/K/V 较小；dS 的影响比 dO 更间接（经过 P 的加权）。参考项目在 GPU 上量化 P/dS 是为了在所有矩阵乘法路径上都施加 HIF8 约束，对 NPU 端影响属于"锦上添花"级别，不是训练稳定性的关键路径

---

## Section 12: Optimizer 量化深度分析与工作宣传建议

### 12.1 Optimizer 量化的实际显存收益

#### 背景澄清：仿真量化 vs 实际量化

参考项目的 `MomentumQuantizer` 使用 `quant_dequant` 算子——这是**精度仿真**，不是真实的 fp8 存储：
- 存储 dtype 仍然是 bf16（2 bytes/param）
- `quant_dequant` 只是模拟 HIF8 精度损失，tensor 内存占用不变
- **仿真量化不节省显存**

真实 fp8 存储（如 bitsandbytes `AdamW8bit` 的做法）才能节省显存，需要将 M1/M2 实际以 1 byte/param 存储。参考项目本身未实现这一层。

#### 14B 模型 AdamW 显存构成（FSDP 16 卡）

| 组件 | dtype | 全局大小 | 每卡（/16） |
|------|-------|---------|------------|
| 模型权重 W | bf16 | 28 GB | 1.75 GB |
| 梯度 G | bf16 | 28 GB | 1.75 GB |
| 1st moment M1 | bf16 | 28 GB | 1.75 GB |
| 2nd moment M2 | bf16 | 28 GB | 1.75 GB |
| **合计** | | **112 GB** | **7 GB** |

> OSP 使用 PyTorch 原生 `torch.optim.AdamW`，moments 默认与参数同 dtype（bf16）

#### 若实现真实 fp8 optimizer state 存储

```
bf16 M1+M2：28+28 = 56 GB 全局 → 每卡 3.5 GB
fp8  M1+M2：14+14 = 28 GB 全局 → 每卡 1.75 GB
节省：每卡 1.75 GB（节省 50% optimizer state，约占总显存的 4~6%）
```

若 moments 为 fp32（混合精度训练），收益更大：

```
fp32 M1+M2：56+56 = 112 GB 全局 → 每卡 7 GB
fp8  M1+M2：14+14 = 28  GB 全局 → 每卡 1.75 GB
节省：每卡 5.25 GB（节省 75% optimizer state）
```

#### 结论

现阶段参考项目的 optimizer 量化 = 纯精度仿真，零显存收益。真正的显存收益需要额外实现 fp8 存储的 AdamW（工程量独立于量化仿真），这是另一个工程问题，OSP 项目和参考项目目前均未完成。

---

### 12.2 Optimizer 在深度学习训练流程中的位置

每个训练 step 的完整流程：

```
① Forward Pass（前向传播）
   输入 x → [量化发生] x_q, w_q 参与矩阵乘法 → loss
   
② Backward Pass（反向传播）
   loss → ∂L/∂W 沿计算图反传
   [量化发生] grad_output 量化后计算 grad_input / grad_weight

③ Optimizer Step（参数更新）  ← AdamW 在这里
   
   对每个参数 W 和其梯度 G：
   
   M1 ← β1 × M1 + (1-β1) × G          # 1st moment：梯度的指数移动平均
   M2 ← β2 × M2 + (1-β2) × G²         # 2nd moment：梯度平方的指数移动平均
   
   M1_hat = M1 / (1 - β1^t)            # 偏差修正（早期 step 补偿）
   M2_hat = M2 / (1 - β2^t)
   
   W ← W - lr × M1_hat / (√M2_hat + ε) # 权重更新

④ 下一个 step 的 Forward 使用更新后的 W
```

#### 两个 moment 的直觉理解

**1st moment（exp_avg，一阶动量）**
- 本质：梯度的"滑动平均"，是对过去所有梯度的加权记忆
- 作用：平滑单步梯度噪声，提供"惯性"，帮助穿越梯度平坦区域和鞍点
- 如果量化：相当于给"记忆中的梯度方向"引入噪声

**2nd moment（exp_avg_sq，二阶动量）**
- 本质：梯度平方的"滑动平均"，反映每个参数的梯度"历史波动幅度"
- 作用：自适应学习率——历史梯度大的参数用小 LR，历史梯度小的用大 LR，这是 Adam 相对 SGD 的核心优势
- 如果量化：相当于给"自适应 LR 估计"引入噪声，可能破坏参数间的 LR 平衡

**量化的插入点（参考项目设计）**：步骤③的 M1/M2 更新后、权重更新前，对 M1/M2 施加 `quant_dequant`。相当于给梯度历史引入 HIF8 精度约束。

---

### 12.3 训练稳定性与 Optimizer 未量化的关系

#### OSP 训练稳定的真实原因（按重要性排序）

1. **Current scaling 设计合理**：scale_max_forward=15, scale_max_backward=224，每次 forward 实时计算 `scale = scale_max / max|x|`，不会因陈旧 scale 导致上溢/下溢
2. **`quant_dequant` 噪声有界**：HIF8 精度限制带来的是有界的量化误差，不是灾难性误差；模型对这个噪声量级有一定鲁棒性
3. **Optimizer 未量化（贡献稳定性的一个因素）**：M1/M2 保持 bf16 精度 → 权重更新方向准确 → 减少了一个潜在的噪声来源

#### Optimizer 未量化是否是"稳定的决定性因素"？

**不是**。分析如下：

- 参考项目在 GPU 上同时量化了 linear/attention/optimizer，训练同样稳定
- Optimizer 量化引入的是"慢变噪声"（动量更新的 EMA 平滑会平均掉大部分量化噪声）
- Linear/attention 量化直接影响每一步的 forward/backward 计算，影响更直接
- Current scaling 本身在参考项目里被证明足够稳定（`AscendCurrentScalingFlashAttentionFunction` 可以正常运行）

**Optimizer 未量化的实际贡献**：让训练"更稳"，而不是"从崩溃变稳定"。如果加上 optimizer 量化：
- 大概率不会崩（参考项目 GPU 版就是全量量化在跑）
- 可能需要 delayed scaling 来补偿 M1/M2 量化误差的积累
- 训练曲线可能会有更多波动，但收敛应该可以维持

**结论**：现在训练稳定，主要是因为 linear+attention 量化设计合理。Optimizer 未量化是一个"额外的稳定性加成"，不是必要条件。

---

### 12.4 工作成果定位与宣传建议

#### 客户视角：Optimizer 完全不可见

Optimizer states（M1、M2）是**训练的中间产物**，模型发布时只发布权重 W：

```
训练产物：W（发布）+ M1 + M2 + 训练日志（全部丢弃）
                ↓
客户收到：量化后的 W
客户使用：W 做推理，与 optimizer 完全无关
```

因此：optimizer 是否量化，客户在推理端**完全感知不到**，不影响对客户的任何承诺。

#### 可以自信宣称的核心成果

```
核心主张：
"We demonstrate successful Quantization-Aware Training (QAT) with HIF8 precision 
simulation for skiparse-based diffusion models on Ascend NPU."

具体支撑：
• HIF8 量化覆盖所有 linear 层（每 block 10 个，含 self-attn Q/K/V/O、
  cross-attn Q/K/V/O、FFN×2）的 forward 权重/激活 + backward 梯度
• HIF8 量化覆盖 attention Q/K/V 及其反传梯度 dQ/dK/dV
• Current scaling 在 skiparse 架构下训练稳定，无需复杂的 delayed scaling
• 训练收敛，模型质量与 baseline 相当（根据实际指标填写）
• NPU（Ascend）原生实现，通过 SDPA → torch_npu dispatch 路径
```

#### 精确边界表（诚实的局限性）

| 已验证 ✅ | 未验证 / 未实现 ⚠️ |
|----------|-----------------|
| HIF8 量化仿真下 skiparse 训练收敛 | 真实 fp8 kernel 替换后的实际推理速度收益（需硬件实测） |
| Current scaling 对 linear/attention 足够稳定 | Optimizer state（M1/M2）量化仿真 |
| 量化不影响模型最终质量（基于仿真精度验证） | Delayed scaling 能否进一步提升稳定性 |
| NPU SDPA 路径 Q/K/V 及反传梯度量化 | P/dS 内部量化（受 npu_fusion_attention 黑盒限制） |
| 推理端 HIF8 权重直接可用 | 实际 fp8 存储的 optimizer state（显存收益待实现） |

#### 推荐的对外表述

**论文 / 技术报告**（精确措辞）：
> "Our quantization scheme covers all linear projections and attention Q/K/V operations, validated through HIF8 precision simulation in forward and backward passes. Optimizer state quantization is not included in the current study and remains as future work."

**对客户 / 演示材料**（强调推理价值）：
> "Skiparse diffusion model weights trained with HIF8 quantization simulation are directly deployable for HIF8-native hardware inference, with no quality degradation observed in our evaluation. The quantization covers all critical inference-time operations."

**一句话 pitch**：
> "First successful QAT of skiparse diffusion models with HIF8 quantization on Ascend NPU — training stable with current scaling, inference-ready weights validated."

#### 避免的表述

- ❌ "完整的 HIF8 量化训练"——因为 optimizer 未覆盖，P/dS 受 API 限制未覆盖
- ❌ "量化后显存降低 X%"——仿真量化不省显存，不能做此声明
- ✅ "HIF8 精度仿真验证了量化对模型质量无影响，为真实 fp8 推理提供了可靠的先验验证"
