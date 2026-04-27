# OSP-Next 1.3B NPU 训练架构总结

## 目录
1. [项目整体架构](#1-项目整体架构)
2. [调用链：train_osp.py → 模型前向](#2-调用链-train_osppy--模型前向)
3. [Self/Cross Attention 与 attention_with_mask / SDPA 的关系](#3-selfcross-attention-与-attention_with_mask--sdpa-的关系)
4. [三种 Block 类型：哪种是标准 Attention，哪种是 Skiparse](#4-三种-block-类型哪种是标准-attention哪种是-skiparse)
5. [Skiparse Attention 的本质：为什么重排就能减少计算量](#5-skiparse-attention-的本质为什么重排就能减少计算量)
6. [NPU mask：作用、来源与特殊处理](#6-npu-mask作用来源与特殊处理)
7. [rearrange 在 NPU 上对应什么底层算子](#7-rearrange-在-npu-上对应什么底层算子)
8. [HIF8 量化改造：Linear层（x+W）与 Attention层（Q/K/V）](#8-hif8-量化改造linear层xw与-attention层qkv)
9. [pre/post_self_attn_all_to_all：Ulysses CP 通信机制](#9-prepost_self_attn_all_to_all-ulysses-cp-通信机制)
10. [maybe_compile(disable=True) 的作用](#10-maybe_compiledisabletrue-的作用)
11. [@staticmethod 与 ctx：autograd.Function 的 Python 基础](#11-staticmethod-与-ctx-autogradfunction-的-python-基础)
12. [多卡 seq 切分：为什么 attention 输入是 16380 而不是 4095](#12-多卡-seq-切分为什么-attention-输入是-16380-而不是-4095)
13. [distributed/ 各文件的作用与协作关系](#13-distributed-各文件的作用与协作关系)
14. [文件索引](#14-文件索引)

---

## 1. 项目整体架构

```
TorchDiff/
├── train/train_osp.py              # 训练入口（OSP/NPU路径）
├── configs/train/npu/
│   ├── osp_1_3b.yaml                      # 原始NPU配置
│   ├── osp_1_3b_hif8.yaml                 # HIF8量化配置（仅Linear）
│   ├── osp_1_3b_hif8_linear.yaml          # 同上，Linear-only快照备份
│   └── osp_1_3b_hif8_linear_atten.yaml    # Linear+Attention双重HIF8（新增）
├── scripts/train/npu/
│   └── train_osp_1_3b.sh           # 启动脚本（8卡单节点）
└── torchdiff/
    ├── modules/
    │   ├── osp_next.py             # OSPNextModel（DiT主体，支持quant+quant_attn）★
    │   ├── osp_next_bak.py         # 原始改动前备份
    │   ├── osp_next_hif8_linear.py # 仅Linear HIF8版备份（加入attention量化前的快照）
    │   ├── attention.py            # Attention后端（FA2/FA3/SDPA）★
    │   ├── skiparse_func.py        # Skiparse序列重排工具函数
    │   ├── hif8_linear.py          # HIF8Linear：x+W双量化；_quant被attention共用★
    │   ├── hif8_attention.py       # HIF8 attention：Q/K/V量化（新增）★
    │   ├── want2v.py               # WanModel（基础DiT，OSPNext继承）
    │   ├── vae.py / t5.py          # VAE & T5 文本编码器
    ├── distributed/
    │   ├── cp_state.py             # Context Parallel状态管理
    │   ├── communication.py        # all_to_all / all_gather
    │   └── fsdp2_wrapper.py        # FSDP2混合精度
    ├── data/                       # 数据集、采样器、collator
    ├── schedulers/flow_matching.py # Flow Matching调度器
    └── utils/                      # 工具函数、梯度裁剪、编码器缓存
```

---

## 2. 调用链：train_osp.py → 模型前向

```
train_osp.py  main()
│
├─ [初始化] WanVAE, T5EncoderModel, OSPNextModel, AdamW
├─ [分布式] FSDP2 mesh(8卡) + SkiparseCP(2) + FullBlocksCP
│
└─ [训练循环]
     ├─ VAE.encode(video)          → latents [B,16,T/4,H/8,W/8]
     ├─ T5Encoder(prompt_ids)      → text_embeddings [B,512,4096]
     ├─ scheduler.q_sample()       → 加噪 interpolated_latents
     │
     └─ model(interpolated_latents, timesteps, text_embeddings)
          └─ OSPNextModel.forward()
               ├─ patch_embedding (Conv3d)
               ├─ time_embedding + text_embedding (nn.Linear)
               └─ blocks[0..29] (OSPNextAttentionBlock × 30)
                    ├─ [Self Attn]  OSPNextSelfAttention
                    ├─ [Cross Attn] OSPNextCrossAttention
                    └─ [FFN]        Linear(dim→ffn_dim) + GELU + Linear(ffn_dim→dim)
```

---

## 3. Self/Cross Attention 与 attention_with_mask / SDPA 的关系

> **结论：它们不是不同层，而是同一条调用链上的不同抽象层次。**

### 完整调用链（每个 block 都走这条链）

```
OSPNextSelfAttention.forward()
    │
    ├─ q/k/v = self.q/k/v(x)          ← Q/K/V Linear投影
    │
    ├─ pre_self_attn_all_to_all()      ← CP: scatter seq to ranks
    │
    ├─ rope_wrapper.apply_rope()       ← RoPE（内含skiparse重排，见第5节）
    │
    ├─ attention_with_mask(q, k, v)    ← 选择后端（设备决定，不是block类型决定）
    │       ├─ [NPU]      scaled_dot_product_attention_with_mask()
    │       │                 └─ torch.nn.functional.scaled_dot_product_attention
    │       ├─ [GPU FA3]  flash_attn_no_pad_v3()
    │       └─ [GPU FA2]  flash_attn_no_pad()
    │
    ├─ post_self_attn_all_to_all()     ← CP: gather seq back
    │
    └─ x = self.o(x)                   ← O Linear投影
```

**FA2/FA3 vs SDPA 的分叉** 在 `attention_with_mask()` 内部，判断条件是 `is_npu_available()`，与 block 类型（Full/Single/Group）无关。

---

## 4. 三种 Block 类型：哪种是标准 Attention，哪种是 Skiparse  【重点1】

| Block 类型 | 数量（1.3B，30层）| Skiparse 重排 | CP Group | Attention 类型 |
|------------|-------------------|---------------|----------|---------------|
| **Full**   | 20层（均匀分布）   | 无（Identity）| FullBlocksCP | 对**完整**序列做 SDPA/FA |
| **Single** | 奇数 skiparse 层  | skiparse_2d_single | SkisparseCP | 对**重排后稀疏**序列做 SDPA/FA |
| **Group**  | 偶数 skiparse 层  | skiparse_2d_group  | SkisparseCP | 对**重排后稀疏**序列做 SDPA/FA |

- Full block = **标准 attention**（等价于原始 WanModel 的 attention）
- Single + Group block = **skiparse attention**（稀疏，通过重排实现）
- 三种 block 都调用同一个 `attention_with_mask()`，底层 attention 算子本身没有任何区别

**1.3B 模型的 block 分布（uniform，num_full_blocks=10）：**
```
层索引:  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
类型:    F  F  S  G  S  G  F  F  S  G  S  G  F  F  S  G  S  G  F  F  S  G  S  G  F  F  S  G  S  G
F=Full, S=Single, G=Group
```

---

## 5. Skiparse Attention 的本质：为什么重排就能减少计算量 【重点2】

### 核心：Attention 的复杂度是 O(N²)，序列长度减半 → 计算量减少 4 倍

**数学推导：**

设原始序列长度 N，batch size B，sparse_ratio=2（2D模式下 p = 2² = 4）。

| | Full Attention | Skiparse（重排后） |
|--|--|--|
| batch size | B | p×B = 4B |
| 序列长度 | N | N/p = N/4 |
| attention 计算量 | ∝ B × N² | ∝ 4B × (N/4)² = B × N²/4 |
| **相对代价** | **1×** | **0.25×** |

**attention 计算量减少 4 倍**，但 batch 维增大 4 倍，两者相比 attention 平方项占主导，整体节省显著。

### Attention 算子"感知不到"稀疏性的原因

```
重排前:  [B, N, C]          → 每个 sample 有 N 个 token，完全稠密

skiparse_2d_single:
  原理: 把空间上每隔 2 个位置的 token 打包成新 batch
  结果: [4B, N/4, C]        → 4 倍更多的 sample，每个只有 N/4 个 token

attention kernel 看到的:
  输入: [4B, N/4, H, D]     ← 它只认形状，不知道这些 token 原来隔着 2 个位置
  计算: 每个 "新 sample" 内的 token 两两做 attention
  效果: 原始序列中每隔 2 个的 token 相互 attend（稀疏！）
```

**Single 和 Group 交替的几何意义（2D空间）：**
```
原始 4×4 patch 网格，sparse_ratio=2：

Single block: 每个 token 只 attend "棋盘格A"（偶数行偶数列）
  ■ □ ■ □
  □ □ □ □
  ■ □ ■ □
  □ □ □ □

Group block:  每个 token 只 attend "棋盘格B"（奇数行奇数列）
  □ □ □ □
  □ ■ □ ■
  □ □ □ □
  □ ■ □ ■

Single + Group 交替 → 两套棋盘格合起来覆盖全部位置，逼近 Full Attention
```

---

## 6. NPU mask：作用、来源与特殊处理 【重点3】

### 6.1 mask 的作用

OSP-Next 的 `attn_mask` 是一个 **布尔型 padding mask**，形状为 `[B, N]`，含义是：
- `True`  → 该位置是有效 token，参与 attention
- `False` → 该位置是填充 token，被 attention 忽略（attention weight → 0）

**为什么会有 padding？**
- Skiparse 重排时，如果 T×H×W 不能被 `sparse_ratio²` 整除，就需要在序列末尾填充哑 token（pad）
- 这些 pad 位置不应该参与 attention，所以要用 mask 屏蔽

### 6.2 WanModel（原始 wan2.1）没有这个 mask 吗？

```python
# want2v.py — WanModel.forward()
seq_lens = torch.tensor(math.prod(grid_sizes), ...).repeat(x.size(0))
context_lens = None  # ← 文本侧也不用 mask
args = [x, e0, seq_lens, grid_size_for_rope, self.freqs, context, context_lens]
```

WanModel 把序列长度通过 `q_lens`/`k_lens` 参数传给 `flash_attention()`，FA2/FA3 内部用 `cu_seqlens` 处理变长序列，**不使用显式 mask**。

OSP-Next 改用 `attention_with_mask()` 并引入显式 mask，是因为：
1. Skiparse 重排后需要屏蔽 padding token
2. `attention_with_mask()` 要同时兼容 NPU（SDPA）和 GPU（FA），而 SDPA 接口只接受 mask，不接受 `cu_seqlens`

### 6.3 mask 的生成（SkiparseMaskPreprocessor）

```python
# osp_next.py — OSPNextModel.forward()
local_single_mask, local_group_mask, global_single_mask, global_group_mask = \
    self.mask_preprocessor.preprocess(patchify_x_shape, grid_sizes, ...)

# Single block 使用:
attn_mask     = global_single_mask   # self-attn Q/KV 共用
cross_attn_mask = local_single_mask  # cross-attn Q 侧

# Full block 使用:
attn_mask = None  # Full block 序列长度整齐，不需要 mask
```

生成逻辑：创建全1 dummy tensor → 经过 skiparse 重排 → all_gather 收集全局视图 → squeeze 成 bool mask。

### 6.4 NPU mask 的特殊处理

```python
# attention.py — scaled_dot_product_attention_with_mask()

# 标准做法（GPU也可用）：
attn_mask = attn_mask[:, None, None, :]   # [B, N] → [B, 1, 1, N]
#   GPU SDPA: 可以从 [B, 1, 1, N] broadcast 到 [B, H, Nq, N]

# NPU 额外展开 Q 维度：
if is_npu_available():
    q_len = q.shape[2]
    attn_mask = attn_mask.expand(-1, -1, q_len, -1)  # [B, 1, 1, N] → [B, 1, Nq, N]
#   NPU SDPA: 不支持 Q 维度广播，必须显式展开
```

**原因**：torch_npu 的 SDPA 算子对 mask 形状要求更严格，要求 Q 维度必须显式展开，而不能依赖 broadcast。

---

## 7. rearrange 在 NPU 上对应什么底层算子

`einops.rearrange` 是纯 Python 库，最终展开为 PyTorch 的 `view` + `permute` 组合。

### Skiparse 重排的分解

以 `skiparse_1d_single: (b, n*p, c) -> (p*b, n, c)` 为例：

```python
rearrange(x, "b (n p) c -> (p b) n c", p=p)
# 等价于：
x = x.view(b, n, p, c)      # view: 无数据移动，只改形状元数据
x = x.permute(2, 0, 1, 3)   # permute: 标记非连续，也无数据移动
x = x.contiguous()           # contiguous: ← 触发实际内存拷贝
x = x.view(p*b, n, c)
```

### NPU 底层对应关系

| PyTorch 操作 | NPU 底层算子 | 是否搬运数据 |
|-------------|-------------|------------|
| `.view()` / `.reshape()`（连续tensor）| 无操作（修改 shape 元数据）| 否 |
| `.permute()` / `.transpose()` | 标记非连续（lazy）| 否（直到 `.contiguous()`）|
| `.contiguous()` | **Transdata**（华为 CANN 算子）| **是** ← 真正的数据搬运 |
| `einops.rearrange`（含维度交换）| view + permute + **Transdata** | 含 Transdata 时是 |

**`skiparse_func.py` 中都调用了 `contiguous()`**，因此每次 skiparse 重排在 NPU 上都会触发一次 **Transdata** 数据搬运。这是 skiparse 在 NPU 上的主要额外开销，通常比 attention 计算的节省小得多，整体仍然是合算的。

---

## 8. HIF8 量化改造：Linear层（x+W）与 Attention层（Q/K/V）

### 8.1 改造范围

每个 `OSPNextAttentionBlock`（共30个）内，以下 10 个 `nn.Linear` 被替换为 `HIF8Linear`：

| 层 | 维度 |
|----|------|
| self_attn.q / .k / .v / .o | dim(1536) ↔ dim(1536) |
| cross_attn.q / .k / .v / .o | dim(1536) ↔ dim(1536) |
| ffn[0] | dim(1536) → ffn_dim(8960) |
| ffn[2] | ffn_dim(8960) → dim(1536) |

共 **300 个 HIF8Linear**（30层 × 10个/层）。

以下 Linear 保持 `nn.Linear` 不变（非 DiT attention/FFN 核心路径）：
- `text_embedding`、`time_embedding`、`time_projection`、`head`

### 8.2 量化流程（前向 + 反向）

```
============================================================
前向（scale_max_forward = 15）：
============================================================

  输入 x  ─────────────────────────────────────────────────
    scale_x  = 15 / max(|x|)           per-tensor scale
    x_q      = quant_dequant_hif8(x * scale_x) / scale_x

  权重 W  ─────────────────────────────────────────────────
    scale_w  = 15 / max(|W|)           per-tensor scale
    w_q      = quant_dequant_hif8(W * scale_w) / scale_w

  输出:   out = x_q @ w_q.T + bias    ← 两个反量化结果相乘

============================================================
反向（scale_max_backward = 224）：
============================================================

  梯度量化:
    scale_g  = 224 / max(|grad_out|)
    grad_q   = quant_dequant_hif8(grad_out * scale_g) / scale_g

  各参数梯度:
    ∂x = grad_q @ w_q                 ← 使用前向保存的 dequant(W)
    ∂W = grad_q.T @ x_q              ← 使用前向保存的 dequant(x)
    ∂b = grad_q.sum(over batch dims)
```

### 8.3 scale_max 选值依据

| 方向 | scale_max | 说明 |
|------|-----------|------|
| 前向（x 和 W）| 15 | HIF8 格式正常范围 ≈ [-15, 15]，接近 fp8 e4m3 |
| 反向（梯度）  | 224 | 梯度分布较宽，HIF8 扩展范围 ≈ 224，类似 fp8 e5m2 |

### 8.4 代码实现（hif8_linear.py 核心逻辑）

```python
class _HIF8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, scale_max_forward, scale_max_backward):
        x_q, _ = _quant(x, scale_max_forward)       # 量化 x
        w_q, _ = _quant(weight, scale_max_forward)   # 量化 W
        ctx.save_for_backward(x_q, w_q, tensor(scale_max_backward))
        return F.linear(x_q, w_q, bias)             # dequant(x) @ dequant(W).T + b

    @staticmethod
    def backward(ctx, grad_output):
        x_q, w_q, smb = ctx.saved_tensors
        grad_q, _ = _quant(grad_output, smb.item()) # 量化梯度
        grad_input  = grad_q @ w_q                  # ∂x
        grad_weight = grad_q.reshape(-1, out).t() @ x_q.reshape(-1, in)  # ∂W
        grad_bias   = grad_q.reshape(-1, out).sum(0)
        return grad_input, grad_weight, grad_bias, None, None
```

### 8.5 启用方式（Linear 量化）

```yaml
# configs/train/npu/osp_1_3b_hif8.yaml  /  osp_1_3b_hif8_linear.yaml
model_config:
  quant: "hif8"              # "hif8" 或 null（null = 原始 nn.Linear）
  scale_max_forward: 15.0
  scale_max_backward: 224.0
```

---

### 8.6 Attention 层 HIF8：Q/K/V 量化（hif8_attention.py）

在 Linear 量化之上，还可以对进入注意力核（SDPA）之前的 Q、K、V 张量再施加一次 HIF8 量化误差，模拟低精度硬件对注意力分数计算的影响。

#### 8.6.1 量化位置

```
OSPNextSelfAttention.forward()
    │
    ├─ Q/K/V = self.q/k/v(x)       ← HIF8Linear（第8.1节，投影层量化）
    │
    ├─ rope_wrapper.apply_rope()
    │
    ├─ hif8_attention_with_mask()   ← ★ 新增：对Q/K/V再次量化
    │       ├─ _HIF8QKVQuantFunction.apply(Q, K, V)
    │       │       前向: Q_q = quant_dequant_hif8(Q * s_q) / s_q
    │       │             K_q, V_q — 同上
    │       │       反向: dQ_q = quant_dequant_hif8(dQ * sg) / sg
    │       │             dK_q, dV_q — 同上
    │       └─ attention_with_mask(Q_q, K_q, V_q)  ← 原attention后端不变
    │
    └─ x = self.o(x)
```

#### 8.6.2 与 Linear 量化的对比

| 维度 | Linear HIF8（8.1节）| Attention HIF8（本节）|
|------|-------------------|---------------------|
| 量化对象 | 激活 x 和权重 W | Q、K、V 张量（在RoPE之后）|
| 模拟的误差 | Linear 投影的量化噪声 | 注意力分数 QKᵀ/√d 的量化噪声 |
| 实现文件 | `hif8_linear.py` | `hif8_attention.py` |
| 配置参数 | `quant: "hif8"` | `quant_attn: "hif8"` |
| 共享的 helper | `_quant()` | 从 `hif8_linear` import `_quant` |

#### 8.6.3 代码实现（hif8_attention.py 核心逻辑）

```python
class _HIF8QKVQuantFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, scale_max_forward, scale_max_backward):
        q_q, _ = _quant(q, scale_max_forward)
        k_q, _ = _quant(k, scale_max_forward)
        v_q, _ = _quant(v, scale_max_forward)
        ctx.save_for_backward(tensor(scale_max_backward))
        return q_q, k_q, v_q        # 返回3个量化后张量

    @staticmethod
    def backward(ctx, grad_q, grad_k, grad_v):
        (smb,) = ctx.saved_tensors
        return _quant(grad_q, smb.item())[0],   # dQ 量化
               _quant(grad_k, smb.item())[0],   # dK 量化
               _quant(grad_v, smb.item())[0],   # dV 量化
               None, None

def hif8_attention_with_mask(q, k, v, ..., scale_max_forward=15.0, scale_max_backward=224.0):
    q_q, k_q, v_q = _HIF8QKVQuantFunction.apply(q, k, v, scale_max_forward, scale_max_backward)
    return attention_with_mask(q_q, k_q, v_q, ...)
```

#### 8.6.4 osp_next.py 改动（quant_attn 参数链路）

```
OSPNextModel.__init__(quant_attn=None)
    └─ OSPNextAttentionBlock.__init__(quant_attn=None)
           └─ OSPNextSelfAttention.__init__(quant_attn=None)
                  self.quant_attn = quant_attn
                  self.scale_max_forward  = scale_max_forward
                  self.scale_max_backward = scale_max_backward

OSPNextSelfAttention.forward():
    if self.quant_attn == "hif8":
        x = hif8_attention_with_mask(q, k, v, ...,
                scale_max_forward=self.scale_max_forward,
                scale_max_backward=self.scale_max_backward)
    else:
        x = attention_with_mask(q, k, v, ...)
```

同样的条件分支也应用于 `OSPNextCrossAttention.forward()`。

#### 8.6.5 启用方式（Linear + Attention 双重量化）

```yaml
# configs/train/npu/osp_1_3b_hif8_linear_atten.yaml
model_config:
  quant: "hif8"          # Linear投影层：x 和 W 量化
  quant_attn: "hif8"     # Attention核：Q/K/V量化（可独立开关）
  scale_max_forward: 15.0
  scale_max_backward: 224.0
```

- `quant` 和 `quant_attn` 可独立设为 `"hif8"` 或 `null`
- 仅 Linear：`quant: "hif8"`, `quant_attn: null`
- 仅 Attention：`quant: null`, `quant_attn: "hif8"`
- 双重量化：两者均设 `"hif8"`（当前 `osp_1_3b_hif8_linear_atten.yaml`）

---

---

## 9. pre/post_self_attn_all_to_all：Ulysses CP 通信机制【重要4】

### 9.1 这两个函数解决什么问题

标准 Attention 要求每个 token 能"看到"全部其他 token（全局注意力）。在 Context Parallel 中，序列被切分到多个 rank 上，单个 rank 只持有序列的一部分。Ulysses CP 的解法是：**用 all_to_all 把"序列切分"暂时转化成"head 切分"**，让每个 rank 在计算 attention 时持有完整序列，但只计算 H/cp_size 个 head。

### 9.2 pre_self_attn_all_to_all（attention 前）

```python
# 输入 Q/K/V: [B, N_local, H, D]    H=12, N_local=16380, D=128
q = all_to_all_4D(q, group=cp_group, scatter_dim=2, gather_dim=1)
```

`all_to_all_4D(scatter_dim=2, gather_dim=1)` 做的事：

```
每个 rank 持有:          [B=1, N_local=16380, H=12, D=128]
                               ↓ all_to_all（cp_size=2 时）
每个 rank 持有:          [B=1, N_full=32760,  H=6,  D=128]
              ↑ 序列维度 gather ×2      ↑ head 维度 scatter ÷2
```

**底层实现（communication.py: _all_to_all_4D）：**
```
[B, N/P, H, D]
  → reshape: [B, N/P, P, H/P, D]
  → transpose(0,2): [P, N/P, B, H/P, D]   ← 变成 P 块，第i块发给rank i
  → dist.all_to_all_single                 ← 实际通信：每块发给对应 rank
  → transpose back → reshape: [B, N, H/P, D]
```

涉及的算子：`view/reshape`（NPU: 零拷贝）+ `transpose`（NPU: 标记非连续）+ **`dist.all_to_all_single`**（NPU: HCCL 集合通信，真正的跨卡数据传输）+ `contiguous()`（NPU: Transdata）

### 9.3 post_self_attn_all_to_all（attention 后）

```python
# 输入 attention output: [B, N_full, H/P, D]
x = all_to_all_4D(x, group=cp_group, scatter_dim=1, gather_dim=2)
```

**逆操作**：scatter 序列维度（÷P），gather head 维度（×P），恢复原始切分：

```
[B, N_full=32760, H/P=6, D=128]
       ↓ all_to_all（scatter seq, gather head）
[B, N_local=16380, H=12, D=128]
```

### 9.4 前向和反向都有 all_to_all

`SeqAllToAll4D`（communication.py:132）是一个 `torch.autograd.Function`：
```python
class SeqAllToAll4D(torch.autograd.Function):
    def forward(ctx, ...):  scatter_dim=2, gather_dim=1 → 前向 all_to_all
    def backward(ctx, ...): scatter_dim=1, gather_dim=2 → 反向 all_to_all（自动互换维度）
```

因此反向传播时，梯度会经过对称的 all_to_all 恢复原始切分，无需手动处理。

### 9.5 Skiparse CP vs Ulysses CP 的 all_to_all 在哪里

| CP 类型 | all_to_all 位置 | 操作维度 |
|---------|----------------|---------|
| **Ulysses CP（Full blocks）** | `pre/post_self_attn_all_to_all` | scatter heads ↔ gather seq |
| **Skiparse CP（Skiparse blocks）** | `SkiparseRearrange._skiparse_cp_scatter/gather` | scatter batch（super-batch）维度 |
| Skiparse Single2Group | `_parallel_skiparse_2d_single_to_group` | 分布式 batch 转置 + all_to_all |

---

## 10. maybe_compile(disable=True) 的作用

### 代码实现

```python
# utils/compile.py
def maybe_compile(disable=False):
    def decorator(func):
        if is_npu_available():
            return func          # NPU 上永远不编译
        if disable:
            return torch.compiler.disable(func)   # 显式禁止被外层 compile 影响
        return torch.compile(func)                # GPU 上正常编译
    return decorator
```

### 三种用法的含义

| 装饰器 | GPU 行为 | NPU 行为 | 用途 |
|--------|---------|---------|------|
| `@maybe_compile()` | `torch.compile` 编译 | 不编译，原函数 | 适合纯数学计算（FFN、投影层） |
| `@maybe_compile(disable=True)` | `torch.compiler.disable` 禁止编译 | 不编译，原函数 | 含分布式通信（all_to_all）、动态 shape 的函数 |
| 无装饰器 | 不编译 | 不编译 | — |

### 为什么 all_to_all 相关函数要 disable=True

`torch.compile` 会把函数内的操作融合成一个 kernel（通过 Triton/inductor）。`dist.all_to_all_single` 是集合通信操作，**不能被 kernel fusion**——编译器无法把它与前后的矩阵运算合并。强制编译会导致：
- 编译失败（graph break）
- 或者动态 shape 触发大量重编译
- NPU 上 `torch_npu` 的 dynamo 支持不完整，一律禁止编译

---

## 11. @staticmethod 与 ctx：autograd.Function 的 Python 基础

### @staticmethod 是什么

`@staticmethod` 是 Python 的内置装饰器，把方法变成**静态方法**：
- 普通方法：第一个参数是 `self`（实例），绑定到某个对象
- 类方法（`@classmethod`）：第一个参数是 `cls`，绑定到类
- **静态方法**：无隐式第一参数，完全独立，只是挂在类命名空间下的函数

```python
class _HIF8LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, ...):   # 不是 self.forward，ctx 由 PyTorch 引擎注入
        ...
    @staticmethod
    def backward(ctx, grad_output):    # 同样，ctx 由引擎注入
        ...
```

`torch.autograd.Function` **要求** `forward` 和 `backward` 都是静态方法，因为 PyTorch 的 autograd 引擎直接调用类的 `forward`/`backward`，不实例化类，所以不能有 `self`。

### ctx 是什么

`ctx`（context）是 PyTorch autograd 引擎在调用 `forward` 时创建并注入的**上下文对象**，本质是一个轻量级字典/对象，用于在前向和反向之间传递状态：

```python
# 前向中：保存信息
ctx.save_for_backward(x_q, w_q, smb_tensor)   # 保存 tensor（支持 grad checkpoint）
ctx.has_bias = True                             # 保存非 tensor 标量/布尔值

# 反向中：取出信息
x_q, w_q, smb = ctx.saved_tensors              # 取出之前保存的 tensor
has_bias = ctx.has_bias                         # 取出标量
```

**为什么用 `save_for_backward` 而不直接 `ctx.x_q = x_q`？**

| 方式 | 内存管理 | Gradient Checkpoint 兼容 | 推荐 |
|------|---------|------------------------|------|
| `ctx.save_for_backward(tensor)` | backward 后自动释放 | 兼容（可重算）| ✅ |
| `ctx.foo = tensor` | 不自动释放，可能泄漏 | 不兼容 | ❌ 只用于标量 |

---

## 12. 多卡 seq 切分：为什么 attention 输入是 16380 而不是 4095

### 配置回顾（8卡单节点，480P，81帧）

```yaml
fsdp_size: 8
cp_size: 4                          # 但 use_context_parallel: False → 实际不启用
skiparse_cp_size: 2                 # use_skiparse_context_parallel: True → 启用
use_context_parallel: False
use_skiparse_context_parallel: True
```

从 train_osp.py 的逻辑推导：
```python
cp_size = 1           # use_context_parallel=False 时不赋值，维持1
skiparse_cp_size = 2  # 来自 yaml
global_cp_size = skiparse_cp_size × cp_size = 2 × 1 = 2

# Mesh:
init_device_mesh("cuda", (8//2, 2, 1), mesh_dim_names=("dp","skiparse_cp","cp"))
#                         ↑              ↑           ↑
#                     dp=4卡          skiparse=2卡  ulysses=1卡(关闭)
```

### FSDP vs CP：操作对象完全不同

```
8张 NPU 卡的分工：
┌──────────────────────────────────────────────────────────┐
│  FSDP (size=8)  →  切分 模型权重 W                        │
│   每张卡存 1/8 的权重，前向时通过 all_gather 拼回完整权重   │
│   与序列长度无关！                                         │
├──────────────────────────────────────────────────────────┤
│  CP (global_cp_size=2)  →  切分 序列 N (激活值)           │
│   每2张卡组成一个 CP group，共同处理同一条序列              │
│   每张卡持有 32760 / 2 = 16380 个 token                   │
└──────────────────────────────────────────────────────────┘
```

### FSDP + CP 联合下的单卡计算：Linear 和 Attention 各是什么

#### Linear 层（以 Q 投影为例）

FSDP 和 CP 是**正交关系**——FSDP 管权重，CP 管序列，两者串行叠加：

```
每卡持有（平时）:  W_q_shard   [1536/8 × 1536]   ← FSDP 分片
                  X_local     [1, 16380, 1536]   ← CP 切了一半序列

前向时：
  step1  FSDP all_gather W_q_shard (8卡)
         → W_q_full  [1536, 1536]              ← 完整权重临时汇聚到每卡
  step2  本地矩阵乘：Y = X_local @ W_q_full.T
         → Y         [1, 16380, 1536]           ← 半序列 × 完整权重
  step3  FSDP 丢弃 W_q_full（释放显存）

结果：Y [1, 16380, 1536] 已经是正确的 Q 投影结果，无需再通信
      ↑ Linear 是 token-independent 的，每 token 的输出只依赖自己的输入，
        CP 切掉的另一半 token 不影响本卡的计算
```

**结论：单卡算 Linear 时是「半序列 × 完整权重」，FSDP all_gather 只在每层前向开始时触发一次，之后立刻释放。（找到了）**

#### Attention 层（Full block，Ulysses CP，full_cp_size=2）

Attention 不是 token-independent 的——每个 token 要 attend 全部其他 token，所以**需要用 all_to_all 临时把序列切分换成 head 切分**，让每卡在 SDPA 时看到完整序列。

完整的单卡前向步骤（含 reshape/transpose/Transdata）：

```
① Q/K/V Linear（FSDP，见上方 Linear 章节）
   FSDP all_gather W_q/k/v → 完整权重
   Y = X_local @ W.T  →  [1, 16380, 1536]
   reshape  →  [1, 16380, H=12, D=128]
   丢弃 W

② RoPE（本地，无通信）

③ pre_self_attn_all_to_all（CP，scatter heads / gather seq）
   输入:  [1, 16380, 12, 128]
   │
   ├─ view:      [1, 16380,  2,  6, 128]   # H→(cp=2, H/cp=6)，零拷贝
   ├─ transpose: [2, 16380,  1,  6, 128]   # 第 i 块将发给 rank i，标记非连续
   ├─ dist.all_to_all_single (HCCL)        # ★ 真正的跨卡数据传输
   │     rank0 把自己的块 0 留下、把块 1 发给 rank1
   │     rank1 把自己的块 0 发给 rank0、把块 1 留下
   ├─ transpose back                        # 标记非连续
   ├─ contiguous()  → NPU Transdata （真正算子不一是这个）  # ★ 真正的本地内存整理
   └─ view:      [1, 32760,  6, 128]       # 序列拼回全长，head 减半
   输出:  [1, 32760, 6, 128]

④ SDPA（本地，无通信）
   输入/输出: [1, 32760, 6, 128]           # 完整 seq，一半 head

⑤ post_self_attn_all_to_all（CP，③的逆操作，scatter seq / gather heads）
   输入:  [1, 32760, 6, 128]
   │
   ├─ view:      [1,  2, 16380, 6, 128]   # N→(cp=2, N/cp)，零拷贝
   ├─ transpose: [2,  1, 16380, 6, 128]
   ├─ dist.all_to_all_single (HCCL)        # ★ 跨卡
   ├─ transpose back
   ├─ contiguous()  → NPU Transdata        # ★ 本地整理
   └─ view:      [1, 16380, 12, 128]      # 恢复半序列，全部 head
   输出:  [1, 16380, 12, 128]

⑥ reshape → [1, 16380, 1536]

⑦ O Linear（FSDP，同①）
   FSDP all_gather W_o → 完整 W_o
   out = x @ W_o.T  →  [1, 16380, 1536]
   丢弃 W_o
```

#### 通信汇总（一次 self-attn block）

| 步骤 | 通信类型 | 参与卡 | 数据量（单侧）| 触发 NPU Transdata？|
|------|---------|-------|-------------|-------------------|
| Q/K/V Linear 前 | FSDP all_gather×3 | 8卡 FSDP group | 1/8 W × 3 | 否（权重连续）|
| pre_all_to_all | CP all_to_all_single | 2卡 CP group | [1,16380,6,128] | 是 |
| post_all_to_all | CP all_to_all_single | 2卡 CP group | [1,16380,6,128] | 是 |
| O Linear 前 | FSDP all_gather×1 | 8卡 FSDP group | 1/8 W | 否 |

反向传播时，FSDP all_gather 再次发生（重建权重算梯度），然后 FSDP reduce_scatter 把梯度碎片分发回各卡；CP all_to_all 通过 SeqAllToAll4D 的 backward 自动执行逆向通信。

### dual_end 模式的 block 分布

`dual_end` 把所有 Full block 集中在模型的**两端**（首部 + 尾部），中间全是 Skiparse block：

```python
# osp_next.py — OSPNextModel.__init__()
if self.skiparse_model_type == SkiparseModelType.DualEnd:
    assert self.num_full_blocks % 4 == 0
    skiparse_start_index = self.num_full_blocks // 2
    skiparse_end_index   = self.num_layers - self.num_full_blocks // 2 - 1
    full_block_indices   = list(range(0, skiparse_start_index)) \
                         + list(range(skiparse_end_index + 1, self.num_layers))
```

**1.3B（30层）+ num_full_blocks=8 + dual_end：**

```
skiparse_start_index = 8 // 2 = 4
skiparse_end_index   = 30 - 4 - 1 = 25
full_block_indices   = [0,1,2,3] + [26,27,28,29]

层索引:  0  1  2  3  4  5  6  7  8  ...  24  25  26  27  28  29
类型:    F  F  F  F  S  G  S  G  S  ...   S   G   F   F   F   F
                     ←──── 22 个 Skiparse block ────→
```

F=Full（完整 attention），S=Single skiparse，G=Group skiparse

对比 uniform 模式（num_full_blocks=10，见第4节）：Full block 均匀散布全模型；dual_end 则把它们全部集中在两端，中间 22 层都是稀疏 attention。

**configs 中的使用情况：**

| 配置文件 | skiparse_model_type | num_full_blocks |
|---------|---------------------|----------------|
| `configs/train/npu/osp_1_3b.yaml` | uniform | 10 |
| `configs/train/gpu/osp_1_3b.yaml` | **dual_end** | **8** |
| `configs/eval/npu/osp_1_3b.yaml` | **dual_end** | **8** |
| `configs/infer/npu/osp_1_3b.yaml` | **dual_end** | **8** |

---

### Profiling：为什么前4个和后4个 attention 耗时显著高于中间层

这是完全正常的现象，根本原因是 **Full block 和 Skiparse block 的 SDPA 计算量相差约 16 倍**（skiparse_cp=2，sparse_ratio=2，2D模式）。

#### 计算量推导

**Full block（Ulysses CP，full_cp_size=2）：**
```
每卡 SDPA 输入: [B=1, N_full=32760, H/2=6, D=128]
计算量 ∝ B × (H/2) × N_full² = 1 × 6 × 32760² ≈ 6.43 × 10⁹
```

**Skiparse block（skiparse_cp=2，sparse_ratio=2，2D → p=sparse_ratio²=4）：**
```
N_local = 16380（CP切分）
skiparse 重排(p=4): [1, 16380, 12, 128] → [4, 4095, 12, 128]（本地）
skiparse_cp scatter(÷2): 每卡拿到 [2, 4095, 12, 128]
每卡 SDPA 输入: [B_eff=2, N_sparse=4095, H=12, D=128]
计算量 ∝ 2 × 12 × 4095² ≈ 4.02 × 10⁸
```

**比值：**
```
Full / Skiparse = (1×6×32760²) / (2×12×4095²)
               = (6/24) × (32760/4095)²
               = (1/4) × 8²
               = 16 倍
```

#### 汇总

| block 类型 | 每卡 SDPA 输入形状 | 相对计算量 |
|-----------|-----------------|-----------|
| Full block（Ulysses CP） | `[1, 32760, 6, 128]` | **16×** |
| Skiparse block（skiparse_cp=2）| `[2, 4095, 12, 128]` | 1× |

所以在 dual_end 布局下，前 4 层（block 0–3）和后 4 层（block 26–29）各是一个 Full block，反向 profiling 中它们的 flash_attention / SDPA 耗时远高于中间 22 个 Skiparse block，**是设计预期，不是异常**。

dual_end 的工程意图：把 Full block（代价高、感受野完整）集中在模型首尾，首端处理 patch embedding 后的原始特征（需要全局上下文建立语义），尾端在输出前做全局整合；中间层用 Skiparse 降低整体计算量。

---

### 480P 81帧的 token 数计算

```
VAE 时序下采样: 81帧 → 21 (causal stride 4: (81-1)//4+1=21)
空间 patch 化: 480×832 / (8×8) / (2×2) = 60×104 / 4 = 30×52
N_total = T × H_patches × W_patches = 21 × 30 × 52 = 32760
```

### 各阶段 shape 变化（Full block 为例，full_cp_size=2）

```
模型输入:   [1, 32760, 1536]        ← 全序列
               ↓ context_preprocessor.preprocess (FullBlocksCP, cp=2)
进入 block: [1, 16380, 1536]        ← 每 rank 持有一半序列
               ↓ Q/K/V 投影
Q/K/V:      [1, 16380, 12, 128]     ← ★ 这是你看到的 shape
               ↓ pre_self_attn_all_to_all (scatter H, gather N, full_cp_size=2)
attention:  [1, 32760,  6, 128]     ← 完整序列，一半 head
               ↓ scaled_dot_product_attention (NPU SDPA)
attn out:   [1, 32760,  6, 128]
               ↓ post_self_attn_all_to_all (scatter N, gather H)
block 出:   [1, 16380, 12, 128]     ← 恢复本 rank 的切分
```

### 为什么不是 32760/8 = 4095

- **4095 = 32760/8**：需要 global_cp_size=8（即 cp_size=8 或 skiparse_cp_size=8），你的配置没有这么设
- **16380 = 32760/2**：来自 global_cp_size=2（skiparse_cp=2 × ulysses_cp=1）
- FSDP=8 切的是权重，不切序列。在你的 8 卡设置里：
  - 4 个独立的 DP 组（每组 2 卡共享同一条序列）
  - 每组 2 卡用 skiparse_cp 协作处理同一序列的不同部分

### 完整的 8 卡分组示意

```
Rank 0 ─┬─ skiparse_cp_group{0,1} ─ 同一条序列的前半 / 后半
Rank 1 ─┘
Rank 2 ─┬─ skiparse_cp_group{2,3} ─ 另一条序列的前半 / 后半
Rank 3 ─┘
Rank 4 ─┬─ skiparse_cp_group{4,5} ─ 另一条序列...
Rank 5 ─┘
Rank 6 ─┬─ skiparse_cp_group{6,7}
Rank 7 ─┘
               ↑ dp_size = 4（每 DP group 处理独立的数据样本）

FSDP: 所有 8 卡共享权重分片（与上述 CP 分组正交）
```

---

## 13. distributed/ 各文件的作用与协作关系

### 13.1 整体分工

```
训练时多卡通信需求:
  ① 权重太大 → FSDP 把权重切分到多卡         → fsdp2_wrapper.py
  ② 序列太长 → CP 把序列切分到多卡           → cp_state.py + communication.py
  ③ 状态同步 → 训练中间状态的保存和恢复        → checkpoint.py
  ④ 模型EMA → FSDP 下的 EMA 参数维护         → fsdp_ema.py
  ⑤ 环境搭建 → 进程组初始化、梯度聚合等       → utils.py
  ⑥ 旧版 CP → 基于 DTensor 的 CP 封装（旧版）→ cp_wrapper.py
```

### 13.2 cp_state.py — CP 全局状态单例

**作用**：维护一个进程级别的单例 `cp_state`，存储所有 CP group 的元信息，供模型前向中的任何位置访问，无需层层传参。

```python
cp_state = ContextParallelState()   # 进程级单例，导入即存在

# 在 train_osp.py 中一次性初始化：
cp_state.reset(
    global_cp_group = ...,   # skiparse_cp × cp 的组合 group
    cp_group        = ...,   # Ulysses CP group（cp_size=1 时也记录）
    skiparse_cp_group = ..., # Skiparse CP group（size=2）
    full_cp_group   = ...,   # Full blocks 用的 Ulysses CP group
)

# 在 OSPNextSelfAttention.forward() 中直接使用：
cp_group, cp_rank, cp_size = cp_state.get_cp_infos_with_type(self.cp_type)
```

**四种 CP group 的区别：**

| group | size（8卡配置） | 使用场景 |
|-------|---------------|---------|
| `global_cp_group` | 2 | 整体 CP 范围（skiparse_cp × ulysses_cp）|
| `cp_group` | 1 | Ulysses CP（Full blocks 以外的，这里实际关闭）|
| `skiparse_cp_group` | 2 | Skiparse blocks 的 batch 维通信 |
| `full_cp_group` | 2 | Full blocks 的 Ulysses CP |

三个全局标志 `USE_CONTEXT_PARALLEL` 等是**缓存变量**（懒计算），第一次调用 `use_context_parallel()` 时计算并缓存，避免每次 forward 都查询 dist group（HCCL 查询有开销）。

### 13.3 communication.py — 带 autograd 的通信原语

所有函数都封装成 `torch.autograd.Function`，前向通信 + 反向自动通信互为逆操作：

| 函数 | 通信原语 | 前向 | 反向 | 用途 |
|------|---------|------|------|------|
| `all_to_all_4D` | `dist.all_to_all_single` | scatter/gather seq↔head | 逆向 scatter/gather | Ulysses CP |
| `all_gather` | `dist.all_gather` | 拼接各 rank 的张量 | narrow 取对应 rank 的梯度 | mask 收集、register tokens |
| `all_to_all_single` | `dist.all_to_all_single` | 按 split_sizes 发送/接收 | 逆向发送/接收 | Skiparse CP 中 _parallel_skiparse |

### 13.4 fsdp2_wrapper.py — FSDP2 混合精度分片

**FSDP2（PyTorch 2.x 新版 FSDP）特点**：
- 权重以分片（DTensor）形式存储在各卡，forward 时 `all_gather` 拼完整权重，backward 后自动 `reduce_scatter` 梯度
- 支持**混合精度**：不同子模块用不同精度策略

三种精度策略应用到 OSPNextModel：
```
OSPNextLayerNorm/RMSNorm  → fp32（high_precision_policy）  # 精度敏感
OSPNextAttentionBlock     → bf16（low_precision_policy）   # 主体，内存效率
整体模型 model            → bf16（low_precision_policy）
```

**与 CP 的关系**：FSDP mesh = `(ddp=1, fsdp=8)`，CP mesh = `(dp=4, skiparse_cp=2, cp=1)`。两个 mesh **正交**：FSDP 切权重，CP 切序列，互不干扰。

### 13.5 checkpoint.py — 分布式检查点

- 保存：`get_model_state_dict` + `get_optimizer_state_dict`（PyTorch DCP API）→ 每个 rank 只保存自己持有的分片
- 加载：自动处理 DTensor 的重分布（不同并行度的 checkpoint 互相兼容）
- 额外保存：RNG 状态（保证可复现）+ dataloader 状态（断点续训）

### 13.6 fsdp_ema.py — FSDP 下的 EMA

EMA 权重保存在 float32 shadow_params（本 rank 只存自己 FSDP 分片的 shadow），更新公式：
```
shadow = decay × shadow + (1 - decay) × param
```
由于 FSDP param 是 DTensor，shadow 也同步按分片更新，无需 all_gather 完整权重。

### 13.7 utils.py — 分布式环境初始化

```python
setup_distributed_env():
    backend = "cpu:gloo, npu:hccl"   # NPU 使用华为 HCCL 通信库
                                      # GPU 使用 NCCL
    dist.init_process_group(...)
    torch.cuda.set_device(local_rank)
```

HCCL（华为集合通信库）对应 NCCL，是 NPU 上的高性能集合通信实现。

### 13.8 cp_wrapper.py — 旧版 CP（基于 DTensor）

这是早期基于 DTensor tensor parallelism API 实现的 CP，通过 `Redistribution` 和 `custom_context_parallelize_module` 实现序列并行。在当前 OSP-Next 主路径中已被 `cp_state.py + communication.py` 的手动实现取代，但代码保留供参考/旧版 WanModel。

---

## 14. 文件索引（完整）

| 文件 | 用途 |
|------|------|
| `train/train_osp.py` | 训练入口，模型初始化、训练循环、分布式设置 |
| `torchdiff/modules/osp_next.py` | OSPNextModel，支持 `quant`（Linear）+ `quant_attn`（Attention）|
| `torchdiff/modules/osp_next_bak.py` | 原始备份（所有HIF8改动前）|
| `torchdiff/modules/osp_next_hif8_linear.py` | Linear-only HIF8备份（加入attention量化前的快照）|
| `torchdiff/modules/attention.py` | Attention后端：FA3/FA2（GPU）/ SDPA（NPU） |
| `torchdiff/modules/skiparse_func.py` | Skiparse序列重排工具函数（1D/2D, single/group） |
| `torchdiff/modules/hif8_linear.py` | HIF8Linear（x+W双量化）；`_quant`被hif8_attention共用 |
| `torchdiff/modules/hif8_attention.py` | HIF8 Q/K/V量化（attention核前量化，新增）|
| `torchdiff/distributed/cp_state.py` | CP状态管理（全局单例）|
| `torchdiff/distributed/communication.py` | all_to_all_4D, all_gather 分布式通信原语 |
| `configs/train/npu/osp_1_3b.yaml` | 原始NPU训练配置 |
| `configs/train/npu/osp_1_3b_hif8.yaml` | HIF8 Linear量化配置 |
| `configs/train/npu/osp_1_3b_hif8_linear.yaml` | Linear-only快照备份（同 osp_1_3b_hif8.yaml）|
| `configs/train/npu/osp_1_3b_hif8_linear_atten.yaml` | Linear + Attention 双重HIF8配置（新增）|
| `scripts/train/npu/train_osp_1_3b.sh` | 8卡单节点启动脚本 |
| `osp_npu_1.3B_summary.md` | 本文档 |
