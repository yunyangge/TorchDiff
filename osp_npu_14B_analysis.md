# OSP-Next 14B NPU 训练架构深度分析

## 目录

1. [14B 模型参数与配置](#1-14b-模型参数与配置)
2. [14B 分布式配置与 Token 数推导](#2-14b-分布式配置与-token-数推导)
3. [dual_end Block 分布（40层 × num_full_blocks=8）](#3-dual_end-block-分布40层--num_full_blocks8)
4. [Skiparse 稀疏化的实现原理](#4-skiparse-稀疏化的实现原理)
5. [sparse_ratio=2 vs sparse_ratio=4 的本质区别](#5-sparse_ratio2-vs-sparse_ratio4-的本质区别)
6. [Single 和 Group 如何互补覆盖全序列](#6-single-和-group-如何互补覆盖全序列)
7. [过渡层的特殊通信：Full→Skiparse 与 Skiparse→Full](#7-过渡层的特殊通信fullskiparse-与-skiparse→full)
8. [Profiling 深度分析：40大FA + 40小FA + all_gather 堵塞](#8-profiling-深度分析40大fa--40小fa--all_gather-堵塞)
9. [为什么 all_gather / reduce_scatter 没有被计算掩盖](#9-为什么-all_gather--reduce_scatter-没有被计算掩盖)
10. [与 1.3B 的关键差异总结](#10-与-13b-的关键差异总结)
11. [Kernel 统计解读：ViewCopy / StrideSlice / HbcScatterNpuKernel](#11-kernel-统计解读viewcopy--strideslice--hbcscatternpukernel)
12. [为什么大 all_gather 出现在反向中间层而非过渡点](#12-为什么大-all_gather-出现在反向中间层而非过渡点)

---

## 1. 14B 模型参数与配置

```yaml
# configs/train/npu/osp_14b.yaml（关键字段）
model_config:
  dim: 5120            # 隐藏维度（1.3B 是 1536）
  ffn_dim: 13824       # FFN 维度（1.3B 是 8960）
  num_heads: 40        # 注意力头数（1.3B 是 12）
  num_layers: 40       # 总 Block 数（1.3B 是 30）
  head_dim: 128        # dim / num_heads = 5120/40 = 128（与1.3B相同）
  skiparse_2d: True    # 2D 空间稀疏化
  sparse_ratio: 4      # ★ 关键：1.3B 是 2，14B 是 4
  skiparse_model_type: "dual_end"
  num_full_blocks: 8

fsdp_size: 16          # 16卡切分权重
skiparse_cp_size: 4    # 4卡切分序列
cp_size: 4             # 设置为4但 use_context_parallel: False → Ulysses CP 关闭
use_skiparse_context_parallel: True
```

**参数规模：**
每个 Block 的主要参数量：

- Self Attn Q/K/V/O：4 × (5120 × 5120) = 4 × 26.2M = 104.9M
- Cross Attn Q/K/V/O：104.9M
- FFN：5120 × 13824 × 2 = 141.6M
- 单 Block 约 351M，40 Block 共约 14.0B 参数

---

## 2. 14B 分布式配置与 Token 数推导

### 2.1 视频分辨率与 Token 数（720P 81帧）

```
输入视频: [B=1, C=16, T=81, H=720, W=1280]

① VAE 时序下采样:   T: 81 → (81-1)//4+1 = 21
② VAE 空间下采样:   H: 720 → 90,  W: 1280 → 160
③ patch_embedding (patch_size=(1,2,2)):
     T: 21 → 21,  H: 90/2 = 45,  W: 160/2 = 80
④ patchify → [B, T×H×W, dim]:
     N_total = 21 × 45 × 80 = 75,600 tokens
```

### 2.2 skiparse_cp=4 下的 2D 空间切分

`ContextParallelPreprocessor._skiparse_2d_params` 的逻辑：

```python
# skiparse_cp_size = 4
# 尽量接近正方形：cp_size_h = 4 // ceil(√4) = 4 // 2 = 2，cp_size_w = 4 // 2 = 2
cp_size_h = 2, cp_size_w = 2

# sub_h = sub_w = sparse_ratio² = 4² = 16（skiparse 2D 的对齐粒度）
sub_h = sub_w = 16

# H方向：
num_sub_h = ceil(45 / 16) = 3     # 45 / 16 取整
seq_h     = ceil(3 / 2) × 16 = 32  # 每卡在 H 方向有 32 patches

# W方向：
num_sub_w = ceil(80 / 16) = 5
seq_w     = ceil(5 / 2) × 16 = 48  # 每卡在 W 方向有 48 patches

每卡局部 grid: T=21 × H_local=32 × W_local=48
N_local = 21 × 32 × 48 = 32,256  （含 padding，>75600/4=18900）
```

### 2.3 16卡的分组逻辑

```
16张 NPU 卡的分工:
┌────────────────────────────────────────────────────────────────┐
│  FSDP (size=16)  →  切分 模型权重 W                              │
│   每张卡存 1/16 的权重，前向时 all_gather 拼回完整权重              │
│   5120×5120 Linear 权重 ~26.2M 参数，每卡存 ~1.6M                │
├────────────────────────────────────────────────────────────────┤
│  skiparse_cp (size=4)  →  2D 空间切分激活值                      │
│   4卡协作处理同一条序列的不同空间子区域                              │
│   每卡持有 32256 token（空间 2×2 分区 + padding）                  │
├────────────────────────────────────────────────────────────────┤
│  DP (data_parallel)  = 16 / 4 = 4                              │
│   4个独立的数据并行组，每组处理不同视频样本                          │
└────────────────────────────────────────────────────────────────┘

rank 0─┬─ skiparse_cp{0,1,2,3} → 同一条序列的4个空间子区域
rank 1─┤
rank 2─┤
rank 3─┘
rank 4─┬─ skiparse_cp{4,5,6,7} → 另一条序列...
...
```

---

## 3. dual_end Block 分布（40层 × num_full_blocks=8）

### 3.1 公式推导

```python
# osp_next.py:1317-1322
assert num_full_blocks % 4 == 0   # 8 % 4 == 0 ✓
skiparse_start_index = 8 // 2 = 4
skiparse_end_index   = 40 - 8 // 2 - 1 = 35
full_block_indices   = [0,1,2,3] + [36,37,38,39]
```

### 3.2 完整 Block 布局

```
层索引: 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19
类型:    F  F  F  F  S  G  S  G  S  G  S  G  S  G  S  G  S  G  S  G

层索引: 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39
类型:    S  G  S  G  S  G  S  G  S  G  S  G  S  G  S  G  F  F  F  F

F=Full（完整 attention），S=Single skiparse，G=Group skiparse

前 4 层（0-3）: Full block → 完整 75600 token SDPA
中 32 层（4-35）: Skiparse 交替（16 对 Single+Group）
后 4 层（36-39）: Full block → 完整 75600 token SDPA
```

### 3.3 边界标志位

每个 Block 构造时会设置两个标志：

| Block 索引       | `is_full2skiparse_block` | `is_skiparse2full_block` | 含义                |
| ---------------- | -------------------------- | -------------------------- | ------------------- |
| 4（首个 Single） | True                       | False                      | Full→Skiparse 过渡 |
| 35（末个 Group） | False                      | True                       | Skiparse→Full 过渡 |
| 其他 Skiparse    | False                      | False                      | 普通稀疏块          |

这两个标志决定在 Block 入口/出口是否需要做**额外的序列重分配通信**（详见第7节）。

---

## 4. Skiparse 稀疏化的实现原理

### 4.1 核心思想：把"稀疏访问"变成"dense计算×小序列"

Skiparse 不修改 attention 算子本身。它通过**空间重排（rearrange）**把稀疏采样模式转化为标准的 dense SDPA：

```
原始序列: [B, N, H, D]         → SDPA 计算量 ∝ B×H×N²

Skiparse 重排后: [p²B, N/p², H, D]  → SDPA 计算量 ∝ p²B×H×(N/p²)² = B×H×N²/p²

节省比例: p² 倍（二维稀疏，p=sparse_ratio）
```

SDPA 算子看到的是普通的 dense 输入，"不知道"这些 token 在原始空间中是不连续的。

### 4.2 2D Skiparse 的空间视角

对于二维空间网格 H×W（忽略时序 T），sparse_ratio=p：

**Single 重排** (`skiparse_2d_single`)：

```python
rearrange(x, 'b (t h p w q) c -> (p q b) (t h w) c',
          p=sparse_ratio, q=sparse_ratio,
          h=H//sparse_ratio, w=W//sparse_ratio)
```

将空间位置 (row, col) 分组：`row = h_idx*p + p_idx，col = w_idx*q + q_idx`

- 超级批次 `(p_idx, q_idx)` 包含所有满足 `row%p == p_idx AND col%p == q_idx` 的 token
- 即：**棋盘格状均匀稀疏采样**，间距为 `p`

**Group 重排** (`skiparse_2d_group`)：

```python
rearrange(x, 'b (txh p1 p2 w q1 q2) c -> (p1 q1 b) (txh p2 w q2) c',
          p1=p, q1=p, p2=p, q2=p, w=W//(p*p))
```

超级批次 `(p1_idx, q1_idx)` 包含所有在**粗粒度 p×p 宏块中位置为 (p1_idx, q1_idx) 的宏块内的全部 token**（宏块内部 dense）

- 即：**以 p×p 密集窗口为单位的稀疏采样**，窗口中心间距为 `p²`

---

## 5. sparse_ratio=2 vs sparse_ratio=4 的本质区别

### 5.1 几何模式对比（仅展示空间 H×W，忽略 T）

**sparse_ratio=2：**

```
8×8 空间网格（S=Single选到, G=Group选到, .=当前子批次未选到）：

Single 超批次(0,0)：      Group 超批次(0,0)：
S . S . S . S .          G G . . G G . .
. . . . . . . .          G G . . G G . .
S . S . S . S .          . . . . . . . .
. . . . . . . .          . . . . . . . .
S . S . S . S .          G G . . G G . .
...                      ...

间距: 2×2 均匀点阵          间距: 2×2 密集块，每隔4跳一次
p²=4 个超批次，各覆盖1/4总token
```

**sparse_ratio=4：**

```
16×16 空间网格：

Single 超批次(0,0)：均匀点阵，间距 4×4，p²=16 个超批次
S . . . S . . . S . . . S . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
. . . . . . . . . . . . . . . .
S . . . S . . . S . . . S . . .
...

Group 超批次(0,0)：密集块，4×4 宏块间距 16
G G G G . . . . G G G G . . . .
G G G G . . . . G G G G . . . .
G G G G . . . . G G G G . . . .
G G G G . . . . G G G G . . . .
. . . . . . . . . . . . . . . .
...
```

### 5.2 计算量与内存比较（14B 720P 81f 实际数值）

|                                   | sparse_ratio=2 | sparse_ratio=4          |
| --------------------------------- | -------------- | ----------------------- |
| 超批次数 p²                      | 4              | 16                      |
| SDPA 序列长度（per super-batch）  | N/4            | N/16                    |
| SDPA 计算量（per block，vs Full） | 1/4            | **1/16**          |
| 每 token 感受野（per block）      | 1/4 总 token   | **1/16** 总 token |
| 需要 S+G 对覆盖全局               | 较少对         | 需要更多 S+G 层         |
| NPU Transdata 数据量              | 较小           | 较大（p²=16 次分组）   |

### 5.3 rearrange 的关键不同

**sparse_ratio=2（2D Single）：**

```python
rearrange(x, 'b (t h p w q) c -> (p q b) (t h w) c', p=2, q=2,
          h=H//2, w=W//2)
# 批次膨胀: 2×2=4倍
# 单超批次序列: T × (H//2) × (W//2)
```

**sparse_ratio=4（2D Single）：**

```python
rearrange(x, 'b (t h p w q) c -> (p q b) (t h w) c', p=4, q=4,
          h=H//4, w=W//4)
# 批次膨胀: 4×4=16倍  ← 数据分成16份
# 单超批次序列: T × (H//4) × (W//4)  ← 每份只有1/16
```

**sparse_ratio=4（2D Group）：**

```python
rearrange(x, 'b (txh p1 p2 w q1 q2) c -> (p1 q1 b) (txh p2 w q2) c',
          p1=4, q1=4, p2=4, q2=4, w=W//16)
# 批次膨胀: p1×q1=16倍（与 Single 相同）
# 序列长度: T×(H//4)×(W//4)（与 Single 相同）
# ← 但 token 的空间排布模式完全不同！
```

两者的区别在于**同一超批次内的 token 空间分布**：

- Single (p=4)：16 个超批次，各自覆盖均匀间距为 4 的稀疏点阵
- Group (p=4)：16 个超批次，各自覆盖密集的 4×4 宏块（宏块间距=16）

---

## 6. Single 和 Group 如何互补覆盖全序列

### 6.1 sparse_ratio=2 的互补原理（***）

以 4×4 网格为例（简化 T=1）：

```
完整空间（16个位置）：
(0,0) (0,1) (0,2) (0,3)
(1,0) (1,1) (1,2) (1,3)
(2,0) (2,1) (2,2) (2,3)
(3,0) (3,1) (3,2) (3,3)

Single 超批次划分（p=2）：
SB(0,0): (0,0)(0,2)(2,0)(2,2) ← 偶行偶列
SB(0,1): (0,1)(0,3)(2,1)(2,3) ← 偶行奇列
SB(1,0): (1,0)(1,2)(3,0)(3,2) ← 奇行偶列
SB(1,1): (1,1)(1,3)(3,1)(3,3) ← 奇行奇列
→ 4个超批次不重叠，完整覆盖所有16个位置    【注意这里实际上每一个还是算到了】

Group 超批次划分（p=2，p1=p2=q1=q2=2，h_outer=H//4=1，w_outer=W//4=1）：
GS(0,0): 位置(0..1,0..1)内的token按p2,q2组合
         → 以2×2密集块为单元，各超批次覆盖不同宏块位置
→ 同样完整覆盖所有16个位置

互补效果：
第 k 层（Single）: 每对 (pos_A, pos_B) 如果在同一 SB 内才能 attend
第 k+1 层（Group）: 每对 (pos_A, pos_B) 如果在同一 GB 内才能 attend
→ 两层合起来，原本在 Single 里不能 attend 的远程局部对，在 Group 里有机会
→ 原本在 Group 里不能 attend 的跨宏块稀疏对，在 Single 里有机会
```

### 6.2 sparse_ratio=4 下每对 token 需要多少层才能间接连通

sparse_ratio=4 时每超批次只有 1/16 的 token，单层覆盖极其稀疏。

理论上，通过连续 Single-Group 交替，经过 O(log p) 量级的层数可以建立全局信息流。14B 模型有 32 个 Skiparse Block（16 对 S+G），对于 sparse_ratio=4 已经足够建立充分的全局感受野。

### 6.3 Single→Group 的中间转换（`skiparse_2d_single_to_group`）

在**每个 Group Block 的入口**，激活值需要从 Single 排列转化为 Group 排列：

```python
def skiparse_2d_single_to_group(x, grid_sizes, sparse_ratio):
    T, H, W = grid_sizes
    return rearrange(
        x, '(p2 q2 b) (txh_p1 p1 w_q1 q1) c -> (p1 q1 b) (txh_p1 p2 w_q1 q2) c',
        p1=sparse_ratio, q1=sparse_ratio, p2=sparse_ratio, q2=sparse_ratio,
        w_q1=W // (sparse_ratio ** 2)
    )
```

本质上是**超批次维度和序列维度之间的转置**：

- 把当前超批次标签 `(p2,q2)` 放进序列维度
- 把序列里的部分维度 `(p1,q1)` 提出来作为新的超批次标签

在分布式（skiparse_cp>1）情况下：使用 `_parallel_skiparse_2d_single_to_group`，内部包含一次 `dist.all_to_all_single`（HCCL），是一次真正的跨卡通信。

**Group→Single 是完全互逆的操作**（代码复用同一个函数），用于 Group Block 出口转回 Single 排列以供下一个 Single Block 使用。

---

## 7. 过渡层的特殊通信：Full→Skiparse 与 Skiparse→Full

### 7.1 为什么过渡层需要额外通信

Full Block 和 Skiparse Block 使用不同的序列分布策略：

- **Full Block**：序列按线性顺序（1D）均分给各卡，每卡持有连续的 **`N/cp` 段**
- **Skiparse Block**：序列按 2D 空间子区域切分，每卡持有一个空间矩形区域

两种切分的含义完全不同，因此在边界必须做一次"全量重分配"：

### 7.2 Full→Skiparse 过渡（Block 3→4，`is_full2skiparse_block=True`）

```python
# osp_next.py:1529-1535
if idx != 0 and block.is_full2skiparse_block:
    # Step 1: postprocess — 从 FullBlocksCP 分布恢复完整序列
    x = self.context_preprocessor.postprocess(
        x, grid_sizes,
        shard_seq_lens=full_block_full_shard_seq_lens,
        cp_type=ContextParallelType.FullBlocksCP
    )
    # 内部执行: all_gather(x, dim=1, group=full_cp_group)
    # [1, 18900, 5120] × 4卡 → [1, 75600, 5120]   ← ★ all_gather：真正的跨卡通信
    # 数据量: B × N_full × dim × 2bytes(bf16) = 1×75600×5120×2 ≈ 740 MB per card

    # Step 2: preprocess — 按 2D 空间子区域重新切分
    x, sub_grid_sizes = self.context_preprocessor.preprocess(
        x, grid_sizes, cp_type=self.main_cp_type  # 即 ContextParallelType.CP
    )
    # 内部执行: 取 x[:, T×H_local×W_local, :] 对应的空间子区域（本地 narrow，无通信）
    # [1, 75600, 5120] → [1, 32256, 5120]
    # 伴随 contiguous() → NPU Transdata（内存重整）
```

**通信代价：**

- all_gather 740 MB 激活值（跨 4 卡）
- 发生时机：紧接在 Block 3 完成之后、Block 4 开始之前
- **无法掩盖**：**Block 4 的计算严格依赖此 all_gather 完成**

### 7.3 Skiparse→Full 过渡（Block 35→36，`is_skiparse2full_block=True`）

```python
# osp_next.py:1563-1569
if idx != len(self.blocks) - 1 and block.is_skiparse2full_block:
    # Step 1: postprocess — 从 skiparse_cp（2D空间）分布恢复完整序列
    x = self.context_preprocessor.postprocess(
        x, grid_sizes,
        shard_seq_lens=full_shard_seq_lens,
        cp_type=self.main_cp_type
    )
    # 内部执行: all_gather(x, dim=1, group=skiparse_cp_group) + 2D 拼图重组
    # [1, 32256, 5120] × 4卡 → [1, 75600, 5120]   ← ★ all_gather
  
    # Step 2: preprocess — 按 1D 线性切分给 Full Block
    x, sub_grid_sizes = self.context_preprocessor.preprocess(
        x, grid_sizes, cp_type=ContextParallelType.FullBlocksCP
    )
    # [1, 75600, 5120] → [1, 18900, 5120]
```

### 7.4 各阶段通信汇总

一个完整前向传播中发生的所有跨卡通信，按时序排列：

```
模型前向开始
│
├─ Block 0 (Full)
│   ├─ [FSDP all_gather] Q/K/V/O Linear 权重（各1/16分片→完整权重）×4次
│   ├─ [CP all_to_all] pre_self_attn_all_to_all（scatter head, gather seq）
│   ├─ [SDPA 本地计算]
│   ├─ [CP all_to_all] post_self_attn_all_to_all（scatter seq, gather head）
│   └─ [FSDP all_gather] O Linear 权重 ×1次
│
├─ Block 1, 2, 3（Full，同上）
│
├─ ★★ [over-all_gather] Full→Skiparse 过渡通信
│   └─ all_gather 完整激活值 75600 × 5120 × bf16 ≈ 740 MB（跨4卡）
│
├─ Block 4 (Single Skiparse, is_full2skiparse_block=True)
│   ├─ [rearrange_input] skiparse_2d_single 重排（Transdata）
│   ├─ [context_rearrange] text/e: Repeat ×p²=16（local scatter）
│   ├─ [FSDP all_gather] Q/K/V/O Linear 权重 ×4次
│   ├─ [skiparse_cp scatter] rearrange后batch维切片（local narrow）
│   ├─ [SDPA 本地计算] [4, 2016, 40, 128]
│   └─ [rearrange_output] Identity（无操作）
│
├─ Block 5 (Group Skiparse)
│   ├─ [rearrange_input] skiparse_2d_single_to_group
│   │   └─ [CP all_to_all] _parallel_skiparse_2d_single_to_group（HCCL）
│   │   └─ [Transdata]
│   ├─ [SDPA 本地计算]
│   └─ [rearrange_output] Identity 或 skiparse_2d_group_reverse
│
├─ ... Block 6-34（交替 Single/Group）...
│
├─ Block 35 (Group, is_skiparse2full_block=True)
│   ├─ [skiparse_2d_single_to_group] 含 all_to_all
│   ├─ [SDPA 本地计算]
│   └─ [rearrange_output] skiparse_2d_group_reverse（Transdata）
│
├─ ★★ [over-all_gather] Skiparse→Full 过渡通信
│   └─ all_gather 完整激活值 + 2D 拼图 ≈ 740 MB（跨4卡）
│
├─ Block 36, 37, 38, 39（Full，同 Block 0-3）
│
└─ 模型前向结束
```

---

## 8. Profiling 深度分析：40大FA + 40小FA + all_gather 堵塞

### 8.1 40个大 FlashAttention（Self Attention）的来源

每个 Block 有一次 Self Attention → 40 个 Block = **40 次 Self Attn FA**。

但它们的计算量天差地别：

**Full Block Self Attn（前4 + 后4 = 共8个）：**

```
SDPA 输入（经 pre_all_to_all 后）: [B=1, N_full=75600, H/cp=10, D=128]
计算量 ∝ B × (H/cp) × N_full²
       = 1 × 10 × 75600²
       ≈ 57.1 × 10⁹ （57 billion）
```

**Skiparse Block Self Attn（中间32个）：**

```
SDPA 输入（skiparse_cp scatter 后）: [B_eff=4, N_sparse=2016, H=40, D=128]
计算量 ∝ B_eff × H × N_sparse²
       = 4 × 40 × 2016²
       ≈ 649 × 10⁶ （649 million）
```

**比值：57100M / 649M ≈ 88倍**

所以在 profiling 中，前4个和后4个 FA（Full Block）的耗时约是中间32个（Skiparse Block）的 **88倍**。

| Block 类型       | 单次 SDPA 计算量 | 相对耗时       |
| ---------------- | ---------------- | -------------- |
| Full（前4/后4）  | ~57B FLOPs/card  | **88×** |
| Skiparse（中32） | ~649M FLOPs/card | 1×            |

### 8.2 40个小 FlashAttention（Cross Attention）的来源

每个 Block 还有一次 Cross Attention（img Query × text Key/Value）：

```python
# OSPNextCrossAttention.forward()
q: [B, N_local, H, D]    # img token，已按 CP 切分
k: [B, text_len=512, H, D]  # text，完整长度（512 token）
v: [B, 512, H, D]

# 注意：cross attn 不需要 all_to_all！
# img Q 已经是 CP-local 的，img 作为 q 天然支持按序列切分
# text_len=512 远小于 N=75600，SDPA 计算量极小
```

Cross Attn SDPA：`N_q × N_kv = N_local × 512 ≈ 18900 × 512 ≈ 9.7M`（微不足道）

因此 40 个 Cross Attn FA 在 profiling 中**一致偏小**，无论是 Full Block 还是 Skiparse Block 的 Cross Attn 耗时都接近（都受 N_kv=512 限制），这也是它们表现均匀的原因。

### 8.3 为什么某些层 all_gather 前有很长的通信堵塞

Profiling 中出现的长时间 all_gather，按来源分三类：

**① 过渡点的激活值 all_gather（最显著）：**

发生在 Block 3→4（Full→Skiparse）和 Block 35→36（Skiparse→Full）：

- 数据量：75600 × 5120 × 2 bytes = ~740 MB
- 在 4 卡 skiparse_cp 组内 all_gather：实际传输 3/4 × 740 MB ≈ 555 MB per link
- HCCL 带宽约 200-400 GB/s（NPU 910B NVLink），传输耗时约 1.4-2.8 ms
- **没有任何计算可以掩盖这段时间**（Block 4 的所有计算都依赖它完成）

**② FSDP all_gather（每个 Linear 层前）：**

每个 Block 有 Q/K/V/O + Cross Q/K/V/O + FFN×2 共 10 个 Linear：

- 每个 Linear 权重：5120 × 5120 × 2 bytes = ~52 MB（完整权重）
- 每卡存 1/16 = ~3.25 MB，all_gather 14 份 × 3.25 MB = ~45 MB per block per linear
- `explicit_prefetching_num_blocks: 0` → **每个 Linear 前都同步等待 all_gather**
- 若设置为 1 或 2，FSDP 会异步预取下一 Block 的权重，覆盖当前 Block 的计算
  （**这里需要修改  重跑profiling， 看是否能有更多掩盖，减少all_gather**）

**③ per-block 内 CP all_to_all（Full Block 专有）：**

Full Block 中 `pre_self_attn_all_to_all` / `post_self_attn_all_to_all`：

- 数据量：[1, 18900, 40, 128] × bf16 = ~192 MB（scatter 和 gather）
- 在 4 卡 full_cp 组内通信，**无法掩盖**（SDPA 强依赖）

---

## 9. 为什么 all_gather / reduce_scatter 没有被计算掩盖

### 9.1 根本原因：串行数据依赖

通信未被掩盖不是"框架不好"，主要是**数据依赖的客观规律**：

```
Linear 层计算链：
  all_gather(W_shard) → W_full         ← 必须完成
         ↓
  Y = X @ W_full.T                     ← 计算
         ↓
  reduce_scatter(grad_W) → grad_shard  ← 反向必须完成
```

Q/K/V 投影结果必须先有完整 W 才能算，SDPA 必须先有完整 K/V 才能算——这些是**硬依赖**，理论上无法并行。

### 9.2 可以被掩盖的部分（但当前配置关闭了）

唯一的掩盖机会是**层间流水：预取下一层的 W，同时计算当前层**：

```
当前层计算 Block k               │  异步预取 Block k+1 的权重
SDPA(Q,K,V)  ─────────────────────╋──────────────────────────────
                                  │  all_gather(W_k+1_shard)   ← 并行！
                                  │  （通信流，不占计算单元）
```

配置文件中 `explicit_prefetching_num_blocks: 0` **明确禁用了预取**，导致每层权重 all_gather 都是同步阻塞的。

改为 `explicit_prefetching_num_blocks: 1` 可以预取 1 个 Block 的权重，理论上接近完全掩盖 FSDP all_gather，但需要额外显存存放预取的完整权重（约 4 × 52 MB × 10 Linear = 2 GB per prefetched block）。

### 9.3 NPU 上 HCCL 掩盖的额外困难

即使代码层面发出了异步通信，NPU+HCCL 比 GPU+NCCL 更难实现真正的计算通信重叠：

| 因素                      | GPU（NCCL）                | NPU（HCCL）                  |
| ------------------------- | -------------------------- | ---------------------------- |
| 计算流 / 通信流独立性     | CUDA stream 天然独立       | CANN stream 独立性较弱       |
| all_gather 与 matmul 并发 | 基本能重叠                 | 受 CANN 调度限制，部分序列化 |
| reduce_scatter 掩盖       | 成熟，NCCL 高度优化        | HCCL 优化程度较低            |
| 编译器辅助                | inductor 可做通信-计算融合 | torch_npu dynamo 支持不完整  |

### 9.4 反向传播中的 reduce_scatter

反向传播时，每个 Linear 层的梯度计算完成后，FSDP2 立即发起 `reduce_scatter(grad_W)` 来分散梯度碎片：

```
反向计算顺序（与前向相反，Block 39→0）：

Block 39 反向：
  [计算] grad_x, grad_W = backward(O_linear)
  [HCCL] reduce_scatter(grad_W_O) ← 同步阻塞，归约后每卡只存1/16梯度
  [计算] grad_x = backward(SDPA)
  [CP all_to_all] post_attn backward（前向的逆）
  [计算] grad_Q, grad_K, grad_V
  [HCCL] reduce_scatter(grad_W_Q/K/V)

... Block 38, 37, 36 同上 ...

★★ Skiparse→Full 反向过渡（Block 35→36 的逆）：
  前向的 all_gather 在反向变为 reduce_scatter（autograd 自动）
  [HCCL] reduce_scatter 激活梯度（740 MB，跨4卡）← 反向中最大通信事件

... Block 35-4 同上 ...

★★ Full→Skiparse 反向过渡（Block 3→4 的逆）：
  [HCCL] reduce_scatter 激活梯度

Block 3→0 反向...
```

每次 `reduce_scatter` 都在计算链的关键路径上，无法通过预取掩盖（当前层的梯度必须先算完才能 scatter）。

### 9.5 为什么是"框架不好"还是"正常行为"的判断

| 现象                     | 原因                       | 是否可优化             |
| ------------------------ | -------------------------- | ---------------------- |
| FSDP all_gather 未掩盖   | `explicit_prefetching=0` | ✅ 开启预取可改善      |
| CP all_to_all 未掩盖     | 严格数据依赖               | ❌ 无法掩盖（强依赖）  |
| 过渡点 all_gather 未掩盖 | 激活值重分配，严格依赖     | ❌ 架构决定            |
| reduce_scatter 未掩盖    | 同步梯度规约               | 部分可通过梯度流水掩盖 |
| NPU 上重叠效果差         | HCCL < NCCL                | 长期优化方向           |

---

## 10. 与 1.3B 的关键差异总结

| 维度                                | 1.3B（NPU train）     | 14B（NPU train）           |
| ----------------------------------- | --------------------- | -------------------------- |
| 层数                                | 30                    | **40**               |
| 隐藏维度                            | 1536                  | **5120**             |
| 注意力头数                          | 12                    | **40**               |
| sparse_ratio                        | 2                     | **4**                |
| skiparse 批次膨胀                   | p²=4                 | **p²=16**           |
| Full Block 数                       | 8（dual_end：前4后4） | **8**（前4后4）      |
| Skiparse Block 数                   | 22                    | **32**               |
| fsdp_size                           | 8                     | **16**               |
| skiparse_cp_size                    | 2                     | **4**                |
| 分辨率                              | 480P 81f              | **720P 81f**         |
| N_total                             | 32,760                | **75,600**           |
| N_local（per card）                 | 16,380                | **32,256**           |
| Full SDPA 计算量（per card）        | 6.4B                  | **57.1B**            |
| Skiparse SDPA 计算量（per card）    | 402M                  | **649M**             |
| Full/Skiparse 计算比                | ~16×                 | **~88×**            |
| Single→Group 转换是否有 all_to_all | 是（skiparse_cp=2）   | 是（skiparse_cp=4，更大）  |
| explicit_prefetching_num_blocks     | 0（profiling关）      | **0**（profiling关） |

### 关键观察

1. **sparse_ratio=4 比 =2 更激进**：Skiparse Block 的序列缩短到 1/16，Full Block 的相对代价从 16× 升至 88×，全模型计算量中 Full Block 的占比更高。
2. **32个 Skiparse Block 已足够**：16 对 Single+Group 对于 sparse_ratio=4 能建立充分的全局感受野，不需要像更小模型那样依赖更多 Full Block。
3. **过渡点通信更重**：14B 的激活值从 32256×5120 → 75600×5120（740 MB），比 1.3B 的 16380×1536（~50 MB）大约 15 倍，过渡点 all_gather 在 profiling 中会非常显眼。
4. **ffn_dim 不成比例**：14B 的 ffn_dim/dim = 13824/5120 ≈ 2.7，而 1.3B 为 8960/1536 ≈ 5.8。14B 的 FFN 相对更窄，attention 权重比例更大。
5. **梯度检查点已开启**：`gradient_checkpointing: True`，反向传播需要重算前向，通信量和计算量翻倍，profiling 中反向会看到更多重复的 all_gather 序列。

---

## 11. Kernel 统计解读：ViewCopy / StrideSlice / HbcScatterNpuKernel

### 11.1 Kernel 统计总览（来自 NPU_14B profiling_result_kernel_statistics.jpg）

图像分辨率限制，以下基于用户补充说明修正：

| Kernel 名                            | Core Type | 时间占比           | 说明                                           |
| ------------------------------------ | --------- | ------------------ | ---------------------------------------------- |
| **reduce_scatter_AicpuKernel** | AI_CPU    | **48.4%** ★ | FSDP2 反向梯度归约+分发                        |
| **allgather_AicpuKernel**      | AI_CPU    | **38.4%** ★ | FSDP2 权重重建（前向+反向重计算）              |
| **FlashAttentionScoreGrad**    | MIX_AIV   | ~3.6%              | FA 反向梯度 kernel                             |
| Cast                                 | MIX_AIV   | ~3.6%              | 数据类型转换（bf16↔fp32）                     |
| Mul                                  | AI_VECTOR | ~1.4%              | 逐元素乘法（modulation gate 等）               |
| **ViewCopy**                   | AI_VECTOR | ~0.41%             | 非连续 tensor → 连续布局拷贝（即"Transdata"） |
| **StrideSlice**                | AI_VECTOR | ~0.38%             | tensor 切片/narrow                             |
| Add                                  | AI_VECTOR | ~0.15%             | 残差连接                                       |
| ReduceSum                            | AI_VECTOR | ~0.16%             | LayerNorm/RMSNorm 内部                         |

**HCCL 合计（reduce_scatter + allgather）= 86.8%，FlashAttention 类 ≈ 4%。模型严重通信瓶颈，AI Core 利用率极低（<15%）。**

### 11.2 reduce_scatter vs allgather 的含义与不对称性

```
reduce_scatter_AicpuKernel (48.4%)：
  触发时机：反向传播，每个 Linear 层的梯度计算完成后
  操作：   跨 16 卡对梯度求和（reduce），再将 1/16 分配给各卡（scatter）
  次数/iteration：10 Linear/block × 40 blocks = 400 次（仅反向）
  数据量/次：完整权重大小 52.4 MB（per Linear），约 49 MB sent per card

allgather_AicpuKernel (38.4%)：
  触发时机：前向 + 反向重计算，每个 block 开始时
  操作：   将本卡的 1/16 权重分片广播给所有 16 卡，重建完整权重
  次数/iteration：10 × 40 × 2（forward + recompute）= 800 次
  数据量/次：52.4 MB（per Linear），约 49 MB received per card

为什么 reduce_scatter > allgather（虽然次数更少）：
  reduce_scatter = gather + reduce（规约计算）+ scatter，比 allgather 多了规约步骤
  规约（16 卡求和）在通信链路上额外消耗时间，导致单次更慢
```

### 11.2 ViewCopy — Transdata 的真实 Kernel 名

之前分析中提到"contiguous() 触发 Transdata"，**在实际 profiler 中这个 kernel 的名字就是 `ViewCopy`**。"Transdata"是 CANN 架构文档的概念名，runtime profiler 里显示的是 ViewCopy。

```
代码:   x.permute(2, 0, 1, 3).contiguous()
PyTorch: 检测到 strides 不连续
CANN:   启动 ViewCopy kernel ← 真正搬运内存，将非连续 layout 复制为连续
```

在本模型中触发 ViewCopy 的主要场景：

```python
# 1. pre/post_self_attn_all_to_all 内（Full Block）
x.view(B, N, 2, H//2, D)
  .transpose(0, 2)           # 标记非连续（无 ViewCopy 此处）
  .contiguous()              # ← ViewCopy！（transposing → contiguous copy）

# 2. _parallel_skiparse_2d_single_to_group 内（每个 Group/Single Block）
recv.view(P2, G, G, b, base, C)
  .permute(0, 2, 1, 3, 4, 5) # 标记非连续
contiguous(recv)              # ← ViewCopy！

# 3. ContextParallelPreprocessor.preprocess 空间切片后
x = x[:, :, start_h:end_h, start_w:end_w, :]
contiguous(x)                 # ← ViewCopy（切片结果通常非连续）
```

Count=1931 在整个 iteration 里相当合理：每个 skiparse block 触发 ~2 次（rearrange_input + all_to_all 后），共 32 blocks × 2 = 64 次/forward，加上 backward recompute 再来 64 次 ≈ 128 次/iteration。其余来自 Full block 的 all_to_all 和其他地方。

### 11.3 StrideSlice — tensor 切片的 NPU 实现

`StrideSlice` 是 CANN 执行**步进切片**的 kernel，对应以下 PyTorch 操作：

```python
# 以下操作在 NPU 上会触发 StrideSlice：
x[:, start:end, :]              # 普通切片
x.narrow(dim, start, length)    # narrow（底层切片）
x.split_with_sizes(sizes, dim)  # 按大小分割
x[::2]                          # 步进 > 1 的索引
```

在本模型中的主要触发来源：

```python
# 1. skiparse_cp 空间子区域提取（每次 preprocess 调用）
x = x[:, :, start_h:end_h, start_w:end_w, :]

# 2. register token 分离（num_register_tokens > 0 时）
register_q, q = q.split_with_sizes([nr, N-nr], dim=1)

# 3. post_self_attn_all_to_all 中的 narrow（如有 variable-length shard）
x = x.narrow(1, rank * chunk, chunk)

# 4. FSDP 分片提取（每个 Linear 参数的 shard 提取）
shard = flat_param.narrow(0, offset, numel)
```

StrideSlice 本身很轻量（单次 < 1 μs），count ~1300 而时间占比仅 0.38% 证实了这一点。它的存在主要说明切片操作频繁，但不是瓶颈。

### 11.4 HbcScatterNpuKernel 占 46% 的含义

`HbcScatterNpuKernel` 是 HCCL（华为集合通信库）ring-allreduce 算法中的底层 scatter 步骤。所有集合通信最终分解为 scatter + gather：

```
all_gather   = 1次 allgather kernel（或 scatter+gather 组合）
all_to_all   = N × scatter + N × gather
reduce_scatter = scatter + reduce
```

**46% 的时间在通信上意味着：**

- 计算设备（AI Core）利用率约 54% 或更低
- 如果 ViewCopy(0.41%) + FlashAttn(~4%) + 其他 compute ≈ 10-15%
- 其余 35-40% 是 **计算在等通信（stall）**
- 这是典型的通信未掩盖导致的低效率

**Count=241 的解释（估算）：**

HCCL 的 ring-allreduce 对于 fsdp_size=16，一次 all_gather 需要 16-1=15 步 scatter + 15 步 gather。如果 HCCL 将多个小的 all_gather 合并成一个大的 bucket all_gather（FSDP2 的 "bucket" 机制），则实际 kernel 调用次数会远少于参数量级的预期。241 次可能对应若干个完整 iteration 的聚合统计，或者是 HCCL bucket 化后的结果。

---

## 12. 为什么大 all_gather 出现在前向中间层（block 16-30）而非过渡点

> **重要前提（已由你确认）**：下面讨论的大 all_gather 和"超长 free"均发生在 **前向（Forward）阶段**，不是反向或重计算阶段。

### 12.1 为什么 block 3→4 过渡点的 all_gather 并不突出

按架构设计，block 3→4（Full→Skiparse 过渡）会触发激活值的 CP 边界 all_gather（将 cp_size 个卡上的分片激活重组为完整激活，约 740 MB），但你观察到此处并不突出。

原因在于 `configs/train/npu/osp_14b.yaml` 中的配置：

```yaml
use_context_parallel: False       # 全局 CP 关闭
use_skiparse_context_parallel: True
```

`use_context_parallel: False` 导致 Full blocks 采用的 Ulysses CP（`full_cp_size`）实际上为 1，即 Full blocks **没有 CP 分片**，自然也就不存在 Full↔Skiparse 过渡时的激活重组 all_gather。此处 block 3→4 你只看到正常的 FSDP 权重 all_gather（约 52 MB per linear），不会特别突出。

同理，block 35→36（Skiparse→Full 过渡）也只有 FSDP 权重 all_gather，没有激活维度的跨卡重组，所以你观察到它"还是挺正常的"。

### 12.2 前向中 checkpoint 激活的累积机制

`gradient_checkpointing: True` + `use_reentrant=False` 的行为：

```
前向执行：block 0 → block 1 → … → block 39

每个 block 执行时：
  ① FSDP all_gather（聚合本层所有 Linear 权重）
  ② 执行 block_forward（FA + FFN）
  ③ FSDP reshard（free 权重，reshard_after_forward=True）
  ④ 将本 block 的输入激活（checkpoint tuple）保存在设备内存中
     大小约：N_local × dim × bf16 = 32256 × 5120 × 2 ≈ 330 MB / block / card
```

随着前向向后推进，**checkpoint 激活不断累积**，到 block k 时设备上保有 blocks 0..k-1 的激活：

```
block  4 时：  4 × 330 MB ≈  1.3 GB    （内存压力尚小）
block 16 时： 16 × 330 MB ≈  5.3 GB    （开始明显影响碎片率）
block 24 时： 24 × 330 MB ≈  7.9 GB    （接近峰值）
block 27 时： 27 × 330 MB ≈  8.9 GB    （碎片率超阈值 → defrag）
block 39 时： 39 × 330 MB ≈ 12.9 GB    （前向结束时最大）
```

### 12.3 为什么 checkpoint 积累会导致 HCCL all_gather 变慢，以及为什么慢的是"断续"的而不是连续的

#### 12.3.1 基础机制：大 buffer 分配失败

FSDP 的 all_gather 和 skiparse 的 all_to_all 都需要从 CANN 内存池申请连续的接收 buffer：

| 操作                                | buffer 大小估算                  | 频率（per block）          |
| ----------------------------------- | -------------------------------- | -------------------------- |
| FSDP all_gather（per linear）       | ~52 MB                           | ×7（attn QKV proj + FFN） |
| Skiparse `dist.all_to_all_single` | ~330 MB（完整 local seq × dim） | ×1                        |

当设备内存中存有大量 checkpoint 激活（分散存放，共占 5-9 GB），**可用连续内存块减少**，导致：

1. CANN 内存池找不到足够大的连续块
2. 触发内部碎片扫描（仅搜索，非 defrag）：耗时 0.5-5 ms
3. 在 profiler 中表现为 all_gather/all_to_all kernel 前有一段空白等待

#### 12.3.2 为什么是"间歇"慢，而不是 block 16 之后全都慢

> **你提的问题**：按照上面逻辑，block 16-17 慢了之后，18、19、20 应该都慢才对，为什么会好一阵子再卡住？

**根本原因：HCCL buffer 有 size-based 缓存策略，同大小的 buffer 会被复用。**

```
HCCL 内部维护一个"上次使用的 buffer"缓存（类似 slab allocator）：
  ① 若本次申请的 buffer size == 上次缓存的 size → 直接复用，0 等待
  ② 若 size 不同（或缓存已被占用）→ 向 CANN 内存池申请新块 → 可能遇到碎片

关键观察：
  - FSDP all_gather（52 MB, ×7/block）每次 size 相同 → 通常命中缓存
  - Skiparse all_to_all（~330 MB, ×1/block）size 与 FSDP buffer 不同
    → 每次 block 都需从内存池重新申请
    → 在内存碎片严重时，这个 330 MB 大块分配就可能失败/慢

因此：并不是"所有 all_gather 都变慢"，而是"特定大小的 buffer 分配偶发失败"。
```

同时，`reshard_after_forward=True` 在每个 block 执行后会 free 掉该层的权重分片（每个 linear ~52 MB × 7 = ~360 MB 被归还），这些刚释放的小块在部分情况下能被下一次分配复用，产生"暂时恢复"的窗口——然后随着 checkpoint 继续积累，下一次大 buffer 申请又可能失败。

**更准确的模型**：内存碎片升高使大 buffer（330 MB 量级）分配的**失败概率**升高，而不是让每次分配都失败。blocks 16-30 的区间内，有些 block 的 all_to_all 申请恰好找到了合适的大块（fast），有些没有（slow）。这就形成了"断续"的 slow 模式，而不是单调变慢。

> 坦白说：具体哪几个 block 会慢，取决于 CANN 内存池的实时状态（不完全可预测）。上述机制可以解释"为何中间层会出现断续的大 all_gather"，但无法精确预测是 block 16 还是 block 17 先出问题。下面 12.8 节给出验证方法。

### 12.4 各观察位置的具体解释

| 你观察到的位置           | 前向阶段对应情况       | 原因                                                                                                         |
| ------------------------ | ---------------------- | ------------------------------------------------------------------------------------------------------------ |
| block 16-17 之间         | 前向执行 block 17 开始 | 已积累 16 个 checkpoint（~5.3 GB），碎片率开始显著影响 HCCL buffer 分配                                      |
| block 21-22 之间         | 前向执行 block 22 开始 | 积累 21 个（~6.9 GB），分配等待进一步加长                                                                    |
| block 24-25 之间         | 前向执行 block 25 开始 | 积累 24 个（~7.9 GB），接近设备可用峰值                                                                      |
| block 27 后"超长 free"   | 前向 block 27-28 之间  | **CANN 内存池触发 defragmentation**，见 12.5 节                                                        |
| block 29-30 之间         | defrag 后首批大分配    | defrag 重建空闲链表后，新一轮 HCCL buffer 分配可能再次引起短暂等待                                           |
| blocks 30-35 "相对正常"  | 前向后半段 Skiparse    | defrag 后内存池干净，all_gather 有所恢复                                                                     |
| block 35→36 过渡 "正常" | Full↔Skiparse 过渡    | 无激活维度跨卡 all_gather（`use_context_parallel=False`），只有普通 FSDP all_gather；defrag 后内存状态较好 |
| 最后4层（36-39）"正常"   | 前向末尾 Full blocks   | Full blocks 无 skiparse all_to_all，通信事件少；defrag 后内存干净                                            |

### 12.5 "超长 free"是什么

```
前向执行到 block 27-28 之间，CANN 设备内存池触发"延迟批量回收"（Defragmentation）：

触发条件：
  设备内存池碎片率 > 阈值（CANN 内部，约 30-50%）
  或某次连续内存分配请求失败（OOM 前的最后补救）

执行过程：
  ① 暂停所有 CANN kernel 提交（profiler 看到空白，即"dead time"）
  ② 遍历内存池，合并相邻的空闲块（compact 碎片）
  ③ 将已释放但尚在池中缓存的内存块归还给 CANN base allocator
  ④ 重建空闲链表，使大连续块重新可用
  → 耗时 1-20 ms，在 profiler 中表现为一段没有任何 NPU kernel 的"超长 free"空白

类比：CPU 世界中 malloc 的 arena consolidation，或 Python 的 gc.collect()。
```

### 12.6 为什么是 block 16 开始而不是更早

blocks 0-15（前向前 1/3 段）对应的 checkpoint 积累量 < ~5 GB，设备总显存（每卡约 60 GB 可用）仍有充足连续空间，HCCL buffer 分配无障碍，all_gather 表现正常。

block 16 之后积累超过 5 GB，CANN 内存管理开始出现碎片问题。临界点因设备内存布局（FSDP 权重常驻部分、激活、通信 buffer 的交错分配）而触发，block 16 只是一个大致临界位置。

### 12.7 优化方向详解

#### 12.7.1 梯度累积（gradient_accumulation_steps > 1）如何减少 reduce_scatter

> **你的疑问**：不是一个前向 all_gather 对应一个反向 reduce_scatter 吗？梯度累积怎么减少 reduce_scatter？

你的理解是对的——在单次 forward+backward 中，每个 Linear 层确实产生 1 个 all_gather（前向聚合权重）和 1 个 reduce_scatter（反向分发梯度）。**梯度累积不减少单次 iteration 内的 all_gather，但它减少的是 reduce_scatter 在总训练时间中的频率**，机制如下：

```
不使用梯度累积（steps=1）：
  iteration 1: forward（all_gather×7×40）+ backward（reduce_scatter×7×40）
  iteration 2: forward + backward
  ...
  每次 backward 都运行 reduce_scatter（7×40=280 次/iteration）

使用梯度累积（steps=4）：
  microbatch 1: forward + backward（使用 model.no_sync()，跳过 reduce_scatter）
  microbatch 2: forward + backward（no_sync，跳过 reduce_scatter）
  microbatch 3: forward + backward（no_sync，跳过 reduce_scatter）
  microbatch 4: forward + backward（正常 backward，执行 reduce_scatter）
  ↑ 4 次 backward 只有 1 次 reduce_scatter，频率降为 1/4
```

FSDP2 通过 `model.no_sync()` context manager 实现：前 N-1 个 microbatch 的 backward 只计算梯度、在本地累加到 `.grad`，不做跨卡通信；第 N 个 microbatch 才触发 reduce_scatter 把累加梯度分发给其他卡。

**实际效果**：如果当前 reduce_scatter 占 48.4%，使用 steps=4 后理论上降至 ~12%（但前向 all_gather 不变，仍是每次 microbatch 都需要）。**代价**：每张卡需要额外保存 N 次 microbatch 的梯度张量（与模型参数同大小），显存开销增加 ~N 倍的梯度空间。14B 模型参数量大，steps=4 时梯度显存可能不可接受。

#### 12.7.2 Skiparse all_to_all 的开销来源

> **你的疑问**：single↔group 转换如何占用通信？"避免冗余 rearrange"怎么调整先说结论：**所谓"避免冗余"的表述在当前架构下几乎无法优化**，single↔group 的交替本身是 skiparse 设计的核心，不能省略。原文措辞过于乐观，以下给出正确理解。

**每次 all_to_all 在做什么（以 `_parallel_skiparse_2d_single_to_group` 为例）**：

```python
# skiparse_func.py 中的实现
def _parallel_skiparse_2d_single_to_group(x, grid_sizes, sparse_ratio, group, group_size):
    # Step 1: 本地 rearrange（纯 compute，ViewCopy）
    x = skiparse_2d_single(x, sub_grid_sizes, P)   # [G*b, T*sub_H*sub_W, C]

    # Step 2: all_to_all_single（跨卡通信）★
    x = contiguous(x).view(group_size, G, G*b, base, C)
    recv = torch.empty_like(x)                      # ← 这里申请 ~330 MB HCCL buffer
    dist.all_to_all_single(recv, x, group=group)    # ← HCCL 通信

    # Step 3: 本地 permute（纯 compute）
    recv = recv.permute(0, 2, 1, 3, 4, 5)

    # Step 4: 本地 rearrange reverse（纯 compute）
    x = skiparse_2d_single_reverse(x, sub_grid_sizes, P)
```

通信量：每次 all_to_all_single 传输张量大小 = G×b × T×sub_H×sub_W × C × bf16
≈ (4/4)×1 × 75600/4 × 5120 × 2 ≈ 330 MB（在 skiparse_cp_size=4 的卡组内）

**频率**：在 32 个 Skiparse blocks（block 4-35）中：

- block 4（第一个 Single）：只做本地 rearrange，**无 all_to_all**
- blocks 5, 6, 7, ..., 35（31 个 block）：每个都有 1 次 all_to_all_single
- 共 **31 次 all_to_all_single**，每次 ~330 MB，总计 ~10 GB 通信量（仅前向）

这 31 次 all_to_all 是架构必须的，不能省略。`explicit_prefetching_num_blocks` 预取的是 FSDP 权重，对 all_to_all 没有帮助。**真正的优化空间**在于确保这些 all_to_all 在 compute 完成后立即发起（减少 CPU 提交延迟），或者通过异步 all_to_all pipeline 与下一层计算重叠（需要改 osp_next.py 的调度逻辑，难度较高）。

#### 12.7.3 其他优化方向汇总

| 问题                                     | 优化方案                                                          | 现实可行性                                              |
| ---------------------------------------- | ----------------------------------------------------------------- | ------------------------------------------------------- |
| checkpoint 积累→内存碎片→all_gather 慢 | `explicit_prefetching_num_blocks: 1` 开启 FSDP 预取             | 高：改一个配置项，可掩盖 FSDP all_gather                |
| HCCL all_to_all buffer 分配失败          | 设置 `HCCL_BUFFSIZE` 或 CANN 内存池预留大块                     | 中：需要查 CANN 环境变量文档                            |
| checkpoint 积累                          | 每 2 层 checkpoint 一次（`checkpoint_activations_granularity`） | 中：需修改 `OSPNextAttentionBlock` 的 checkpoint 逻辑 |
| reduce_scatter 占 48.4%                  | `gradient_accumulation_steps: 4`（见 12.7.1 节）                | 低：显存代价大，14B 可能无法承受                        |

### 12.8 如何在 NPU 服务器上验证"内存碎片导致 all_gather 断续变慢"假设

> **你的问题**：假如上面的逻辑是对的，我该怎么在 NPU 服务器上验证？观察 profiling 的哪些特征？

#### 12.8.1 方法一：在 forward 中加入 block 级内存日志（最直接）

在 `OSPNextModel.forward()` 的 block 循环里插入一行日志：

```python
# torchdiff/modules/osp_next.py — OSPNextModel.forward() 的主循环
for i, block in enumerate(self.blocks):
    # ↓ 加在每个 block 前
    if dist.get_rank() == 0:
        alloc = torch.npu.memory_allocated() / 1024**3
        reserved = torch.npu.memory_reserved() / 1024**3
        frag = (reserved - alloc) / reserved * 100 if reserved > 0 else 0
        print(f"[block {i:02d}] alloc={alloc:.2f}GB reserved={reserved:.2f}GB frag={frag:.1f}%")
    x = block(x, ...)
```

**验证标准**：

- 若"慢 block"（16、21、24、27、29）处 `frag%` 明显高于"快 block"（18、19、20）→ 碎片假设成立
- 若 `frag%` 全程均匀增长没有明显波动 → 碎片不是主因，可能是 HCCL 流调度或 NPU 热节流

#### 12.8.2 方法二：CANN Profiler 开启内存事件（精确但数据量大）

```python
# 在训练代码的 profiler 配置中加入 profile_memory=True
with torch_npu.profiler.profile(
    activities=[torch_npu.profiler.ProfilerActivity.NPU],
    profile_memory=True,          # ← 记录每次 malloc/free 的时间戳
    with_stack=False,
) as prof:
    model_step()

prof.export_chrome_trace("trace.json")
```

在 Chrome trace viewer（`chrome://tracing`）中，内存 malloc/free 事件会与 HCCL kernel 事件出现在同一时间轴。**观察**：慢 all_gather 前是否有大量连续的 malloc 尝试（表现为多个短暂的 `malloc` 条目密集出现后才有 HCCL kernel 开始）。

#### 12.8.3 方法三：消融实验——关闭 checkpoint 或减少层数（最强证据）

**实验 A：禁用 gradient_checkpointing**

```yaml
# configs/train/npu/osp_14b.yaml
gradient_checkpointing: False   # 改为 False
```

理论预测：

- 前向中不再累积 checkpoint 激活 → 内存碎片消失
- blocks 16-30 的断续大 all_gather 应消失或显著缩短
- 代价：显存大幅增加（可能 OOM）；可以先在 1.3B 版本上验证

**实验 B：只对后半段层开启 checkpoint**

在 `OSPNextAttentionBlock.forward()` 中，修改 checkpoint 触发条件，让 blocks 0-19 不做 checkpoint：

```python
use_ckpt = self.gradient_checkpointing and (self.block_idx >= 20)
if use_ckpt:
    x = checkpoint(self._block_forward, x, ...)
else:
    x = self._block_forward(x, ...)
```

理论预测：前向到 block 20 时只有 0 个 checkpoint 激活（而非 20 个），碎片率低 → 慢 all_gather 的起始点应从 block 16 推迟到 block 36 附近。

#### 12.8.4 方法四：观察 npu-smi 中的内存使用曲线

哈哈 在另一个终端实时观察：

```bash
watch -n 0.5 npu-smi info  # 每 0.5 秒刷新
# 或
npu-smi info -t usages -i 0 -c 0   # 指定卡号和 chip
```

**验证标准**：

- 在训练 forward 阶段，内存占用应呈单调上升趋势（checkpoint 累积）
- 在 block 27 左右应出现一次"内存下降后回升"或"占用短暂不变但 reserved 不变"事件（对应 defrag）
- 注意 `npu-smi` 的时间分辨率较低（0.5s 级别），只能看到整体趋势

#### 12.8.5 方法五：固定 HCCL buffer（最直接消除等待）

设置 CANN 环境变量来预分配固定 HCCL buffer，使 HCCL 不再每次动态申请：

```bash
export HCCL_BUFFSIZE=512        # 单位 MB，设为大于 330 的值
export HCCL_ALGO=level0:fullmesh   # 可选：调整 HCCL 通信算法
```

**验证标准**：设置后若 profiler 中 blocks 16-30 的大 all_gather 消失/缩短 → 确认是 HCCL buffer 分配导致的延迟。若无变化 → 问题在别处（通信带宽不足或 CANN 调度）。

#### 12.8.6 推荐验证顺序

```
1. 先做「方法一」（加一行内存日志）：成本最低，30 分钟内出结果
   → 若碎片率与慢 block 相关：继续做方法五（HCCL_BUFFSIZE）
   → 若碎片率不相关：重新考虑其他假设（NPU 热节流？HCCL 流调度？）

2. 同时做「方法三 A」（禁用 checkpoint）验证因果关系，在 1.3B 上做即可

3. 若确认碎片是根因：长期方案选择方法三 B（selective checkpoint）或 HCCL buffer 预分配
```
