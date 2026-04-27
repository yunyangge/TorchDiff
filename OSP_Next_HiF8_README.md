# OSP-Next-14B HiFloat8 Inference on NPU

## 概述

OSP-Next 是基于 [Wan2.1](https://github.com/Wan-Video/Wan2.1) 文生视频扩散模型改造的新一代视频生成架构，核心创新为 **Skiparse（稀疏跳跃注意力）** 机制：将标准全量 Attention 替换为交替 Single / Group 稀疏注意力块，在保持视频质量的前提下大幅降低序列计算量。

本样例提供的 OSP-Next-14B 权重经过 **HiFloat8（HIF8）量化感知训练（QAT）**。由于 Atlas A2 系列（Ascend 910B）硬件本身不原生支持 HIF8 数据格式，训练阶段在 Atlas A2 上以 bf16 进行分布式训练，同时借助 `quant_cy_npu` 仿真算子对所有 Linear 投影层（Q/K/V/O/FFN）及 Attention Q/K/V 输入施加 HIF8 精度仿真，使权重收敛至 HIF8 精度范围内。Ascend 950 A5 原生支持 HIF8 数据格式，所得权重可在 A5 上直接进行原生 HIF8 量化推理，无需任何格式转换。

- 本样例的并行策略与性能优化详情可参见 [TODO: 填写技术文档或论文链接]()。

## 硬件要求

- 产品型号：Ascend 950 A5（单卡即可运行 14B 推理）
- 操作系统：Linux ARM
- 镜像版本：TODO: 填写 docker 镜像名称和版本（如 `cann8.0_pt2.8.0_aarch_image:v0.x`）
- 驱动版本：Ascend HDK TODO（运行 `npu-smi info` 确认当前版本）

> 使用 `npu-smi info` 检查 Ascend NPU 固件和驱动是否正常安装。如果未安装或者版本不是要求的版本，请下载[固件和驱动](TODO: 填写驱动下载链接)，然后根据指引自行安装。

## 快速启动

### 下载源码

在各节点上执行如下命令下载 cann-recipes-infer 源码：

```bash
mkdir -p /home/code; cd /home/code/
git clone https://gitcode.com/cann/cann-recipes-infer.git
cd cann-recipes-infer
```

### 下载权重

从 [TODO: 填写权重下载地址（ModelScope/HuggingFace）]() 下载 OSP-Next-14B HIF8 完整权重包，并上传到各节点某个固定的路径下，比如 `/data/models/osp-next-14b-hif8`。

下载完成后，各节点目录结构参考如下：

```
/data/models/osp-next-14b-hif8/
├── ema_model_state_dict.pt            # DiT 主模型权重（HIF8 QAT 训练）
├── Wan2.1_VAE.pth                     # VAE 权重
├── models_t5_umt5-xxl-enc-bf16.pth   # T5 文本编码器权重
└── google/
    └── umt5-xxl/                      # T5 tokenizer
```

### 获取 docker 镜像

从 [ARM 镜像地址](TODO: 填写镜像下载链接) 下载 docker 镜像，然后通过如下命令将镜像导入到 A5 服务器的每个节点上：

```bash
docker load -i TODO_镜像文件名.tar
```

### 拉起 docker 容器

通过如下脚本拉起容器；默认容器名为 `cann_recipes_infer`，注意：需要将权重路径和源码路径挂载到容器中。OSP-Next-14B 在单张 Ascend 950 A5 上即可完成推理，无需多卡。

```bash
docker run -u root -itd \
    --name cann_recipes_infer \
    --ulimit nproc=65535:65535 \
    --ipc=host \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /data/models:/data/models \
    -v /home/code/cann-recipes-infer:/home/code/cann-recipes-infer \
    --shm-size=128g \
    --privileged \
    TODO_镜像名称 /bin/bash
```

在各节点上通过如下命令进入容器：

```bash
docker attach cann_recipes_infer
cd /home/code/cann-recipes-infer/models/osp-next-14b-hif8
```

### 修改代码

修改 `config/osp_next_14b_hif8.yaml` 中的权重路径（对应上一步下载的目录）：

```yaml
# 单卡推理，所有并行度设为 1
fsdp_size: 1
cp_size: 1
skiparse_cp_size: 1
use_context_parallel: False
use_skiparse_context_parallel: False

model_config:
  pretrained_model_dir_or_checkpoint: "/data/models/osp-next-14b-hif8/ema_model_state_dict.pt"

vae_config:
  vae_path: "/data/models/osp-next-14b-hif8/Wan2.1_VAE.pth"

text_encoder_config:
  checkpoint_path: "/data/models/osp-next-14b-hif8/models_t5_umt5-xxl-enc-bf16.pth"
  text_tokenizer_path: "/data/models/osp-next-14b-hif8/google/umt5-xxl"
```

修改推理提示词文件 `prompts.txt`（每行一条文生视频描述）：

```
A majestic eagle soaring over snow-capped mountains at sunrise, cinematic 4K.
Close-up of ocean waves crashing against rocky cliffs, slow motion, golden hour.
```

### 拉起推理

在各节点上同步执行如下命令即可拉起推理任务：

```bash
bash infer.sh
```

## 附录

### YAML 配置参数说明

| 参数                    | 说明                                      | 默认值                        |
| ----------------------- | ----------------------------------------- | ----------------------------- |
| `prompt_txt`          | 提示词文件路径，每行一条描述              | `prompts.txt`               |
| `output_dir`          | 生成视频保存目录                          | `samples/osp_next_14b_hif8` |
| `num_frames`          | 生成帧数                                  | `81`（约 5 秒 @16fps）      |
| `height` / `width`  | 输出分辨率                                | `720` / `1280`（720p）    |
| `num_inference_steps` | 扩散采样步数                              | `50`                        |
| `guidance_scale`      | 分类器自由引导强度                        | `5.0`                       |
| `quant`               | Linear 层量化类型，`"hif8"` 启用 HIF8   | `"hif8"`                    |
| `quant_attn`          | Attention Q/K/V 量化类型，`"hif8"` 启用 | `"hif8"`                    |
| `scale_max_forward`   | HIF8 前向激活/权重缩放上界                | `15.0`                      |

### HiFloat8 量化覆盖范围

每个 OSPNextAttentionBlock（共 40 层）中，以下计算节点被 HIF8 精度约束覆盖：

| 子模块                                 | 量化类型                   | 数量/层 |
| -------------------------------------- | -------------------------- | ------- |
| Self-Attention Q/K/V/O 投影（Linear）  | 激活 × 权重双量化         | 4       |
| Cross-Attention Q/K/V/O 投影（Linear） | 激活 × 权重双量化         | 4       |
| FFN Linear1 / Linear2                  | 激活 × 权重双量化         | 2       |
| Attention Q/K/V → SDPA 输入           | per-tensor Current Scaling | 每层    |

> **说明**：HIF8 采用 Current Scaling 策略（`scale_max_forward=15`），即每次前向实时计算 per-tensor 缩放因子，无需预热步骤，在 NPU 上通过 `quant_cy_npu` 算子实现。

### FAQ

- **`ImportError: quant_cy_npu`**：HiFloat8 NPU 算子包未编译安装。请参考 [TODO: 补充 quant_cy_npu 编译安装说明链接]() 完成编译（`bash build.sh`）。若仅希望验证视频生成效果而不使用 HIF8，可将 YAML 中 `quant` 和 `quant_attn` 均设为 `null`，此时退回标准 bf16 推理。
- **`HCCL_BUFFSIZE` 不足问题**：如果报错日志中出现关键字 `"HCCL_BUFFSIZE is too SMALL, ..., NEEDED_HCCL_BUFFSIZE=..., HCCL_BUFFSIZE=200MB, ..."`，可通过配置环境变量 `export HCCL_BUFFSIZE=实际需要的大小` 解决，所有 Rank 上该环境变量需保持一致。HCCL_BUFFSIZE 参数介绍可参考[昇腾资料](TODO: 填写链接)中的详细描述。
- **显存不足（OOM）**：OSP-Next-14B 设计为在单张 Ascend 950 A5 上运行，**无需多卡**。若遇到 OOM，请首先确认使用的是 A5 而非显存较小的其他型号；其次检查 `fsdp_size`、`cp_size`、`skiparse_cp_size` 是否均已设为 `1`，`use_skiparse_context_parallel` 是否为 `False`。
- **自定义算子导入失败**：如果报错日志中出现类似关键字 `"_OpNamespace 'custom' object has no attribute"`，可参考[自定义算子指南](TODO: 填写链接)编译所需算子。
