#!/bin/bash
# 批量测试多个 LoRA checkpoint 的推理脚本
# 遇到错误立即停止执行后续命令（按需开启）
set -e

pkill -9 -f infer.py || true  # 加上 || true 防止没有进程可杀时脚本直接退出
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0
export ACL_DEVICE_SYNC_TIMEOUT=3600
export ASCEND_LAUNCH_BLOCKING=1

# ======== 分布式环境变量：兼容 ModelArts 训练作业自动注入 ========
# ModelArts 不会直接给 NODE_RANK / MASTER_ADDR 这种 torchrun 原生名字，
# 需要从 MA_* / VC_* / VC_WORKER_HOSTS 等平台变量派生。
# 若本地/手动启动，可直接通过 NODE_RANK=xx MASTER_ADDR=yy 方式覆盖。

# 1) 总节点数 NNODES
if [ -z "${NNODES}" ]; then
    if [ -n "${MA_NUM_HOSTS}" ]; then
        NNODES=${MA_NUM_HOSTS}
    elif [ -n "${VC_WORKER_NUM}" ]; then
        NNODES=${VC_WORKER_NUM}
    else
        NNODES=1
    fi
fi

# 2) 当前节点编号 NODE_RANK
if [ -z "${NODE_RANK}" ]; then
    if [ -n "${VC_TASK_INDEX}" ]; then
        NODE_RANK=${VC_TASK_INDEX}
    elif [ -n "${MA_TASK_INDEX}" ]; then
        NODE_RANK=${MA_TASK_INDEX}
    elif [ -n "${VK_TASK_INDEX}" ]; then
        NODE_RANK=${VK_TASK_INDEX}
    elif [ -n "${RANK}" ] && [ -z "${LOCAL_RANK}" ]; then
        # 某些老镜像用 RANK 表示节点编号（注意不要和 torchrun 的 RANK 冲突）
        NODE_RANK=${RANK}
    else
        NODE_RANK=0
    fi
fi

# 3) 主节点地址 MASTER_ADDR
#    ModelArts 常见：VC_WORKER_HOSTS="worker-0.xxx,worker-1.xxx,..."
if [ -z "${MASTER_ADDR}" ]; then
    if [ -n "${VC_WORKER_HOSTS}" ]; then
        MASTER_ADDR=$(echo "${VC_WORKER_HOSTS}" | cut -d',' -f1)
    elif [ -n "${MA_VJ_NAME}" ] && [ -n "${VC_TASK_INDEX}" ]; then
        # 兜底：构造 worker-0 的 hostname
        MASTER_ADDR="${MA_VJ_NAME}-worker-0"
    else
        MASTER_ADDR="127.0.0.1"
    fi
fi

# 4) 主节点端口
MASTER_PORT=${MASTER_PORT:-29505}

# 5) 单机进程数（NPU/卡数）
if [ -z "${NPRC_PER_NODE}" ]; then
    if [ -n "${MA_NUM_GPUS}" ]; then
        NPRC_PER_NODE=${MA_NUM_GPUS}
    elif [ -n "${NPU_NUM}" ]; then
        NPRC_PER_NODE=${NPU_NUM}
    else
        NPRC_PER_NODE=16
    fi
fi

# 是否为主节点：只有 NODE_RANK=0 负责生成临时 yaml / 打印列表
IS_MAIN_NODE=0
if [ "${NODE_RANK}" = "0" ]; then
    IS_MAIN_NODE=1
fi

echo "[Dist] NNODES=${NNODES}  NODE_RANK=${NODE_RANK}  NPRC_PER_NODE=${NPRC_PER_NODE}"
echo "[Dist] MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

# LoRA checkpoint 根目录
LORA_ROOT="/home/ma-user/work/xianyi/osp_next/TorchDiff/output/osp_next_14b_lr1e_5_81f720p_sparse2d2_sp2_ssp4_grpo4_node7_lr1e04_bf16"

# 仅测试 step >= MIN_STEP 的 checkpoint（默认 32，即从 step-32 开始，28 及之前跳过）
# 如需覆盖，可通过 MIN_STEP=xx bash ... 传入；设为 0 表示全部跑。
MIN_STEP=${MIN_STEP:-0}

# 基础 yaml 配置文件
BASE_CONFIG="configs/infer/npu/osp_14b_rl.yaml"

# 输出根目录（step 子目录会动态拼接）
OUTPUT_ROOT="samples/osp_next_mixgrpo/moviegen/lr104_bf16_2"

# 临时生成 yaml 的目录
TMP_CONFIG_DIR="configs/infer/npu/_sweep_tmp"
mkdir -p "${TMP_CONFIG_DIR}"

# 日志目录
LOG_DIR="logs/sweep_rl"
mkdir -p "${LOG_DIR}"

# 收集所有 lora-checkpoint-* 目录，按 step 数字升序排序
mapfile -t CKPT_DIRS < <(find "${LORA_ROOT}" -maxdepth 1 -mindepth 1 -type d -name 'lora-checkpoint-*' \
    | awk -F'lora-checkpoint-' '{print $2"\t"$0}' \
    | sort -n -k1,1 \
    | cut -f2-)

if [ ${#CKPT_DIRS[@]} -eq 0 ]; then
    echo "未在 ${LORA_ROOT} 下找到任何 lora-checkpoint-* 目录，退出。"
    exit 1
fi

echo "[NODE_RANK=${NODE_RANK}] 共找到 ${#CKPT_DIRS[@]} 个 LoRA checkpoint 待测试："
if [ "${IS_MAIN_NODE}" = "1" ]; then
    for d in "${CKPT_DIRS[@]}"; do
        echo "  - ${d}"
    done
fi

TASK_IDX=0
for CKPT_DIR in "${CKPT_DIRS[@]}"; do
    TASK_IDX=$((TASK_IDX + 1))

    # 从目录名中提取 step 号，如 lora-checkpoint-28 -> 28
    STEP=$(basename "${CKPT_DIR}" | sed -E 's/^lora-checkpoint-//')

    # 过滤掉小于 MIN_STEP 的 checkpoint
    # 用 10 进制强制展开，避免 step 带前导 0 被当成八进制
    if [ $((10#${STEP})) -lt $((10#${MIN_STEP})) ]; then
        echo "[跳过] step${STEP} < MIN_STEP=${MIN_STEP}"
        continue
    fi

    LORA_PATH="${CKPT_DIR}/adapter_model.bin"
    if [ ! -f "${LORA_PATH}" ]; then
        echo "[跳过] 未找到 ${LORA_PATH}"
        continue
    fi

    OUTPUT_DIR="${OUTPUT_ROOT}/step${STEP}"
    TMP_CONFIG="${TMP_CONFIG_DIR}/osp_14b_rl_step${STEP}.yaml"
    # 日志文件按 NODE_RANK 区分，避免多机 tee 写同一个文件互相覆盖
    LOG_FILE="${LOG_DIR}/task_step${STEP}_node${NODE_RANK}.log"

    # 基于基础 yaml 生成本次运行的 yaml：替换 lora_path 和 output_dir
    # 使用 python 来安全地修改 yaml，避免 sed 引号转义问题
    # 仅由主节点写入临时 yaml，其它节点等待文件就绪，避免多机同时写共享盘造成竞争
    if [ "${IS_MAIN_NODE}" = "1" ]; then
        python - <<PY_EOF
import yaml
with open("${BASE_CONFIG}", "r") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)
cfg["lora_path"] = "${LORA_PATH}"
cfg["output_dir"] = "${OUTPUT_DIR}"
with open("${TMP_CONFIG}", "w") as f:
    yaml.dump(cfg, f, indent=4, sort_keys=False)
PY_EOF
    else
        # 非主节点等待主节点生成 yaml（假定 TMP_CONFIG_DIR 是各节点共享/已同步的路径）
        for _ in $(seq 1 120); do
            [ -f "${TMP_CONFIG}" ] && break
            sleep 1
        done
        if [ ! -f "${TMP_CONFIG}" ]; then
            echo "[NODE_RANK=${NODE_RANK}] 等待 ${TMP_CONFIG} 超时，退出。"
            exit 1
        fi
    fi

    echo "========== [NODE_RANK=${NODE_RANK}] 开始执行任务 ${TASK_IDX}: step${STEP} =========="
    echo "  LoRA:   ${LORA_PATH}"
    echo "  Output: ${OUTPUT_DIR}"
    echo "  Config: ${TMP_CONFIG}"
    echo "  Log:    ${LOG_FILE}"

    # 每个任务前清理残留的 infer 进程，避免端口/显存占用
    pkill -9 -f infer.py || true
    sleep 2

    torchrun \
        --nproc_per_node=${NPRC_PER_NODE} \
        --nnodes=${NNODES} \
        --node_rank=${NODE_RANK} \
        --master_addr=${MASTER_ADDR} \
        --master_port=${MASTER_PORT} \
        infer/infer_osp.py \
        --config "${TMP_CONFIG}" 2>&1 | tee "${LOG_FILE}"

    echo "========== 任务 ${TASK_IDX} (step${STEP}) 执行完毕 =========="
done

echo "========== 所有 ${#CKPT_DIRS[@]} 个 LoRA checkpoint 任务执行完毕 =========="
