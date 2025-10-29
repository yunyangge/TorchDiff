export WANDB_MODE="online"
export WANDB_API_KEY="720d886d8c437c2142c88056a1eab8ef78d64a1f"
wandb login $WANDB_API_KEY


export TOKENIZERS_PARALLELISM=false

# export NCCL_IB_TC=136
# export NCCL_IB_SL=5
# export NCCL_IB_GID_INDEX=3
# export NCCL_SOCKET_IFNAME=eth
# export NCCL_IB_HCA=mlx5
# export NCCL_IB_TIMEOUT=22
# export NCCL_IB_QPS_PER_CONNECTION=8
# export NCCL_NET_PLUGIN=none
# export NCCL_DEBUG=WARN

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
NPRC_PER_NODE=${NPRC_PER_NODE:-8}
NNODES=${NNODES:-1}
WORLD_SIZE=$(($NNODES * $NPRC_PER_NODE))

torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train.py \
  --config configs/train/npu/test_train_fsdp_npu.yaml