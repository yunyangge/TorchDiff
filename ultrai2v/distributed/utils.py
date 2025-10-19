import os
import logging
import torch
import torch.distributed as dist
from datetime import timedelta
from torch.distributed.tensor import DTensor, Replicate, Shard
def setup_distributed_env(backend: str = "nccl", timeout: int = 300):
    """ Initialize distributed environment. """
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)

def cleanup_distributed_env():
    """ Clean up distributed environment. """
    dist.destroy_process_group()
   
def set_modules_to_forward_prefetch(main_block_list, num_to_forward_prefetch=2):
    for i, block in enumerate(main_block_list):
        if i >= len(main_block_list) - num_to_forward_prefetch:
            break
        blocks_to_prefetch = [
            main_block_list[i + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        block.set_modules_to_forward_prefetch(blocks_to_prefetch)


def set_modules_to_backward_prefetch(main_block_list, num_to_backward_prefetch=2):
    for i, block in enumerate(main_block_list):
        if i < num_to_backward_prefetch:
            continue
        blocks_to_prefetch = [
            main_block_list[i - j] for j in range(1, num_to_backward_prefetch + 1)
        ]
        block.set_modules_to_backward_prefetch(blocks_to_prefetch)

def gather_data_from_all_ranks(data, dim=0):
    """ gather data from all ranks, return a tensor with data from all ranks """
    world_size = dist.get_world_size()
    if world_size == 1:
        return data
    gather_list = [torch.zeros_like(data) for _ in range(world_size)]
    dist.all_gather(gather_list, data)
    return torch.stack(gather_list, dim=dim)

def all_gather_tensor(tensor, dim=0, device_mesh=None):
    tensor = DTensor.from_local(
        tensor,
        device_mesh=device_mesh,
        placements=(Shard(dim),)
    )
