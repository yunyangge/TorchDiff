import os
import logging
import torch
import torch.distributed as dist
from datetime import timedelta
from torch.distributed.tensor import DTensor, Replicate, Shard
from typing import Iterable
from ultrai2v.utils.utils import str_to_precision, precision_to_str

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

def gather_data_from_all_ranks(data, dim=0, group=None):
    """ gather data from all ranks, return a tensor with data from all ranks """
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return data
    gather_list = [torch.empty_like(data) for _ in range(world_size)]
    dist.all_gather(gather_list, data, group=group)
    return torch.stack(gather_list, dim=dim)

def broadcast_tensor_list(tensors, group_src=0, group=None):
    """ Broadcast a list of tensors from source rank to all other ranks. """
    if group is None:
        group = dist.group.WORLD
    group_rank = dist.get_rank(group)
    # broadcast tensor list length
    if group_rank == group_src:
        n = torch.tensor(len(tensors), device=tensors[0].device, dtype=torch.int)
    else:
        n = torch.tensor(0, device=torch.device("cuda"), dtype=torch.int)
    dist.broadcast(n, group=group, group_src=group_src)
    n = int(n.item())

    tensors = tensors if group_rank == group_src else [None] * n

    for i in range(n):
        # broadcast tensor ndim
        if group_rank == group_src:
            ndim = torch.tensor(len(tensors[i].shape), device=tensors[i].device, dtype=torch.int)
        else:
            ndim = torch.tensor(0, device="cuda", dtype=torch.int)
        dist.broadcast(ndim, group=group, group_src=group_src)
        ndim = int(ndim.item())

        # broadcast tensor shape
        if group_rank == group_src:
            shape = torch.tensor(tensors[i].shape, device=tensors[i].device, dtype=torch.int)
        else:
            shape = torch.empty((ndim, ), device="cuda", dtype=torch.int)
        dist.broadcast(shape, group=group, group_src=group_src)
        shape = tuple(shape.tolist())

        # broadcast tensor dtype
        if group_rank == group_src:
            dtype_str = [precision_to_str(tensors[i].dtype)]
        else:
            dtype_str = [None]
        dist.broadcast_object_list(dtype_str, group=group, group_src=group_src)

        # broadcast tensor data
        dtype = str_to_precision(dtype_str[0])
        if group_rank != group_src:
            tensors[i] = torch.empty(shape, device="cuda", dtype=dtype)
        dist.broadcast(tensors[i], group=group, group_src=group_src)

    return tensors