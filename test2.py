import os
import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.device_mesh import init_device_mesh
from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env

if __name__ == "__main__":

    setup_distributed_env()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = torch.float32

    samples = torch.arange(0, 16, dtype=weight_dtype, device=device).reshape(2, 4, 2)
    if rank == 0:
        print(f"original samples: {samples}")
    device_mesh = init_device_mesh("cuda", (2, 2, 2), mesh_dim_names=("dp", "cp", "ccp"))
    cp_mesh = device_mesh
    samples = DTensor.from_local(samples, device_mesh=cp_mesh, placements=(Replicate(), Replicate()))
    samples = samples.redistribute(device_mesh=cp_mesh, placements=(Shard(1), Shard(2))).to_local()
    print(f"{'=' * 10}After Sharding{'=' * 10}")
    print(f"rank={rank}, samples={samples}")
    print(f"{'=' * (10 + len('After Sharding') + 10)}")
    samples = DTensor.from_local(samples, device_mesh=cp_mesh, placements=(Shard(1), Shard(2)))
    samples = samples.redistribute(device_mesh=cp_mesh, placements=(Replicate(), Replicate())).to_local()
    print(f"{'=' * 10}After Gather{'=' * 10}")
    print(f"rank={rank}, samples={samples}")
    print(f"{'=' * (10 + len('After Gather') + 10)}")
    cleanup_distributed_env()