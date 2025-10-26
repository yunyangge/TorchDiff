import torch
import logging
from torch.distributed.fsdp import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    fully_shard,
)

def FSDP2_mix_warpper(
    model,
    dp_mesh=None,
    weight_dtype=torch.bfloat16,
    main_block_to_half=None,
    blocks_to_float=None,
    reshard_after_forward=None,
    cpu_offload=False,
):
    is_rank_zero = torch.distributed.get_rank() == 0
    if is_rank_zero:
        logging.info("Parallelize Module with FSDP2...")
    low_precision_policy = MixedPrecisionPolicy(
        param_dtype=weight_dtype,
        reduce_dtype=torch.float32,
        output_dtype=weight_dtype,
    )
    high_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=weight_dtype,
    )

    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if blocks_to_float is not None and len(blocks_to_float) > 0:
        for module in model.modules():
            for block in blocks_to_float:
                if isinstance(module, block):
                    if is_rank_zero:
                        logging.info(f"FSDP {block} Module with High Precision.")
                    fully_shard(module, mp_policy=high_precision_policy, **fsdp_kwargs)
    if main_block_to_half is not None:
        for module in model.modules():
            if isinstance(module, main_block_to_half):
                if is_rank_zero:
                    logging.info(f"FSDP {main_block_to_half} Module with Low Precision.")
                fully_shard(module, mp_policy=low_precision_policy, **fsdp_kwargs)
    if is_rank_zero:
        logging.info(f"FSDP Other Modules.")
    fully_shard(model, mp_policy=low_precision_policy, **fsdp_kwargs)
    if is_rank_zero:
        logging.info("FSDP Down!")


def FSDP2_fp32_warpper(
    model,
    dp_mesh=None,
    main_block=None,
    reshard_after_forward=None,
    cpu_offload=False,
):
    fp32_precision_policy = MixedPrecisionPolicy(
        param_dtype=torch.float32,
        reduce_dtype=torch.float32,
        output_dtype=torch.float32,
    )
    fsdp_kwargs = {
        "reshard_after_forward": reshard_after_forward,
        "mesh": dp_mesh,
    }  # dp_mesh is None means distributed to all nodes.

    if cpu_offload:
        fsdp_kwargs["offload_policy"] = CPUOffloadPolicy()

    if main_block is not None:
        for module in model.modules():
            if isinstance(module, main_block):
                fully_shard(module, mp_policy=fp32_precision_policy, **fsdp_kwargs)
    fully_shard(model, mp_policy=fp32_precision_policy, **fsdp_kwargs)

if __name__ == "__main__":
    from torch.distributed.device_mesh import init_device_mesh
    from ultrai2v.modules.model import wan_model, wan_model_blocks_to_float, wan_model_main_block
    from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env
    
    setup_distributed_env()

    ddp_fsdp_mesh = init_device_mesh(
        "cuda",
        (8, 1),
        mesh_dim_names=("ddp", "fsdp"),
    )
    print("ddp_fsdp_mesh:", ddp_fsdp_mesh)

    model_name = "wan_t2v"
    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    dtype = torch.bfloat16

    latents = torch.randn(1, 16, 16, 64, 64).to(device=device, dtype=dtype)
    text_embeddings = torch.randn(1, 512, 4096).to(device=device, dtype=dtype)
    timesteps = torch.randint(0, 1000, (1,)).to(device=device)

    ddp_model = wan_model[model_name]().to(device=device, dtype=dtype)
    ddp_model = torch.nn.parallel.DistributedDataParallel(ddp_model)

    fsdp_model = wan_model[model_name]()
    ddp_fsdp_model = wan_model[model_name]()

    FSDP2_mix_warpper(
        ddp_fsdp_model,
        dp_mesh=ddp_fsdp_mesh,
        weight_dtype=dtype,
        main_block_to_half=wan_model_main_block[model_name],
        blocks_to_float=wan_model_blocks_to_float[model_name],
        reshard_after_forward=True,
        cpu_offload=False,
    )

    FSDP2_mix_warpper(
        fsdp_model,
        dp_mesh=None,
        weight_dtype=dtype,
        main_block_to_half=wan_model_main_block[model_name],
        blocks_to_float=wan_model_blocks_to_float[model_name],
        reshard_after_forward=True,
        cpu_offload=False,
    )
    with torch.no_grad():
        with torch.autocast(device_type='cuda', dtype=dtype):
            fsdp_output = fsdp_model(latents, timesteps, text_embeddings)
            ddp_output = ddp_model(latents, timesteps, text_embeddings)
            ddp_fsdp_output = ddp_fsdp_model(latents, timesteps, text_embeddings)
    print("fsdp_output - ddp_output MSE:", 1000 * torch.mean((fsdp_output.float() - ddp_output.float()) ** 2))
    print("fsdp_output - ddp_fsdp_output MSE:", 1000 * torch.mean((fsdp_output.float() - ddp_fsdp_output.float()) ** 2))
    print("ddp_output - ddp_fsdp_output MSE:", 1000 * torch.mean((ddp_output.float() - ddp_fsdp_output.float()) ** 2))

    cleanup_distributed_env()