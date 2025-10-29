import os
import yaml
import torch
from argparse import ArgumentParser

from torch.distributed.device_mesh import init_device_mesh

from ultrai2v.distributed.utils import setup_distributed_env, cleanup_distributed_env
from ultrai2v.modules import (
    WanVAE, 
    T5EncoderModel, 
    models, 
    models_main_block, 
    models_blocks_to_float,
    models_cp_plan,
)
from ultrai2v.schedulers import schedulers
from ultrai2v.distributed.checkpoint import Checkpointer
from ultrai2v.utils.utils import str_to_precision, get_memory_allocated
from ultrai2v.utils.log_utils import get_logger, log_on_main_process
from ultrai2v.pipelines import pipelines

def main(config):
    logger = get_logger()

    # config analysis
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model", "wan_t2v")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})

    # inference config
    pipeline_name = config.get("pipeline_name", "t2v")
    prompt_txt = config.get("prompt_txt", None)
    num_frames = config.get("num_frames", 49)
    height = config.get("height", 480)
    width = config.get("width", 832)

    # save config
    output_dir = config.get("output_dir", "./output")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

    # distributed setup
    setup_distributed_env()
    
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}")
    weight_dtype = str_to_precision(weight_dtype)

    # init fsdp config
    fsdp_size = config.get("fsdp_size", 8)
    ddp_size = config.get("ddp_size", world_size // fsdp_size)
    ddp_fsdp_mesh = init_device_mesh("cuda", (ddp_size, fsdp_size), mesh_dim_names=("ddp", "fsdp"))
    logger.info(f"rank {rank} use ddp mesh {ddp_fsdp_mesh['ddp']} and fsdp mesh {ddp_fsdp_mesh['fsdp']}")

    # init cp mesh if use context parallel
    cp_size = 1
    use_context_parallel = use_context_parallel and config.get("cp_size", 1) > 1
    if use_context_parallel:
        # cp size == model parallel (FSDP) size
        cp_size = config.get("cp_size", fsdp_size)
        if cp_size == fsdp_size:
            cp_mesh = ddp_fsdp_mesh["fsdp"]
            dp_group = ddp_fsdp_mesh["ddp"].get_group()
        # cp size != model parallel (FSDP) size
        else:
            dp_cp_mesh = init_device_mesh("cuda", (world_size // cp_size, cp_size), mesh_dim_names=("dp", "cp"))
            cp_mesh = dp_cp_mesh["cp"]
            dp_group = dp_cp_mesh["dp"].get_group()
        log_on_main_process(logger, f"We use context parallel, cp_size: {cp_size}")
        logger.info(f"rank {rank} use cp mesh {cp_mesh}")

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)

    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device # for vae, we do not use fsdp
    )
    log_on_main_process(logger, f"VAE model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing text encoder model...")
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device, # when no fsdp, we init the text_encoder on device
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False), # when using fsdp, we shard the text encoder by ddp_fsdp mesh
        device_mesh=ddp_fsdp_mesh if text_encoder_config.get("use_fsdp", False) else None,
    )
    log_on_main_process(logger, f"Text encoder model initialized, memory allocated: {get_memory_allocated()} GiB")

    log_on_main_process(logger, "Initializing diffusion model and scheduler...")

    scheduler = schedulers[scheduler_config.pop("scheduler_name", "flow_matching")](**scheduler_config)

    pretrained_model_dir_or_checkpoint = model_config.get("pretrained_model_dir_or_checkpoint", None)
    if pretrained_model_dir_or_checkpoint is not None and os.path.isdir(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_dir {pretrained_model_dir_or_checkpoint}")
        model = models[model_name].from_pretrained(pretrained_model_dir_or_checkpoint)
    elif pretrained_model_dir_or_checkpoint is not None and os.path.isfile(pretrained_model_dir_or_checkpoint):
        log_on_main_process(logger, f"Load model from pretrained_model_checkpoint {pretrained_model_dir_or_checkpoint}")
        model = models[model_name](**model_config)
        checkpointer = Checkpointer(folder=output_dir, dcp_api=save_with_dcp_api)
        checkpointer.load_model_from_path(model, pretrained_model_dir_or_checkpoint)
    else:
        raise ValueError(f"In inference mode, pretrained_model_dir_or_checkpoint must be specified!")

    if use_context_parallel and model.num_heads % cp_size != 0:
        raise ValueError(f"When using context parallel, num_heads {model.num_heads} mush be mutiple of cp_size {cp_size}!")



    cleanup_distributed_env()






if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/t2v.yaml")
    args = parser.parse_args()
    if not os.path.exists(args.config):
        raise ValueError
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    main(config)
