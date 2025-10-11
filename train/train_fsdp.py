import os
import yaml
from tqdm import tqdm
import torch
import torch.distributed as dist
from argparse import ArgumentParser

from ultrai2v.data import ultra_datasets, ultra_samplers, ultra_collators
from torch.utils.data import DataLoader

from ultrai2v.utils.log_utils import get_logger, log_on_main_process, verify_min_gpu_count
from ultrai2v.utils.random import set_seed
from ultrai2v.distributed.utils import (
    setup_distributed_env, 
    cleanup_distributed_env, 
    set_modules_to_forward_prefetch, 
    set_modules_to_backward_prefetch, 
    gather_data_from_all_ranks
)
from ultrai2v.distributed.fsdp2_warpper import FSDP2_mix_warpper
from ultrai2v.distributed.fsdp_ema import FSDPEMAModel as EMAModel

from ultrai2v.modules import WanVAE, T5EncoderModel, models, models_main_block, models_blocks_to_float
from ultrai2v.schedulers import schedulers

from ultrai2v.distributed.checkpoint import Checkpointer
from ultrai2v.utils.constant import VIDEO, PROMPT_IDS, PROMPT_MASK
from ultrai2v.utils.utils import str_to_precision, params_nums_to_str



def main(config):
    # config analysis
    seed = config.get("seed", 42)

    # model config
    model_name = config.get("model", "wan_t2v")
    model_config = config.get("model_config", {})
    vae_config = config.get("vae_config", {})
    text_encoder_config = config.get("text_encoder_config", {})
    scheduler_config = config.get("scheduler_config", {})

    # data config
    data_config = config.get("data_config", {})

    # optimizer config
    optimizer_config = config.get("optimizer_config", {})

    # training config
    training_iteration = config.get("training_iteration", 1000000)
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    grad_norm_threshold = config.get("grad_norm_threshold", 1.0)
    log_interval = config.get("log_interval", 1)
    save_interval = config.get("save_interval", 1000)
    weight_dtype = config.get("weight_dtype", "bfloat16")
    reshard_after_forward = config.get("reshard_after_forward", None)
    model_cpu_offload = config.get("model_cpu_offload", False)
    ema_decay = config.get("ema_decay", 0.9999)
    ema_update_interval = config.get("ema_update_interval", 1)
    explicit_prefetching_num_blocks = config.get("explicit_prefetching_num_blocks", 0)
    
    # save config
    output_dir = config.get("output_dir", "./output")
    save_with_dcp_api = config.get("save_with_dcp_api", False)

    # distributed setup
    setup_distributed_env()
    verify_min_gpu_count()

    rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = torch.distributed.get_world_size()
    device = torch.device(f"cuda:{rank}")
    weight_dtype = str_to_precision(weight_dtype)

    logger = get_logger()

    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    
    set_seed(seed, device_specific=False) # for init
    # init model
    log_on_main_process(logger, "Initializing VAE model...")
    vae = WanVAE(
        vae_pth=vae_config.get("vae_path", None),
        dtype=str_to_precision(vae_config.get("dtype", "fp32")),
        device=device # for vae, we do not use fsdp
    )

    log_on_main_process(logger, "Initializing text encoder model...")
    text_encoder = T5EncoderModel(
        text_len=text_encoder_config.get("text_len", 512),
        dtype=text_encoder_config.get("dtype", weight_dtype),
        device=device, # when no fsdp, we init the text_encoder on gpu
        checkpoint_path=text_encoder_config.get("checkpoint_path", None),
        use_fsdp=text_encoder_config.get("use_fsdp", False) # when using fsdp, we init the text_encoder on cpu, and use fsdp to shard the model to gpu
    )

    # vae.to(device)
    # if not text_encoder_config.get("use_fsdp", False):
    #     text_encoder.to(device)
    # vae.requires_grad_(False) # in vae
    # text_encoder.requires_grad_(False) # in text_encoder

    log_on_main_process(logger, "Initializing diffusion model and scheduler...")

    scheduler = schedulers[scheduler_config.pop("scheduler_name", "flow_matching")](**scheduler_config)

    if model_config.get("pretrained_model_dir", None) is not None:
        checkpoint_dir = model_config.get("pretrained_model_dir")
        model = models[model_name].from_pretrained(checkpoint_dir)
    else:
        model = models[model_name](**model_config)

    FSDP2_mix_warpper(
        model,
        dp_mesh=None,
        weight_dtype=weight_dtype,
        main_block_to_half=models_main_block[model_name],
        blocks_to_float=models_blocks_to_float[model_name],
        reshard_after_forward=reshard_after_forward,
        cpu_offload=model_cpu_offload,
    )
    ema_model = EMAModel(model, decay=ema_decay, update_interval=ema_update_interval)

    model.train()

    if explicit_prefetching_num_blocks > 0:
        set_modules_to_forward_prefetch(model.blocks, num_to_forward_prefetch=explicit_prefetching_num_blocks)
        set_modules_to_backward_prefetch(model.blocks, num_to_backward_prefetch=explicit_prefetching_num_blocks)

    checkpointer = Checkpointer(folder=output_dir, dcp_api=save_with_dcp_api)
    if checkpointer.last_training_iteration is not None:
        log_on_main_process(logger, "Loading model checkpoint...")
        checkpointer.load_model(model)
        log_on_main_process(logger, "Loading EMA model checkpoint...")
        load_ema_model = model.clone()
        checkpointer.load_model(load_ema_model, ema=True)
        ema_model.model_copy_to_ema(load_ema_model)
        
    log_on_main_process(logger, "Initializing and loading optimizer checkpoint...")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        optimizer_config.get("lr", 1e-4),
        betas=optimizer_config.get("betas", (0.9, 0.999)),
        weight_decay=optimizer_config.get("weight_decay", 1e-2),
        eps=optimizer_config.get("eps", 1e-15),
    )

    if checkpointer.last_training_iteration is not None:
        checkpointer.load_optim(model, optimizer)
    current_iteration = 0 if checkpointer.last_training_iteration is None else checkpointer.last_training_iteration

    set_seed(seed, device_specific=True) # for training
    log_on_main_process(logger, "Initializing dataset, sampler and dataloader...")
    dataset = ultra_datasets[data_config.pop("dataset_name", "t2v_random")](**data_config.get("dataset_config", {}))
    batch_size = data_config.get("batch_size", 1)
    sampler = ultra_samplers[data_config.pop("sampler_name", "stateful_distributed")](
        dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=data_config.get("shuffle", True),
        consumed_samples=current_iteration * batch_size * gradient_accumulation_steps,
        drop_last=data_config.get("drop_last", True),
    )
    collator = ultra_collators[data_config.pop("collator_name", "wan_t2v")](**data_config.get("collator_config", {}))
    dataloader = DataLoader(
        dataset,
        batch_size=data_config.get("batch_size", 1),
        sampler=sampler,
        collate_fn=collator,
        num_workers=data_config.get("num_workers", 16),
        pin_memory=data_config.get("pin_memory", True),
    )

    tqdm_bar = tqdm(total=training_iteration, disable=(rank != 0), initial=current_iteration, desc="Training")
    gathered_avg_loss = 0.0
    current_batch_nums = current_iteration * gradient_accumulation_steps
    start_training_logs = f"""
    =============================Start Training=============================
    Before FSDP sharding,
    Model has {params_nums_to_str(sum(p.numel() for p in model.parameters() if p.requires_grad))} trainable parameters
    After FSDP sharding,
    Model has {params_nums_to_str(sum(p._local_tensor.numel() for p in model.parameters() if p.requires_grad))} trainable parameters
    world_size: {world_size} GPUs
    Training iterations: {training_iteration}
    Current iteration: {current_iteration}
    Batch size per GPU: {batch_size}
    Gradient accumulation steps: {gradient_accumulation_steps}
    Effective batch size (world_size x batch_size x gradient_accumulation_steps): {world_size * batch_size * gradient_accumulation_steps}
    Save model to {output_dir} every {save_interval} iterations
    Training ...
    =======================================================================
    """
    log_on_main_process(logger, start_training_logs)
    while current_iteration < training_iteration:
        for batch in dataloader:
            current_batch_nums += 1
            video = batch[VIDEO].to(torch.float32).to(device)
            prompt_ids = batch[PROMPT_IDS].to(device)
            prompt_mask = batch[PROMPT_MASK].to(device)

            with torch.no_grad():
                latents = vae.encode(video)
                text_embeddings = text_encoder(prompt_ids, prompt_mask)

            q_sample_results = scheduler.q_sample(latents, sigmas=None, prior_dist=None)
            interpolated_latents = q_sample_results["x_t"]
            prior_dist = q_sample_results["prior_dist"]
            sigmas = q_sample_results["sigmas"]
            timesteps = q_sample_results["timesteps"]

            model_output = model(
                interpolated_latents,
                timesteps,
                text_embeddings,
            )

            loss = scheduler.training_losses(model_output, latents, prior_dist)[0]
            loss = loss / gradient_accumulation_steps # default value of gradient_accumulation_steps is 1
            loss.backward()
            loss_for_log = loss.detach().clone()
            gathered_loss = gather_data_from_all_ranks(loss_for_log, dim=0)
            gathered_avg_loss += gathered_loss.mean().item()

            if (current_batch_nums) % gradient_accumulation_steps == 0:
                if grad_norm_threshold > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm_threshold)
                optimizer.step()
                optimizer.zero_grad()

                ema_model.update(model, current_iteration)

                if current_iteration % log_interval == 0:
                    tqdm_bar.update(log_interval)
                    tqdm_bar.set_postfix({"loss": gathered_avg_loss, "lr": optimizer.param_groups[0]['lr']})

                if current_iteration % save_interval == 0 or current_iteration == training_iteration:
                    log_on_main_process(logger, f"Saving model checkpoint at iteration {current_iteration}...")
                    checkpointer.save(model, optimizer, current_iteration)
                
                gathered_avg_loss = 0.0    
                current_iteration += 1

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
