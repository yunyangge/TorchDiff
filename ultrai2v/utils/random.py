import logging
import numpy as np
import random
import torch
import torch.distributed as dist

# adapted from https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/random.py#L39
def set_seed(
    seed: int,
    device_specific: bool = False, 
    deterministic: bool = False,
    process_group: dist.ProcessGroup = None
):
    """
    Helper function for reproducible behavior to set the seed in `random`, `numpy`, `torch`.

    Args:
        seed (`int`):
            The seed to set.
        device_specific (`bool`, *optional*, defaults to `False`):
            Whether to differ the seed on each device slightly with `self.process_index`.
        deterministic (`bool`, *optional*, defaults to `False`):
            Whether to use deterministic algorithms where available. Can slow down training.
    """
    if device_specific:
        if process_group is None:
            if not dist.is_initialized():
                raise ValueError("`device_specific` can only be set to `True` when using distributed.")
            process_group = dist.group.WORLD
        rank = dist.get_rank(process_group)
        seed += rank
        logging.info(f"Using device specific seed {seed} on local rank {rank} and global rank {dist.get_rank()}")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)