import logging
import torch
import torch.nn as nn
from torch.distributed.tensor.parallel import parallelize_module
from torch.distributed.device_mesh import DeviceMesh

def CP_warpper(model: nn.Module, all_cp_plans: dict, cp_mesh: DeviceMesh):
    is_rank_zero = torch.distributed.get_rank() == 0
    if is_rank_zero:
        logging.info("Parallelize Module with Context Parallel...")
    for module in model.modules():
        for module_cls, cp_plan in all_cp_plans.items():
            if isinstance(module, module_cls):
                if is_rank_zero:
                    logging.info(f"Parallelize {module_cls}.")
                parallelize_module(
                    module,
                    device_mesh=cp_mesh,
                    parallelize_plan=cp_plan
                )
    if is_rank_zero:
        logging.info("Context Parallel Down!")