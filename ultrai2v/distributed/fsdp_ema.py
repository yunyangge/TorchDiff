import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, FSDPModule
from torch.distributed.tensor import DTensor

class FSDPEMAModel:
    def __init__(
        self,
        model: FSDPModule, # The model must be an instance of FSDPModule
        decay: float = 0.9999,
        update_interval: int = 1,
    ):
        if not isinstance(model, FSDPModule):
            raise ValueError("Model must be an instance of FSDPModule")
        self.decay = decay
        self.one_minus_decay = 1.0 - decay
        self.update_interval = update_interval
        self.shadow_params = {
            name: param.clone().detach().float() for name, param in model.named_parameters()
        }
        self.backup = {}

    def get_shadow_params(self):
        return self.shadow_params

    @torch.no_grad()
    def update(self, model: FSDPModule, step: int):
        if step % self.update_interval != 0:
            return
        for name, param in model.named_parameters():
            shadow_param = self.shadow_params[name]
            assert isinstance(param, DTensor), f"{name}"
            assert isinstance(shadow_param, DTensor), f"{name}"
            if param.requires_grad:
                shadow_param.data.sub_(self.one_minus_decay * (shadow_param.data.float() - param.data.float()))
            else:
                shadow_param.data.copy_(param.data)

    def ema_copy_to_model(self, model: FSDPModule):
        for name, param in model.named_parameters():
            shadow_param = self.shadow_params[name]
            assert isinstance(param, DTensor), f"{name}"
            assert isinstance(shadow_param, DTensor), f"{name}"
            param.data.copy_(shadow_param.data)

    def model_copy_to_ema(self, model: FSDPModule):
        for name, param in model.named_parameters():
            shadow_param = self.shadow_params[name]
            assert isinstance(param, DTensor), f"{name}"
            assert isinstance(shadow_param, DTensor), f"{name}"
            shadow_param.data.copy_(param.data)

    def store(self, model: FSDPModule):
        for name, param in model.named_parameters():
            self.backup[name] = param.clone().detach()

    def restore(self, model: FSDPModule):
        for name, param in model.named_parameters():
            assert name in self.backup
            param.data.copy_(self.backup[name].data)
        self.backup = {}