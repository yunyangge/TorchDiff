from typing import Any, Optional, Union

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    DeviceMesh,
    distribute_module,
    distribute_tensor,
    DTensor,
    Replicate,
    Shard,
)
from torch.distributed.tensor.placement_types import Placement
from torch.distributed.tensor.parallel import ParallelStyle

class Redistribute(ParallelStyle):
    def __init__(
        self,
        *,
        original_layouts: Union[Placement, tuple[Placement]],
        target_layouts: Union[Placement, tuple[Placement]],
        use_local_output: bool = True,
    ):
        self.original_layouts = (
            (original_layouts,)
            if isinstance(original_layouts, Placement)
            else original_layouts
        )
        self.target_layouts = (
            (target_layouts,)
            if isinstance(target_layouts, Placement)
            else target_layouts
        )
        self.use_local_output = use_local_output
        assert len(self.original_layouts) == len(self.target_layouts), (
            "original_layout and target_layout should have same length!"
        )

    def _redistribute(self, outputs, device_mesh):
        target_outputs = []
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        if len(outputs) != len(self.target_layouts):
            raise ValueError(
                "module outputs and target_layouts should have same length!"
            )
        for out, out_layout, desired_out_layout in zip(
            outputs, self.original_layouts, self.target_layouts
        ):
            if out_layout is not None:
                if isinstance(out, DTensor):
                    # TODO: re-enable the check once we fix the compile path
                    # assert out.placements[0] == out_layout
                    dt_out = out
                else:
                    dt_out = DTensor.from_local(
                        out, device_mesh, (out_layout,), run_check=False
                    )

                if out_layout != desired_out_layout:
                    dt_out = dt_out.redistribute(placements=(desired_out_layout,))
                prepared_outputs.append(
                    dt_out.to_local() if self.use_local_output else dt_out
                )
            else:
                prepared_outputs.append(out)
        if len(prepared_outputs) == 1:
            return prepared_outputs[0]
        else:
            return tuple(prepared_outputs)

    def _apply(self, module: nn.Module, device_mesh: DeviceMesh) -> nn.Module:
        
        assert (module, nn.Identity()), f"Redistribute should be bound to nn.Identifiy() to perform redistribution, but module is an instance of {module.__class__.__name__}!"

        module.register_forward_hook(
            lambda _, inputs, outputs: self._redistribute(outputs, device_mesh)
        )  # type: ignore[misc, call-arg]
        return module

    def __repr__(self) -> str:
        tmpstr = self.__class__.__name__ + "("
        tmpstr += f"original_layout={self.original_layouts}, "
        tmpstr += f"target_layout={self.target_layouts}, "
        tmpstr += f"use_local_output={self.use_local_output}"
        tmpstr += ")"
        return tmpstr