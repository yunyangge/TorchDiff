import torch
import logging

def str_to_precision(s):
    if s == "bfloat16" or s == "bf16":
        return torch.bfloat16
    elif s == "float16" or s == "fp16":
        return torch.float16
    elif s == "float32" or s == "float" or s == "fp32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported precision string: {s}")

def params_nums_to_str(params_num):
    if params_num >= 1e9:
        return f"{params_num / 1e9:.2f}B"
    elif params_num >= 1e6:
        return f"{params_num / 1e6:.2f}M"
    elif params_num >= 1e3:
        return f"{params_num / 1e3:.2f}K"
    else:
        return str(params_num)

def get_memory_allocated():
    return f"{torch.cuda.memory_allocated() / 1024**3:.2f}"  # GiB

def is_npu_available():
    is_available = True
    try:
        import torch_npu
    except:
        is_available = False
    return is_available