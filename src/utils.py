import torch
from torch import nn


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_device(preferred: str = "cuda") -> str:
    if preferred == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"
