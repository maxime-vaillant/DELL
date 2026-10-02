from torch import nn


def freeze(module: nn.Module) -> None:
    """Stop gradient updates for `module`; used for DECOLLE's fixed random readout heads."""
    for parameter in module.parameters():
        parameter.requires_grad_(False)
