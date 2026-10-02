import torch
from spikingjelly.activation_based import layer
from torch import nn


class SemanticHead(nn.Module):
    """3x3 convolution to `num_classes` logits, averaged over time into a (B, num_classes, H, W) map.

    There is no neuron after the convolution: the logits must stay real-valued.
    """

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()

        self.conv = layer.Conv2d(in_channels=in_channels, out_channels=num_classes, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.conv(x), dim=0)
