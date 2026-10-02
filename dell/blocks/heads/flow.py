import torch
from spikingjelly.activation_based import layer
from torch import nn


class FlowHead(nn.Module):
    """3x3 convolution to two flow channels, averaged over time into a (B, 2, H, W) map.

    There is no neuron after the convolution: the output must stay real-valued.
    """

    def __init__(self, in_channels: int):
        super().__init__()

        self.conv = layer.Conv2d(in_channels=in_channels, out_channels=2, kernel_size=3, padding=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(self.conv(x), dim=0)
