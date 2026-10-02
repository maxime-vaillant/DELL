import torch
from spikingjelly.activation_based import layer
from torch import nn

from dell.utils.neuron import get_neuron


class SeparableConv2d(nn.Module):
    """Depthwise convolution, optional neuron, then 1x1 pointwise convolution.

    Drop-in replacement for `layer.Conv2d` with fewer parameters. The neuron between the two stages
    is only created when `activation` is True.
    """

    def __init__(
            self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True,
            activation: bool = False,
    ):
        super().__init__()

        self.depthwise = layer.Conv2d(
            in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=bias,
        )
        self.act = get_neuron() if activation else nn.Identity()
        self.pointwise = layer.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.act(self.depthwise(x)))
