from torch import nn

from dell.blocks.separable_conv import SeparableConv2d
from dell.utils.neuron import get_neuron


class SEWBlock(nn.Module):
    """Spike-element-wise residual block of two separable convolutions, with an identity shortcut."""

    def __init__(
            self,
            channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            use_separable_activation: bool = False,
    ):
        super().__init__()

        conv_kwargs = dict(
            in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=True,
            activation=use_separable_activation,
        )
        self.conv1 = SeparableConv2d(**conv_kwargs)
        self.act1 = get_neuron()

        self.conv2 = SeparableConv2d(**conv_kwargs)
        self.act2 = get_neuron()

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = self.act2(self.conv2(out))

        return out + x
