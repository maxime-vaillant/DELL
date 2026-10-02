from spikingjelly.activation_based import layer
from torch import nn

from dell.blocks.separable_conv import SeparableConv2d
from dell.utils.neuron import get_neuron


class EncoderBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 5,
            padding: int = 2,
            pool: bool = True,
            use_separable_conv: bool = False,
            use_separable_activation: bool = False,
    ):
        super().__init__()

        conv_cls = SeparableConv2d if use_separable_conv else layer.Conv2d
        conv_kwargs = dict(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=True)
        if use_separable_conv:
            conv_kwargs.update(activation=use_separable_activation)
        self.conv = conv_cls(**conv_kwargs)

        self.act = get_neuron()
        self.pool = layer.MaxPool2d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        skip = self.act(self.conv(x))
        down = self.pool(skip)

        return down, skip
