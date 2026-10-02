import torch
from spikingjelly.activation_based import layer
from torch import nn

from dell.blocks.separable_conv import SeparableConv2d
from dell.utils.neuron import get_neuron


class DecoderBlock(nn.Module):
    """Nearest-neighbour upsampling, concatenation with the skip connection, then convolution and neuron."""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            skip_channels: int,
            kernel_size: int = 3,
            padding: int = 1,
            use_separable_conv: bool = False,
            use_separable_activation: bool = False,
    ):
        super().__init__()

        self.upsample = layer.Upsample(scale_factor=2, mode='nearest')

        conv_cls = SeparableConv2d if use_separable_conv else layer.Conv2d
        fusion_kwargs = dict(in_channels=in_channels + skip_channels, out_channels=out_channels,
                             kernel_size=kernel_size, padding=padding, bias=True)
        if use_separable_conv:
            fusion_kwargs.update(activation=use_separable_activation)
        self.fusion = conv_cls(**fusion_kwargs)
        self.fusion_act = get_neuron()

    def forward(self, x, skip):
        up = torch.cat([self.upsample(x), skip], dim=2)

        return self.fusion_act(self.fusion(up))
