from spikingjelly.activation_based import layer
from torch import nn

from dell.utils.neuron import get_neuron


class LocalExtraConv(nn.Module):
    """Extra convolution for a local head on a block that has no distinct skip to fuse.

    The stem (encoder level 0) does not pool, so its output and skip are the same tensor, and the
    bottleneck produces no skip. Upsampling and concatenating would only duplicate channels, so
    this block adds one more convolution at the block's own width instead.

    `forward` accepts and ignores `skip` so it can be called like a `DecoderBlock`.
    """

    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1):
        super().__init__()

        self.conv = layer.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=True)
        self.act = get_neuron()

    def forward(self, x, skip=None):
        return self.act(self.conv(x))
