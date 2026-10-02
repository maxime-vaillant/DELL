import torch
from torch import nn

from dell.blocks.decoder import DecoderBlock
from dell.blocks.encoder import EncoderBlock
from dell.blocks.sew import SEWBlock

_ENCODER_LARGE_KERNEL_MAX_LEVEL = 2
_DECODER_LARGE_KERNEL_MAX_LEVEL = 1


class UNet(nn.Module):
    """Spiking U-Net: `depth + 1` encoder stages, a SEW bottleneck and `depth` decoder stages.

    Only the deepest encoder stage, the first decoder stage and the bottleneck use separable
    convolutions. The first skip connection feeds the head, which is why the head takes twice
    `base_channels` input channels.
    """

    def __init__(
            self,
            in_channels,
            head: nn.Module,
            base_channels: int = 32,
            depth: int = 4,
            use_separable_activation: bool = False,
    ):
        super().__init__()

        widths = [base_channels * (2 ** level) for level in range(depth + 1)]
        stage_in_channels = [in_channels] + widths[:-1]

        self.encoder = nn.ModuleList([
            EncoderBlock(
                in_channels=stage_in_channels[level],
                out_channels=widths[level],
                kernel_size=5 if level <= _ENCODER_LARGE_KERNEL_MAX_LEVEL else 3,
                padding=2 if level <= _ENCODER_LARGE_KERNEL_MAX_LEVEL else 1,
                pool=level > 0,
                use_separable_conv=level == depth,
                use_separable_activation=use_separable_activation,
            )
            for level in range(depth + 1)
        ])

        self.bottleneck = SEWBlock(
            channels=widths[-1],
            kernel_size=3,
            padding=1,
            use_separable_activation=use_separable_activation,
        )

        self.decoder = nn.ModuleList([
            DecoderBlock(
                in_channels=widths[level],
                out_channels=widths[level - 1],
                skip_channels=widths[level],
                kernel_size=5 if level <= _DECODER_LARGE_KERNEL_MAX_LEVEL else 3,
                padding=2 if level <= _DECODER_LARGE_KERNEL_MAX_LEVEL else 1,
                use_separable_conv=level == depth,
                use_separable_activation=use_separable_activation,
            )
            for level in range(depth, 0, -1)
        ])

        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []
        for encoder_block in self.encoder:
            x, skip = encoder_block(x)
            skips.append(skip)

        x = self.bottleneck(x)

        for decoder_block, skip in zip(self.decoder, reversed(skips[1:])):
            x = decoder_block(x, skip)

        x = torch.cat([x, skips[0]], dim=2)

        return self.head(x)
