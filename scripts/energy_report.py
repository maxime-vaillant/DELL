"""Theoretical energy per inference on DSEC optical flow: this network against E-RAFT and Cuadrado et al.

Operation counts come from each architecture (shape-only rebuilds of the two baselines, see below)
and are priced with figures from real chips: Loihi 1 for the spiking networks (23.6 pJ per
spike-gated synaptic operation, Davies et al. 2018) and Eyeriss v1 for the ANN (~20 pJ per MAC,
derived, see `utils/energy_estimate.py` for the caveats). Every spiking network is assumed to fire
at a uniform 10%, as no trained checkpoints exist for the baselines.

    python scripts/energy_report.py

The two baselines are rebuilt shape-for-shape from their papers and published code. They live in
this file because the comparison depends on them.
"""
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from dell.utils.energy_estimate import (
    CorrVolumeSpec, EYERISS_V1_PJ_PER_MAC, LOIHI1_PJ_PER_SYNOP, assign_uniform_firing_rate,
    corr_volume_energy, eyeriss_energy, extract_layers, find_dense_input_layers, loihi_energy,
    snn_synops,
)

_BASELINE_FIRING_RATE = 0.10  # uniform activity assumed for all compared networks

# ═════════════════════════════════════════════════════════════════════════════════
# E-RAFT (Gehrig et al. 2021, arXiv:2108.10552) -- shape-only rebuild
#
# Mirrors the real published source (github.com/uzh-rpg/E-RAFT, model/{eraft,extractor,
# corr,update}.py) line-for-line in shape: BasicEncoder (fnet, shared weights, run on
# both event packets; cnet, separate weights, run once), BasicMotionEncoder, SepConvGRU,
# FlowHead, and the mask head -- every constant below (hdim=cdim=128, fnet
# output_dim=256, corr_levels=4, corr_radius=4, mask upsample factor 8) is copied from
# eraft.py's own __init__, not guessed. N_k=12 is E-RAFT's own fixed iteration count
# (paper Sec. 5), independent of input length.
#
# No GPU and no real/trained weights needed: every module below is built only to get
# correct tensor *shapes* through its real Conv2d layers, so a plain zero input is
# enough. The correlation volume/lookup themselves are NOT Conv2d (see
# utils/energy_estimate.py's own CorrVolumeSpec) -- priced separately, not by this class.
# ═════════════════════════════════════════════════════════════════════════════════

ERAFT_HIDDEN_DIM = 128
ERAFT_CONTEXT_DIM = 128
ERAFT_FEATURE_DIM = 256
ERAFT_CORR_LEVELS = 4
ERAFT_CORR_RADIUS = 4
ERAFT_CORR_PLANES = ERAFT_CORR_LEVELS * (2 * ERAFT_CORR_RADIUS + 1) ** 2  # 324
ERAFT_UPSAMPLE_FACTOR = 8


class _ERAFTResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1)
        self.downsample = nn.Conv2d(in_planes, planes, 1, stride=stride) if stride != 1 else None

    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(x + y)


class _ERAFTBasicEncoder(nn.Module):
    def __init__(self, in_channels, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.layer1 = nn.Sequential(_ERAFTResidualBlock(64, 64, 1), _ERAFTResidualBlock(64, 64, 1))
        self.layer2 = nn.Sequential(_ERAFTResidualBlock(64, 96, 2), _ERAFTResidualBlock(96, 96, 1))
        self.layer3 = nn.Sequential(_ERAFTResidualBlock(96, 128, 2), _ERAFTResidualBlock(128, 128, 1))
        self.conv2 = nn.Conv2d(128, output_dim, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.conv2(x)


class _ERAFTBasicMotionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convc1 = nn.Conv2d(ERAFT_CORR_PLANES, 256, 1)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv = nn.Conv2d(64 + 192, 128 - 2, 3, padding=1)

    def forward(self, flow, corr):
        cor = F.relu(self.convc2(F.relu(self.convc1(corr))))
        flo = F.relu(self.convf2(F.relu(self.convf1(flow))))
        out = F.relu(self.conv(torch.cat([cor, flo], dim=1)))
        return torch.cat([out, flow], dim=1)


class _ERAFTSepConvGRU(nn.Module):
    def __init__(self, hidden_dim, input_dim):
        super().__init__()
        self.convz1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convr1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convq1 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (1, 5), padding=(0, 2))
        self.convz2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convr2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))
        self.convq2 = nn.Conv2d(hidden_dim + input_dim, hidden_dim, (5, 1), padding=(2, 0))

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q

        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r * h, x], dim=1)))
        h = (1 - z) * h + z * q
        return h


class _ERAFTFlowHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)

    def forward(self, x):
        return self.conv2(F.relu(self.conv1(x)))


class _ERAFTMaskHead(nn.Module):
    def __init__(self, input_dim, up_factor=ERAFT_UPSAMPLE_FACTOR):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, 256, 3, padding=1)
        self.conv2 = nn.Conv2d(256, up_factor * up_factor * 9, 1)

    def forward(self, x):
        return 0.25 * self.conv2(F.relu(self.conv1(x)))


class ERAFT(nn.Module):
    """Shape-only rebuild of eraft.py's ERAFT for MAC counting -- see this section's
    own header comment. Real values are never used anywhere (zeros throughout); only
    which Conv2d each tensor passes through, and at what spatial shape, matters."""

    def __init__(self, in_channels: int, num_iters: int):
        super().__init__()
        self.num_iters = num_iters
        self.fnet = _ERAFTBasicEncoder(in_channels, ERAFT_FEATURE_DIM)
        self.cnet = _ERAFTBasicEncoder(in_channels, ERAFT_HIDDEN_DIM + ERAFT_CONTEXT_DIM)
        self.motion_encoder = _ERAFTBasicMotionEncoder()
        self.gru = _ERAFTSepConvGRU(ERAFT_HIDDEN_DIM, ERAFT_HIDDEN_DIM + ERAFT_HIDDEN_DIM)
        self.flow_head = _ERAFTFlowHead(ERAFT_HIDDEN_DIM)
        self.mask_head = _ERAFTMaskHead(ERAFT_HIDDEN_DIM)

    def forward(self, image1, image2):
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)  # shared weights: same module, called twice

        net, inp = torch.split(self.cnet(image2), [ERAFT_HIDDEN_DIM, ERAFT_CONTEXT_DIM], dim=1)
        net, inp = torch.tanh(net), F.relu(inp)

        b, _, h8, w8 = fmap1.shape
        flow = torch.zeros(b, 2, h8, w8)
        mask = None
        for _ in range(self.num_iters):
            corr = torch.zeros(b, ERAFT_CORR_PLANES, h8, w8)  # shape stand-in
            motion_features = self.motion_encoder(flow, corr)
            x = torch.cat([motion_features, inp], dim=1)
            net = self.gru(net, x)
            flow = self.flow_head(net)
            mask = self.mask_head(net)

        return fmap1, fmap2, flow, mask, (h8, w8)


# ═════════════════════════════════════════════════════════════════════════════════
# Cuadrado et al. 2023 (Frontiers in Neuroscience, doi:10.3389/fnins.2023.1160034) --
# shape-only rebuild
#
# Source: github.com/j-cuadrado/of_ev_snn, network_3d/{blocks.py,poolingNet_cat_1res.py}
# (class NeuronPool_Separable_Pool3d), mirrored shape-for-shape. num_steps=1 throughout
# (the "T=21" input axis is a Conv3d *depth* axis consumed within one forward call, not
# a repeated timestep -- 5 stacked valid kernel-5 convs reduce it 21->17->13->9->5->1
# exactly, down to one 2D frame at the bottleneck); resolution is likewise fixed at
# DSEC's native 480x640 (every decoder/predictor block's up_size is hardcoded in the
# real source, not derived from input shape).
# ═════════════════════════════════════════════════════════════════════════════════

CUADRADO_T_IN = 21
CUADRADO_H_IN, CUADRADO_W_IN = 480, 640


class _CuadradoPoolingEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7):
        super().__init__()
        self.depthwise = nn.Conv3d(in_channels, in_channels, kernel_size=(5, kernel_size, kernel_size),
                                    padding=(0, kernel_size // 2, kernel_size // 2), groups=in_channels, bias=False)
        self.pointwise = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

    def forward(self, x):
        return self.pool(self.pointwise(self.depthwise(x)))


class _CuadradoSeparableSEWResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.conv1_depthwise = nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=kernel_size // 2, bias=False)
        self.conv1_pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.conv2_depthwise = nn.Conv2d(channels, channels, kernel_size, groups=channels, padding=kernel_size // 2, bias=False)
        self.conv2_pointwise = nn.Conv2d(channels, channels, 1, bias=False)

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1_pointwise(self.conv1_depthwise(x)))   # relu stands in for IFNode -- shape-only
        out = F.relu(self.conv2_pointwise(self.conv2_depthwise(out)))
        return out + identity


class _CuadradoSeparableDecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_size, kernel_size=7):
        super().__init__()
        self.up_size = (up_size[0] + kernel_size - 1, up_size[1] + kernel_size - 1)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, size=self.up_size, mode='nearest')
        return F.relu(self.pointwise(self.depthwise(x)))


class _CuadradoSeparablePredictorBlock(nn.Module):
    def __init__(self, in_channels, up_size, kernel_size=7):
        super().__init__()
        self.up_size = (up_size[0] + kernel_size - 1, up_size[1] + kernel_size - 1)
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, 2, 1, bias=False)

    def forward(self, x):
        x = F.interpolate(x, size=self.up_size, mode='nearest')
        return self.pointwise(self.depthwise(x))  # terminal head -- no neuron after this


class CuadradoNet(nn.Module):
    """Shape-only rebuild of NeuronPool_Separable_Pool3d -- see this section's own
    header comment."""

    def __init__(self):
        super().__init__()
        self.bottom = nn.Conv3d(2, 32, kernel_size=(5, 7, 7), padding=(0, 3, 3), bias=False)
        self.conv1 = _CuadradoPoolingEncoder(32, 64)
        self.conv2 = _CuadradoPoolingEncoder(64, 128)
        self.conv3 = _CuadradoPoolingEncoder(128, 256)
        self.conv4 = _CuadradoPoolingEncoder(256, 512)
        self.res1 = _CuadradoSeparableSEWResBlock(512)
        self.deconv4 = _CuadradoSeparableDecoderBlock(512, 256, up_size=(60, 80))
        self.deconv3 = _CuadradoSeparableDecoderBlock(512, 128, up_size=(120, 160))
        self.deconv2 = _CuadradoSeparableDecoderBlock(256, 64, up_size=(240, 320))
        self.deconv1 = _CuadradoSeparableDecoderBlock(128, 32, up_size=(480, 640))
        self.pred4 = _CuadradoSeparablePredictorBlock(512, up_size=(480, 640))
        self.pred3 = _CuadradoSeparablePredictorBlock(256, up_size=(480, 640))
        self.pred2 = _CuadradoSeparablePredictorBlock(128, up_size=(480, 640))
        self.pred1 = _CuadradoSeparablePredictorBlock(64, up_size=(480, 640))

    def forward(self, x):
        in_conv1 = F.relu(self.bottom(x))
        out_conv1 = self.conv1(in_conv1)
        out_conv2 = self.conv2(out_conv1)
        out_conv3 = self.conv3(out_conv2)
        out_conv4 = self.conv4(out_conv3)

        in_res1 = torch.squeeze(out_conv4, 2)
        out_res1 = self.res1(in_res1)

        in_deconv4 = out_res1 + out_conv4[:, :, -1]
        out_deconv4 = self.deconv4(in_deconv4)
        in_deconv3 = torch.cat((out_deconv4, out_conv3[:, :, -1]), dim=1)
        out_deconv3 = self.deconv3(in_deconv3)
        in_deconv2 = torch.cat((out_deconv3, out_conv2[:, :, -1]), dim=1)
        out_deconv2 = self.deconv2(in_deconv2)
        in_deconv1 = torch.cat((out_deconv2, out_conv1[:, :, -1]), dim=1)
        out_deconv1 = self.deconv1(in_deconv1)

        up_4 = self.pred4(in_deconv3)
        up_3 = self.pred3(in_deconv2)
        up_2 = self.pred2(in_deconv1)
        up_1 = self.pred1(torch.cat((out_deconv1, in_conv1[:, :, -1]), dim=1))

        return up_4 + up_3 + up_2 + up_1


_CUADRADO_HAS_NEURON = {
    "bottom": True,
    "conv1.pointwise": True, "conv2.pointwise": True, "conv3.pointwise": True, "conv4.pointwise": True,
    "res1.conv1_pointwise": True, "res1.conv2_pointwise": True,
    "deconv4.pointwise": True, "deconv3.pointwise": True, "deconv2.pointwise": True, "deconv1.pointwise": True,
}


# ═════════════════════════════════════════════════════════════════════════════════
# Build each network's LayerSpec list, then price it
# ═════════════════════════════════════════════════════════════════════════════════

def _build_ours_uniform(mid_activation=False):
    """This network under the same uniform 10% activity as the baselines.

    A pointwise stage fed directly by a depthwise stage sees a dense, real-valued input, so
    `find_dense_input_layers` pins its input rate to 1.0. `mid_activation=True` sets
    `use_separable_activation=True`: a spiking neuron between the depthwise and pointwise stage of
    every separable convolution, which makes every stage spike-fed. Four sites are separable (see
    `models/unet.py`): the deepest encoder stage, the first decoder stage, and the two
    convolutions of the bottleneck."""
    from dell.modules.e2e.optical_flow import EndToEndOpticalFlow
    model = EndToEndOpticalFlow(
        in_channels=2, base_channels=32, depth=4, use_separable_activation=mid_activation,
    ).unet
    model.eval()
    num_steps = 16
    dummy = torch.zeros(num_steps, 1, 2, 192, 256)
    layers = extract_layers(model, dummy)
    dense = find_dense_input_layers(model)
    assign_uniform_firing_rate(layers, rate=_BASELINE_FIRING_RATE, num_steps=num_steps, dense_input_names=dense)
    return layers, num_steps


def _build_cuadrado():
    model = CuadradoNet()
    model.eval()
    dummy = torch.zeros(1, 2, CUADRADO_T_IN, CUADRADO_H_IN, CUADRADO_W_IN)
    layers = extract_layers(model, dummy)
    # Every *.pointwise conv is fed directly by its own depthwise sibling's dense,
    # real-valued output (no neuron in between) -- the same pattern
    # find_dense_input_layers detects automatically when a real spikingjelly neuron
    # object is present to find (CuadradoNet's own IFNode stand-ins are plain F.relu).
    dense = {l.name for l in layers if l.name.endswith(".pointwise")}
    assign_uniform_firing_rate(layers, rate=_BASELINE_FIRING_RATE, num_steps=1, dense_input_names=dense)
    # theta_out: only a layer with a real neuron immediately after it produces spikes;
    # the 4 terminal predictor pointwise convs and every depthwise stage have none --
    # zero them (assign_uniform_firing_rate has no has_neuron concept and would
    # otherwise overcount a membrane-reset term on these).
    for l in layers:
        if not _CUADRADO_HAS_NEURON.get(l.name, False):
            l.theta_out = 0.0
    return layers, 1


def _build_eraft():
    num_iters = 12
    height, width = 480, 640  # DSEC native resolution
    model = ERAFT(in_channels=2, num_iters=num_iters)
    model.eval()
    dummy = (torch.zeros(1, 2, height, width), torch.zeros(1, 2, height, width))
    layers = extract_layers(model, dummy)
    with torch.no_grad():
        _, _, _, _, (h8, w8) = model(*dummy)
    corr = CorrVolumeSpec(
        name="eraft_corr", H=h8, W=w8, dim=ERAFT_FEATURE_DIM,
        num_levels=ERAFT_CORR_LEVELS, radius=ERAFT_CORR_RADIUS, iters=num_iters,
    )
    return layers, corr, (height, width)


def _eraft_upsample_macs(height, width, up_factor=8):
    """eraft.py's final convex-upsampling stage (unfold + softmax-weighted sum, 1/8 ->
    full res): not a Conv2d (no hook catches it) and not expressible as a
    CorrVolumeSpec either -- one weighted-sum term per output pixel x 9 neighbours x 2
    flow channels, priced at the same flat eyeriss pJ/MAC rate as everything else here."""
    return height * width * 2 * 9


def print_full_comparison():
    print("\n" + "=" * 100)
    print(f"THEORETICAL ENERGY PER INFERENCE -- E-RAFT / Cuadrado et al. / Ours "
          f"(uniform {_BASELINE_FIRING_RATE * 100:.0f}% activity, unmeasured)")
    print("Loihi 1 (23.6 pJ per synaptic operation) for the SNNs, Eyeriss v1 (~20 pJ per MAC) for the ANN.")
    print("See utils/energy_estimate.py for the sourcing and caveats of both constants.")
    print("=" * 100)

    layers_ours, _ = _build_ours_uniform()
    layers_ours_mid, _ = _build_ours_uniform(mid_activation=True)
    layers_cuad, _ = _build_cuadrado()
    layers_eraft, corr, (h, w) = _build_eraft()

    def snn_row(name, layers):
        ops = sum(snn_synops(l) * l.n_calls for l in layers)
        energy_pj = sum(loihi_energy(l, LOIHI1_PJ_PER_SYNOP) * l.n_calls for l in layers)
        print(f"{name:<44s} {ops / 1e9:>10.1f} GAC {energy_pj / 1e12:>10.3f} J   (Loihi 1)")

    conv_macs = sum(l.n_synapses * l.n_calls for l in layers_eraft)
    other_macs = corr.total_ops + _eraft_upsample_macs(h, w)
    eraft_pj = (
        sum(eyeriss_energy(l, EYERISS_V1_PJ_PER_MAC) * l.n_calls for l in layers_eraft)
        + corr_volume_energy(corr, EYERISS_V1_PJ_PER_MAC) + _eraft_upsample_macs(h, w) * EYERISS_V1_PJ_PER_MAC
    )

    print(f"\n{'Model':<44s} {'Ops / inference':>14s} {'Energy':>12s}")
    print(f"{'E-RAFT (ANN)':<44s} {(conv_macs + other_macs) / 1e9:>10.1f} GMAC {eraft_pj / 1e12:>10.3f} J   (Eyeriss v1)")
    print(f"   of which convolutions {conv_macs / 1e9:.1f} GMAC, correlation volume and upsampling {other_macs / 1e9:.1f} GMAC")
    snn_row("Cuadrado et al. (SNN)", layers_cuad)
    snn_row("Ours (SNN), dense pointwise inputs", layers_ours)
    snn_row("Ours (SNN), neuron between depthwise/pointwise", layers_ours_mid)

    print("\nIn the Cuadrado row, as in 'Ours, dense pointwise inputs', every pointwise convolution is fed by a")
    print("depthwise convolution with no neuron in between, so its input is dense and counted at rate 1.0.")
    print("With a neuron in between (the last row) every stage is spike-gated at the assumed rate.")
    print("Nothing here has run on neuromorphic or ANN-accelerator silicon: an unvalidated proxy.")


def main():
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()

    # the networks are shape-only builds that never run on a GPU; set here rather than at
    # import time so importing this module does not change the backend
    import dell.utils.neuron as neuron_utils
    neuron_utils.NEURON_KWARGS['backend'] = 'torch'
    print_full_comparison()


if __name__ == "__main__":
    main()
