"""Per-inference energy estimation from the model's operation counts.

Every number is a count of real operations extracted from the model multiplied by a flat pJ per
operation figure from real silicon. No memory hierarchy is modeled.

Two chips are used:
  * SNN: Loihi 1 (Davies et al. 2018, 23.6 pJ per synaptic operation).
  * ANN: Eyeriss v1 (Chen et al. 2016), whose figure is derived from the Eyeriss v2 paper (see
    `EYERISS_V1_PJ_PER_MAC`).

`snn_synops(layer)` and `layer.n_synapses` give raw operation counts, so another pJ figure can be
applied to them.
"""
import math
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

import torch
import torch.nn as nn
from spikingjelly.activation_based import functional
from spikingjelly.activation_based import neuron as sj_neuron

_SPIKE_TYPES = (sj_neuron.LIFNode,)


# ── energy per operation ─────────────────────────────────────────────────────────

# SNN, priced per spike-triggered synaptic operation (see `snn_synops`).
# Loihi 1: Davies et al. 2018, IEEE Micro, "Loihi: A Neuromorphic Manycore Processor with On-Chip
# Learning", energy per synaptic operation as stated in the paper's tables.
LOIHI1_PJ_PER_SYNOP = 23.6

# ANN, priced per dense MAC (see `LayerSpec.n_synapses`). Derived, not stated directly.
# Chen et al. 2019 (JETCAS, "Eyeriss v2") report, for sparse MobileNet (width 0.5, 128x128 input,
# 65 nm, batch 1), 2560.3 inferences/J for v2 and 2.5x less for v1 (1024.1 inferences/J).
# Converting to pJ/MAC needs that network's MAC count, estimated at ~49M from the paper's
# "14.7x reduction in nominal MACs vs. AlexNet" and AlexNet's ~724M MACs (Krizhevsky et al. 2012).
#   v1: 1 / 1024.1 J / 49M MACs ~= 20 pJ/MAC
# The absolute value carries the uncertainty of the MAC estimate (roughly +/-15%).
EYERISS_V1_PJ_PER_MAC = 20.0


# ── per-layer spec ────────────────────────────────────────────────────────────────

class LayerType(Enum):
    CONV2D = "conv2d"
    CONV3D = "conv3d"
    CONV2D_TRANSPOSE = "conv2d_transpose"   # needs its own fan-out rule in `snn_synops`
    LINEAR = "linear"


@dataclass
class LayerSpec:
    name: str
    layer_type: LayerType
    C_in: int = 0
    C_out: int = 0
    H_in: int = 0
    W_in: int = 0
    H_out: int = 0
    W_out: int = 0
    H_k: int = 0
    W_k: int = 0
    T_in: int = 1     # Conv3d only: extra depth axis, 1 for Conv2d and Linear
    T_out: int = 1
    T_k: int = 1
    stride: int = 1
    groups: int = 1
    has_bias: bool = True
    N_in: int = 0
    N_out: int = 0
    theta_in: float = 0.0    # mean spikes (or nonzero elements) arriving per inference
    theta_out: float = 0.0   # mean spikes leaving this layer's own neuron per inference
    n_calls: int = 1    # forward calls per inference, >1 for weight-shared modules; counts below are per call

    @property
    def is_conv(self):
        return self.layer_type in (LayerType.CONV2D, LayerType.CONV3D, LayerType.CONV2D_TRANSPOSE)

    @property
    def is_transpose(self):
        return self.layer_type == LayerType.CONV2D_TRANSPOSE

    @property
    def is_depthwise(self):
        return self.is_conv and self.groups == self.C_in and self.groups > 1

    @property
    def n_synapses(self):
        """Dense MAC count of one call, the ANN-side counterpart of `snn_synops`."""
        if self.is_conv:
            return (self.C_out * (self.C_in // self.groups) * self.T_k * self.H_k * self.W_k
                    * self.T_out * self.H_out * self.W_out)
        return self.N_in * self.N_out

    @property
    def n_in_elements(self):
        """Number of input elements per inference; a firing rate is a fraction of this."""
        return self.C_in * self.T_in * self.H_in * self.W_in if self.is_conv else self.N_in

    @property
    def n_out_elements(self):
        return self.C_out * self.T_out * self.H_out * self.W_out if self.is_conv else self.N_out


# ── architecture extraction ──────────────────────────────────────────────────────

def extract_layers(model: nn.Module, dummy_input: torch.Tensor) -> List[LayerSpec]:
    """Run one forward pass and return the shape of every Conv/Linear layer.

    Works for single-step (B, C, H, W) and multi-step (T, B, C, H, W) inputs. Modules called several
    times per inference only get one entry (shapes are identical across calls), with `n_calls` set.
    """
    spatial = OrderedDict()
    call_count: Dict[str, int] = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            spatial[name] = (list(x.shape), list(out.shape))
            call_count[name] = call_count.get(name, 0) + 1
        return hook

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.Linear)):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        functional.reset_net(model)
        if isinstance(dummy_input, (tuple, list)):
            model(*dummy_input)   # multi-argument forward (e.g. E-RAFT's BasicUpdateBlock)
        else:
            model(dummy_input)
    for h in hooks:
        h.remove()

    layers = []
    for name, module in model.named_modules():
        if name not in spatial:
            continue
        in_s, out_s = spatial[name]
        if isinstance(module, nn.Conv3d):
            # input is (N, C, D, H, W)
            kt, hk, wk = module.kernel_size
            st = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            layers.append(LayerSpec(
                name=name, layer_type=LayerType.CONV3D,
                C_in=module.in_channels, C_out=module.out_channels,
                T_in=in_s[-3], H_in=in_s[-2], W_in=in_s[-1],
                T_out=out_s[-3], H_out=out_s[-2], W_out=out_s[-1],
                T_k=kt, H_k=hk, W_k=wk, stride=st, groups=module.groups,
                has_bias=module.bias is not None, n_calls=call_count[name]))
        elif isinstance(module, nn.Conv2d):
            hk = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            wk = module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size
            st = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            layers.append(LayerSpec(
                name=name, layer_type=LayerType.CONV2D,
                C_in=module.in_channels, C_out=module.out_channels,
                H_in=in_s[-2], W_in=in_s[-1], H_out=out_s[-2], W_out=out_s[-1],
                H_k=hk, W_k=wk, stride=st, groups=module.groups,
                has_bias=module.bias is not None, n_calls=call_count[name]))
        elif isinstance(module, nn.ConvTranspose2d):
            # same fields as Conv2d; only the fan-out in `snn_synops` differs for transposed convs
            hk = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
            wk = module.kernel_size[1] if isinstance(module.kernel_size, tuple) else module.kernel_size
            st = module.stride[0] if isinstance(module.stride, tuple) else module.stride
            layers.append(LayerSpec(
                name=name, layer_type=LayerType.CONV2D_TRANSPOSE,
                C_in=module.in_channels, C_out=module.out_channels,
                H_in=in_s[-2], W_in=in_s[-1], H_out=out_s[-2], W_out=out_s[-1],
                H_k=hk, W_k=wk, stride=st, groups=module.groups,
                has_bias=module.bias is not None, n_calls=call_count[name]))
        elif isinstance(module, nn.Linear):
            layers.append(LayerSpec(
                name=name, layer_type=LayerType.LINEAR,
                N_in=module.in_features, N_out=module.out_features,
                has_bias=module.bias is not None, n_calls=call_count[name]))
    return layers


# ── activity assumptions ─────────────────────────────────────────────────────────

def find_dense_input_layers(model: nn.Module) -> set:
    """Names of the Conv/Linear layers whose input is not the output of a spiking neuron.

    The typical case is the pointwise stage of a `SeparableConv2d` fed directly by its depthwise
    stage with no neuron in between: its input is dense, so a spike rate would understate it.
    Layers are traversed in registration order, which is the forward order for the networks here.
    """
    dense = set()
    last_compute = None
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv3d, nn.Linear)):
            if last_compute is not None:
                dense.add(name)  # previous conv never reached a neuron before this one
            last_compute = name
        elif isinstance(module, _SPIKE_TYPES) and last_compute is not None:
            last_compute = None  # previous conv's spikes consumed -- next conv is spike-fed
    return dense


def assign_uniform_firing_rate(
        layers: List[LayerSpec], rate: float, num_steps: int = 1, dense_input_names: set = frozenset(),
) -> List[LayerSpec]:
    """Set theta_in and theta_out of every layer from an assumed constant firing rate.

    Layers in `dense_input_names` (see `find_dense_input_layers`) get a rate of 1.0 for their
    input, as it is not thresholded by a neuron. `num_steps` scales the single-frame element
    counts to a whole inference.
    """
    for l in layers:
        in_rate = 1.0 if l.name in dense_input_names else rate
        l.theta_in = in_rate * l.n_in_elements * num_steps
        l.theta_out = rate * l.n_out_elements * num_steps
    return layers


# ── operation counts ────────────────────────────────────────────────────────────

def snn_synops(l: LayerSpec) -> float:
    """Spike-gated synaptic operations of one call: `theta_in` times the fan-out of one input spike.

    The fan-out is the kernel footprint times the output channels per group (kernel footprint only
    for depthwise convolutions). A strided convolution divides each kernel dimension by the stride
    (rounded up), as a spike only lands on a subset of the output grid. A transposed convolution
    upsamples, so a spike reaches the full kernel window. Multiply by `l.n_calls` when summing.
    """
    if l.is_transpose:
        fan = l.T_k * l.H_k * l.W_k
        if not l.is_depthwise:
            fan *= (l.C_out // l.groups)
        return l.theta_in * fan
    if l.is_conv:
        fan = math.ceil(l.T_k / l.stride) * math.ceil(l.H_k / l.stride) * math.ceil(l.W_k / l.stride)
        if not l.is_depthwise:
            fan *= (l.C_out // l.groups)
        return l.theta_in * fan
    return l.theta_in * l.N_out


# ── pricing: real fabricated chips only (see module docstring) ──────────────────

def loihi_energy(l: LayerSpec, pj_per_synop: float = LOIHI1_PJ_PER_SYNOP) -> float:
    """SNN energy in pJ: synaptic operations times `pj_per_synop`."""
    return snn_synops(l) * pj_per_synop


def eyeriss_energy(l: LayerSpec, pj_per_mac: float = EYERISS_V1_PJ_PER_MAC) -> float:
    """ANN energy in pJ: dense MACs times `pj_per_mac`."""
    return l.n_synapses * pj_per_mac


# ── correlation volume (RAFT-family models) ───────────────────────────────────────
#
# The correlation volume of E-RAFT (https://github.com/uzh-rpg/E-RAFT, model/corr.py) is not a
# Conv2d or Linear, so `extract_layers` does not see it. It has two parts:
#   1. `CorrBlock.corr`: all-pairs dot products between two feature maps, computed once per
#      inference. This is RAFT's O((H*W)^2) memory cost.
#   2. `CorrBlock.__call__`: bilinear lookups in a 4-level pyramid of that volume (4 corner reads
#      per sample), repeated at each of the 12 refinement iterations. Priced at the same flat
#      pJ/MAC as a dense operation, a simplification.

@dataclass
class CorrVolumeSpec:
    name: str
    H: int              # feature-map height after the encoder stride, e.g. 480 / 8 = 60
    W: int              # feature-map width, e.g. 640 / 8 = 80
    dim: int            # feature channels when correlating (E-RAFT: 256)
    num_levels: int = 4  # E-RAFT defaults
    radius: int = 4
    iters: int = 12

    @property
    def build_macs(self) -> int:
        return self.H * self.W * self.H * self.W * self.dim

    @property
    def _window(self) -> int:
        return (2 * self.radius + 1) ** 2

    @property
    def lookup_ops_per_iter(self) -> int:
        """4 corner reads per bilinear sample, at every sampled offset."""
        return self.H * self.W * self.num_levels * self._window * 4

    @property
    def total_ops(self) -> int:
        """Total operations of one inference: one build plus `iters` lookups."""
        return self.build_macs + self.iters * self.lookup_ops_per_iter


def corr_volume_energy(c: CorrVolumeSpec, pj_per_mac: float = EYERISS_V1_PJ_PER_MAC) -> float:
    return c.total_ops * pj_per_mac
