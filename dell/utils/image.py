import torch
import torch.nn.functional as F


def to_uint8(img: torch.Tensor) -> torch.Tensor:
    """Stretch to 0-255 between the 1st and 99th percentiles, so hot pixels do not dominate."""
    img = img.float()
    lo = torch.quantile(img, 0.01)
    hi = torch.quantile(img, 0.99).clamp(min=lo + 1e-6)
    img = (img - lo).clamp(min=0) / (hi - lo)
    return (img * 255).round().clamp(0, 255).byte()


def to_uint8_fixed(img: torch.Tensor, vmax) -> torch.Tensor:
    """Map [0, vmax] linearly to 0-255, so brightness stays comparable across frames and epochs."""
    img = img.float()
    vmax = max(float(vmax), 1e-6)
    normalized = (img.clamp(min=0) / vmax).clamp(max=1.0)
    return (normalized * 255).round().byte()


PANEL_GAP_PX = 4   # white gap between the columns of a comparison panel

_EVENT_WHITE = (255.0, 255.0, 255.0)
_EVENT_BLUE = (0.0, 0.0, 255.0)   # positive/ON polarity
_EVENT_RED = (255.0, 0.0, 0.0)    # negative/OFF polarity


def event_rgb_on_white(events_sample: torch.Tensor, panel_hw: tuple[int, int]) -> torch.Tensor:
    """Render events as blue (ON) and red (OFF) on a white background.

    events_sample: (T, 2, H_e, W_e), channel 0 = OFF counts, channel 1 = ON counts. Counts are summed
    over time, resized (nearest) to `panel_hw`, and stretched with `to_uint8`.
    """
    counts = events_sample.sum(dim=0)  # (2, H_e, W_e)
    if counts.shape[-2:] != panel_hw:
        counts = F.interpolate(counts[None].float(), size=panel_hw, mode='nearest')[0]

    neg_norm = to_uint8(counts[0]).float().unsqueeze(-1) / 255.0  # (H, W, 1)
    pos_norm = to_uint8(counts[1]).float().unsqueeze(-1) / 255.0

    white = counts.new_tensor(_EVENT_WHITE)
    blue = counts.new_tensor(_EVENT_BLUE)
    red = counts.new_tensor(_EVENT_RED)

    rgb = white - pos_norm * (white - blue) - neg_norm * (white - red)
    return rgb.round().clamp(0, 255).byte()


def event_rgb_net_polarity(events_sample: torch.Tensor, panel_hw: tuple[int, int]) -> torch.Tensor:
    """Render the net polarity (ON minus OFF) as blue/red toward black, zero net as white.

    The magnitude is log-normalized by the frame's peak. Pixels with comparable ON and OFF counts
    net to zero and look like quiet pixels. events_sample: (T, 2, H_e, W_e), same channel order as
    `event_rgb_on_white`.
    """
    counts = events_sample.sum(dim=0).float()  # (2, H_e, W_e)
    if counts.shape[-2:] != panel_hw:
        counts = F.interpolate(counts[None], size=panel_hw, mode='nearest')[0]

    net = counts[1] - counts[0]
    abs_net = net.abs()
    peak = abs_net.max()
    scale = torch.log1p(peak) if peak > 0 else abs_net.new_tensor(1.0)
    normalized = torch.log1p(abs_net) / scale * 255.0
    zeros = torch.zeros_like(normalized)

    rgb = torch.zeros((*net.shape, 3), dtype=torch.float32, device=net.device)
    rgb[..., 2] = torch.where(net > 0, normalized, zeros)  # blue: positive/ON
    rgb[..., 0] = torch.where(net < 0, normalized, zeros)  # red: negative/OFF
    rgb[net == 0] = 255.0
    return rgb.round().clamp(0, 255).byte()
