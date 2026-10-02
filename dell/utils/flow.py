import math

import torch
import torch.nn.functional as F

from dell.utils.image import PANEL_GAP_PX, event_rgb_on_white, to_uint8_fixed


def split_flow_target(flow_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a (u, v, valid) grid into (flow, mask), zeroing the flow where it is invalid.

    DSEC decodes unwritten pixels to -256, which would otherwise count as real motion in
    anything that reduces over the whole grid, such as the panel color scale.
    """
    flow, valid = flow_gt[:, :2].float(), flow_gt[:, 2:3] > 0
    mask = valid.expand_as(flow)
    return flow * mask, mask


def resize_flow_target(flow_gt: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Resample a (u, v, valid) grid to `target_hw` by averaging the valid pixels in each window.

    Preferable to nearest-neighbor when downsampling sparse ground truth, which would lose most
    valid points. The flow is rescaled by the resize ratio, since it is measured in pixels.
    """
    orig_h, orig_w = flow_gt.shape[-2:]
    target_h, target_w = target_hw

    flow, valid = flow_gt[:, :2], flow_gt[:, 2:3].float()
    flow_masked = flow * valid

    flow_avg = F.adaptive_avg_pool2d(flow_masked, target_hw)
    valid_avg = F.adaptive_avg_pool2d(valid, target_hw)

    valid_mask = (valid_avg > 0).float()
    flow_pooled = flow_avg / valid_avg.clamp(min=1e-6)
    flow_pooled = flow_pooled * valid_mask  # zero flow where no valid source pixel existed

    scale = flow_pooled.new_tensor([target_w / orig_w, target_h / orig_h])
    flow_pooled = flow_pooled * scale.view(1, 2, 1, 1)

    return torch.cat([flow_pooled, valid_mask], dim=1)


def upsample_flow_prediction(prediction: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly upsample a flow prediction to `target_hw`, rescaling (u, v) by the resize ratio.

    Inverse of the rescaling done by `resize_flow_target`: displacements are in pixels of the grid
    they were predicted on.
    """
    pred_h, pred_w = prediction.shape[-2:]
    target_h, target_w = target_hw

    upsampled = F.interpolate(prediction.float(), size=target_hw, mode='bilinear', align_corners=False)
    scale = upsampled.new_tensor([target_w / pred_w, target_h / pred_h])

    return upsampled * scale.view(1, 2, 1, 1)


def cos_angle(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """Cosine of the angle between flow vectors, using Barron et al.'s (u, v, 1) embedding.

    The extra component avoids an undefined angle where the ground-truth flow is zero.
    """
    ones = torch.ones_like(pred[:, :1])
    pred_3d = torch.cat([pred, ones], dim=1)
    gt_3d = torch.cat([gt, ones], dim=1)

    return (pred_3d * gt_3d).sum(dim=1) / (
        pred_3d.norm(dim=1) * gt_3d.norm(dim=1)
    ).clamp(min=1e-6)  # (B, H, W)


def angular_error_degrees(cos: torch.Tensor) -> torch.Tensor:
    """Degrees between two vectors given their cosine (Barron et al.), clamped against acos's domain edges."""
    return cos.clamp(-1 + 1e-7, 1 - 1e-7).acos().rad2deg()


def modulus_loss(prediction: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean Euclidean distance between predicted and ground-truth (u, v) over valid pixels."""
    error = (prediction - flow).norm(dim=1)  # (B, H, W)
    return error[mask[:, 0]].mean()


def angular_loss_arccos(
    prediction: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor, eps: float = 1e-7,
) -> torch.Tensor:
    """Mean arccos angle (radians) between predicted and ground-truth flow over valid pixels."""
    dot = (prediction * flow).sum(dim=1)
    norms = prediction.norm(dim=1) * flow.norm(dim=1)
    cos = ((dot + eps) / (norms + eps)).clamp(-1 + eps, 1 - eps)

    return cos.acos()[mask[:, 0]].mean()  # (B, H, W) -> scalar, radians


def flow_losses_modulus_angular(
    prediction: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unweighted (modulus, angular) loss pair. Training sums them."""
    return modulus_loss(prediction, flow, mask), angular_loss_arccos(prediction, flow, mask)


PE_THRESHOLDS_PX = (1, 2, 3)  # DSEC's NPE outlier-rate thresholds, N in {1, 2, 3}


def flow_metrics(
    prediction: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor]]:
    """Average endpoint error (px), average angular error (deg) and NPE outlier rates (%).

    NPE is the share of valid pixels whose endpoint error exceeds N px, for N in `PE_THRESHOLDS_PX`.
    All tensors must be on the ground truth's pixel grid, so call this through
    `evaluate_at_gt_resolution`: angular error is not recoverable from rescaled flow.
    """
    endpoint_error = (prediction - flow).norm(dim=1)   # (B, H, W)
    valid_error = endpoint_error[mask[:, 0]]
    epe = valid_error.mean()

    angular_error = angular_error_degrees(cos_angle(prediction, flow))  # (B, H, W)
    ae = angular_error[mask[:, 0]].mean()

    pe_rates = {n: (valid_error > n).float().mean() * 100 for n in PE_THRESHOLDS_PX}

    return epe, ae, pe_rates


def evaluate_at_gt_resolution(
    prediction: torch.Tensor, flow_gt_full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Score a prediction against the unresampled ground truth, on the ground truth's own grid.

    Returns (epe, ae, pe_rates, prediction_full, flow_full, mask_full). The full-resolution
    tensors are what the panels are drawn from, so pictures match the scored comparison.
    """
    flow, mask = split_flow_target(flow_gt_full)
    prediction_full = upsample_flow_prediction(prediction, flow.shape[-2:])
    epe, ae, pe_rates = flow_metrics(prediction_full, flow, mask)

    return epe, ae, pe_rates, prediction_full, flow, mask


def evaluate_at_gt_resolution_averaged(
    predictions: list[torch.Tensor], flow_gt_full: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[int, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
    """Like `evaluate_at_gt_resolution`, for several predictions at different resolutions.

    Each prediction is upsampled separately, then they are averaged and scored.
    """
    flow, mask = split_flow_target(flow_gt_full)
    target_hw = flow.shape[-2:]

    upsampled = torch.stack([upsample_flow_prediction(prediction, target_hw) for prediction in predictions])
    prediction_full = upsampled.mean(dim=0)
    epe, ae, pe_rates = flow_metrics(prediction_full, flow, mask)

    return epe, ae, pe_rates, prediction_full, flow, mask


def _flow_colorwheel() -> torch.Tensor:
    """Middlebury color wheel (Baker et al. 2007): 55 RGB entries in 0-255, as in RAFT's flow_viz."""
    RY, YG, GC, CB, BM, MR = 15, 6, 4, 11, 13, 6
    wheel = torch.zeros(RY + YG + GC + CB + BM + MR, 3)
    col = 0

    wheel[0:RY, 0] = 255
    wheel[0:RY, 1] = torch.floor(255 * torch.arange(RY) / RY)
    col += RY

    wheel[col:col + YG, 0] = 255 - torch.floor(255 * torch.arange(YG) / YG)
    wheel[col:col + YG, 1] = 255
    col += YG

    wheel[col:col + GC, 1] = 255
    wheel[col:col + GC, 2] = torch.floor(255 * torch.arange(GC) / GC)
    col += GC

    wheel[col:col + CB, 1] = 255 - torch.floor(255 * torch.arange(CB) / CB)
    wheel[col:col + CB, 2] = 255
    col += CB

    wheel[col:col + BM, 2] = 255
    wheel[col:col + BM, 0] = torch.floor(255 * torch.arange(BM) / BM)
    col += BM

    wheel[col:col + MR, 2] = 255 - torch.floor(255 * torch.arange(MR) / MR)
    wheel[col:col + MR, 0] = 255

    return wheel  # (55, 3), values in [0, 255]


_COLORWHEEL = _flow_colorwheel()


def flow_to_rgb(flow_uv: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Encode flow as Middlebury colors: hue is direction, saturation is magnitude / `scale`.

    Zero flow is white. Magnitudes above `scale` are drawn at full saturation, dimmed by 0.75.
    """
    wheel = _COLORWHEEL.to(flow_uv.device)
    ncols = wheel.shape[0]

    u, v = flow_uv[0] / scale, flow_uv[1] / scale
    rad = (u ** 2 + v ** 2).sqrt()
    angle = torch.atan2(-v, -u) / math.pi  # (-1, 1]

    fk = (angle + 1) / 2 * (ncols - 1)
    k0 = fk.floor().long()
    k1 = (k0 + 1) % ncols
    f = (fk - k0).unsqueeze(-1)  # (H, W, 1)

    col0 = wheel[k0] / 255.0  # (H, W, 3)
    col1 = wheel[k1] / 255.0
    col = (1 - f) * col0 + f * col1  # full-saturation hue for this pixel's direction

    in_range = (rad <= 1.0).unsqueeze(-1)
    rad = rad.clamp(max=1.0).unsqueeze(-1)
    col = torch.where(in_range, 1 - rad * (1 - col), col * 0.75)

    return (col * 255).round().clamp(0, 255).byte()  # (H, W, 3)


def build_flow_comparison_panel(
    events_sample: torch.Tensor, prediction_sample: torch.Tensor,
    flow_sample: torch.Tensor, mask_sample: torch.Tensor,
) -> torch.Tensor:
    """Side-by-side panel: events | flow_gt | flow_pred | magnitude error | angle error.

    events_sample: (T, 2, H_e, W_e). prediction/flow/mask_sample: (2, H, W) for one sample, with the
    prediction already on the ground truth's grid. Returns a (H, 5W + gaps, 3) uint8 image.
    """
    mask = mask_sample[0].float()
    # The color scale is the 99th percentile of the valid ground-truth magnitudes, so one fast
    # object does not wash out the rest. Ground truth and prediction share it.
    valid_magnitude = flow_sample.norm(dim=0)[mask.bool()]
    scale = (torch.quantile(valid_magnitude, 0.99) if valid_magnitude.numel() else flow_sample.new_tensor(0.0)).clamp(min=1e-6)
    panel_hw = flow_sample.shape[-2:]

    event_rgb = event_rgb_on_white(events_sample, panel_hw)

    # Invalid ground truth is black, which keeps it distinct from valid stationary flow (white).
    flow_rgb = flow_to_rgb(flow_sample, scale)
    flow_rgb[mask == 0] = 0
    pred_rgb = flow_to_rgb(prediction_sample, scale)

    # Error maps use fixed scales so brightness is comparable across frames and epochs.
    magnitude_diff = (prediction_sample.norm(dim=0) - flow_sample.norm(dim=0)).abs() * mask
    magnitude_diff_rgb = to_uint8_fixed(magnitude_diff, scale).unsqueeze(-1).expand(-1, -1, 3)

    cos = cos_angle(prediction_sample.unsqueeze(0), flow_sample.unsqueeze(0))[0]
    angle_diff = angular_error_degrees(cos) * mask
    angle_diff_rgb = to_uint8_fixed(angle_diff, 180.0).unsqueeze(-1).expand(-1, -1, 3)

    gap = torch.full((event_rgb.shape[0], PANEL_GAP_PX, 3), 255, dtype=torch.uint8, device=event_rgb.device)

    return torch.cat(
        [event_rgb, gap, flow_rgb, gap, pred_rgb, gap, magnitude_diff_rgb, gap, angle_diff_rgb], dim=1
    )
