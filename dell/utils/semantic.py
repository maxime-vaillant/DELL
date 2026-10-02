from typing import NamedTuple

import torch
import torch.nn.functional as F

from dell.utils.image import PANEL_GAP_PX, event_rgb_on_white


IGNORE_INDEX = 255   # label of pixels without ground truth
NUM_SEMANTIC_CLASSES = 19   # Cityscapes classes, as predicted by InternImage (the M3ED pseudo-labeler)

# Cityscapes class index -> RGB (mmsegmentation's cityscapes_palette, also used by M3ED's debug videos).
CITYSCAPES_PALETTE = torch.tensor([
    [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
    [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
    [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
    [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100],
    [0, 0, 230], [119, 11, 32],
], dtype=torch.uint8)

# Same order as CITYSCAPES_PALETTE.
CITYSCAPES_CLASS_NAMES = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign",
    "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train",
    "motorcycle", "bicycle",
]

# Merge of the 19 Cityscapes classes into 11, used for label_taxonomy="cityscapes-11".
# Index = 19-class id, value = 11-class id: background = terrain + sky, person = person + rider,
# car = car + truck + bus + train + motorcycle + bicycle, traffic-sign = traffic light + sign.
CITYSCAPES_19_TO_11 = [5, 6, 1, 9, 2, 4, 10, 10, 7, 0, 0, 3, 3, 8, 8, 8, 8, 8, 8]
NUM_SEMANTIC_CLASSES_11 = 11

# Each 11-class reuses the color of its first 19-class source.
CITYSCAPES_11_PALETTE = torch.tensor([
    [152, 251, 152],   # background
    [70, 70, 70],      # building
    [190, 153, 153],   # fence
    [220, 20, 60],     # person
    [153, 153, 153],   # pole
    [128, 64, 128],    # road
    [244, 35, 232],    # sidewalk
    [107, 142, 35],    # vegetation
    [0, 0, 142],       # car
    [102, 102, 156],   # wall
    [220, 220, 0],     # traffic-sign
], dtype=torch.uint8)

# Same order as CITYSCAPES_11_PALETTE.
CITYSCAPES_11_CLASS_NAMES = [
    "background", "building", "fence", "person", "pole", "road", "sidewalk", "vegetation",
    "car", "wall", "traffic-sign",
]


def class_names_for(num_classes: int) -> list[str]:
    """Class names for the 19-class or 11-class taxonomy; bare indices for any other count."""
    if num_classes == NUM_SEMANTIC_CLASSES:
        return CITYSCAPES_CLASS_NAMES
    if num_classes == NUM_SEMANTIC_CLASSES_11:
        return CITYSCAPES_11_CLASS_NAMES
    return [str(cls) for cls in range(num_classes)]


def resize_semantic_target(
    target: torch.Tensor, target_hw: tuple[int, int], num_classes: int = NUM_SEMANTIC_CLASSES,
) -> torch.Tensor:
    """Nearest-neighbor downsample of a (B, 1, H, W) label grid to `target_hw`.

    `num_classes` is unused and only kept so callers can swap this with other resize functions.
    """
    H, W = target.shape[-2:]
    target_h, target_w = target_hw
    if (H, W) == (target_h, target_w):
        return target.long()

    return F.interpolate(target.float(), size=target_hw, mode='nearest').long()


def resize_semantic_target_soft(
    target: torch.Tensor, target_hw: tuple[int, int], num_classes: int = NUM_SEMANTIC_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Downsample labels to `target_hw` as soft per-class shares, ignoring IGNORE_INDEX pixels.

    Returns (soft_target, valid_fraction). `soft_target` is (B, num_classes, target_h, target_w):
    each class's share of the valid source pixels in the block, all zeros where none is valid.
    `valid_fraction` is (B, 1, target_h, target_w), the share of valid source pixels; losses must
    ignore blocks where it is 0. Uses adaptive pooling, so `target_hw` need not divide the source size.
    """
    target = target.long().squeeze(1)   # (B, H, W)
    H, W = target.shape[-2:]
    target_h, target_w = target_hw
    if target_h > H or target_w > W:
        raise ValueError(
            f"resize_semantic_target_soft: target {target_h}x{target_w} is larger than source {H}x{W} "
            "(this downsamples labels onto the trunk's resolution, never upsamples them)"
        )

    valid = (target != IGNORE_INDEX).unsqueeze(1).float()   # (B, 1, H, W)

    # one_hot rejects IGNORE_INDEX, so relabel those pixels to class 0 and zero their mass via `valid`
    safe_target = target.masked_fill(valid.squeeze(1) == 0, 0)
    one_hot = F.one_hot(safe_target, num_classes=num_classes).permute(0, 3, 1, 2).float() * valid   # (B, C, H, W)

    class_mass_avg = F.adaptive_avg_pool2d(one_hot, output_size=target_hw)   # (B, C, target_h, target_w)
    valid_frac = F.adaptive_avg_pool2d(valid, output_size=target_hw)         # (B, 1, target_h, target_w)

    # Both are 0 on all-invalid windows, so the clamp gives 0 instead of NaN.
    soft_target = class_mass_avg / valid_frac.clamp(min=1e-12)
    return soft_target, valid_frac


def upsample_semantic_logits(logits: torch.Tensor, target_hw: tuple[int, int]) -> torch.Tensor:
    """Bilinearly upsample a (B, num_classes, H, W) logit map to `target_hw`."""
    return F.interpolate(logits.float(), size=target_hw, mode='bilinear', align_corners=False)


def semantic_smooth_l1_loss_soft(
    logits: torch.Tensor, soft_target: torch.Tensor, valid_frac: torch.Tensor, beta: float = 1.0,
) -> torch.Tensor:
    """Smooth-L1 between logits and a soft per-class target, over pixels with `valid_frac` > 0.

    soft_target and valid_frac are the two outputs of `resize_semantic_target_soft`, on the logits' grid.
    """
    valid = valid_frac.squeeze(1) > 0   # (B, H, W)
    per_pixel = F.smooth_l1_loss(logits, soft_target, beta=beta, reduction='none').sum(dim=1)  # (B, H, W)
    return per_pixel[valid].mean()


class PerClassCounts(NamedTuple):
    """Per-class counts of shape (num_classes,): intersection, union and ground-truth pixels.

    Unlike per-batch IoU ratios, they can be summed over an epoch, which is the unbiased way to
    get an epoch-level per-class IoU.
    """
    intersection: torch.Tensor
    union: torch.Tensor
    gt_count: torch.Tensor


def semantic_metrics(
    logits: torch.Tensor, target: torch.Tensor, num_classes: int = NUM_SEMANTIC_CLASSES,
    extra_exclude: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PerClassCounts]:
    """Compute mIoU, pixel accuracy, mean accuracy and frequency-weighted IoU (Long et al. 2015).

    Only pixels != IGNORE_INDEX count, and, if given, outside `extra_exclude` (a (B, H, W) bool mask).
    Also returns the per-class IoU (NaN for classes absent from the union) and the underlying
    `PerClassCounts`. `logits` and `target` must share the target's resolution: use
    `evaluate_at_gt_resolution`.
    """
    prediction = logits.argmax(dim=1)   # (B, H, W)
    target = target.squeeze(1)          # (B, H, W)
    valid = target != IGNORE_INDEX
    if extra_exclude is not None:
        valid = valid & ~extra_exclude
    total_valid = valid.sum().clamp(min=1)

    pixel_accuracy = (prediction[valid] == target[valid]).float().mean()

    per_class_iou = logits.new_full((num_classes,), float('nan'))
    per_class_accuracy = logits.new_full((num_classes,), float('nan'))
    per_class_freq = logits.new_zeros((num_classes,))
    per_class_intersection = logits.new_zeros((num_classes,))
    per_class_union = logits.new_zeros((num_classes,))
    per_class_gt_count = logits.new_zeros((num_classes,))
    for cls in range(num_classes):
        gt_cls = valid & (target == cls)
        pred_cls = valid & (prediction == cls)
        gt_count = gt_cls.sum()
        union = (gt_cls | pred_cls).sum()
        intersection = (gt_cls & pred_cls).sum()

        per_class_intersection[cls] = intersection
        per_class_union[cls] = union
        per_class_gt_count[cls] = gt_count

        if union > 0:
            per_class_iou[cls] = intersection.float() / union.float()
        if gt_count > 0:
            per_class_accuracy[cls] = intersection.float() / gt_count.float()
            per_class_freq[cls] = gt_count.float() / total_valid.float()

    iou_present = ~per_class_iou.isnan()
    mean_iou = per_class_iou[iou_present].mean() if iou_present.any() else logits.new_tensor(0.0)

    acc_present = ~per_class_accuracy.isnan()
    mean_accuracy = per_class_accuracy[acc_present].mean() if acc_present.any() else logits.new_tensor(0.0)

    freq_weighted_iou = (per_class_freq * per_class_iou.nan_to_num(0.0)).sum()

    per_class_counts = PerClassCounts(per_class_intersection, per_class_union, per_class_gt_count)
    return mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, per_class_iou, per_class_counts


def semantic_metrics_from_counts(counts: PerClassCounts) -> tuple[float, float, float, float]:
    """Same four metrics as `semantic_metrics`, from `PerClassCounts` summed over an epoch.

    This is a ratio of sums, unlike the `{stage}/miou` metrics Lightning logs, which average the
    per-batch values.
    """
    intersection, union, gt_count = counts.intersection, counts.union, counts.gt_count

    per_class_iou = torch.where(union > 0, intersection / union.clamp(min=1), torch.full_like(union, float('nan')))
    iou_present = ~per_class_iou.isnan()
    mean_iou = per_class_iou[iou_present].mean().item() if iou_present.any() else 0.0

    per_class_accuracy = torch.where(gt_count > 0, intersection / gt_count.clamp(min=1), torch.full_like(gt_count, float('nan')))
    acc_present = ~per_class_accuracy.isnan()
    mean_accuracy = per_class_accuracy[acc_present].mean().item() if acc_present.any() else 0.0

    total_valid = gt_count.sum().clamp(min=1)
    pixel_accuracy = (intersection.sum() / total_valid).item()

    per_class_freq = gt_count / total_valid
    freq_weighted_iou = (per_class_freq * per_class_iou.nan_to_num(0.0)).sum().item()

    return mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou


def evaluate_at_gt_resolution(
    logits: torch.Tensor, target_full: torch.Tensor, num_classes: int = NUM_SEMANTIC_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PerClassCounts, torch.Tensor]:
    """Score trunk-resolution logits against the unresampled label, on the label's own grid.

    Returns (mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, per_class_iou,
    per_class_counts, prediction_full), where `prediction_full` is the (B, H, W) argmax used for the
    comparison panel. `num_classes` must match the model's head.
    """
    logits_full = upsample_semantic_logits(logits, target_full.shape[-2:])
    mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, per_class_iou, per_class_counts = semantic_metrics(
        logits_full, target_full, num_classes=num_classes,
    )
    prediction_full = logits_full.argmax(dim=1)
    return mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, per_class_iou, per_class_counts, prediction_full


def evaluate_at_trunk_resolution(
    logits: torch.Tensor, target_full: torch.Tensor, num_classes: int = NUM_SEMANTIC_CLASSES,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, PerClassCounts]:
    """Score logits on their own grid, against the label downsampled with `resize_semantic_target`.

    Returns the first six values of `evaluate_at_gt_resolution`.
    """
    target_local = resize_semantic_target(target_full, logits.shape[-2:], num_classes=num_classes)
    mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, per_class_iou, per_class_counts = semantic_metrics(
        logits, target_local, num_classes=num_classes,
    )
    return mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, per_class_iou, per_class_counts


def colorize_labels(labels: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
    """Map (H, W) int labels to (H, W, 3) uint8 colors; labels outside the palette are black."""
    palette = palette.to(labels.device)
    in_range = (labels >= 0) & (labels < palette.shape[0])
    colors = palette[labels.clamp(0, palette.shape[0] - 1)]
    colors[~in_range] = 0
    return colors


def build_semantic_comparison_panel(
    events_sample: torch.Tensor, prediction_sample: torch.Tensor, target_sample: torch.Tensor,
    num_classes: int = NUM_SEMANTIC_CLASSES,
) -> torch.Tensor:
    """Side-by-side panel: events | label_gt | label_pred, as a (H, 3W + gaps, 3) uint8 image.

    events_sample: (T, 2, H_e, W_e). prediction_sample: (H, W) class indices on the label's grid.
    target_sample: (1, H, W). `num_classes` selects the 11-class or the native palette.
    """
    panel_hw = target_sample.shape[-2:]
    palette = CITYSCAPES_11_PALETTE if num_classes == 11 else CITYSCAPES_PALETTE

    event_rgb = event_rgb_on_white(events_sample, panel_hw)

    gt_rgb = colorize_labels(target_sample[0].long(), palette)
    pred_rgb = colorize_labels(prediction_sample.long(), palette)

    gap = torch.full((event_rgb.shape[0], PANEL_GAP_PX, 3), 255, dtype=torch.uint8, device=event_rgb.device)

    return torch.cat([event_rgb, gap, gt_rgb, gap, pred_rgb], dim=1)
