import mlflow
import torch

from dell.utils.semantic import PerClassCounts, semantic_metrics_from_counts


class PerClassIoUMixin:
    """Per-class IoU table for semantic-segmentation modules, logged once per epoch.

    Each row has the class IoU, its share of the valid ground-truth pixels (`support_pct`) and a
    `low_support` flag. IoU is computed from counts summed over the whole epoch, as averaging
    per-batch IoU is biased for rare classes.

    Subclasses must set `num_classes` and `class_names`. They call `_reset_per_class_iou(stage)`
    at the start of each epoch, `_accumulate_per_class_iou(stage, counts)` at each step and
    `_log_per_class_iou_table(stage)` at the end of each epoch.

    `resolution` keeps independent tables for the same stage: 'full' is the ground-truth
    resolution; 'downsampled' is the network's own resolution and gets its own artifact folder
    and `*_downsampled` scalar names.
    """

    def _reset_per_class_iou(self, stage: str, resolution: str = 'full') -> None:
        per_class_counts = getattr(self, '_per_class_counts', None)
        if per_class_counts is None:
            per_class_counts = {}
            self._per_class_counts = per_class_counts

        per_class_counts[(stage, resolution)] = PerClassCounts(
            intersection=torch.zeros(self.num_classes),
            union=torch.zeros(self.num_classes),
            gt_count=torch.zeros(self.num_classes),
        )

    def _accumulate_per_class_iou(self, stage: str, counts: PerClassCounts, resolution: str = 'full') -> None:
        key = (stage, resolution)
        total = self._per_class_counts[key]
        self._per_class_counts[key] = PerClassCounts(
            intersection=total.intersection + counts.intersection.detach().cpu(),
            union=total.union + counts.union.detach().cpu(),
            gt_count=total.gt_count + counts.gt_count.detach().cpu(),
        )

    def _log_per_class_iou_table(
        self, stage: str, low_support_threshold_pct: float = 0.1, resolution: str = 'full',
    ) -> None:
        """Log the per-class table as a JSON artifact, plus mIoU, Acc, mAcc and fIoU scalars.

        The table ends with "[mIoU]", "[Acc]", "[mAcc]" and "[fIoU]" rows computed from the summed
        counts. The scalars (`{stage}/miou_true`, ...) are the same values. These differ from the
        `{stage}/miou` metrics, which average per-batch values.

        `mlflow.log_dict` is used instead of `mlflow.log_table`: every epoch has a new artifact
        path, and `log_table` records each path in a run tag that eventually overflows and crashes
        the run.
        """
        total = self._per_class_counts[(stage, resolution)]
        total_valid = total.gt_count.sum().clamp(min=1).item()

        rows = {"class": [], "iou_pct": [], "support_pct": [], "low_support": []}
        for cls, name in enumerate(self.class_names):
            union = total.union[cls].item()
            support_pct = 100 * total.gt_count[cls].item() / total_valid
            rows["class"].append(name)
            rows["iou_pct"].append(100 * total.intersection[cls].item() / union if union > 0 else float('nan'))
            rows["support_pct"].append(support_pct)
            rows["low_support"].append(support_pct < low_support_threshold_pct)

        mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou = semantic_metrics_from_counts(total)
        for label, value in [
            ("[mIoU]", mean_iou), ("[Acc]", pixel_accuracy), ("[mAcc]", mean_accuracy), ("[fIoU]", freq_weighted_iou),
        ]:
            rows["class"].append(label)
            rows["iou_pct"].append(100 * value)
            rows["support_pct"].append(float('nan'))
            rows["low_support"].append(False)

        artifact_dir = 'per_class_iou' if resolution == 'full' else f'per_class_iou_{resolution}'
        mlflow.log_dict(rows, artifact_file=f"{artifact_dir}/{stage}_epoch_{self.current_epoch:04d}.json")

        suffix = '' if resolution == 'full' else f'_{resolution}'
        self.log(f'{stage}/miou_true{suffix}', mean_iou, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{stage}/acc_true{suffix}', pixel_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{stage}/macc_true{suffix}', mean_accuracy, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{stage}/fiou_true{suffix}', freq_weighted_iou, on_step=False, on_epoch=True, sync_dist=True)
