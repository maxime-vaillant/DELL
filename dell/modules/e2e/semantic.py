import mlflow
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional

from dell.blocks.heads.semantic import SemanticHead
from dell.models.unet import UNet
from dell.utils.firing_rate_logging import FiringRateMixin
from dell.utils.loss_trend_logging import LossTrendMixin
from dell.utils.memory_logging import PeakMemoryMixin
from dell.utils.per_class_iou_logging import PerClassIoUMixin
from dell.utils.semantic import (
    build_semantic_comparison_panel, class_names_for, evaluate_at_gt_resolution,
    evaluate_at_trunk_resolution, resize_semantic_target_soft,
    semantic_smooth_l1_loss_soft,
)
from dell.utils.video_logging import SemanticValidationVideoMixin


class EndToEndSemantic(
    SemanticValidationVideoMixin, PerClassIoUMixin, PeakMemoryMixin,
    LossTrendMixin, FiringRateMixin, LightningModule,
):
    """Semantic segmentation with a spiking U-Net trained end to end: one loss, one optimizer.

    The loss is smooth-L1 on soft labels. With `use_deep_supervision`, every decoder block gets its
    own head and the loss is the mean of all head losses. Metrics always score the trunk head alone,
    at the label's resolution.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        depth: int = 3,
        num_classes: int = 19,
        lr: float = 1e-3,
        max_epochs: int = 100,
        weight_decay: float = 1e-4,
        use_deep_supervision: bool = False,
        log_images_every_n_epochs: int = 10,
        log_video: bool = True,
        log_last_epoch_only: bool = True,
        validation_video_fps: float = 8.0,
        max_video_frames: int = 200,
        loss_trend_window: int = 10,
        val_stages: tuple[str, ...] = ('val',),
    ):
        super().__init__()
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.class_names = class_names_for(num_classes)
        self.use_deep_supervision = use_deep_supervision
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.log_video = log_video
        self.log_last_epoch_only = log_last_epoch_only
        self.validation_video_fps = validation_video_fps
        self.max_video_frames = max_video_frames
        self.loss_trend_window = loss_trend_window
        self.val_stages = val_stages
        self._training_video_frames: list = []
        self._validation_video_frames: list = []
        self._loss_history: dict[str, list[float]] = {}
        self._current_stage: str = 'train'

        self.unet = UNet(
            in_channels=in_channels, head=SemanticHead(base_channels * 2, num_classes=num_classes),
            base_channels=base_channels, depth=depth,
        )
        self._register_firing_rate_hooks()

        if self.use_deep_supervision:
            # output widths of the UNet decoder blocks
            widths = [base_channels * (2 ** level) for level in range(depth + 1)]
            decoder_out_channels = [widths[level - 1] for level in range(depth, 0, -1)]
            self.decoder_heads = nn.ModuleList(
                [SemanticHead(channels, num_classes=num_classes) for channels in decoder_out_channels]
            )

        functional.set_step_mode(self, step_mode='m')

        self._log_media_this_epoch = False
        self._train_peak_logged = False

    def _local_loss(self, logits: torch.Tensor, target_full: torch.Tensor) -> torch.Tensor:
        """Smooth-L1 loss at the logits' resolution, against the label resized down to the prediction."""
        soft_target, valid_frac = resize_semantic_target_soft(target_full, logits.shape[-2:], num_classes=self.num_classes)
        return semantic_smooth_l1_loss_soft(logits, soft_target, valid_frac)

    def _forward(self, events: torch.Tensor) -> tuple[torch.Tensor, list]:
        """Return (trunk logits, decoder logits); the latter is empty without deep supervision.

        The graph stays connected, so a single backward pass trains everything.
        """
        if not self.use_deep_supervision:
            return self.unet(events), []

        skips = []
        x = events
        for encoder_block in self.unet.encoder:
            x, skip = encoder_block(x)
            skips.append(skip)

        x = self.unet.bottleneck(x)

        decoder_predictions = []
        for decoder_block, head, skip in zip(self.unet.decoder, self.decoder_heads, reversed(skips[1:])):
            x = decoder_block(x, skip)
            decoder_predictions.append(head(x))

        x = torch.cat([x, skips[0]], dim=2)
        logits = self.unet.head(x)

        return logits, decoder_predictions

    def _combined_loss(
        self, logits: torch.Tensor, decoder_predictions: list, target: torch.Tensor, stage: str,
    ) -> torch.Tensor:
        """Trunk head loss, or with deep supervision the mean of it and every decoder head loss."""
        head_loss = self._local_loss(logits, target)
        if not self.use_deep_supervision:
            return head_loss

        losses = [head_loss]
        for layer_idx, decoder_prediction in enumerate(decoder_predictions):
            decoder_loss = self._local_loss(decoder_prediction, target)
            self.log(f'{stage}/dec_loss_layer_{layer_idx}', decoder_loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
            losses.append(decoder_loss)

        return torch.stack(losses).mean()

    def _loss_keys(self) -> list[str]:
        """Names of the train losses tracked by the trend metric, matching `LocalSemantic`'s."""
        if not self.use_deep_supervision:
            return ['loss']
        return [f'dec_loss_layer_{i}' for i in range(len(self.unet.decoder))] + ['loss']

    def _evaluate(self, logits: torch.Tensor, target: torch.Tensor):
        """Score the trunk head alone, at the label's resolution."""
        return evaluate_at_gt_resolution(logits.detach(), target, num_classes=self.num_classes)

    def _shared_step(self, batch, stage: str, metrics_stage: str | None = None):
        """Run one step. `stage` is the literal 'train' or 'val', used for firing-rate and energy.

        `metrics_stage` (default `stage`) names the logged metrics; in validation it is one of
        `self.val_stages`, so each validation dataloader is reported separately.
        """
        metrics_stage = metrics_stage or stage
        self._current_stage = stage
        self._current_metrics_stage = metrics_stage
        events, target = batch
        functional.reset_net(self)

        logits, decoder_predictions = self._forward(events)
        self._log_overall_firing_rate(stage, metrics_stage)

        loss = self._combined_loss(logits, decoder_predictions, target, metrics_stage)
        self.log(f'{metrics_stage}/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        # metrics use the label's native resolution
        mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, _, per_class_counts, prediction_full = self._evaluate(
            logits, target,
        )
        self.log(f'{metrics_stage}/miou', mean_iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{metrics_stage}/acc', pixel_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{metrics_stage}/macc', mean_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{metrics_stage}/fiou', freq_weighted_iou, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self._accumulate_per_class_iou(metrics_stage, per_class_counts)

        # second pass at the trunk's own resolution, with the label downsampled
        downsampled_mean_iou, downsampled_pixel_accuracy, downsampled_mean_accuracy, downsampled_freq_weighted_iou, _, downsampled_per_class_counts = evaluate_at_trunk_resolution(
            logits.detach(), target, num_classes=self.num_classes,
        )
        self.log(f'{metrics_stage}/miou_downsampled', downsampled_mean_iou, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{metrics_stage}/acc_downsampled', downsampled_pixel_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{metrics_stage}/macc_downsampled', downsampled_mean_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{metrics_stage}/fiou_downsampled', downsampled_freq_weighted_iou, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self._accumulate_per_class_iou(metrics_stage, downsampled_per_class_counts, resolution='downsampled')

        return events, loss, (prediction_full, target)

    def training_step(self, batch, batch_idx):
        events, loss, final_sample = self._shared_step(batch, stage='train')

        if self._log_media_this_epoch and final_sample is not None:
            prediction_full, target = final_sample
            self._accumulate_training_video_frames(events, prediction_full, target)

        return loss

    def on_train_epoch_start(self) -> None:
        self._reset_peak_memory()
        self._train_peak_logged = False
        self._log_media_this_epoch = self._should_log_media()
        self._training_video_frames = []
        self._reset_per_class_iou('train')
        self._reset_per_class_iou('train', resolution='downsampled')

    def on_train_epoch_end(self) -> None:
        self._log_loss_trends(stage='train', keys=self._loss_keys())

        if self._log_media_this_epoch and self._training_video_frames:
            self._log_training_video()
        self._training_video_frames = []
        self._log_per_class_iou_table('train')
        self._log_per_class_iou_table('train', resolution='downsampled')

        # fallback for epochs without validation; otherwise already logged by on_validation_epoch_start
        if not self._train_peak_logged:
            self._log_peak_memory('train')

    def on_validation_epoch_start(self) -> None:
        self._validation_video_frames = []
        for metrics_stage in self.val_stages:
            self._reset_per_class_iou(metrics_stage)
            self._reset_per_class_iou(metrics_stage, resolution='downsampled')

        if not self.trainer.sanity_checking:
            self._log_peak_memory('train')
            self._train_peak_logged = True
        self._reset_peak_memory()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        # dataloader_idx indexes self.val_stages (the datamodule's VAL_STAGES); panels and videos
        # come from the first dataloader only
        metrics_stage = self.val_stages[dataloader_idx]
        events, _, final_sample = self._shared_step(batch, stage='val', metrics_stage=metrics_stage)

        if dataloader_idx == 0 and self._log_media_this_epoch and final_sample is not None:
            prediction_full, target = final_sample
            if batch_idx == 0:
                panel = build_semantic_comparison_panel(events[:, 0], prediction_full[0], target[0], num_classes=self.num_classes)
                mlflow.log_image(panel.cpu().numpy(), artifact_file=f"val_images/epoch_{self.current_epoch:04d}.png")
            self._accumulate_validation_video_frames(events, prediction_full, target)

    def on_validation_epoch_end(self) -> None:
        if not self.trainer.sanity_checking:
            for metrics_stage in self.val_stages:
                self._log_loss_trends(stage=metrics_stage, keys=self._loss_keys())
                self._log_per_class_iou_table(metrics_stage)
                self._log_per_class_iou_table(metrics_stage, resolution='downsampled')

        if self._log_media_this_epoch and self._validation_video_frames:
            self._log_validation_video()
        self._validation_video_frames = []
        self._log_peak_memory('val')

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.01
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
