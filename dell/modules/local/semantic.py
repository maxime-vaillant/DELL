import mlflow
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional

from dell.blocks.decoder import DecoderBlock
from dell.blocks.heads.semantic import SemanticHead
from dell.blocks.local.deconv import LocalExtraConv
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


class LocalSemantic(
    SemanticValidationVideoMixin, PerClassIoUMixin, PeakMemoryMixin,
    LossTrendMixin, FiringRateMixin, LightningModule,
):
    """Semantic segmentation with block-local learning: every block trains on its own loss and optimizer.

    The twin of `LocalOpticalFlow`: same `UNet` trunk as `EndToEndSemantic`, same wiring, with
    semantic heads and smooth-L1 losses on soft labels. Metrics always score the trunk head at the
    label's resolution.
    `use_deep_supervision` has no effect and is only accepted for CLI parity.

    `configure_optimizers` returns one AdamW per encoder block, the bottleneck, each decoder block
    and the trunk head, in that order, each covering the block's own head too.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        depth: int = 3,
        num_classes: int = 19,
        lr: float = 1e-3,
        max_epochs: int = 100,
        use_deep_supervision: bool = False,
        log_images_every_n_epochs: int = 10,
        log_video: bool = True,
        log_last_epoch_only: bool = True,
        validation_video_fps: float = 8.0,
        max_video_frames: int = 200,
        loss_trend_window: int = 10,
        gradient_clip_val: float = 1.0,
        val_stages: tuple[str, ...] = ('val',),
    ):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.max_epochs = max_epochs
        self.num_classes = num_classes
        self.class_names = class_names_for(num_classes)
        self.use_deep_supervision = use_deep_supervision
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.log_video = log_video
        self.log_last_epoch_only = log_last_epoch_only
        self.validation_video_fps = validation_video_fps
        self.max_video_frames = max_video_frames
        self.loss_trend_window = loss_trend_window
        self.gradient_clip_val = gradient_clip_val
        self.val_stages = val_stages
        self._training_video_frames: list = []
        self._validation_video_frames: list = []
        self._loss_history: dict[str, list[float]] = {}
        self._current_stage: str = 'train'

        # same widths as the UNet: depth + 1 encoder stages, then depth decoder stages
        widths = [base_channels * (2 ** level) for level in range(depth + 1)]
        decoder_out_channels = [widths[level - 1] for level in range(depth, 0, -1)]
        self.unet = UNet(
            in_channels=in_channels, head=SemanticHead(base_channels * 2, num_classes=num_classes),
            base_channels=base_channels, depth=depth,
        )
        self._register_firing_rate_hooks()

        # level 0 does not pool, so it gets a same-width conv; deeper levels get a DecoderBlock
        self.encoder_deconvs = nn.ModuleList(
            [LocalExtraConv(widths[0])]
            + [
                DecoderBlock(in_channels=widths[level], out_channels=widths[level - 1], skip_channels=widths[level])
                for level in range(1, depth + 1)
            ]
        )
        encoder_head_channels = [widths[0]] + widths[:depth]
        self.encoder_heads = nn.ModuleList(
            [SemanticHead(channels, num_classes=num_classes) for channels in encoder_head_channels]
        )

        # the bottleneck does not pool either, so it also gets a plain extra conv
        self.bottleneck_deconv = LocalExtraConv(widths[-1])
        self.bottleneck_head = SemanticHead(widths[-1], num_classes=num_classes)
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

    def _log_layer_loss(self, stage: str, prefix: str, layer_idx: int, loss: torch.Tensor) -> None:
        """Log a block's loss. There is no per-block mIoU/Acc: blocks predict at different
        resolutions. Only the trunk head is scored on the label's grid."""
        self.log(f'{stage}/{prefix}_loss_layer_{layer_idx}', loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

    def _block_loss_keys(self) -> list[str]:
        """Names of every block's loss, in the order of the optimizers."""
        num_encoder_blocks, num_decoder_blocks = len(self.unet.encoder), len(self.unet.decoder)
        return (
            [f'enc_loss_layer_{i}' for i in range(num_encoder_blocks)]
            + ['bottleneck_loss']
            + [f'dec_loss_layer_{i}' for i in range(num_decoder_blocks)]
            + ['loss']
        )

    def _clip_gradients(self, optimizer: torch.optim.Optimizer) -> None:
        """Clip the gradient norm of one block.

        Trainer-level clipping does not work with manual optimization, so each block is clipped
        separately, which is looser than `EndToEndSemantic`'s single clip over the whole network.
        """
        if self.gradient_clip_val:
            self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm='norm')

    def _step_encoder(self, current: torch.Tensor, target_full: torch.Tensor, optimizers: list, stage: str):
        """Train every encoder block on its own loss.

        Returns the detached final `down` and every block's detached `skip` (shallowest first). A
        block's `down` and `skip` come from the same call, so fusing them in its head does not let
        gradients cross blocks.
        """
        skip_connections = []

        encoder_iter = zip(self.unet.encoder, self.encoder_deconvs, self.encoder_heads, optimizers)
        for layer_idx, (encoder_block, deconv, head, optimizer) in enumerate(encoder_iter):
            if optimizer is not None:
                optimizer.zero_grad()

            down, skip = encoder_block(current)
            logits = head(deconv(down, skip))

            loss = self._local_loss(logits, target_full)
            if optimizer is not None:
                self.manual_backward(loss)
                self._clip_gradients(optimizer)
                optimizer.step()

            self._log_layer_loss(stage, 'enc', layer_idx, loss)

            # detached so the next block's optimizer never reaches this block's graph
            skip_connections.append(skip.detach() if skip is not None else None)
            current = down.detach()

        return current, skip_connections

    def _step_bottleneck(self, features: torch.Tensor, target_full: torch.Tensor, optimizer, stage: str):
        """Train the bottleneck on its own loss, reading through `bottleneck_deconv`."""
        if optimizer is not None:
            optimizer.zero_grad()

        features = self.unet.bottleneck(features)
        logits = self.bottleneck_head(self.bottleneck_deconv(features))

        loss = self._local_loss(logits, target_full)
        if optimizer is not None:
            self.manual_backward(loss)
            self._clip_gradients(optimizer)
            optimizer.step()

        self.log(f'{stage}/bottleneck_loss', loss, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        return features.detach()

    def _step_decoder(self, features: torch.Tensor, skip_connections: list, target_full: torch.Tensor, optimizers: list, stage: str):
        """Train every decoder block on its own loss, deepest first, and return the last detached features.

        The shallowest skip is left out, as in `UNet.forward`: it belongs to the head.
        """
        decoder_skips = list(reversed(skip_connections[1:]))

        decoder_iter = zip(self.unet.decoder, self.decoder_heads, optimizers)
        for layer_idx, (decoder_block, head, optimizer) in enumerate(decoder_iter):
            if optimizer is not None:
                optimizer.zero_grad()

            features = decoder_block(features, decoder_skips[layer_idx])
            logits = head(features)

            loss = self._local_loss(logits, target_full)
            if optimizer is not None:
                self.manual_backward(loss)
                self._clip_gradients(optimizer)
                optimizer.step()

            self._log_layer_loss(stage, 'dec', layer_idx, loss)

            features = features.detach()

        return features

    def _evaluate(self, logits: torch.Tensor, target_full: torch.Tensor):
        """Score the trunk head alone, at the label's resolution."""
        return evaluate_at_gt_resolution(logits.detach(), target_full, num_classes=self.num_classes)

    def _step_head(
        self, features: torch.Tensor, skip: torch.Tensor,
        target_full: torch.Tensor, optimizer, stage: str,
    ):
        """Train the trunk head on the last decoder features and the first skip.

        Returns (prediction_full, target) at the label's resolution.
        """
        if optimizer is not None:
            optimizer.zero_grad()

        features = torch.cat([features, skip], dim=2)
        logits = self.unet.head(features)

        loss = self._local_loss(logits, target_full)
        if optimizer is not None:
            self.manual_backward(loss)
            self._clip_gradients(optimizer)
            optimizer.step()

        self.log(f'{stage}/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)

        # metrics use the label's native resolution
        mean_iou, pixel_accuracy, mean_accuracy, freq_weighted_iou, _, per_class_counts, prediction_full = self._evaluate(
            logits, target_full,
        )
        self.log(f'{stage}/miou', mean_iou, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{stage}/acc', pixel_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{stage}/macc', mean_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{stage}/fiou', freq_weighted_iou, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self._accumulate_per_class_iou(stage, per_class_counts)

        # second pass at the trunk's own resolution, with the label downsampled
        downsampled_mean_iou, downsampled_pixel_accuracy, downsampled_mean_accuracy, downsampled_freq_weighted_iou, _, downsampled_per_class_counts = evaluate_at_trunk_resolution(
            logits.detach(), target_full, num_classes=self.num_classes,
        )
        self.log(f'{stage}/miou_downsampled', downsampled_mean_iou, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{stage}/acc_downsampled', downsampled_pixel_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{stage}/macc_downsampled', downsampled_mean_accuracy, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self.log(f'{stage}/fiou_downsampled', downsampled_freq_weighted_iou, on_step=False, on_epoch=True, sync_dist=True, add_dataloader_idx=False)
        self._accumulate_per_class_iou(stage, downsampled_per_class_counts, resolution='downsampled')

        return prediction_full, target_full

    def _split_optimizers(self, optimizers: list | None):
        """Split the optimizer list into encoder, bottleneck, decoder and head optimizers."""
        num_encoder_blocks, num_decoder_blocks = len(self.unet.encoder), len(self.unet.decoder)

        if optimizers is None:
            return [None] * num_encoder_blocks, None, [None] * num_decoder_blocks, None

        encoder_optimizers = optimizers[:num_encoder_blocks]
        bottleneck_optimizer = optimizers[num_encoder_blocks]
        decoder_optimizers = optimizers[num_encoder_blocks + 1:num_encoder_blocks + 1 + num_decoder_blocks]
        head_optimizer = optimizers[-1]

        return encoder_optimizers, bottleneck_optimizer, decoder_optimizers, head_optimizer

    def _shared_step(self, batch, optimizers: list | None, stage: str, metrics_stage: str | None = None):
        """Run one step. `stage` is the literal 'train' or 'val', used for firing-rate and energy.

        `metrics_stage` (default `stage`) names the logged metrics; in validation it is one of
        `self.val_stages`, so each validation dataloader is reported separately.
        """
        metrics_stage = metrics_stage or stage
        self._current_stage = stage
        self._current_metrics_stage = metrics_stage
        events, target = batch
        functional.reset_net(self)

        encoder_optimizers, bottleneck_optimizer, decoder_optimizers, head_optimizer = self._split_optimizers(optimizers)

        features, skip_connections = self._step_encoder(events, target, encoder_optimizers, metrics_stage)
        features = self._step_bottleneck(features, target, bottleneck_optimizer, metrics_stage)
        features = self._step_decoder(features, skip_connections, target, decoder_optimizers, metrics_stage)
        final_sample = self._step_head(
            features, skip_connections[0], target, head_optimizer, metrics_stage,
        )
        self._log_overall_firing_rate(stage, metrics_stage)

        return events, final_sample

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        events, final_sample = self._shared_step(batch, optimizers, stage='train')

        if self._log_media_this_epoch and final_sample is not None:
            prediction_full, target = final_sample
            self._accumulate_training_video_frames(events, prediction_full, target)

    def on_train_epoch_start(self) -> None:
        self._reset_peak_memory()
        self._train_peak_logged = False
        self._log_media_this_epoch = self._should_log_media()
        self._training_video_frames = []
        self._reset_per_class_iou('train')
        self._reset_per_class_iou('train', resolution='downsampled')

    def on_train_epoch_end(self) -> None:
        self._log_loss_trends(stage='train', keys=self._block_loss_keys())

        schedulers = self.lr_schedulers()
        schedulers = schedulers if isinstance(schedulers, list) else [schedulers]
        for scheduler in schedulers:
            scheduler.step()

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

        # validation runs inside the training epoch, so log the train peak before resetting the counter
        if not self.trainer.sanity_checking:
            self._log_peak_memory('train')
            self._train_peak_logged = True
        self._reset_peak_memory()

    def validation_step(self, batch, batch_idx, dataloader_idx: int = 0):
        # dataloader_idx indexes self.val_stages (the datamodule's VAL_STAGES); panels and videos
        # come from the first dataloader only
        metrics_stage = self.val_stages[dataloader_idx]
        events, final_sample = self._shared_step(batch, optimizers=None, stage='val', metrics_stage=metrics_stage)

        if dataloader_idx == 0 and self._log_media_this_epoch and final_sample is not None:
            prediction_full, target = final_sample
            if batch_idx == 0:
                panel = build_semantic_comparison_panel(events[:, 0], prediction_full[0], target[0], num_classes=self.num_classes)
                mlflow.log_image(panel.cpu().numpy(), artifact_file=f"val_images/epoch_{self.current_epoch:04d}.png")
            self._accumulate_validation_video_frames(events, prediction_full, target)

    def on_validation_epoch_end(self) -> None:
        # the sanity check would add a pre-training entry to the loss history
        if not self.trainer.sanity_checking:
            for metrics_stage in self.val_stages:
                self._log_loss_trends(stage=metrics_stage, keys=self._block_loss_keys())
                self._log_per_class_iou_table(metrics_stage)
                self._log_per_class_iou_table(metrics_stage, resolution='downsampled')

        if self._log_media_this_epoch and self._validation_video_frames:
            self._log_validation_video()
        self._validation_video_frames = []
        self._log_peak_memory('val')

    def configure_optimizers(self):
        encoder_optimizers = [
            torch.optim.AdamW(
                list(encoder_block.parameters()) + list(deconv.parameters()) + list(head.parameters()),
                lr=self.lr, weight_decay=1e-4,
            )
            for encoder_block, deconv, head in zip(self.unet.encoder, self.encoder_deconvs, self.encoder_heads)
        ]
        bottleneck_optimizer = torch.optim.AdamW(
            list(self.unet.bottleneck.parameters()) + list(self.bottleneck_deconv.parameters())
            + list(self.bottleneck_head.parameters()),
            lr=self.lr, weight_decay=1e-4,
        )
        decoder_optimizers = [
            torch.optim.AdamW(list(decoder_block.parameters()) + list(head.parameters()), lr=self.lr, weight_decay=1e-4)
            for decoder_block, head in zip(self.unet.decoder, self.decoder_heads)
        ]
        head_optimizer = torch.optim.AdamW(self.unet.head.parameters(), lr=self.lr, weight_decay=1e-4)

        optimizers = [*encoder_optimizers, bottleneck_optimizer, *decoder_optimizers, head_optimizer]
        # manual optimization disables Lightning's scheduler stepping; on_train_epoch_end steps them
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.01)
            for optimizer in optimizers
        ]

        return optimizers, schedulers
