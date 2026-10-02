import mlflow
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional

from dell.blocks.heads.flow import FlowHead
from dell.models.unet import UNet
from dell.utils.flow import (
    build_flow_comparison_panel, evaluate_at_gt_resolution, evaluate_at_gt_resolution_averaged,
    flow_losses_modulus_angular, resize_flow_target, split_flow_target, upsample_flow_prediction,
)
from dell.utils.firing_rate_logging import FiringRateMixin
from dell.utils.loss_trend_logging import LossTrendMixin
from dell.utils.memory_logging import PeakMemoryMixin
from dell.utils.video_logging import ValidationVideoMixin


class EndToEndOpticalFlow(ValidationVideoMixin, PeakMemoryMixin, LossTrendMixin, FiringRateMixin, LightningModule):
    """Optical flow with a spiking U-Net trained end to end: one loss, one optimizer.

    With `use_deep_supervision`, each decoder block also has a flow head and the loss is the mean
    of the head losses, backpropagated once. Metrics are scored at the ground truth's resolution.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        depth: int = 3,
        lr: float = 1e-3,
        max_epochs: int = 100,
        weight_decay: float = 1e-4,
        use_separable_activation: bool = False,
        use_deep_supervision: bool = True,
        log_images_every_n_epochs: int = 10,
        log_video: bool = True,
        log_last_epoch_only: bool = True,
        validation_video_fps: float = 8.0,
        max_video_frames: int = 200,
        loss_trend_window: int = 10,
    ):
        super().__init__()
        self.lr = lr
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.use_deep_supervision = use_deep_supervision
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.log_video = log_video
        self.log_last_epoch_only = log_last_epoch_only
        self.validation_video_fps = validation_video_fps
        self.max_video_frames = max_video_frames
        self.loss_trend_window = loss_trend_window
        self._training_video_frames: list = []
        self._validation_video_frames: list = []
        self._loss_history: dict[str, list[float]] = {}
        self._current_stage: str = 'train'

        self.unet = UNet(
            in_channels=in_channels, head=FlowHead(base_channels * 2),
            base_channels=base_channels, depth=depth, use_separable_activation=use_separable_activation,
        )
        self._register_firing_rate_hooks()

        if self.use_deep_supervision:
            # output widths of the UNet decoder blocks
            widths = [base_channels * (2 ** level) for level in range(depth + 1)]
            decoder_out_channels = [widths[level - 1] for level in range(depth, 0, -1)]
            self.decoder_heads = nn.ModuleList([FlowHead(channels) for channels in decoder_out_channels])

        functional.set_step_mode(self, step_mode='m')

        self._log_media_this_epoch = False
        self._train_peak_logged = False

    def _local_loss(self, prediction: torch.Tensor, flow_gt_full: torch.Tensor) -> torch.Tensor:
        """Loss at the prediction's resolution: the label is resized down to the prediction."""
        flow_gt = resize_flow_target(flow_gt_full, prediction.shape[-2:])
        flow, mask = split_flow_target(flow_gt)

        modulus_loss, angular_loss = flow_losses_modulus_angular(prediction, flow, mask)
        return modulus_loss + angular_loss

    def _forward(self, events: torch.Tensor) -> tuple[torch.Tensor, list]:
        """Return (trunk prediction, decoder predictions); the latter is empty without deep supervision.

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
        prediction = self.unet.head(x)

        return prediction, decoder_predictions

    @torch.no_grad()
    def predict(self, events: torch.Tensor, output_hw: tuple[int, int] = (480, 640)) -> torch.Tensor:
        """Predict flow upsampled to `output_hw`, without ground truth (e.g. for the DSEC test set).

        Works outside a Lightning loop. With deep supervision, the upsampled predictions of the
        decoder heads and the trunk head are averaged, as in `_evaluate`.
        """
        functional.reset_net(self)
        prediction, decoder_predictions = self._forward(events)

        if not self.use_deep_supervision:
            return upsample_flow_prediction(prediction, output_hw)

        predictions = decoder_predictions + [prediction]
        return torch.stack([upsample_flow_prediction(p, output_hw) for p in predictions]).mean(dim=0)

    def _combined_loss(
        self, prediction: torch.Tensor, decoder_predictions: list, flow_gt: torch.Tensor, stage: str,
    ) -> torch.Tensor:
        """Trunk head loss, or with deep supervision the mean of it and every decoder head loss."""
        head_loss = self._local_loss(prediction, flow_gt)
        if not self.use_deep_supervision:
            return head_loss

        losses = [head_loss]
        for layer_idx, decoder_prediction in enumerate(decoder_predictions):
            decoder_loss = self._local_loss(decoder_prediction, flow_gt)
            self.log(f'{stage}/dec_loss_layer_{layer_idx}', decoder_loss, on_step=False, on_epoch=True, sync_dist=True)
            losses.append(decoder_loss)

        return torch.stack(losses).mean()

    def _loss_keys(self) -> list[str]:
        """Names of the train losses tracked by the trend metric, matching `LocalOpticalFlow`'s."""
        if not self.use_deep_supervision:
            return ['loss']
        return [f'dec_loss_layer_{i}' for i in range(len(self.unet.decoder))] + ['loss']

    def _evaluate(self, prediction: torch.Tensor, decoder_predictions: list, flow_gt: torch.Tensor):
        """Score at the ground truth's resolution; with deep supervision, average all head predictions."""
        if not self.use_deep_supervision:
            return evaluate_at_gt_resolution(prediction.detach(), flow_gt)

        predictions = [decoder_prediction.detach() for decoder_prediction in decoder_predictions] + [prediction.detach()]
        return evaluate_at_gt_resolution_averaged(predictions, flow_gt)

    def _shared_step(self, batch, stage: str):
        self._current_stage = stage
        events, flow_gt = batch
        functional.reset_net(self)

        prediction, decoder_predictions = self._forward(events)
        self._log_overall_firing_rate(stage)

        loss = self._combined_loss(prediction, decoder_predictions, flow_gt, stage)
        self.log(f'{stage}/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # metrics use the original resolution so they are comparable with published DSEC numbers
        epe, ae, pe_rates, prediction_full, flow_full, mask_full = self._evaluate(prediction, decoder_predictions, flow_gt)
        self.log(f'{stage}/epe', epe, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{stage}/ae', ae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for n, pe in pe_rates.items():
            self.log(f'{stage}/{n}pe', pe, on_step=False, on_epoch=True, sync_dist=True)

        return events, loss, (prediction_full, flow_full, mask_full)

    def training_step(self, batch, batch_idx):
        events, loss, final_sample = self._shared_step(batch, stage='train')

        if self._log_media_this_epoch and final_sample is not None:
            prediction, flow, mask = final_sample
            self._accumulate_training_video_frames(events, prediction, flow, mask)

        return loss

    def on_train_epoch_start(self) -> None:
        self._reset_peak_memory()
        self._train_peak_logged = False
        self._log_media_this_epoch = self._should_log_media()
        self._training_video_frames = []

    def on_train_epoch_end(self) -> None:
        self._log_loss_trends(stage='train', keys=self._loss_keys())

        if self._log_media_this_epoch and self._training_video_frames:
            self._log_training_video()
        self._training_video_frames = []

        # fallback for epochs without validation; otherwise already logged by on_validation_epoch_start
        if not self._train_peak_logged:
            self._log_peak_memory('train')

    def on_validation_epoch_start(self) -> None:
        self._validation_video_frames = []

        # validation runs inside the training epoch, so log the train peak before resetting the counter
        if not self.trainer.sanity_checking:
            self._log_peak_memory('train')
            self._train_peak_logged = True
        self._reset_peak_memory()

    def validation_step(self, batch, batch_idx):
        events, _, final_sample = self._shared_step(batch, stage='val')

        if self._log_media_this_epoch and final_sample is not None:
            prediction, flow, mask = final_sample
            if batch_idx == 0:
                panel = build_flow_comparison_panel(events[:, 0], prediction[0], flow[0], mask[0])
                mlflow.log_image(panel.cpu().numpy(), artifact_file=f"val_images/epoch_{self.current_epoch:04d}.png")
            self._accumulate_validation_video_frames(events, prediction, flow, mask)

    def on_validation_epoch_end(self) -> None:
        # the sanity check would add a pre-training entry to the loss history
        if not self.trainer.sanity_checking:
            self._log_loss_trends(stage='val', keys=self._loss_keys())

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
