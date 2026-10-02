import mlflow
import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from spikingjelly.activation_based import functional

from dell.blocks.heads.flow import FlowHead
from dell.models.unet import UNet
from dell.utils.decolle import freeze
from dell.utils.flow import (
    build_flow_comparison_panel, evaluate_at_gt_resolution, evaluate_at_gt_resolution_averaged,
    flow_losses_modulus_angular, resize_flow_target, split_flow_target, upsample_flow_prediction,
)
from dell.utils.firing_rate_logging import FiringRateMixin
from dell.utils.loss_trend_logging import LossTrendMixin
from dell.utils.memory_logging import PeakMemoryMixin
from dell.utils.video_logging import ValidationVideoMixin


class DecolleOpticalFlow(ValidationVideoMixin, PeakMemoryMixin, LossTrendMixin, FiringRateMixin, LightningModule):
    """Optical flow with DECOLLE-style learning: like `LocalOpticalFlow` with fixed random readouts.

    Same `UNet` trunk, per-block detached losses, optimizers and bookkeeping, with two differences:

    1. The heads of the encoder blocks, bottleneck and decoder blocks are frozen random
       convolutions (`freeze`). Gradients still flow through them into their block, so each block
       trains; only the head weights never change. The optimizers therefore cover blocks only. The
       trunk head is the exception: it is trained, and its prediction is the model's output.
    2. There is no deconv: each head reads its block's output directly.

    With deep supervision, the decoder predictions are averaged with the trunk prediction for the
    metrics and `predict`, as in `LocalOpticalFlow`.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        depth: int = 3,
        lr: float = 1e-3,
        max_epochs: int = 100,
        use_deep_supervision: bool = True,
        log_images_every_n_epochs: int = 10,
        log_video: bool = True,
        log_last_epoch_only: bool = True,
        validation_video_fps: float = 8.0,
        max_video_frames: int = 200,
        loss_trend_window: int = 10,
        gradient_clip_val: float = 1.0,
    ):
        super().__init__()
        self.automatic_optimization = False
        self.lr = lr
        self.max_epochs = max_epochs
        self.use_deep_supervision = use_deep_supervision
        self.log_images_every_n_epochs = log_images_every_n_epochs
        self.log_video = log_video
        self.log_last_epoch_only = log_last_epoch_only
        self.validation_video_fps = validation_video_fps
        self.max_video_frames = max_video_frames
        self.loss_trend_window = loss_trend_window
        self.gradient_clip_val = gradient_clip_val
        self._training_video_frames: list = []
        self._validation_video_frames: list = []
        self._loss_history: dict[str, list[float]] = {}
        self._current_stage: str = 'train'

        # same widths as the UNet: depth + 1 encoder stages, then depth decoder stages
        widths = [base_channels * (2 ** level) for level in range(depth + 1)]
        decoder_out_channels = [widths[level - 1] for level in range(depth, 0, -1)]

        self.unet = UNet(in_channels=in_channels, head=FlowHead(base_channels * 2), base_channels=base_channels, depth=depth)
        self._register_firing_rate_hooks()

        self.encoder_heads = nn.ModuleList([FlowHead(channels) for channels in widths])
        self.bottleneck_head = FlowHead(widths[-1])
        self.decoder_heads = nn.ModuleList([FlowHead(channels) for channels in decoder_out_channels])
        for head in [*self.encoder_heads, self.bottleneck_head, *self.decoder_heads]:
            freeze(head)

        functional.set_step_mode(self, step_mode='m')

        self._log_media_this_epoch = False
        self._train_peak_logged = False

    def _local_loss(self, prediction: torch.Tensor, flow_gt_full: torch.Tensor) -> torch.Tensor:
        """Loss at the prediction's resolution: the label is resized down to the block's prediction."""
        flow_gt = resize_flow_target(flow_gt_full, prediction.shape[-2:])
        flow, mask = split_flow_target(flow_gt)

        modulus_loss, angular_loss = flow_losses_modulus_angular(prediction, flow, mask)
        return modulus_loss + angular_loss

    def _log_layer_loss(self, stage: str, prefix: str, layer_idx: int, loss: torch.Tensor) -> None:
        """Log a block's loss. There is no per-block EPE/AE: blocks predict at different resolutions,
        where those metrics are not comparable. Only the trunk head is scored on the ground truth's grid."""
        self.log(f'{stage}/{prefix}_loss_layer_{layer_idx}', loss, on_step=False, on_epoch=True, sync_dist=True)

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
        separately, which is looser than `EndToEndOpticalFlow`'s single clip over the whole network.
        """
        if self.gradient_clip_val:
            self.clip_gradients(optimizer, gradient_clip_val=self.gradient_clip_val, gradient_clip_algorithm='norm')

    def _step_encoder(self, current: torch.Tensor, flow_gt: torch.Tensor, optimizers: list, stage: str):
        """Train every encoder block on its own loss, through its frozen head.

        Returns the detached final `down`, every block's detached `skip` (shallowest first) and every
        block's detached prediction.
        """
        skip_connections = []
        predictions = []

        encoder_iter = zip(self.unet.encoder, self.encoder_heads, optimizers)
        for layer_idx, (encoder_block, head, optimizer) in enumerate(encoder_iter):
            if optimizer is not None:
                optimizer.zero_grad()

            down, skip = encoder_block(current)
            prediction = head(down)

            loss = self._local_loss(prediction, flow_gt)
            if optimizer is not None:
                self.manual_backward(loss)
                self._clip_gradients(optimizer)
                optimizer.step()

            self._log_layer_loss(stage, 'enc', layer_idx, loss)
            predictions.append(prediction.detach())

            # detached so the next block's optimizer never reaches this block's graph
            skip_connections.append(skip.detach() if skip is not None else None)
            current = down.detach()

        return current, skip_connections, predictions

    def _step_bottleneck(self, features: torch.Tensor, flow_gt: torch.Tensor, optimizer, stage: str):
        """Train the bottleneck on its own loss, through its frozen head."""
        if optimizer is not None:
            optimizer.zero_grad()

        features = self.unet.bottleneck(features)
        prediction = self.bottleneck_head(features)

        loss = self._local_loss(prediction, flow_gt)
        if optimizer is not None:
            self.manual_backward(loss)
            self._clip_gradients(optimizer)
            optimizer.step()

        self.log(f'{stage}/bottleneck_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return features.detach(), prediction.detach()

    def _step_decoder(self, features: torch.Tensor, skip_connections: list, flow_gt: torch.Tensor, optimizers: list, stage: str):
        """Train every decoder block on its own loss, deepest first.

        Returns the last detached features and every block's detached prediction. The shallowest
        skip is left out, as in `UNet.forward`: it belongs to the head.
        """
        decoder_skips = list(reversed(skip_connections[1:]))
        predictions = []

        decoder_iter = zip(self.unet.decoder, self.decoder_heads, optimizers)
        for layer_idx, (decoder_block, head, optimizer) in enumerate(decoder_iter):
            if optimizer is not None:
                optimizer.zero_grad()

            features = decoder_block(features, decoder_skips[layer_idx])
            prediction = head(features)

            loss = self._local_loss(prediction, flow_gt)
            if optimizer is not None:
                self.manual_backward(loss)
                self._clip_gradients(optimizer)
                optimizer.step()

            self._log_layer_loss(stage, 'dec', layer_idx, loss)
            predictions.append(prediction.detach())

            features = features.detach()

        return features, predictions

    def _evaluate(self, prediction: torch.Tensor, sub_predictions: list, flow_gt: torch.Tensor):
        """Score the trunk head at the ground truth's resolution.

        With deep supervision, the other predictions are averaged in after upsampling. This only
        changes the reported metrics, not training.
        """
        if not self.use_deep_supervision:
            return evaluate_at_gt_resolution(prediction.detach(), flow_gt)

        return evaluate_at_gt_resolution_averaged(sub_predictions + [prediction.detach()], flow_gt)

    def _step_head(
        self, features: torch.Tensor, skip: torch.Tensor, sub_predictions: list,
        flow_gt: torch.Tensor, optimizer, stage: str,
    ):
        """Train the trunk head, the one trained readout, on the last decoder features and the first skip.

        Returns (prediction, flow, mask) at the ground truth's resolution.
        """
        if optimizer is not None:
            optimizer.zero_grad()

        features = torch.cat([features, skip], dim=2)
        prediction = self.unet.head(features)

        loss = self._local_loss(prediction, flow_gt)
        if optimizer is not None:
            self.manual_backward(loss)
            self._clip_gradients(optimizer)
            optimizer.step()

        self.log(f'{stage}/loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        # metrics use the original resolution, as in the DSEC leaderboard
        epe, ae, pe_rates, prediction_full, flow_full, mask_full = self._evaluate(prediction, sub_predictions, flow_gt)
        self.log(f'{stage}/epe', epe, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f'{stage}/ae', ae, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        for n, pe in pe_rates.items():
            self.log(f'{stage}/{n}pe', pe, on_step=False, on_epoch=True, sync_dist=True)

        return prediction_full, flow_full, mask_full

    @torch.no_grad()
    def predict(self, events: torch.Tensor, output_hw: tuple[int, int] = (480, 640)) -> torch.Tensor:
        """Predict flow upsampled to `output_hw`, without ground truth (e.g. for the DSEC test set).

        Follows the training wiring without losses or optimizers, and works outside a Lightning
        loop. With deep supervision, the decoder and trunk predictions are averaged after upsampling.
        """
        functional.reset_net(self)

        current = events
        skip_connections = []
        for encoder_block in self.unet.encoder:
            down, skip = encoder_block(current)
            skip_connections.append(skip)
            current = down

        features = self.unet.bottleneck(current)

        decoder_skips = list(reversed(skip_connections[1:]))
        decoder_predictions = []
        for layer_idx, (decoder_block, head) in enumerate(zip(self.unet.decoder, self.decoder_heads)):
            features = decoder_block(features, decoder_skips[layer_idx])
            decoder_predictions.append(head(features))

        features = torch.cat([features, skip_connections[0]], dim=2)
        prediction = self.unet.head(features)

        if not self.use_deep_supervision:
            return upsample_flow_prediction(prediction, output_hw)

        predictions = decoder_predictions + [prediction]
        return torch.stack([upsample_flow_prediction(p, output_hw) for p in predictions]).mean(dim=0)

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

    def _shared_step(self, batch, optimizers: list | None, stage: str):
        self._current_stage = stage
        events, flow_gt = batch
        functional.reset_net(self)

        encoder_optimizers, bottleneck_optimizer, decoder_optimizers, head_optimizer = self._split_optimizers(optimizers)

        features, skip_connections, encoder_predictions = self._step_encoder(events, flow_gt, encoder_optimizers, stage)
        features, bottleneck_prediction = self._step_bottleneck(features, flow_gt, bottleneck_optimizer, stage)
        features, decoder_predictions = self._step_decoder(features, skip_connections, flow_gt, decoder_optimizers, stage)
        sub_predictions = decoder_predictions  # only decoder predictions are averaged under deep supervision
        final_sample = self._step_head(
            features, skip_connections[0], sub_predictions, flow_gt, head_optimizer, stage,
        )
        self._log_overall_firing_rate(stage)

        return events, final_sample

    def training_step(self, batch, batch_idx):
        optimizers = self.optimizers()
        optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        events, final_sample = self._shared_step(batch, optimizers, stage='train')

        if self._log_media_this_epoch and final_sample is not None:
            prediction, flow, mask = final_sample
            self._accumulate_training_video_frames(events, prediction, flow, mask)

    def on_train_epoch_start(self) -> None:
        self._reset_peak_memory()
        self._train_peak_logged = False
        self._log_media_this_epoch = self._should_log_media()
        self._training_video_frames = []

    def on_train_epoch_end(self) -> None:
        self._log_loss_trends(stage='train', keys=self._block_loss_keys())

        schedulers = self.lr_schedulers()
        schedulers = schedulers if isinstance(schedulers, list) else [schedulers]
        for scheduler in schedulers:
            scheduler.step()

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
        events, final_sample = self._shared_step(batch, optimizers=None, stage='val')

        if self._log_media_this_epoch and final_sample is not None:
            prediction, flow, mask = final_sample
            if batch_idx == 0:
                panel = build_flow_comparison_panel(events[:, 0], prediction[0], flow[0], mask[0])
                mlflow.log_image(panel.cpu().numpy(), artifact_file=f"val_images/epoch_{self.current_epoch:04d}.png")
            self._accumulate_validation_video_frames(events, prediction, flow, mask)

    def on_validation_epoch_end(self) -> None:
        # the sanity check would add a pre-training entry to the loss history
        if not self.trainer.sanity_checking:
            self._log_loss_trends(stage='val', keys=self._block_loss_keys())

        if self._log_media_this_epoch and self._validation_video_frames:
            self._log_validation_video()
        self._validation_video_frames = []
        self._log_peak_memory('val')

    def configure_optimizers(self):
        encoder_optimizers = [
            torch.optim.AdamW(encoder_block.parameters(), lr=self.lr, weight_decay=1e-4)
            for encoder_block in self.unet.encoder
        ]
        bottleneck_optimizer = torch.optim.AdamW(self.unet.bottleneck.parameters(), lr=self.lr, weight_decay=1e-4)
        decoder_optimizers = [
            torch.optim.AdamW(decoder_block.parameters(), lr=self.lr, weight_decay=1e-4)
            for decoder_block in self.unet.decoder
        ]
        head_optimizer = torch.optim.AdamW(self.unet.head.parameters(), lr=self.lr, weight_decay=1e-4)

        optimizers = [*encoder_optimizers, bottleneck_optimizer, *decoder_optimizers, head_optimizer]
        # manual optimization disables Lightning's scheduler stepping; on_train_epoch_end steps them
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.lr * 0.01)
            for optimizer in optimizers
        ]

        return optimizers, schedulers
