import os
import shutil
import subprocess
import tempfile

import imageio
import mlflow
import torch

from dell.utils.flow import build_flow_comparison_panel
from dell.utils.semantic import build_semantic_comparison_panel


class VideoLoggingMixin:
    """Encodes accumulated comparison panels into an mp4 with ffmpeg and logs it to MLflow.

    Subclasses accumulate panels in `_training_video_frames` and `_validation_video_frames`.
    """

    def _encode_and_log_video(self, frames: list, artifact_path: str) -> None:
        module_name = type(self).__name__
        ffmpeg_path = shutil.which('ffmpeg')
        if ffmpeg_path is None:
            print(f"[{module_name}] ffmpeg not found on PATH — skipping {artifact_path} video logging")
            return

        with tempfile.TemporaryDirectory() as frame_directory:
            for frame_index, frame in enumerate(frames):
                imageio.v2.imwrite(os.path.join(frame_directory, f"frame_{frame_index:05d}.png"), frame)

            video_path = os.path.join(frame_directory, f"epoch_{self.current_epoch:04d}.mp4")
            try:
                subprocess.run(
                    [
                        ffmpeg_path, "-y",
                        "-framerate", str(self.validation_video_fps),
                        "-i", os.path.join(frame_directory, "frame_%05d.png"),
                        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                        "-c:v", "libx264",
                        "-pix_fmt", "yuv420p",
                        video_path,
                    ],
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as error:
                print(f"[{module_name}] ffmpeg encoding failed — skipping {artifact_path} video logging\n{error.stderr.decode(errors='replace')}")
                return

            mlflow.log_artifact(video_path, artifact_path=artifact_path)

    def _log_training_video(self) -> None:
        self._encode_and_log_video(self._training_video_frames, "train_videos")

    def _log_validation_video(self) -> None:
        self._encode_and_log_video(self._validation_video_frames, "val_videos")

    def _accumulate_subsampled(self, stage: str, frame_list: list, new_frames: list) -> None:
        """Append evenly spaced frames, keeping at most `max_video_frames` per stage.

        Buffering every panel of a large epoch can run out of memory, so frames are subsampled
        across the whole stage rather than taken from its first batches.
        """
        if not new_frames:
            return
        num_batches = self.trainer.num_training_batches if stage == 'train' else self.trainer.num_val_batches
        if isinstance(num_batches, (list, tuple)):
            num_batches = sum(num_batches)
        total_frames = (num_batches or 0) * len(new_frames)
        stride = max(1, total_frames // self.max_video_frames) if total_frames else 1

        index_attr = f"_{stage}_frame_index"
        index = getattr(self, index_attr, 0)
        frame_list.extend(frame for offset, frame in enumerate(new_frames) if (index + offset) % stride == 0)
        setattr(self, index_attr, index + len(new_frames))

    def _should_log_media(self) -> bool:
        """Whether to log panels and videos this epoch.

        Evaluated in `on_train_epoch_start`, so training frames can be collected from the first
        batch. By default only the final epoch is logged; otherwise every
        `log_images_every_n_epochs`, skipping epoch 0 (an untrained network).
        """
        if not self.log_video:
            return False
        if self.log_last_epoch_only:
            return self.current_epoch == self.trainer.max_epochs - 1
        return self.current_epoch > 0 and self.current_epoch % self.log_images_every_n_epochs == 0


class ValidationVideoMixin(VideoLoggingMixin):
    """Optical-flow comparison panels for the video logging."""

    def _panel_frames(self, events: torch.Tensor, prediction: torch.Tensor,
                       flow: torch.Tensor, mask: torch.Tensor) -> list:
        batch_size = events.shape[1]
        return [
            build_flow_comparison_panel(
                events[:, sample_index], prediction[sample_index], flow[sample_index], mask[sample_index],
            ).cpu().numpy()
            for sample_index in range(batch_size)
        ]

    def _accumulate_training_video_frames(self, events: torch.Tensor, prediction: torch.Tensor,
                                           flow: torch.Tensor, mask: torch.Tensor) -> None:
        self._accumulate_subsampled('train', self._training_video_frames, self._panel_frames(events, prediction, flow, mask))

    def _accumulate_validation_video_frames(self, events: torch.Tensor, prediction: torch.Tensor,
                                             flow: torch.Tensor, mask: torch.Tensor) -> None:
        self._accumulate_subsampled('val', self._validation_video_frames, self._panel_frames(events, prediction, flow, mask))


class SemanticValidationVideoMixin(VideoLoggingMixin):
    """Semantic-segmentation comparison panels for the video logging."""

    def _panel_frames(self, events: torch.Tensor, prediction_full: torch.Tensor, target: torch.Tensor) -> list:
        batch_size = events.shape[1]
        return [
            build_semantic_comparison_panel(
                events[:, sample_index], prediction_full[sample_index], target[sample_index],
                num_classes=self.num_classes,
            ).cpu().numpy()
            for sample_index in range(batch_size)
        ]

    def _accumulate_training_video_frames(self, events: torch.Tensor, prediction_full: torch.Tensor,
                                           target: torch.Tensor) -> None:
        self._accumulate_subsampled('train', self._training_video_frames, self._panel_frames(events, prediction_full, target))

    def _accumulate_validation_video_frames(self, events: torch.Tensor, prediction_full: torch.Tensor,
                                             target: torch.Tensor) -> None:
        self._accumulate_subsampled('val', self._validation_video_frames, self._panel_frames(events, prediction_full, target))
