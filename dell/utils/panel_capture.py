"""Helpers for scripts that run validation only to collect rendered panels or frames."""
import copy

import mlflow

from dell.datasets.augmentations import Compose, FrameResize


def silence_mlflow_artifacts() -> None:
    """Turn the MLflow artifact functions into no-ops, so a read-only pass writes nothing to the run."""
    for name in ("log_image", "log_dict", "log_text", "log_artifact"):
        setattr(mlflow, name, lambda *args, **kwargs: None)


def capture_validation_frames(model) -> list:
    """Make `trainer.validate(model)` collect a frame for every validation sample.

    Returns a list that is filled when validation ends. Panel collection is normally enabled in
    `on_train_epoch_start` and subsampled; here it is forced on and unbounded. The model's own video
    encoding and upload is replaced by the capture.
    """
    captured_frames: list = []
    model._log_media_this_epoch = True
    model.max_video_frames = 10 ** 9
    # capture when the video would be logged, as the frames are cleared right after
    model._log_validation_video = lambda: captured_frames.extend(model._validation_video_frames)
    return captured_frames


def validation_dataset(datamodule):
    """The (first) validation dataset of a datamodule."""
    val_loader = datamodule.val_dataloader()
    if isinstance(val_loader, list):
        val_loader = val_loader[0]
    return val_loader.dataset


def native_resolution_validation_dataset(datamodule, distorted: bool = False):
    """The validation dataset without its `FrameResize` step, to render events at sensor resolution.

    Event windows are the same as the model saw. Without `FrameResize`, `tonic`'s `ToFrame` returns
    a numpy array, so callers convert it to a tensor. With `distorted`, rectification is turned off
    on datasets that support it (DSEC), giving raw event coordinates.
    """
    dataset = validation_dataset(datamodule)
    native_dataset = copy.copy(dataset)
    native_dataset.transform = Compose(
        [step for step in dataset.transform.transforms if not isinstance(step, FrameResize)]
    )
    if distorted and hasattr(native_dataset, "rectify"):
        native_dataset.rectify = False
    return native_dataset
