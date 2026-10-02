"""Lookup tables used by run.py and infer.py.

To add a model or dataset, register it in `MODULES` or `DATASETS` (and `TASK_DATASETS` for a new
task/dataset pairing). `MODEL_ARGS` and `DATA_ARGS` list the CLI options forwarded to the model
and datamodule constructors.
"""
from pytorch_lightning import LightningDataModule, LightningModule

from dell.datasets.dsec import DSECDataModule
from dell.datasets.m3ed import M3EDDataModule
from dell.modules.decolle.optical_flow import DecolleOpticalFlow
from dell.modules.decolle.semantic import DecolleSemantic
from dell.modules.e2e.optical_flow import EndToEndOpticalFlow
from dell.modules.e2e.semantic import EndToEndSemantic
from dell.modules.local.optical_flow import LocalOpticalFlow
from dell.modules.local.semantic import LocalSemantic

MODULES: dict[tuple[str, str], type[LightningModule]] = {
    ("e2e", "optical_flow"): EndToEndOpticalFlow,
    ("local", "optical_flow"): LocalOpticalFlow,
    ("decolle", "optical_flow"): DecolleOpticalFlow,
    ("e2e", "semantic"): EndToEndSemantic,
    ("local", "semantic"): LocalSemantic,
    ("decolle", "semantic"): DecolleSemantic,
}

DATASETS: dict[str, type[LightningDataModule]] = {
    "dsec": DSECDataModule,
    "m3ed": M3EDDataModule,
}

# Datasets with ground truth for each task.
TASK_DATASETS: dict[str, set[str]] = {
    "optical_flow": {"dsec"},
    "semantic": {"dsec", "m3ed"},
}

MODEL_ARGS = {
    "base_channels", "depth", "lr", "weight_decay", "use_deep_supervision", "use_separable_activation",
    "log_images_every_n_epochs", "log_video", "log_last_epoch_only", "validation_video_fps",
    "max_video_frames", "gradient_clip_val",
}
DATA_ARGS = {
    "batch_size", "data_path", "num_workers", "num_steps",
    "target_type", "window_duration_s", "target_interval_s", "label_taxonomy",
}
