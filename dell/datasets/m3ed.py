import os
from typing import List, Optional

import tonic.transforms as transforms
from torch.utils.data import DataLoader

from dell.datasets.augmentations import Compose, FrameResize, PairedCompose, PairedRandomFlipLR, PairedRandomShift, SafeToFrame
from dell.datasets.base import BaseSpikeDataModule
from dell.datasets.download import download_file_with_retries
from dell.datasets.raw.m3ed import M3EDDataset, aws_url
from dell.utils.semantic import IGNORE_INDEX


class M3EDDataModule(BaseSpikeDataModule):
    """M3ED semantic-segmentation data module.

    Each sample is one temporal window of left Prophesee Gen4 events (1280 x 720), resized to
    144 x 256 and paired with an InternImage pseudo-label warped into the event camera frame
    (not human-annotated; see `datasets.raw.m3ed.M3EDDataset`).

    Output shapes:
        events: (T, B, 2, 144, 256)
        target: (B, 1, 720, 1280), int64 class indices 0-18 (0-10 with
                label_taxonomy="cityscapes-11"), 255 = ignore

    M3ED has no tonic support, so `download()` fetches each recording's `data.h5` and
    `semantics.h5` from M3ED's S3 bucket. Files can be tens of GB. Existing files are skipped.

    Only the `car_urban_day_*` recordings are used, with a held-out validation set. M3ED's own test
    recordings have no public `semantics.h5`. `val_dataloader` returns a one-element list so that
    modules can index `val_stages[dataloader_idx]`.

    Parameters:
        download:          fetch missing recordings in `setup()`.
        event_view:        'left' (default), 'right' or 'stereo'. Ground truth is always
                           left-referenced.
        window_duration_s: event window length (default: the time between two frames).
        target_interval_s: minimum time between two targets, e.g. 2.0 keeps one per 2 s.
        label_taxonomy:    'cityscapes-11' (default) or 'cityscapes-19'; sets `NUM_CLASSES`.
    """

    _OUTPUT_SIZE = (144, 256)     # exact 5x downsample of 720 x 1280
    INPUT_DIM = 144 * 256 * 2
    CHANNELS = 2
    _SENSOR_SIZE = M3EDDataset.sensor_size   # (1280, 720, 2)
    VAL_STAGES = ('val_car',)
    NUM_CLASSES = 11   # overridden per instance for label_taxonomy="cityscapes-19"

    _TRAIN_RECORDINGS: List[str] = [
        'car_urban_day_city_hall',
        'car_urban_day_penno_small_loop',
        'car_urban_day_rittenhouse',
        'car_urban_day_ucity_small_loop',
    ]
    _VAL_CAR_RECORDINGS: List[str] = [
        'car_urban_day_horse',
        'car_urban_day_penno_big_loop',
    ]

    def __init__(
        self,
        batch_size: int,
        data_path: Optional[str] = None,
        num_workers: int = 8,
        num_steps: int = 10,
        *,
        download: bool = True,
        event_view: str = "left",
        window_duration_s: Optional[float] = None,
        target_interval_s: Optional[float] = None,
        label_taxonomy: str = "cityscapes-11",
    ):
        super().__init__(batch_size=batch_size, data_path=data_path, num_workers=num_workers, num_steps=num_steps)
        self._download = download
        self._event_view = event_view
        self._window_duration_s = window_duration_s
        self._target_interval_s = target_interval_s
        self._label_taxonomy = label_taxonomy
        self.NUM_CLASSES = 11 if label_taxonomy == "cityscapes-11" else 19

    def download(self):
        """Download each recording's data and semantics HDF5 files, skipping existing ones."""
        os.makedirs(self._m3ed_root, exist_ok=True)
        for recording in self._TRAIN_RECORDINGS + self._VAL_CAR_RECORDINGS:
            self._download_recording_file(recording, "data.h5")
            self._download_recording_file(recording, "semantics.h5")

    def _download_recording_file(self, recording: str, suffix: str) -> None:
        local_path = os.path.join(self._m3ed_root, f"{recording}_{suffix}")
        if os.path.isfile(local_path):
            return

        download_file_with_retries(aws_url(recording, suffix), local_path, f"M3ED {recording}/{suffix}")

    @property
    def _m3ed_root(self) -> str:
        return os.path.join(self.data_path, "M3ED")

    def setup(self, stage=None):
        if self._download:
            self.download()

        train_transform = Compose([
            SafeToFrame(transforms.ToFrame(sensor_size=self._SENSOR_SIZE, n_time_bins=self.num_steps)),
            FrameResize(self._OUTPUT_SIZE),
        ])
        test_transform = Compose([
            SafeToFrame(transforms.ToFrame(sensor_size=self._SENSOR_SIZE, n_time_bins=self.num_steps)),
            FrameResize(self._OUTPUT_SIZE),
        ])

        # applied once events and target are loaded, so both are transformed together; shifted-in
        # pixels have no label and must be ignored, as 0 is a real class
        train_joint_transform = PairedCompose([
            PairedRandomFlipLR(p=0.5),
            PairedRandomShift(shift_x=0.2, shift_y=0.2, p=0.5, target_fill_value=IGNORE_INDEX),
        ])

        shared_dataset_kwargs = {
            "event_view": self._event_view,
            "window_duration_s": self._window_duration_s,
            "target_interval_s": self._target_interval_s,
            "label_taxonomy": self._label_taxonomy,
        }

        self.train_ds = M3EDDataset(
            m3ed_root=self._m3ed_root,
            recordings=self._TRAIN_RECORDINGS,
            transform=train_transform,
            joint_transform=train_joint_transform,
            **shared_dataset_kwargs,
        )
        self.val_car_ds = M3EDDataset(
            m3ed_root=self._m3ed_root,
            recordings=self._VAL_CAR_RECORDINGS,
            transform=test_transform,
            **shared_dataset_kwargs,
        )

    def val_dataloader(self) -> List[DataLoader]:
        """One loader per entry of `VAL_STAGES`."""
        return [self._make_loader(self.val_car_ds, shuffle=False)]
