import os
import zipfile
from typing import List, Literal, Optional

import tonic
import tonic.transforms as transforms

from dell.datasets.augmentations import Compose, FrameResize, PairedRandomFlipLR, SafeToFrame
from dell.datasets.base import BaseSpikeDataModule
from dell.datasets.download import download_file_with_retries
from dell.datasets.raw.common import normalize_event_view
from dell.datasets.raw.dsec import DSECDataset


_SEMANTIC_LABELS_BASE_URL = "https://download.ifi.uzh.ch/rpg/DSEC/semantic"


def _extract_semantic_labels(zip_path: str, split: str, recordings: List[str], dsec_root: str) -> None:
    """Extract the members of `recordings` from a `{split}_semantic_segmentation.zip`.

    Files go to `{dsec_root}/{recording}/semantic/...`, next to the event folders downloaded by tonic.
    """
    wanted_recordings = set(recordings)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = member.filename.split("/")
            if len(parts) < 3 or parts[0] != split or parts[1] not in wanted_recordings:
                continue
            destination = os.path.join(dsec_root, parts[1], "semantic", *parts[2:])
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            with archive.open(member) as source, open(destination, "wb") as target_file:
                target_file.write(source.read())


# Recordings with optical-flow ground truth, split by hand so that near-duplicate sequences
# (e.g. zurich_city_11_*) do not end up on both sides.
_FLOW_TRAIN_RECORDINGS: List[str] = [
    "zurich_city_01_a",
    "zurich_city_02_a",
    "zurich_city_02_c",
    "zurich_city_02_e",
    "zurich_city_05_a",
    "zurich_city_05_b",
    "zurich_city_06_a",
    "zurich_city_07_a",
    "zurich_city_09_a",
    "zurich_city_10_a",
    "zurich_city_10_b",
    "zurich_city_11_a",
    "zurich_city_11_c",
]

_FLOW_VAL_RECORDINGS: List[str] = [
    "thun_00_a",
    "zurich_city_02_d",
    "zurich_city_03_a",
    "zurich_city_08_a",
    "zurich_city_11_b",
]

# The 11 recordings with DSEC-Semantic labels, in DSEC's official split. Train and test recordings
# are downloaded with separate tonic calls, as tonic rejects a call mixing the two.
_SEMANTIC_TRAIN_RECORDINGS: List[str] = [
    "zurich_city_00_a", "zurich_city_01_a", "zurich_city_02_a", "zurich_city_04_a",
    "zurich_city_05_a", "zurich_city_06_a", "zurich_city_07_a", "zurich_city_08_a",
]
_SEMANTIC_TEST_RECORDINGS: List[str] = [
    "zurich_city_13_a", "zurich_city_14_c", "zurich_city_15_a",
]

class DSECDataModule(BaseSpikeDataModule):
    """DSEC optical-flow and semantic-segmentation data module.

    Each sample is one temporal window (~100 ms) of left DVXplorer events (640 x 480), resized to
    192 x 256 and paired with a ground-truth frame. Semantic ground truth is always left-referenced,
    whatever `event_view` is.

    Output shapes:
        events:               (T, B, 2, 192, 256)
        target optical_flow:  (B, 3, 480, 640), [u_px, v_px, valid]
        target semantic:      (B, 1, 440, 640), int64 Cityscapes trainId, 255 = ignore

    Parameters:
        target_type:       'optical_flow' (default) or 'semantic'.
        download:          fetch missing recordings in `setup()` (tonic skips existing files).
        event_view:        'left' (default), 'right' or 'stereo'.
        window_duration_s: event window length (default: the time between two frames).
        target_interval_s: minimum time between two targets, e.g. 2.0 keeps one per 2 s.
        rectify:           undistort events with the rectify map (default True).
        label_taxonomy:    'cityscapes-11' (default) or 'cityscapes-19'; semantic only. Selects
                           the label folder to read and sets `NUM_CLASSES` to 11 or 19.
    """

    _OUTPUT_SIZE = (192, 256)     # 480 × 640 → 192 × 256, preserving 4 : 3 aspect ratio
    INPUT_DIM    = 192 * 256 * 2
    CHANNELS     = 2
    _SENSOR_SIZE = DSECDataset.sensor_size   # (640, 480, 2)
    NUM_CLASSES  = -1                        # regression; set per instance for semantic targets

    # Hand-picked train / val split of the flow recordings
    _FLOW_TRAIN = _FLOW_TRAIN_RECORDINGS
    _FLOW_TEST  = _FLOW_VAL_RECORDINGS

    # DSEC-Semantic's official split
    _SEMANTIC_TRAIN = _SEMANTIC_TRAIN_RECORDINGS
    _SEMANTIC_TEST  = _SEMANTIC_TEST_RECORDINGS

    def __init__(
        self,
        *args,
        target_type: Literal["optical_flow", "semantic"] = "optical_flow",
        download: bool = True,
        event_view: str = "left",
        window_duration_s: Optional[float] = None,
        target_interval_s: Optional[float] = None,
        rectify: bool = True,
        label_taxonomy: str = "cityscapes-11",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.target_type = target_type
        self._download = download
        self._event_view = event_view
        self._window_duration_s = window_duration_s
        self._target_interval_s = target_interval_s
        self._rectify = rectify
        self._label_taxonomy = label_taxonomy
        if target_type == "semantic":
            self.NUM_CLASSES = 11 if label_taxonomy == "cityscapes-11" else 19

    def download(self):
        """Download the required files via tonic; semantic targets also need DSEC-Semantic's label zips."""
        if self.target_type == "semantic":
            data_selection = [f"events_{side}" for side in normalize_event_view(self._event_view)]
            # separate calls: tonic rejects a call mixing train and test recordings
            tonic.datasets.DSEC(save_to=self.data_path, split=self._SEMANTIC_TRAIN, data_selection=data_selection)
            tonic.datasets.DSEC(save_to=self.data_path, split=self._SEMANTIC_TEST, data_selection=data_selection)
            self._download_semantic_labels()
            return

        target_selection = [
            "optical_flow_forward_event",
            "optical_flow_forward_timestamps",
        ]

        data_selection = [f"events_{side}" for side in normalize_event_view(self._event_view)]

        tonic.datasets.DSEC(
            save_to=self.data_path,
            split="train",
            data_selection=data_selection,
            target_selection=target_selection,
        )

    def _download_semantic_labels(self) -> None:
        """Download and extract the DSEC-Semantic label zips, skipping splits already extracted."""
        for split, recordings in (("train", self._SEMANTIC_TRAIN), ("test", self._SEMANTIC_TEST)):
            already_extracted = all(
                os.path.isfile(os.path.join(self._dsec_root, recording, "semantic", f"{recording}_semantic_timestamps.txt"))
                for recording in recordings
            )
            if already_extracted:
                continue

            os.makedirs(self._dsec_root, exist_ok=True)
            zip_path = os.path.join(self._dsec_root, f"_{split}_semantic_segmentation.zip")
            url = f"{_SEMANTIC_LABELS_BASE_URL}/{split}_semantic_segmentation.zip"
            download_file_with_retries(url, zip_path, f"DSEC-Semantic {split} labels")
            _extract_semantic_labels(zip_path, split, recordings, self._dsec_root)
            os.remove(zip_path)

    @property
    def _dsec_root(self) -> str:
        return os.path.join(self.data_path, "DSEC")

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

        # applied once events and target are loaded, so both are flipped together
        train_joint_transform = PairedRandomFlipLR(
            p=0.5, negate_flow_u=(self.target_type == "optical_flow"),
        )

        if self.target_type == "optical_flow":
            train_recordings = self._FLOW_TRAIN
            test_recordings  = self._FLOW_TEST
        else:
            train_recordings = self._SEMANTIC_TRAIN
            test_recordings  = self._SEMANTIC_TEST

        shared_dataset_kwargs = {
            "target_type": self.target_type,
            "event_view": self._event_view,
            "window_duration_s": self._window_duration_s,
            "target_interval_s": self._target_interval_s,
            "rectify": self._rectify,
            "label_taxonomy": self._label_taxonomy,
        }

        self.train_ds = DSECDataset(
            dsec_root=self._dsec_root,
            recordings=train_recordings,
            transform=train_transform,
            joint_transform=train_joint_transform,
            **shared_dataset_kwargs,
        )
        self.test_ds = DSECDataset(
            dsec_root=self._dsec_root,
            recordings=test_recordings,
            transform=test_transform,
            **shared_dataset_kwargs,
        )
