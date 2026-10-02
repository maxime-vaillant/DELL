import os
import random
import warnings
from typing import Any, Callable, Dict, List, Literal, Optional, Set, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dell.datasets.raw.common import DEFAULT_WINDOW_DURATION_US, normalize_event_view


# Recordings of the optical-flow benchmark's held-out test split, which has no ground truth.
FLOW_TEST_RECORDINGS = [
    "interlaken_00_b", "interlaken_01_a", "thun_01_a", "thun_01_b",
    "zurich_city_12_a", "zurich_city_14_c", "zurich_city_15_a",
]

# DSEC-Semantic ships labels for both taxonomies, so no merge is needed (unlike M3ED).
_LABEL_TAXONOMY_TO_FOLDER = {"cityscapes-11": "11classes", "cityscapes-19": "19classes"}


def _ensure_hdf5_plugin_ready() -> None:
    """Strip invalid HDF5_PLUGIN_PATH entries and inject the real hdf5plugin path."""
    current = os.environ.get("HDF5_PLUGIN_PATH", "")
    valid_paths: List[str] = [
        path for path in current.split(os.pathsep) if path and os.path.isdir(path)
    ]
    try:
        import hdf5plugin
        if os.path.isdir(hdf5plugin.PLUGIN_PATH) and hdf5plugin.PLUGIN_PATH not in valid_paths:
            valid_paths.insert(0, hdf5plugin.PLUGIN_PATH)
    except ImportError as exc:
        raise RuntimeError(
            "DSEC event files require hdf5plugin. Install it with `pip install hdf5plugin`."
        ) from exc
    os.environ["HDF5_PLUGIN_PATH"] = os.pathsep.join(valid_paths)


class DSECDataset(Dataset):
    """`DSEC <https://dsec.ifi.uzh.ch/>`_ event dataset: one event window and one target per sample.

    Event HDF5 files stay open for the dataset's lifetime. Events are optionally undistorted with
    each recording's `rectify_map.h5`.

    Targets:
        'optical_flow': float tensor (3, 480, 640) with channels [u_px, v_px, valid_mask], decoded
            from DSEC's 16-bit PNGs.
        'semantic': int64 tensor (1, 440, 640) of Cityscapes trainIds from DSEC-Semantic's labels,
            0-10 for 'cityscapes-11' or 0-18 for 'cityscapes-19', 255 = ignore. The labels have 440
            rows because the frame camera never captured the bottom 40 rows.
        'optical_flow_test': the benchmark's held-out test split, which has no ground truth.
            `__getitem__` returns `(events, file_index)` instead of `(events, target)`.

    Parameters:
        dsec_root:         the `DSEC/` directory created by tonic, i.e. `{save_to}/DSEC`.
        recordings:        recording names to include.
        target_type:       see above.
        transform:         applied to the raw event array after loading.
        joint_transform:   optional `(events, target) -> (events, target)` applied last, so spatial
                           augmentations such as flips stay consistent between both. Events are a
                           dict of per-side tensors for `event_view='stereo'`.
        event_view:        'left' (default), 'right' or 'stereo' (returns a {'left', 'right'} dict).
        window_duration_s: if set, use a fixed window ending at the target timestamp instead of the
                           window between two frames. The flow label's own interval is unchanged.
        target_interval_s: minimum time between two kept targets; None keeps all of them.
        rectify:           undistort events with `rectify_map.h5` when present. It is read on every
                           `__getitem__`, so it can be toggled on an existing instance.
        label_taxonomy:    'cityscapes-11' (default) or 'cityscapes-19'; semantic only, selects the
                           label folder to read.
    """

    sensor_size = (640, 480, 2)   # DVXplorer left/right cameras
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])

    def __init__(
        self,
        dsec_root: str,
        recordings: List[str],
        target_type: Literal["optical_flow", "semantic", "optical_flow_test"] = "optical_flow",
        transform: Optional[Callable] = None,
        joint_transform: Optional[Callable[[Any, torch.Tensor], Tuple[Any, torch.Tensor]]] = None,
        event_view: str = "left",
        window_duration_s: Optional[float] = None,
        target_interval_s: Optional[float] = None,
        rectify: bool = True,
        label_taxonomy: str = "cityscapes-11",
    ):
        self.dsec_root = dsec_root
        self.target_type = target_type
        self.transform = transform
        self.joint_transform = joint_transform
        self.rectify = rectify
        self.label_taxonomy = label_taxonomy
        self.window_duration_us = (
            int(round(window_duration_s * 1_000_000)) if window_duration_s is not None else None
        )
        self.target_interval_us = (
            int(round(target_interval_s * 1_000_000)) if target_interval_s is not None else None
        )

        self._event_sides: Tuple[str, ...] = normalize_event_view(event_view)
        self._is_stereo = len(self._event_sides) == 2

        _ensure_hdf5_plugin_ready()

        # Init before any code path that can raise so __del__ is always safe.
        self._events_files: Dict[str, Dict[str, h5py.File]] = {}
        self._event_stores: Dict[str, Dict[str, Any]] = {}
        self._recording_meta: Dict[str, Dict[str, Any]] = {}

        self._index = self._build_index(recordings)
        active_recordings: Set[str] = {entry[0] for entry in self._index}
        self._recording_meta = self._build_recording_meta(active_recordings)
        self._open_events_files()

    def _recording_path(self, recording: str) -> str:
        return os.path.join(self.dsec_root, recording)

    def _list_root_contents(self) -> str:
        if not os.path.isdir(self.dsec_root):
            return "(directory does not exist)"
        entries = sorted(os.listdir(self.dsec_root))
        preview = entries[:6]
        suffix = f" … (+{len(entries) - 6} more)" if len(entries) > 6 else ""
        return str(preview) + suffix

    def _build_index(self, recordings: List[str]) -> List[Tuple]:
        index_fn = {
            "optical_flow": self._index_flow,
            "semantic": self._index_semantic,
            "optical_flow_test": self._index_flow_test,
        }[self.target_type]

        index: List[Tuple] = []
        skipped: List[str] = []
        for recording in recordings:
            entries = index_fn(recording)
            if entries:
                index.extend(entries)
            else:
                skipped.append(recording)

        if skipped:
            warnings.warn(
                f"DSECDataset: {len(skipped)} recording(s) had no "
                f"'{self.target_type}' files and were skipped: {skipped}",
                stacklevel=3,
            )

        if not index:
            example_recording = recordings[0] if recordings else "<none>"
            if self.target_type == "optical_flow":
                example_path = os.path.join(
                    self.dsec_root, example_recording,
                    "optical_flow_forward_timestamps",
                    f"{example_recording}_optical_flow_forward_timestamps.txt",
                )
            elif self.target_type == "optical_flow_test":
                example_path = os.path.join(
                    self.dsec_root, example_recording, "test_forward_flow_timestamps.csv",
                )
            else:
                example_path = os.path.join(
                    self.dsec_root, example_recording,
                    "semantic", f"{example_recording}_semantic_timestamps.txt",
                )
            raise RuntimeError(
                f"No samples found under '{self.dsec_root}' "
                f"for target_type='{self.target_type}'.\n"
                f"Checked {len(recordings)} recording(s).\n"
                f"Example expected path:\n  {example_path}\n"
                f"Root directory exists: {os.path.isdir(self.dsec_root)}\n"
                f"Root contents: {self._list_root_contents()}\n\n"
                "Fix options:\n"
                "  1. Run DSECDataModule(..., download=True).setup() once to fetch missing files.\n"
                "  2. Verify data_path points to the parent of the DSEC/ folder\n"
                "     (e.g. data_path='/home/user/data' → '/home/user/data/DSEC/<recording>/').\n"
                "  3. If tonic stored data elsewhere, pass dsec_root= explicitly to DSECDataset."
            )
        return index

    def _index_flow(self, recording: str) -> List[Tuple]:
        base = self._recording_path(recording)
        timestamps_file = os.path.join(
            base, "optical_flow_forward_timestamps",
            f"{recording}_optical_flow_forward_timestamps.txt",
        )
        flow_directory = os.path.join(base, "optical_flow_forward_event")

        if not os.path.exists(timestamps_file) or not os.path.isdir(flow_directory):
            return []

        with open(timestamps_file) as file:
            lines = [line for line in file.readlines()[1:] if line.strip()]   # skip header
        timestamps = [
            (int(line.split(", ")[0]), int(line.split(", ")[1].strip())) for line in lines
        ]
        flow_files = sorted(
            os.path.join(flow_directory, filename)
            for filename in os.listdir(flow_directory)
            if filename.endswith(".png")
        )
        entries: List[Tuple] = []
        last_kept_timestamp: Optional[int] = None
        for (native_t_start, t_stop), path in zip(timestamps, flow_files):
            if (
                self.target_interval_us is not None
                and last_kept_timestamp is not None
                and t_stop - last_kept_timestamp < self.target_interval_us
            ):
                continue

            # t_stop stays pinned to the native value: it anchors the flow label's
            # displacement interval. Only the input event window start is overridden.
            t_start = (
                t_stop - self.window_duration_us
                if self.window_duration_us is not None
                else native_t_start
            )
            entries.append((recording, t_start, t_stop, path))
            last_kept_timestamp = t_stop
        return entries

    def _index_flow_test(self, recording: str) -> List[Tuple]:
        """Indexes the DSEC optical-flow *benchmark*'s own held-out test split: these recordings
        ship no ground-truth flow PNGs, only a `test_forward_flow_timestamps.csv` (columns
        `from_timestamp_us, to_timestamp_us, file_index`, see
        https://dsec.ifi.uzh.ch/optical-flow-submission-format/) naming which windows the
        evaluation server scores and which index to name each prediction file after. t_stop stays
        pinned to `to_timestamp_us` -- same convention as _index_flow's own native t_stop -- so
        window_duration_s only ever overrides t_start. The file itself is expected at
        `{dsec_root}/{recording}/test_forward_flow_timestamps.csv` (see download_dsec_test.py,
        which lays test recordings out this way already)."""
        base = self._recording_path(recording)
        timestamps_file = os.path.join(base, "test_forward_flow_timestamps.csv")
        if not os.path.exists(timestamps_file):
            return []

        with open(timestamps_file) as file:
            lines = [line for line in file if line.strip() and not line.startswith("#")]

        entries: List[Tuple] = []
        for line in lines:
            from_us, to_us, file_index = (int(value) for value in line.split(","))
            t_start = (
                to_us - self.window_duration_us
                if self.window_duration_us is not None
                else from_us
            )
            entries.append((recording, t_start, to_us, file_index))
        return entries

    def _index_single_timestamp_target(
        self, recording: str, timestamps_file: str, target_directory: str,
    ) -> List[Tuple]:
        """Indexing logic for semantic targets, which have one timestamp + one PNG per frame:
        t_stop is the frame's own timestamp (one per line in
        timestamps_file); t_start falls back to the previous frame's own t_stop (or
        DEFAULT_WINDOW_DURATION_US before the very first frame) unless window_duration_s
        overrides it."""
        if not os.path.exists(timestamps_file) or not os.path.isdir(target_directory):
            return []

        with open(timestamps_file) as file:
            timestamps = [int(line.strip()) for line in file if line.strip()]
        target_files = sorted(
            os.path.join(target_directory, filename)
            for filename in os.listdir(target_directory)
            if filename.endswith(".png")
        )
        entries: List[Tuple] = []
        last_kept_timestamp: Optional[int] = None
        for frame_index, (path, t_stop) in enumerate(zip(target_files, timestamps)):
            if (
                self.target_interval_us is not None
                and last_kept_timestamp is not None
                and t_stop - last_kept_timestamp < self.target_interval_us
            ):
                continue

            if self.window_duration_us is not None:
                t_start = t_stop - self.window_duration_us
            elif frame_index > 0:
                t_start = timestamps[frame_index - 1]
            else:
                t_start = t_stop - DEFAULT_WINDOW_DURATION_US
            entries.append((recording, t_start, t_stop, path))
            last_kept_timestamp = t_stop
        return entries

    def _index_semantic(self, recording: str) -> List[Tuple]:
        base = self._recording_path(recording)
        timestamps_file = os.path.join(base, "semantic", f"{recording}_semantic_timestamps.txt")
        label_directory = os.path.join(base, "semantic", _LABEL_TAXONOMY_TO_FOLDER[self.label_taxonomy])
        return self._index_single_timestamp_target(recording, timestamps_file, label_directory)

    def _build_recording_meta(self, recordings: Set[str]) -> Dict[str, Dict[str, Any]]:
        expected_rectify_shape = (self.sensor_size[1], self.sensor_size[0], 2)
        side_to_folder = {"left": "events_left", "right": "events_right"}
        meta: Dict[str, Dict[str, Any]] = {}

        for recording in recordings:
            base = self._recording_path(recording)
            side_metadata: Dict[str, Any] = {}

            for side in self._event_sides:
                folder_name = side_to_folder[side]
                events_path = os.path.join(base, folder_name, "events.h5")
                rectify_map_path = os.path.join(base, folder_name, "rectify_map.h5")

                with h5py.File(events_path, "r") as events_file:
                    t_offset = int(events_file["t_offset"][()])
                    ms_to_idx = events_file["ms_to_idx"][()]
                    total_events = int(events_file["events"]["t"].shape[0])

                rectify_map = None
                if os.path.isfile(rectify_map_path):
                    with h5py.File(rectify_map_path, "r") as rectify_file:
                        loaded_map = rectify_file["rectify_map"][()]
                    if loaded_map.shape == expected_rectify_shape:
                        rectify_map = loaded_map
                    else:
                        warnings.warn(
                            f"Unexpected rectify_map shape {loaded_map.shape} for "
                            f"{recording}/{side} (expected {expected_rectify_shape}), "
                            f"skipping rectification for this side."
                        )

                side_metadata[side] = {
                    "events_path": events_path,
                    "t_offset": t_offset,
                    "ms_to_idx": ms_to_idx,
                    "total_events": total_events,
                    "rectify_map": rectify_map,
                }

            meta[recording] = side_metadata
        return meta

    def _open_events_files(self) -> None:
        for recording, side_metadata in self._recording_meta.items():
            self._events_files[recording] = {}
            self._event_stores[recording] = {}
            for side, metadata in side_metadata.items():
                opened_file = h5py.File(metadata["events_path"], "r")
                events_group = opened_file["events"]
                self._events_files[recording][side] = opened_file
                self._event_stores[recording][side] = {
                    "x": events_group["x"],
                    "y": events_group["y"],
                    "t": events_group["t"],
                    "p": events_group["p"],
                }

    @staticmethod
    def _ms_to_event_index(ms_to_idx: np.ndarray, millisecond: int, total_events: int) -> int:
        if millisecond <= 0:
            return 0
        if millisecond >= ms_to_idx.shape[0]:
            return total_events
        return int(ms_to_idx[millisecond])

    def _load_event_slice(
        self,
        recording: str,
        side: str,
        start_us: int,
        end_us: int,
    ) -> np.ndarray:
        metadata = self._recording_meta[recording][side]
        t_offset = metadata["t_offset"]
        ms_to_idx = metadata["ms_to_idx"]
        total_events = metadata["total_events"]
        event_store = self._event_stores[recording][side]

        local_start_us = start_us - t_offset
        local_end_us = end_us - t_offset

        if local_end_us <= 0:
            return self._empty_events()

        coarse_start_ms = int(max(0, local_start_us // 1000))
        coarse_end_ms = int(max(0, local_end_us // 1000))

        coarse_start_idx = self._ms_to_event_index(ms_to_idx, coarse_start_ms, total_events)
        coarse_stop_idx = self._ms_to_event_index(ms_to_idx, coarse_end_ms + 1, total_events)

        if coarse_stop_idx <= coarse_start_idx:
            return self._empty_events()

        # Fine-grained binary search within the coarse bucket
        t_chunk = event_store["t"][coarse_start_idx:coarse_stop_idx]
        fine_start_offset = int(np.searchsorted(t_chunk, local_start_us, side="left"))
        fine_end_offset = int(np.searchsorted(t_chunk, local_end_us, side="right"))
        start_idx = coarse_start_idx + fine_start_offset
        end_idx = coarse_start_idx + fine_end_offset

        x_raw = event_store["x"][start_idx:end_idx]
        y_raw = event_store["y"][start_idx:end_idx]
        t_raw = event_store["t"][start_idx:end_idx]
        p_raw = event_store["p"][start_idx:end_idx]

        events = np.empty(len(x_raw), dtype=self.dtype)
        events["x"] = x_raw.astype(np.int16)
        events["y"] = y_raw.astype(np.int16)
        events["t"] = t_raw.astype(np.int64) + t_offset
        events["p"] = p_raw.astype(bool)
        return events

    def _empty_events(self) -> np.ndarray:
        return np.empty(0, dtype=self.dtype)

    def _undistort_events(self, events: np.ndarray, rectify_map: np.ndarray) -> np.ndarray:
        if events.shape[0] == 0:
            return events

        x_coords = events["x"].astype(np.int64)
        y_coords = events["y"].astype(np.int64)

        within_bounds = (
            (x_coords >= 0) & (x_coords < self.sensor_size[0])
            & (y_coords >= 0) & (y_coords < self.sensor_size[1])
        )
        if not np.all(within_bounds):
            x_coords = x_coords[within_bounds]
            y_coords = y_coords[within_bounds]
            timestamps = events["t"][within_bounds]
            polarities = events["p"][within_bounds]
        else:
            timestamps = events["t"]
            polarities = events["p"]

        if x_coords.shape[0] == 0:
            return self._empty_events()

        rectified_xy = rectify_map[y_coords, x_coords]
        x_rectified = np.clip(
            np.rint(rectified_xy[:, 0]).astype(np.int64), 0, self.sensor_size[0] - 1
        ).astype(np.int16)
        y_rectified = np.clip(
            np.rint(rectified_xy[:, 1]).astype(np.int64), 0, self.sensor_size[1] - 1
        ).astype(np.int16)

        rectified_events = np.empty(len(x_rectified), dtype=self.dtype)
        rectified_events["x"] = x_rectified
        rectified_events["y"] = y_rectified
        rectified_events["t"] = timestamps
        rectified_events["p"] = polarities
        return rectified_events

    def _load_flow_target(self, path: str) -> torch.Tensor:
        import imageio
        # DSEC stores flow as 16-bit RGBA PNG: channel 0-1 = (value + 2^15) / 128, channel 2 = valid
        raw = imageio.v2.imread(path, format="PNG-FI").astype(np.float32)   # (H, W, 4)
        flow_uv = (raw[:, :, :2] - 2**15) / 128.0      # decode to pixels/frame  (H, W, 2)
        validity_mask = raw[:, :, 2:3]                  # binary mask             (H, W, 1)
        hwc = np.concatenate([flow_uv, validity_mask], axis=2)
        return torch.from_numpy(hwc.transpose(2, 0, 1))  # (3, H, W)

    def _load_semantic_target(self, path: str) -> torch.Tensor:
        from PIL import Image
        # 8-bit greyscale PNG; each pixel is already the Cityscapes trainId DSEC-Semantic's own
        # labels.py assigns (0..10 or 0..18 depending on label_taxonomy), 255 = ignore -- no
        # decoding/merging needed, unlike M3ED's own InternImage pseudo-labels.
        labels = np.array(Image.open(path)).astype(np.int64)   # (H, W)
        return torch.from_numpy(labels).unsqueeze(0)   # (1, H, W)

    @staticmethod
    def _capture_rng_state() -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": None,
        }
        try:
            import torch as torch_module
            state["torch"] = torch_module.random.get_rng_state()
        except ImportError:
            pass
        return state

    @staticmethod
    def _restore_rng_state(state: Dict[str, Any]) -> None:
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        if state.get("torch") is not None:
            try:
                import torch as torch_module
                torch_module.random.set_rng_state(state["torch"])
            except ImportError:
                pass

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int):
        recording, t_start, t_stop, target_path = self._index[idx]

        events_by_side: Dict[str, np.ndarray] = {}
        for side in self._event_sides:
            side_events = self._load_event_slice(recording, side, t_start, t_stop)
            rectify_map = self._recording_meta[recording][side]["rectify_map"]
            if self.rectify and rectify_map is not None:
                side_events = self._undistort_events(side_events, rectify_map)
            events_by_side[side] = side_events

        if self.transform is not None:
            if self._is_stereo:
                # Apply the same random transform to both sides by resetting the RNG.
                rng_state = self._capture_rng_state()
                events_by_side["left"] = self.transform(events_by_side["left"])
                self._restore_rng_state(rng_state)
                events_by_side["right"] = self.transform(events_by_side["right"])
            else:
                side = self._event_sides[0]
                events_by_side[side] = self.transform(events_by_side[side])

        if self.target_type == "optical_flow":
            target = self._load_flow_target(target_path)
        elif self.target_type == "semantic":
            target = self._load_semantic_target(target_path)
        else:  # "optical_flow_test" -- target_path is already the int file_index; no label to load
            target = target_path

        events_output = events_by_side if self._is_stereo else events_by_side[self._event_sides[0]]

        if self.joint_transform is not None:
            events_output, target = self.joint_transform(events_output, target)

        return events_output, target

    def _close_open_files(self) -> None:
        for per_side_files in self._events_files.values():
            for opened_file in per_side_files.values():
                try:
                    opened_file.close()
                except Exception:
                    pass
        self._events_files.clear()
        self._event_stores.clear()

    def __del__(self):
        self._close_open_files()
