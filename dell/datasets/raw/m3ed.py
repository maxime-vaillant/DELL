import os
import random
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from dell.datasets.raw.common import DEFAULT_WINDOW_DURATION_US, normalize_event_view
from dell.utils.semantic import CITYSCAPES_19_TO_11


_MS_BUCKET_US = 1_000   # granularity of ms_map_idx

BASE_URL = "https://m3ed-dist.s3.us-west-2.amazonaws.com"   # official S3 bucket, per m3ed's tools/download_m3ed.py


def aws_url(filename: str, suffix: str) -> str:
    """Mirrors the official `aws_url` helper in daniilidis-group/m3ed's tools/download_m3ed.py --
    e.g. aws_url('car_urban_day_horse', 'data.h5') / aws_url('car_urban_day_horse', 'semantics.h5')."""
    return f"{BASE_URL}/processed/{filename}/{filename}_{suffix}"


# M3ED recordings with InternImage semantic labels, mapped to upstream's `is_test_file` flag.
#
# These are pseudo-labels from a Cityscapes-pretrained InternImage run on the RGB camera and warped
# into the event camera frame, not human annotations: mIoU measures agreement with that model.
_SEMANTIC_RECORDINGS: Dict[str, bool] = {   # recording -> is_test_file
    "car_urban_day_city_hall": False,
    "car_urban_day_horse": False,
    "car_urban_day_penno_big_loop": False,
    "car_urban_day_penno_small_loop": False,
    "car_urban_day_rittenhouse": False,
    "car_urban_day_ucity_big_loop": True,
    "car_urban_day_ucity_small_loop": False,
    "car_urban_day_schuylkill_tunnel": True,
    "falcon_outdoor_day_penno_parking_1": False,
    "falcon_outdoor_day_penno_parking_2": False,
    "falcon_outdoor_day_penno_plaza": False,
    "falcon_outdoor_day_penno_cars": False,
    "falcon_outdoor_day_penno_parking_3": True,
    "falcon_outdoor_day_penno_trees": False,
    "falcon_outdoor_day_fast_flight_1": False,
    "falcon_outdoor_day_fast_flight_3": True,
    "falcon_outdoor_day_fast_flight_2": False,
    "spot_outdoor_day_art_plaza_loop": False,
    "spot_outdoor_day_srt_green_loop": False,
    "spot_outdoor_day_skatepark_1": False,
    "spot_outdoor_day_skatepark_2": False,
    "spot_outdoor_day_skatepark_3": True,
    "spot_outdoor_day_rocky_steps": False,
    "spot_outdoor_day_srt_under_bridge_1": False,
    "spot_outdoor_day_srt_under_bridge_2": False,
    "spot_outdoor_day_penno_building_loop": True,
    "spot_outdoor_day_penno_short_loop": False,
}


@dataclass
class _MsIndex:
    """Coarse time index of one recording side, from M3ED's `ms_map_idx`.

    `ms_map_idx[k]` is the index of the first event in millisecond bucket `k`, counted from the
    side's first event. It is small enough to keep in RAM, unlike the full timestamp column.
    `valid` says whether `_validate_ms_index` confirmed this convention; if not, callers must use
    `M3EDDataset._fallback_t_array`.
    """
    ms_map_idx: np.ndarray   # (n_ms,) int64
    t0: int                  # this side's first event timestamp -- ms_map_idx's own zero reference
    n_events: int
    valid: bool


class M3EDDataset(Dataset):
    """`M3ED-Semantic <https://m3ed.io/>`_ event dataset: one event window and one label per sample.

    Only the requested window is read from disk. Windows are located with M3ED's per-side
    `ms_map_idx` (event index of the first event of each millisecond), then refined with
    `np.searchsorted`, like `DSECDataset`. Since the convention of `ms_map_idx` is undocumented, each
    side is checked against its real timestamps when opened. If the check fails or `ms_map_idx` is
    missing, the full timestamp column is cached in RAM instead, with a warning.

    Target: (1, H, W) int64 Cityscapes class indices 0-18, 255 where the calibration warp from the
    RGB camera covers no pixel. The labels are InternImage pseudo-labels already aligned to the left
    event camera, whatever `event_view` is.

    Parameters:
        m3ed_root:          the `M3ED/` directory created by `M3EDDataModule.download()`, holding
                            `<recording>_data.h5` and `<recording>_semantics.h5`.
        recordings:         recording names to include (keys of `_SEMANTIC_RECORDINGS`).
        transform:          applied to the raw event array after loading.
        joint_transform:    optional `(events, target) -> (events, target)` applied last, so spatial
                            augmentations such as flips stay consistent between both.
        event_view:         'left' (default), 'right' or 'stereo' (returns a {'left', 'right'} dict).
        window_duration_s:  if set, use a fixed window ending at the label's timestamp instead of the
                            window between two frames.
        target_interval_s:  minimum time between two kept targets; None keeps all of them.
        label_taxonomy:     'cityscapes-11' (default) or 'cityscapes-19'. 'cityscapes-11' merges rare classes
                            with `utils.semantic.CITYSCAPES_19_TO_11`. 255 is kept either way.
    """

    sensor_size = (1280, 720, 2)   # Prophesee Gen4, per M3ED's own sensor suite (m3ed.io)
    dtype = np.dtype([("x", np.int16), ("y", np.int16), ("t", np.int64), ("p", bool)])

    def __init__(
        self,
        m3ed_root: str,
        recordings: List[str],
        transform: Optional[Callable] = None,
        joint_transform: Optional[Callable[[Any, torch.Tensor], Tuple[Any, torch.Tensor]]] = None,
        event_view: str = "left",
        window_duration_s: Optional[float] = None,
        target_interval_s: Optional[float] = None,
        label_taxonomy: str = "cityscapes-11",
    ):
        if label_taxonomy not in ("cityscapes-19", "cityscapes-11"):
            raise ValueError(f"label_taxonomy must be 'cityscapes-19' or 'cityscapes-11', got {label_taxonomy!r}")

        self.m3ed_root = m3ed_root
        self.transform = transform
        self.joint_transform = joint_transform
        self.window_duration_us = (
            int(round(window_duration_s * 1_000_000)) if window_duration_s is not None else None
        )
        self.target_interval_us = (
            int(round(target_interval_s * 1_000_000)) if target_interval_s is not None else None
        )

        # lookup table applying the 19 -> 11 merge; other values (IGNORE_INDEX) are unchanged
        self._label_remap_table: Optional[np.ndarray] = None
        if label_taxonomy == "cityscapes-11":
            table = np.arange(256, dtype=np.int64)
            table[:len(CITYSCAPES_19_TO_11)] = CITYSCAPES_19_TO_11
            self._label_remap_table = table

        self._event_sides: Tuple[str, ...] = normalize_event_view(event_view)
        self._is_stereo = len(self._event_sides) == 2

        # Init before any code path that can raise so __del__ is always safe.
        self._data_files: Dict[str, h5py.File] = {}
        self._semantics_files: Dict[str, h5py.File] = {}
        self._ms_index: Dict[str, Dict[str, _MsIndex]] = {}
        self._events_t_fallback: Dict[str, Dict[str, np.ndarray]] = {}

        self._index = self._build_index(recordings)

    def _data_h5_path(self, recording: str) -> str:
        return os.path.join(self.m3ed_root, f"{recording}_data.h5")

    def _semantics_h5_path(self, recording: str) -> str:
        return os.path.join(self.m3ed_root, f"{recording}_semantics.h5")

    def _open_recording(self, recording: str) -> None:
        if recording in self._data_files:
            return
        self._data_files[recording] = h5py.File(self._data_h5_path(recording), "r")
        self._semantics_files[recording] = h5py.File(self._semantics_h5_path(recording), "r")

        # 'left' is always indexed: ts_map_prophesee_left_t indexes into it regardless of
        # event_view (ground truth is always left-referenced, see class docstring).
        sides_to_cache = set(self._event_sides) | {"left"}
        self._ms_index[recording] = {}
        for side in sides_to_cache:
            self._ms_index[recording][side] = self._build_ms_index(recording, side)

    def _build_ms_index(self, recording: str, side: str) -> _MsIndex:
        group = self._data_files[recording][f"/prophesee/{side}"]
        t_dataset = group["t"]
        n_events = int(t_dataset.shape[0])
        if n_events == 0:
            return _MsIndex(ms_map_idx=np.zeros(0, dtype=np.int64), t0=0, n_events=0, valid=True)

        t0 = int(t_dataset[0])

        if "ms_map_idx" not in group:
            warnings.warn(
                f"M3EDDataset: recording {recording!r} side {side!r} has no 'ms_map_idx' "
                "dataset; falling back to a full in-RAM timestamp cache for this side "
                "(RAM-heavy -- 8 bytes/event).",
                stacklevel=4,
            )
            return _MsIndex(ms_map_idx=np.zeros(0, dtype=np.int64), t0=t0, n_events=n_events, valid=False)

        ms_map_idx = np.asarray(group["ms_map_idx"][:]).astype(np.int64)
        valid = self._validate_ms_index(ms_map_idx, t_dataset, t0, n_events)
        if not valid:
            warnings.warn(
                f"M3EDDataset: recording {recording!r} side {side!r}'s 'ms_map_idx' failed a "
                "consistency check against its own event timestamps; falling back to a full "
                "in-RAM timestamp cache for this side (RAM-heavy -- 8 bytes/event) to preserve "
                "correctness.",
                stacklevel=4,
            )
        return _MsIndex(ms_map_idx=ms_map_idx, t0=t0, n_events=n_events, valid=valid)

    @staticmethod
    def _validate_ms_index(ms_map_idx: np.ndarray, t_dataset: Any, t0: int, n_events: int) -> bool:
        """Cheap, self-contained sanity check of the defining ms-bucket invariant --
        ``ms_map_idx[k]`` must be the index of the *first* event whose timestamp falls in
        ``[t0 + k*1000us, t0 + (k+1)*1000us)`` -- against this side's own real timestamps, at a
        handful of buckets spread across the whole recording. Only ever touches a few individual
        events (cheap local h5py reads), never the full ``t`` column."""
        if ms_map_idx.ndim != 1 or ms_map_idx.shape[0] == 0:
            return False
        if np.any(np.diff(ms_map_idx) < 0):
            return False
        if ms_map_idx[0] < 0 or ms_map_idx[0] > n_events or ms_map_idx[-1] > n_events:
            return False

        n_ms = ms_map_idx.shape[0]
        sample_buckets = sorted({int(round(fraction * (n_ms - 1))) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)})
        for bucket in sample_buckets:
            event_idx = int(ms_map_idx[bucket])
            if event_idx >= n_events:
                continue   # trailing empty bucket(s) at the very end of the recording -- fine
            t_at = int(t_dataset[event_idx])
            if t_at - t0 >= (bucket + 1) * _MS_BUCKET_US:
                return False   # this bucket's first event actually belongs to a later bucket
            if event_idx > 0:
                t_before = int(t_dataset[event_idx - 1])
                if t_before - t0 >= bucket * _MS_BUCKET_US:
                    return False   # the preceding event should still belong to an earlier bucket
        return True

    def _build_index(self, recordings: List[str]) -> List[Tuple]:
        index: List[Tuple] = []
        skipped: List[str] = []
        for recording in recordings:
            if recording not in _SEMANTIC_RECORDINGS:
                raise ValueError(
                    f"Unknown M3ED semantic recording {recording!r}. "
                    f"Known recordings: {sorted(_SEMANTIC_RECORDINGS)}"
                )
            if not (os.path.isfile(self._data_h5_path(recording)) and os.path.isfile(self._semantics_h5_path(recording))):
                skipped.append(recording)
                continue

            self._open_recording(recording)
            entries = self._index_recording(recording)
            if entries:
                index.extend(entries)
            else:
                skipped.append(recording)

        if skipped:
            warnings.warn(
                f"M3EDDataset: {len(skipped)} recording(s) missing data/semantics files "
                f"and were skipped: {skipped}",
                stacklevel=3,
            )

        if not index:
            example_recording = recordings[0] if recordings else next(iter(_SEMANTIC_RECORDINGS))
            raise RuntimeError(
                f"No samples found under '{self.m3ed_root}'.\n"
                f"Checked {len(recordings)} recording(s): {recordings}.\n"
                f"Example expected paths:\n"
                f"  {self._data_h5_path(example_recording)}\n"
                f"  {self._semantics_h5_path(example_recording)}\n\n"
                "Fix: run M3EDDataModule(..., download=True).setup() once to fetch missing "
                "files, or verify m3ed_root points at the parent of the M3ED/ folder "
                "(e.g. data_path='/home/user/data' -> '/home/user/data/M3ED/<recording>_data.h5')."
            )
        return index

    def _index_recording(self, recording: str) -> List[Tuple]:
        ts_map = np.asarray(self._semantics_files[recording]["ts_map_prophesee_left_t"][:]).astype(np.int64)
        n_events = self._ms_index[recording]["left"].n_events

        # Every event index this recording's index construction will ever need is already exactly
        # one of ts_map's own (valid) entries: a frame's own anchor (`ts_map[frame_index]`) and the
        # *previous* frame's anchor (`ts_map[frame_index - 1]`) are the same values, just read at
        # adjacent positions. So one batched lookup over ts_map's unique valid values covers both.
        valid_event_indices = ts_map[(ts_map >= 0) & (ts_map < n_events)]
        t_lookup: Dict[int, int] = {}
        if valid_event_indices.size:
            unique_indices = np.unique(valid_event_indices)
            values = self._left_t_values(recording, unique_indices)
            t_lookup = dict(zip(unique_indices.tolist(), values.tolist()))

        entries: List[Tuple] = []
        last_kept_timestamp: Optional[int] = None
        for frame_index, event_idx in enumerate(ts_map):
            event_idx = int(event_idx)
            if event_idx not in t_lookup:
                continue   # frame falls outside this recording's decoded event range
            t_stop = t_lookup[event_idx]

            if (
                self.target_interval_us is not None
                and last_kept_timestamp is not None
                and t_stop - last_kept_timestamp < self.target_interval_us
            ):
                continue

            if self.window_duration_us is not None:
                t_start = t_stop - self.window_duration_us
            elif frame_index > 0:
                prev_event_idx = int(ts_map[frame_index - 1])
                t_start = t_lookup.get(prev_event_idx, t_stop - DEFAULT_WINDOW_DURATION_US)
            else:
                t_start = t_stop - DEFAULT_WINDOW_DURATION_US

            entries.append((recording, t_start, t_stop, frame_index))
            last_kept_timestamp = t_stop
        return entries

    def _left_t_values(self, recording: str, event_indices: np.ndarray) -> np.ndarray:
        """Batched ``t`` lookup for a set of 'left'-side event indices -- one fancy h5py read (or
        a plain in-RAM index) instead of one file read per index."""
        ms_index = self._ms_index[recording]["left"]
        t_dataset = self._data_files[recording]["/prophesee/left/t"]
        if not ms_index.valid:
            t_full = self._fallback_t_array(recording, "left", t_dataset, ms_index.n_events)
            return t_full[event_indices]

        unique_indices, inverse = np.unique(event_indices, return_inverse=True)
        values = np.asarray(t_dataset[unique_indices.tolist()], dtype=np.int64)
        return values[inverse]

    def _fallback_t_array(self, recording: str, side: str, t_dataset: Any, n_events: int) -> np.ndarray:
        cache = self._events_t_fallback.setdefault(recording, {})
        if side not in cache:
            cache[side] = (
                np.ascontiguousarray(t_dataset[:], dtype=np.int64) if n_events > 0
                else np.zeros(0, dtype=np.int64)
            )
        return cache[side]

    @staticmethod
    def _ms_to_event_index(ms_map_idx: np.ndarray, millisecond: int, n_events: int) -> int:
        if millisecond <= 0:
            return 0
        if millisecond >= ms_map_idx.shape[0]:
            return n_events
        return int(ms_map_idx[millisecond])

    def _coarse_to_fine_range(self, ms_index: _MsIndex, t_dataset: Any, start_us: int, end_us: int) -> Tuple[int, int]:
        local_start_us = start_us - ms_index.t0
        local_end_us = end_us - ms_index.t0

        coarse_start_ms = max(0, local_start_us // _MS_BUCKET_US)
        coarse_end_ms = max(0, local_end_us // _MS_BUCKET_US)

        coarse_start_idx = self._ms_to_event_index(ms_index.ms_map_idx, coarse_start_ms, ms_index.n_events)
        coarse_stop_idx = self._ms_to_event_index(ms_index.ms_map_idx, coarse_end_ms + 1, ms_index.n_events)

        if coarse_stop_idx <= coarse_start_idx:
            return coarse_start_idx, coarse_start_idx

        # Fine-grained binary search within the coarse bucket range.
        t_chunk = t_dataset[coarse_start_idx:coarse_stop_idx]
        fine_start_offset = int(np.searchsorted(t_chunk, start_us, side="left"))
        fine_end_offset = int(np.searchsorted(t_chunk, end_us, side="right"))
        return coarse_start_idx + fine_start_offset, coarse_start_idx + fine_end_offset

    def _load_event_slice(self, recording: str, side: str, start_us: int, end_us: int) -> np.ndarray:
        ms_index = self._ms_index[recording][side]
        if ms_index.n_events == 0:
            return self._empty_events()

        group = self._data_files[recording][f"/prophesee/{side}"]
        t_dataset = group["t"]

        if ms_index.valid:
            if end_us <= ms_index.t0 or start_us - ms_index.t0 > ms_index.ms_map_idx.shape[0] * _MS_BUCKET_US:
                return self._empty_events()
            start_idx, end_idx = self._coarse_to_fine_range(ms_index, t_dataset, start_us, end_us)
        else:
            t_full = self._fallback_t_array(recording, side, t_dataset, ms_index.n_events)
            if len(t_full) == 0 or end_us <= t_full[0] or start_us > t_full[-1]:
                return self._empty_events()
            start_idx = int(np.searchsorted(t_full, start_us, side="left"))
            end_idx = int(np.searchsorted(t_full, end_us, side="right"))

        if end_idx <= start_idx:
            return self._empty_events()

        x = group["x"][start_idx:end_idx]
        y = group["y"][start_idx:end_idx]
        t = t_dataset[start_idx:end_idx]
        p = group["p"][start_idx:end_idx]

        events = np.empty(end_idx - start_idx, dtype=self.dtype)
        events["x"] = x.astype(np.int16)
        events["y"] = y.astype(np.int16)
        events["t"] = t
        events["p"] = p.astype(bool)
        return events

    def _empty_events(self) -> np.ndarray:
        return np.empty(0, dtype=self.dtype)

    def _load_semantic_target(self, recording: str, frame_index: int) -> torch.Tensor:
        labels = self._semantics_files[recording]["predictions"][frame_index]   # (H, W, 1) uint8
        labels = np.asarray(labels).squeeze(-1).astype(np.int64)   # (H, W); 0-18 class, 255 = ignore
        if self._label_remap_table is not None:
            labels = self._label_remap_table[labels]   # 19-class -> 11-class, see label_taxonomy
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
        recording, t_start, t_stop, frame_index = self._index[idx]

        events_by_side: Dict[str, np.ndarray] = {
            side: self._load_event_slice(recording, side, t_start, t_stop)
            for side in self._event_sides
        }

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

        target = self._load_semantic_target(recording, frame_index)
        events_output = events_by_side if self._is_stereo else events_by_side[self._event_sides[0]]

        if self.joint_transform is not None:
            events_output, target = self.joint_transform(events_output, target)

        return events_output, target

    def _close_open_files(self) -> None:
        for opened_file in list(self._data_files.values()) + list(self._semantics_files.values()):
            try:
                opened_file.close()
            except Exception:
                pass
        self._data_files.clear()
        self._semantics_files.clear()

    def __del__(self):
        self._close_open_files()
