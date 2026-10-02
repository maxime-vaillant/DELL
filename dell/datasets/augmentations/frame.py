import numpy as np
import torch
import torch.nn.functional as F


class SafeToFrame:
    """Wraps a tonic.transforms.ToFrame instance to work around a real shape bug in tonic 1.6.0's
    own ToFrame.__call__: for an EMPTY event slice (len(events) == 0 -- which legitimately happens
    here, e.g. a recording's first frame, whose backward-looking window starts before the
    recording itself does -- see Compose's own docstring below), ToFrame.__call__'s zero-fallback
    builds the frame as shape (n_bins, P, sensor_size[0], sensor_size[1]) = (n_bins, P, W, H) --
    it never reverses sensor_size the way tonic.functional.to_frame_numpy does for every real,
    non-empty slice (frames = np.zeros((len(event_slices), *sensor_size[::-1]), ...) = (n_bins, P,
    H, W)). So one empty window comes back W/H-swapped relative to every populated window out of
    the exact same transform -- confirmed via inspect.getsource(tonic.transforms.ToFrame) (tonic
    1.6.0) and reproduced directly against both DSEC (sensor_size (640, 480, 2)) and M3ED
    ((1280, 720, 2)): an empty window returns (n_bins, P, W, H) while every populated one returns
    (n_bins, P, H, W). One wrongly-shaped frame is enough to corrupt an entire rendered video --
    ffmpeg's image2 demuxer locks a video's dimensions to its first frame and stretches every
    later, correctly-shaped frame to match, which is what made
    generate_validation_videos.py's M3ED events_*.mp4 come out portrait-shaped with landscape
    content squeezed into it.
    """

    def __init__(self, to_frame):
        self._to_frame = to_frame

    def __call__(self, events):
        if len(events) == 0:
            sensor_w, sensor_h, sensor_p = self._to_frame.sensor_size
            n_bins = self._to_frame.n_time_bins or self._to_frame.n_event_bins or 1
            return np.zeros((n_bins, sensor_p, sensor_h, sensor_w), dtype=np.int16)
        return self._to_frame(events)


def _as_tensor(frames):
    return frames if torch.is_tensor(frames) else torch.from_numpy(frames).float()


class FrameResize:
    """Resizes event frames to output_size via adaptive avg-pooling, preserving total event counts.

    adaptive_avg_pool2d returns the *mean* per output pixel; multiplying by (source pixels /
    output pixels) converts that back to an (approximate) sum, so total event count is roughly
    preserved regardless of output_size. Uses true division for that scale factor, not `//`: with
    integer division, any dimension where output_size exceeds the source size floors to 0 (e.g.
    346 // 352 == 0), silently zeroing the *entire* tensor -- not just under-scaling it, an event
    frame of all zeros, indistinguishable from "no events happened". Confirmed to silence every
    spiking neuron downstream this way (0% firing rate network-wide) when configured with an
    output_size larger than the source in any dimension. True division has no such collapse, and
    is also more accurate than `//` even in the ordinary downsampling case (`//` truncates, e.g.
    computing scale 4 instead of the true 6.25 for a 480->192, 640->256 resize).
    """

    def __init__(self, output_size: tuple[int, int]):
        self.output_size = output_size

    def __call__(self, frames):
        tensor = _as_tensor(frames)
        source_h, source_w = tensor.shape[-2:]
        target_h, target_w = self.output_size
        scale = (source_h / target_h) * (source_w / target_w)
        return F.adaptive_avg_pool2d(tensor, self.output_size) * scale


class Compose:
    """Sequentially applies a list of transforms -- a minimal, non-short-circuiting stand-in for
    ``tonic.transforms.Compose``.

    ``tonic.transforms.Compose.__call__`` breaks out of its loop the moment ``len(events) == 0``
    and returns whatever it has so far *unchanged* -- for every DataModule in this repo that means
    it returns the raw structured event array without ever reaching ``tonic.transforms.ToFrame``
    (always the first step here, normally wrapped in ``SafeToFrame`` -- see that class's own
    docstring). A window with zero events is not an edge case worth special-casing away -- it
    happens legitimately (e.g. a recording's first frame, whose backward-looking window starts
    before the recording itself does) -- so skipping ``ToFrame`` there leaks a raw structured numpy
    array all the way to ``default_collate``, which then fails with a cryptic ``"can't convert
    np.ndarray of type numpy.void"`` TypeError instead of just collating an all-zero frame like
    every other empty-ish window. NOTE: plain ``tonic.transforms.ToFrame`` on its own does *not*
    handle zero-event input correctly (tonic 1.6.0) -- its empty-input fallback returns a
    (W, H)-shaped zero frame instead of the (H, W) every populated window returns, silently
    corrupting anything downstream that assumes a consistent shape (see ``SafeToFrame``). Always
    wrap ``ToFrame`` in ``SafeToFrame`` before handing it to this ``Compose``.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events):
        for transform in self.transforms:
            events = transform(events)
        return events


class PairedCompose:
    """Sequentially applies a list of (events, target) -> (events, target) joint transforms --
    the paired counterpart of Compose, for chaining several PairedRandom* transforms into one
    joint_transform callable (M3EDDataset/DSECDataset each accept only a single one)."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, events, target):
        for transform in self.transforms:
            events, target = transform(events, target)
        return events, target


class PairedRandomFlipLR:
    """Randomly flips events and target horizontally in sync.

    Meant to run after events have already been converted to frames (so the
    flip is a plain width-axis mirror, independent of events/target having
    different resolutions). For optical-flow targets, channel 0 (u, the
    horizontal flow component) is negated to stay geometrically consistent
    with the mirrored frame; semantic targets are just mirrored.
    """

    def __init__(self, p: float = 0.5, negate_flow_u: bool = False):
        self.p = p
        self.negate_flow_u = negate_flow_u

    def __call__(self, events, target):
        if torch.rand(()) > self.p:
            return events, target

        if isinstance(events, dict):
            events = {side: torch.flip(frames, dims=[-1]) for side, frames in events.items()}
        else:
            events = torch.flip(events, dims=[-1])

        target = torch.flip(target, dims=[-1])
        if self.negate_flow_u:
            target[0] = -target[0]

        return events, target


class PairedRandomShift:
    """Randomly shifts events and target frames together along x/y, in sync -- the translation
    counterpart of PairedRandomFlipLR.

    shift_x/shift_y bound the shift as a fraction of each frame's own width/height, not an
    absolute pixel count -- one (frac_x, frac_y) pair is drawn per call and rescaled by each
    tensor's own H/W, so the same range stays geometrically consistent even when events and
    target don't share a resolution.

    Unlike datasets.augmentations.event.Rolling, there's no wraparound (no modulo): content
    shifted past an edge is dropped rather than reappearing on the opposite side, and the edge it
    vacates is filled instead -- events frames always with 0.0 (an event *count*, so "no event" is
    never anything else), target frames with target_fill_value (0.0 by default -- pass
    target_fill_value=utils.semantic.IGNORE_INDEX for semantic targets, since 0 is a real class
    there, not "no label").
    """

    def __init__(self, shift_x: float = 0.1, shift_y: float = 0.1, p: float = 0.5, target_fill_value: float = 0.0):
        self.shift_x = shift_x
        self.shift_y = shift_y
        self.p = p
        self.target_fill_value = target_fill_value

    @staticmethod
    def _shift(frames: torch.Tensor, dx: int, dy: int, fill_value: float) -> torch.Tensor:
        if dx == 0 and dy == 0:
            return frames

        shifted = torch.full_like(frames, fill_value)
        h, w = frames.shape[-2:]

        src_x0, src_x1 = max(0, -dx), min(w, w - dx)
        dst_x0, dst_x1 = max(0, dx), min(w, w + dx)
        src_y0, src_y1 = max(0, -dy), min(h, h - dy)
        dst_y0, dst_y1 = max(0, dy), min(h, h + dy)

        shifted[..., dst_y0:dst_y1, dst_x0:dst_x1] = frames[..., src_y0:src_y1, src_x0:src_x1]
        return shifted

    @classmethod
    def _shift_by_fraction(cls, frames: torch.Tensor, frac_x: float, frac_y: float, fill_value: float) -> torch.Tensor:
        h, w = frames.shape[-2:]
        return cls._shift(frames, round(frac_x * w), round(frac_y * h), fill_value)

    def __call__(self, events, target):
        if torch.rand(()) > self.p:
            return events, target

        frac_x = ((torch.rand(()) * 2 - 1) * self.shift_x).item()
        frac_y = ((torch.rand(()) * 2 - 1) * self.shift_y).item()

        if isinstance(events, dict):
            events = {
                side: self._shift_by_fraction(frames, frac_x, frac_y, fill_value=0.0)
                for side, frames in events.items()
            }
        else:
            events = self._shift_by_fraction(events, frac_x, frac_y, fill_value=0.0)

        target = self._shift_by_fraction(target, frac_x, frac_y, fill_value=self.target_fill_value)

        return events, target