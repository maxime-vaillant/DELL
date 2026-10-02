"""Render events, ground truth and predictions as separate mp4 videos over a whole validation set.

Each video is a standalone component (`events`, `gt`, `prediction`, `masked_prediction`, and for
optical flow `magnitude_diff` and `angle_diff`), at the ground truth's native resolution. This is
the whole-validation-set counterpart of `generate_panel_columns.py`.

    python scripts/generate_validation_videos.py --mode e2e --task optical_flow --dataset dsec \\
        --checkpoint checkpoints/OF_E2E/model.ckpt --out-dir out/videos/e2e_of

    python scripts/generate_validation_videos.py --mode e2e --task semantic --dataset m3ed \\
        --checkpoint checkpoints/SEG_E2E_M3ED/model.ckpt --components events gt prediction

    # sample count, resolutions and a size estimate for the whole validation set, without inference
    python scripts/generate_validation_videos.py --mode e2e --task optical_flow --dataset dsec --dry-run

The model flags must match the training run, as for `infer.py`. The model runs through
`Trainer.validate()` with frame collection forced on and unbounded (see `panel_capture`), so every
validation sample is visited in dataset order. The model's `_panel_frames` is replaced to build a
dict of independent images instead of one stitched panel.

Resolutions: `gt`, `prediction` and `masked_prediction` are on the ground truth's grid (480x640 for
DSEC flow, 440x640 for DSEC semantic, 720x1280 for M3ED). `events` is rendered from the dataset
without `FrameResize`, at sensor resolution (640x480 for DSEC, 1280x720 for M3ED), as upsampling
the model input would look pixelated. Each video is sized independently.

Frame rate: by default one sample is one real time step, so playback is real time. The frame rate is
inferred from the median gap between targets, and `--target-interval-s` defaults to None (every
target is kept) instead of the training default of 2 s, which would play back about 20x too fast.

Memory: all rendered images are kept in memory until the validation pass ends. Use `--max-samples`
for a quick test on long recordings.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from pytorch_lightning import Trainer, seed_everything

from dell.cli import build_infer_parser, build_model_and_datamodule, resolve_checkpoint_path
from dell.utils.flow import angular_error_degrees, cos_angle, flow_to_rgb
from dell.utils.image import event_rgb_net_polarity, event_rgb_on_white, to_uint8_fixed
from dell.utils.panel_capture import (
    capture_validation_frames, native_resolution_validation_dataset, silence_mlflow_artifacts,
    validation_dataset,
)
from dell.utils.semantic import CITYSCAPES_11_PALETTE, CITYSCAPES_PALETTE, IGNORE_INDEX, colorize_labels

VALID_TASKS = ("optical_flow", "semantic")

# The error maps only exist for optical flow, and are opt-in.
DEFAULT_COMPONENTS = ("events", "gt", "prediction", "masked_prediction")
OPTICAL_FLOW_ONLY_COMPONENTS = ("magnitude_diff", "angle_diff")
ALL_COMPONENT_CHOICES = DEFAULT_COMPONENTS + OPTICAL_FLOW_ONLY_COMPONENTS

# Rough libx264 (crf 23) bits per pixel per frame, only used for the --dry-run size estimate.
# Events are mostly white and compress best; error maps are mostly dark.
BPP_RANGE = {
    "events": (0.015, 0.05),
    "gt": (0.05, 0.13),
    "prediction": (0.05, 0.13),
    "masked_prediction": (0.04, 0.11),
    "magnitude_diff": (0.02, 0.07),
    "angle_diff": (0.02, 0.07),
}


def _flow_components(prediction_sample: torch.Tensor, flow_sample: torch.Tensor,
                      mask_sample: torch.Tensor, components: set) -> dict:
    """One (H, W, 3) uint8 image per requested optical-flow component.

    prediction, flow and mask are (2, H, W) on the ground truth's grid. Same per-column rendering as
    `build_flow_comparison_panel`. `masked_prediction` blanks the invalid ground truth pixels.
    """
    mask = mask_sample[0].float()
    valid_magnitude = flow_sample.norm(dim=0)[mask.bool()]
    scale = (torch.quantile(valid_magnitude, 0.99) if valid_magnitude.numel()
             else flow_sample.new_tensor(0.0)).clamp(min=1e-6)

    images: dict = {}

    if "gt" in components:
        gt_rgb = flow_to_rgb(flow_sample, scale)
        gt_rgb[mask == 0] = 0
        images["gt"] = gt_rgb.cpu().numpy()

    pred_rgb = None
    if "prediction" in components or "masked_prediction" in components:
        pred_rgb = flow_to_rgb(prediction_sample, scale)
    if "prediction" in components:
        images["prediction"] = pred_rgb.cpu().numpy()
    if "masked_prediction" in components:
        masked_rgb = pred_rgb.clone()
        masked_rgb[mask == 0] = 0
        images["masked_prediction"] = masked_rgb.cpu().numpy()

    if "magnitude_diff" in components:
        magnitude_diff = (prediction_sample.norm(dim=0) - flow_sample.norm(dim=0)).abs() * mask
        images["magnitude_diff"] = to_uint8_fixed(magnitude_diff, scale).unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()

    if "angle_diff" in components:
        cos = cos_angle(prediction_sample.unsqueeze(0), flow_sample.unsqueeze(0))[0]
        angle_diff = angular_error_degrees(cos) * mask
        images["angle_diff"] = to_uint8_fixed(angle_diff, 180.0).unsqueeze(-1).expand(-1, -1, 3).cpu().numpy()

    return images


def _semantic_components(prediction_sample: torch.Tensor, target_sample: torch.Tensor,
                          num_classes: int, components: set) -> dict:
    """One (H, W, 3) uint8 image per requested semantic component.

    prediction_sample is (H, W) class indices and target_sample (1, H, W) labels, both on the label's
    grid. Same per-column rendering as `build_semantic_comparison_panel`.
    """
    palette = CITYSCAPES_11_PALETTE if num_classes == 11 else CITYSCAPES_PALETTE

    images: dict = {}

    if "gt" in components:
        images["gt"] = colorize_labels(target_sample[0].long(), palette).cpu().numpy()

    pred_rgb = None
    if "prediction" in components or "masked_prediction" in components:
        pred_rgb = colorize_labels(prediction_sample.long(), palette)
    if "prediction" in components:
        images["prediction"] = pred_rgb.cpu().numpy()
    if "masked_prediction" in components:
        masked_rgb = pred_rgb.clone()
        masked_rgb[target_sample[0] == IGNORE_INDEX] = 0
        images["masked_prediction"] = masked_rgb.cpu().numpy()

    return images


_EVENT_RENDERERS = {
    "blend": event_rgb_on_white,
    "net": event_rgb_net_polarity,
}


def _event_image(events_native, style: str) -> np.ndarray:
    events_native = torch.as_tensor(events_native).float()
    render = _EVENT_RENDERERS[style]
    return render(events_native, events_native.shape[-2:]).cpu().numpy()


def _infer_native_fps(dataset) -> float | None:
    """Frames per second matching the median gap between targets, from the dataset's `_index`.

    Gaps are computed within each recording. Returns None if the dataset has no such index.
    """
    index = getattr(dataset, "_index", None)
    if not index:
        return None

    by_recording: dict = {}
    for entry in index:
        by_recording.setdefault(entry[0], []).append(entry[2])

    gaps_us = []
    for stops in by_recording.values():
        stops = sorted(stops)
        gaps_us.extend(b - a for a, b in zip(stops, stops[1:]) if b > a)

    if not gaps_us:
        return None
    return 1_000_000.0 / float(np.median(gaps_us))


def _encode_video(frames_dir: Path, fps: float, crf: int, out_path: Path) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        raise SystemExit("ffmpeg not found on PATH -- required to encode videos")

    subprocess.run(
        [
            ffmpeg_path, "-y",
            "-framerate", str(fps),
            "-i", str(frames_dir / "frame_%06d.png"),
            "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-c:v", "libx264", "-crf", str(crf),
            "-pix_fmt", "yuv420p",
            str(out_path),
        ],
        check=True, capture_output=True,
    )


def _print_size_estimate(component: str, height: int, width: int, num_frames: int, fps: float) -> None:
    lo_bpp, hi_bpp = BPP_RANGE[component]
    lo_mb = lo_bpp * height * width * num_frames / 8 / 1e6
    hi_mb = hi_bpp * height * width * num_frames / 8 / 1e6
    duration_s = num_frames / fps
    print(
        f"    {component:<18} {height}x{width}  {num_frames} frames  "
        f"~{duration_s:.1f}s @ {fps:.2f}fps  ~{lo_mb:.0f}-{hi_mb:.0f} MB (rough libx264 estimate)"
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_infer_parser()
    parser.description, parser.epilog = __doc__, None
    # native cadence by default, see the module docstring
    parser.set_defaults(target_interval_s=None)

    video = parser.add_argument_group("video export")
    video.add_argument("--components", nargs="+", choices=ALL_COMPONENT_CHOICES, default=None,
                        help="videos to generate (default: events gt prediction masked_prediction); "
                             "magnitude_diff and angle_diff are error maps, optical_flow only")
    video.add_argument("--event-style", choices=("blend", "net"), default="blend",
                        help="how to draw events: 'blend' paints ON/OFF counts on white (default), 'net' "
                             "paints the signed net polarity on black")
    video.add_argument("--event-distorted", action="store_true",
                        help="draw events from raw, unrectified coordinates; DSEC only")
    video.add_argument("--label",
                        help="name used in <component>_<label>.mp4 (default: the MLflow run id or the checkpoint name)")
    video.add_argument("--out-dir", type=Path, default=Path("out/videos"))
    video.add_argument("--fps", type=float, default=None,
                        help="playback frame rate (default: inferred from the gap between targets, "
                             "10 fps for DSEC flow)")
    video.add_argument("--crf", type=int, default=23,
                        help="libx264 quality, 0 is lossless and 51 the worst (default: %(default)s)")
    video.add_argument("--max-samples", type=int, default=None,
                        help="render only the first N validation samples, for a quick test")
    video.add_argument("--dry-run", action="store_true",
                        help="only print the sample count, resolutions, fps and a size estimate; "
                             "needs no checkpoint")
    args = parser.parse_args(argv)

    if args.task not in VALID_TASKS:
        raise SystemExit(f"--task {args.task} has no video to render (only optical_flow/semantic do)")

    components = args.components or list(DEFAULT_COMPONENTS)
    if args.task != "optical_flow":
        bad = [c for c in components if c in OPTICAL_FLOW_ONLY_COMPONENTS]
        if bad:
            raise SystemExit(f"--components {bad} are optical_flow only "
                              f"(semantic has no error map to visualize)")
    if args.event_distorted and args.dataset != "dsec":
        print(f"[generate_validation_videos] --event-distorted has no effect for --dataset "
              f"{args.dataset} (no per-recording rectify_map to disable)", file=sys.stderr)

    checkpoint_path = None if args.dry_run else resolve_checkpoint_path(args)

    label = args.label or args.mlflow_run_id or (Path(checkpoint_path).stem if checkpoint_path else args.dataset)

    seed_everything(args.seed, workers=True)

    model, datamodule, _, _ = build_model_and_datamodule(args, checkpoint_path=checkpoint_path)
    model.eval()

    if args.dry_run:
        datamodule.setup()
        val_dataset = validation_dataset(datamodule)
        num_samples = len(val_dataset) if args.max_samples is None else min(args.max_samples, len(val_dataset))

        _, one_target = val_dataset[0]
        gt_h, gt_w = one_target.shape[-2:]

        sensor_w, sensor_h = datamodule._SENSOR_SIZE[:2]   # (W, H, 2)
        events_hw = (sensor_h, sensor_w)

        fps = args.fps or _infer_native_fps(val_dataset) or 10.0

        print(f"[generate_validation_videos] --dry-run: {num_samples} validation sample(s), "
              f"fps={fps:.3f} ({'explicit' if args.fps else 'auto-inferred'})")
        resolutions = {name: (gt_h, gt_w) for name in ALL_COMPONENT_CHOICES}
        resolutions["events"] = events_hw
        for component in components:
            height, width = resolutions[component]
            _print_size_estimate(component, height, width, num_samples, fps)
        return

    captured_frames = capture_validation_frames(model)

    # replace the stitched panel by a dict of independent component images per sample
    non_event_components = set(components) - {"events"}
    if args.task == "optical_flow":
        model._panel_frames = lambda events, prediction, flow, mask: [
            _flow_components(prediction[i], flow[i], mask[i], non_event_components)
            for i in range(events.shape[1])
        ]
    else:
        num_classes = model.num_classes
        model._panel_frames = lambda events, prediction_full, target: [
            _semantic_components(prediction_full[i], target[i], num_classes, non_event_components)
            for i in range(events.shape[1])
        ]

    silence_mlflow_artifacts()

    trainer_kwargs = {}
    if args.max_samples is not None:
        trainer_kwargs["limit_val_batches"] = (args.max_samples - 1) // args.batch_size + 1

    trainer = Trainer(
        accelerator="auto", devices=args.devices, logger=False, enable_checkpointing=False,
        **trainer_kwargs,
    )
    trainer.validate(model, datamodule=datamodule)

    if args.max_samples is not None:
        captured_frames = captured_frames[:args.max_samples]
    if not captured_frames:
        raise SystemExit("no validation samples captured -- check --dataset/--task match the checkpoint")

    fps = args.fps
    if fps is None:
        fps = _infer_native_fps(validation_dataset(datamodule)) or 10.0
        print(f"[generate_validation_videos] auto-inferred fps={fps:.3f} from the validation "
              f"dataset's own median inter-target gap", file=sys.stderr)

    native_val_dataset = (
        native_resolution_validation_dataset(datamodule, distorted=args.event_distorted)
        if "events" in components else None
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    num_frames = len(captured_frames)

    with tempfile.TemporaryDirectory() as tmp_root:
        tmp_root = Path(tmp_root)
        component_dirs = {component: tmp_root / component for component in components}
        for directory in component_dirs.values():
            directory.mkdir()

        for idx, frame_components in enumerate(captured_frames):
            if "events" in components:
                events_native, _ = native_val_dataset[idx]
                imageio.imwrite(component_dirs["events"] / f"frame_{idx:06d}.png",
                                 _event_image(events_native, args.event_style))

            for component, image in frame_components.items():
                imageio.imwrite(component_dirs[component] / f"frame_{idx:06d}.png", image)

            if (idx + 1) % 200 == 0 or idx + 1 == num_frames:
                print(f"[generate_validation_videos] rendered {idx + 1}/{num_frames} frames", file=sys.stderr)

        del captured_frames   # free memory before encoding

        for component, directory in component_dirs.items():
            out_path = args.out_dir / f"{component}_{label}.mp4"
            _encode_video(directory, fps, args.crf, out_path)
            size_mb = out_path.stat().st_size / 1e6
            print(f"[generate_validation_videos] wrote {out_path} ({size_mb:.1f} MB, "
                  f"{num_frames} frames @ {fps:.3f}fps)")


if __name__ == "__main__":
    main()
