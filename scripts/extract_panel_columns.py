"""Crop the event, ground-truth and prediction columns out of a panel comparison video.

The videos are logged during training: optical flow panels have 5 columns (event | flow_gt |
flow_pred | magnitude_diff | angle_diff), semantic ones 3 (event | label_gt | label_pred). For each
frame index, this writes `event_<idx>.png`, `gt_<idx>.png` and `pred_<label>_<idx>.png`.

    python scripts/extract_panel_columns.py local_of.mp4 --num-columns 5 --indices 0 50 99 --label local
    python scripts/extract_panel_columns.py local_sem.mp4 --num-columns 3 --out-dir out/panels/semantic

`--num-columns` must match the panel type; it is only used to find the column width. `--indices`
are 0-based frame numbers; omit it to extract every frame.

Columns keep the colors baked into the video, so videos logged with an older color scheme cannot be
recolored. For those, or checkpoints without a video, use `generate_panel_columns.py`.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from dell.utils.image import PANEL_GAP_PX

GAP_PX = PANEL_GAP_PX


def _require_tool(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise SystemExit(f"{name} not found on PATH")
    return path


def _probe_size(path: Path) -> tuple[int, int]:
    ffprobe = _require_tool("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    return stream["width"], stream["height"]


def _probe_frame_count(path: Path) -> int:
    """Exact frame count, from a full decode because some muxers write a wrong `nb_frames`."""
    ffprobe = _require_tool("ffprobe")
    result = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", str(path)],
        capture_output=True, text=True, check=True,
    )
    return int(result.stdout.strip())


def _extract_frame_png(path: Path, out_path: Path, frame_index: int, x: int, width: int) -> None:
    ffmpeg = _require_tool("ffmpeg")
    # select the frame by index, then crop one column; -vsync 0 stops ffmpeg from duplicating or
    # dropping frames, which would make -frames:v 1 grab the wrong one
    vf = f"select='eq(n\\,{frame_index})',crop={width}:ih:{x}:0"
    result = subprocess.run(
        [ffmpeg, "-y", "-i", str(path), "-vf", vf, "-vsync", "0", "-frames:v", "1", str(out_path)],
        capture_output=True,
    )
    if result.returncode != 0:
        raise SystemExit(f"ffmpeg failed extracting frame {frame_index} from {path} -> {out_path}:\n"
                          f"{result.stderr.decode(errors='replace')}")


def _column_x(index: int, col_width: int) -> int:
    return index * (col_width + GAP_PX)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("video", type=Path, help="panel video to extract columns from")
    parser.add_argument("--num-columns", type=int, required=True, choices=[3, 5],
                         help="3 for a semantic panel, 5 for an optical_flow one")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                         help="0-based frame indices, e.g. 0 50 99 (default: every frame)")
    parser.add_argument("--label", help="name used in pred_<label>_<idx>.png (default: the video's filename)")
    parser.add_argument("--out-dir", type=Path, default=Path("out/panels"))
    args = parser.parse_args(argv)

    label = args.label or args.video.stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    width, _ = _probe_size(args.video)
    indices = args.indices if args.indices is not None else range(_probe_frame_count(args.video))

    col_width = (width - (args.num_columns - 1) * GAP_PX) // args.num_columns
    print(f"[extract_panel_columns] column width={col_width}px "
          f"(from video width={width}px, --num-columns={args.num_columns})")

    for idx in indices:
        outputs = [
            (_column_x(0, col_width), args.out_dir / f"event_{idx}.png"),
            (_column_x(1, col_width), args.out_dir / f"gt_{idx}.png"),
            (_column_x(2, col_width), args.out_dir / f"pred_{label}_{idx}.png"),
        ]
        for x, out_path in outputs:
            _extract_frame_png(args.video, out_path, idx, x, col_width)
            print(f"[extract_panel_columns] wrote {out_path}")


if __name__ == "__main__":
    main()
