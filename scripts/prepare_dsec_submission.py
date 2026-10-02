"""Write DSEC optical-flow benchmark predictions from a trained checkpoint, and zip them for upload.

The format is described at https://dsec.ifi.uzh.ch/optical-flow-submission-format/. The test split
must be on disk first (`python scripts/download_dsec_test.py --data-path <data_path>`).

    python scripts/prepare_dsec_submission.py --mode e2e --task optical_flow --dataset dsec \\
        --checkpoint checkpoints/OF_E2E/model.ckpt --data-path ~/data --output-dir out/dsec_submission

The model flags must match the training run, as for `infer.py`. `--task optical_flow --dataset dsec`
is required. Only `--num-steps`, `--batch-size` and `--window-duration-s` matter among the data
flags, and they must match training too; the others are ignored, as the test recordings are read
directly with `DSECDataset(target_type="optical_flow_test")`.

Before uploading, run `scripts/check_dsec_submission.py` (a vendored copy of DSEC's checker) on the
output directory. It is also worth running `scripts/check_predict.py` once on the same checkpoint
to confirm that `predict()` reproduces the validation metrics.
"""
import argparse
import os
import struct
import tempfile
import zipfile
import zlib

import numpy as np
import torch
import tonic.transforms as tonic_transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dell.cli import build_model_and_datamodule, build_parser
from dell.datasets.augmentations import Compose, FrameResize
from dell.datasets.dsec import DSECDataModule
from dell.datasets.raw.dsec import FLOW_TEST_RECORDINGS, DSECDataset

TEST_SEQUENCES = FLOW_TEST_RECORDINGS


def build_submission_parser() -> argparse.ArgumentParser:
    parser = build_parser()
    parser.description, parser.epilog = __doc__, None
    submission = parser.add_argument_group("dsec submission")
    submission.add_argument("--checkpoint", required=True, help="path to the .ckpt file")
    submission.add_argument("--output-dir", default="out/dsec_submission",
                            help="where PNGs are written, one folder per sequence (default: %(default)s)")
    submission.add_argument("--sequences", nargs="+", default=TEST_SEQUENCES, choices=TEST_SEQUENCES,
                            help="test sequences to run (default: all 7)")
    submission.add_argument("--zip-path", help="submission zip to write (default: <output-dir>.zip)")
    return parser


def _flow_to_16bit(flow_hw2: np.ndarray) -> np.ndarray:
    """Encode (H, W, 2) flow in pixels as (H, W, 3) uint16: flow * 128 + 2**15, and a valid channel of 1.

    Values are clipped so extreme predictions saturate instead of wrapping around.
    """
    encoded = np.empty((*flow_hw2.shape[:2], 3), dtype=np.uint16)
    encoded[..., :2] = np.clip(np.round(flow_hw2 * 128 + 2 ** 15), 0, 65535).astype(np.uint16)
    encoded[..., 2] = 1
    return encoded


def _write_png16_rgb_stdlib(path: str, image: np.ndarray) -> None:
    """Write an (H, W, 3) uint16 array as a 16-bit RGB PNG, using only the standard library.

    Fallback for machines where imageio's FreeImage plugin does not load (e.g. Apple Silicon). Pillow
    and imageio's default plugin would silently truncate the array to 8 bits.
    """
    assert image.dtype == np.uint16 and image.ndim == 3 and image.shape[2] == 3
    height, width, _ = image.shape

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    ihdr = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 16, 2, 0, 0, 0))

    big_endian = image.astype(">u2")  # PNG requires network (big-endian) byte order for 16-bit samples
    row_bytes = big_endian.tobytes()
    stride = width * 3 * 2
    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0 (None)
        raw += row_bytes[row * stride:(row + 1) * stride]
    idat = chunk(b"IDAT", zlib.compress(bytes(raw), level=6))

    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ihdr + idat + chunk(b"IEND", b""))


def _make_png_writer():
    """Pick the PNG writer: imageio's FreeImage plugin (recommended by DSEC) if it works, else stdlib."""
    try:
        import imageio
        with tempfile.NamedTemporaryFile(suffix=".png") as probe_file:
            imageio.imwrite(probe_file.name, np.zeros((1, 1, 3), dtype=np.uint16), format="PNG-FI")

        print("[prepare_dsec_submission] writing PNGs via imageio (format='PNG-FI')")

        def _write(path: str, image: np.ndarray) -> None:
            imageio.imwrite(path, image, format="PNG-FI")

        return _write
    except Exception as exc:
        print(f"[prepare_dsec_submission] imageio's FreeImage plugin isn't usable here ({exc!r}); "
              f"falling back to a dependency-free stdlib PNG writer")
        return _write_png16_rgb_stdlib


@torch.no_grad()
def _write_sequence(
    model, recording: str, dsec_test_root: str, transform, window_duration_s: float, batch_size: int,
    num_workers: int, device: torch.device, output_dir: str, png_writer,
) -> None:
    dataset = DSECDataset(
        dsec_root=dsec_test_root, recordings=[recording], target_type="optical_flow_test",
        transform=transform, event_view="left", rectify=True, window_duration_s=window_duration_s,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=DSECDataModule._collate,
    )

    sequence_dir = os.path.join(output_dir, recording)
    os.makedirs(sequence_dir, exist_ok=True)

    for events, file_indices in tqdm(loader, desc=recording):
        events = events.to(device)
        predictions = model.predict(events)                          # (B, 2, 480, 640)
        predictions = predictions.permute(0, 2, 3, 1).cpu().numpy()   # (B, 480, 640, 2)

        for prediction, file_index in zip(predictions, file_indices.tolist()):
            encoded = _flow_to_16bit(prediction)
            png_writer(os.path.join(sequence_dir, f"{file_index:06d}.png"), encoded)


def main(argv: list[str] | None = None) -> None:
    args = build_submission_parser().parse_args(argv)

    if args.task != "optical_flow" or args.dataset != "dsec":
        raise SystemExit("this script only ever prepares a DSEC optical-flow submission; "
                          "pass --task optical_flow --dataset dsec")

    model, _, _, _ = build_model_and_datamodule(args, checkpoint_path=args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    data_path = getattr(args, "data_path", None) or os.path.expanduser("~/data")
    dsec_test_root = os.path.join(data_path, "DSEC", "test")
    if not os.path.isdir(dsec_test_root):
        raise SystemExit(f"{dsec_test_root} doesn't exist -- run "
                          f"'python scripts/download_dsec_test.py --data-path {data_path}' first")

    # same transform as DSECDataModule's test transform, so the model sees its training input format
    transform = Compose([
        tonic_transforms.ToFrame(sensor_size=DSECDataModule._SENSOR_SIZE, n_time_bins=args.num_steps),
        FrameResize(DSECDataModule._OUTPUT_SIZE),
    ])

    png_writer = _make_png_writer()

    os.makedirs(args.output_dir, exist_ok=True)
    for recording in args.sequences:
        _write_sequence(
            model, recording, dsec_test_root, transform, args.window_duration_s, args.batch_size,
            args.num_workers, device, args.output_dir, png_writer,
        )
        print(f"[prepare_dsec_submission] wrote {recording}")

    zip_path = args.zip_path or f"{args.output_dir.rstrip('/')}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for recording in args.sequences:
            sequence_dir = os.path.join(args.output_dir, recording)
            for filename in sorted(os.listdir(sequence_dir)):
                archive.write(os.path.join(sequence_dir, filename), arcname=os.path.join(recording, filename))

    print(f"[prepare_dsec_submission] wrote {zip_path}")
    print(f"[prepare_dsec_submission] verify before uploading: "
          f"python scripts/check_dsec_submission.py {args.output_dir} <test_forward_flow_timestamps_dir>")


if __name__ == "__main__":
    main()
