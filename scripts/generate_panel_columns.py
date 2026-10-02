"""Write the event, ground-truth and prediction columns of validation panels by running inference.

Unlike `extract_panel_columns.py`, this does not need a logged video, and always uses the current
color scheme.

    python scripts/generate_panel_columns.py --mode e2e --task optical_flow --dataset dsec \\
        --checkpoint checkpoints/e2e_of/model.ckpt --label e2e --indices 0 50 99
    python scripts/generate_panel_columns.py --mode e2e --task semantic --dataset m3ed \\
        --mlflow-run-id <run_id> --label e2e --indices 20 --out-dir out/panels/semantic

The model flags must match the training run, as for `infer.py`. The model runs through
`Trainer.validate()` with panel collection forced on and unbounded, so every validation sample gets
the panel built during training, in dataset order; `--indices` selects among them. Validation stops
after the batch containing the largest index. MLflow artifact logging is disabled, so nothing is
written back to the run.
"""
from pathlib import Path

import imageio.v2 as imageio
import torch
from pytorch_lightning import Trainer, seed_everything

from dell.cli import build_infer_parser, build_model_and_datamodule, resolve_checkpoint_path
from dell.utils.image import PANEL_GAP_PX, event_rgb_on_white
from dell.utils.panel_capture import (
    capture_validation_frames, native_resolution_validation_dataset, silence_mlflow_artifacts,
)

GAP_PX = PANEL_GAP_PX
NUM_COLUMNS = {"optical_flow": 5, "semantic": 3}


def _column_slice(panel, column_index: int, col_width: int):
    x = column_index * (col_width + GAP_PX)
    return panel[:, x:x + col_width]


def _masked_prediction(gt_column, pred_column):
    """Blank the prediction wherever the ground truth is invalid.

    Only the rendered panel is available here, in which invalid ground truth is pure black and
    nothing else is (valid zero flow is white).
    """
    masked = pred_column.copy()
    masked[(gt_column == 0).all(axis=-1)] = 0
    return masked


def main(argv: list[str] | None = None) -> None:
    parser = build_infer_parser()
    parser.description, parser.epilog = __doc__, None
    parser.add_argument("--indices", type=int, nargs="+", required=True,
                         help="0-based validation sample indices (dataset order), e.g. 0 50 99")
    parser.add_argument("--label", default=None,
                         help="name used in pred_<label>_<idx>.png (default: the MLflow run id or the checkpoint name)")
    parser.add_argument("--out-dir", type=Path, default=Path("out/panels"))
    args = parser.parse_args(argv)

    if args.task not in NUM_COLUMNS:
        raise SystemExit(f"--task {args.task} has no comparison panel to extract columns from "
                          f"(only optical_flow/semantic do)")

    checkpoint_path = resolve_checkpoint_path(args)

    label = args.label or args.mlflow_run_id or Path(checkpoint_path).stem
    args.out_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(args.seed, workers=True)

    model, datamodule, _, _ = build_model_and_datamodule(args, checkpoint_path=checkpoint_path)
    model.eval()

    captured_frames = capture_validation_frames(model)
    silence_mlflow_artifacts()

    needed_batches = max(args.indices) // args.batch_size + 1
    trainer = Trainer(
        accelerator="auto", devices=args.devices, logger=False, enable_checkpointing=False,
        limit_val_batches=needed_batches,
    )
    trainer.validate(model, datamodule=datamodule)

    if max(args.indices) >= len(captured_frames):
        raise SystemExit(
            f"only {len(captured_frames)} validation sample(s) were run (indices 0.."
            f"{len(captured_frames) - 1}) but index {max(args.indices)} was requested -- pass a "
            f"smaller index, or check --dataset matches what you expect"
        )

    num_columns = NUM_COLUMNS[args.task]
    total_width = captured_frames[0].shape[1]
    col_width = (total_width - (num_columns - 1) * GAP_PX) // num_columns

    native_val_dataset = native_resolution_validation_dataset(datamodule)

    for idx in args.indices:
        panel = captured_frames[idx]
        gt_column = _column_slice(panel, 1, col_width)
        pred_column = _column_slice(panel, 2, col_width)
        events_native, _ = native_val_dataset[idx]
        events_native = torch.as_tensor(events_native).float()
        event_image = event_rgb_on_white(events_native, events_native.shape[-2:]).cpu().numpy()
        outputs = [
            (event_image, args.out_dir / f"event_{idx}.png"),
            (gt_column, args.out_dir / f"gt_{idx}.png"),
            (pred_column, args.out_dir / f"pred_{label}_{idx}.png"),
        ]
        if args.task == "optical_flow":
            outputs.append(
                (_masked_prediction(gt_column, pred_column), args.out_dir / f"pred_masked_{label}_{idx}.png")
            )
        for image, out_path in outputs:
            imageio.imwrite(out_path, image)
            print(f"[generate_panel_columns] wrote {out_path}")


if __name__ == "__main__":
    main()
