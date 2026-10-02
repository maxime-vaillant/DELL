"""Check that `predict()`, used for the DSEC submission, matches the validation metrics.

Runs the checkpoint on the labeled DSEC validation split two ways and prints both: the official
`trainer.validate()` (EPE, AE and NPE logged by `validation_step`) and `model.predict()` scored with
the same `flow_metrics`. Run this once per checkpoint before trusting a submission: a wrongly wired
`predict()` would otherwise only show up as a worse leaderboard score.

    python scripts/check_predict.py --mode e2e --task optical_flow --dataset dsec \\
        --checkpoint checkpoints/OF_E2E/model.ckpt --data-path ~/data

The model flags must match the training run, as for `infer.py`. The two rows should agree within a
fraction of a percent (floating-point summation order differs). A larger gap means either the flags
do not match the training run, or `predict()` is wrong.
"""
import torch
from pytorch_lightning import Trainer
from tqdm import tqdm

from dell.cli import build_model_and_datamodule, build_parser
from dell.utils.flow import flow_metrics, split_flow_target


def build_check_parser():
    parser = build_parser()
    parser.description, parser.epilog = __doc__, None
    check = parser.add_argument_group("check-predict")
    check.add_argument("--checkpoint", required=True, help="path to the .ckpt file")
    check.add_argument("--max-batches", type=int, default=None,
                       help="only score the first N validation batches with predict() "
                            "(default: all); use --overfit-batches to limit trainer.validate() too")
    return parser


@torch.no_grad()
def _predict_metrics(model, loader, device: torch.device, max_batches: int | None):
    """Mean of the metrics over the validation set, weighted by batch size like Lightning's epoch average."""
    epe_sum = ae_sum = 0.0
    pe_sums: dict[int, float] = {}
    n_samples = 0

    for batch_idx, (events, flow_gt) in enumerate(tqdm(loader, desc="predict() pass")):
        if max_batches is not None and batch_idx >= max_batches:
            break
        events, flow_gt = events.to(device), flow_gt.to(device)
        batch_size = events.shape[1]  # (T, B, 2, H, W)

        flow, mask = split_flow_target(flow_gt)
        prediction_full = model.predict(events, output_hw=flow.shape[-2:])
        epe, ae, pe_rates = flow_metrics(prediction_full, flow, mask)

        epe_sum += epe.item() * batch_size
        ae_sum += ae.item() * batch_size
        for n, pe in pe_rates.items():
            pe_sums[n] = pe_sums.get(n, 0.0) + pe.item() * batch_size
        n_samples += batch_size

    return {
        "epe": epe_sum / n_samples,
        "ae": ae_sum / n_samples,
        **{f"{n}pe": total / n_samples for n, total in pe_sums.items()},
    }


def main(argv: list[str] | None = None) -> None:
    args = build_check_parser().parse_args(argv)

    model, datamodule, _, _ = build_model_and_datamodule(args, checkpoint_path=args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    datamodule.setup()

    trainer = Trainer(
        accelerator="auto", devices=args.devices, logger=False,
        enable_checkpointing=False, overfit_batches=args.overfit_batches,
    )
    official = trainer.validate(model, datamodule=datamodule)[0]

    # Trainer.validate() moves the model back to the CPU when it returns
    model = model.to(device).eval()
    official = {k.removeprefix("val/"): v for k, v in official.items() if k.startswith("val/")}

    predicted = _predict_metrics(model, datamodule.val_dataloader(), device, args.max_batches)

    keys = sorted(set(official) | set(predicted), key=lambda k: (k not in ("epe", "ae"), k))
    print(f"\n{'metric':<8}{'trainer.validate()':>20}{'predict()':>16}{'diff':>12}")
    for key in keys:
        official_value = official.get(key)
        predicted_value = predicted.get(key)
        diff = "" if official_value is None or predicted_value is None else f"{predicted_value - official_value:+.4f}"
        official_str = "-" if official_value is None else f"{official_value:.4f}"
        predicted_str = "-" if predicted_value is None else f"{predicted_value:.4f}"
        print(f"{key:<8}{official_str:>20}{predicted_str:>16}{diff:>12}")


if __name__ == "__main__":
    main()
