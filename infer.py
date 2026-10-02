"""Evaluate a trained checkpoint on the validation set. Run `python infer.py --help` for the options."""
from pytorch_lightning import Trainer, seed_everything

from dell.cli import build_infer_parser, build_model_and_datamodule, resolve_checkpoint_path


def main(argv: list[str] | None = None) -> None:
    args = build_infer_parser().parse_args(argv)

    checkpoint_path = resolve_checkpoint_path(args)

    seed_everything(args.seed, workers=True)

    model, datamodule, _, _ = build_model_and_datamodule(args, checkpoint_path=checkpoint_path)
    model.eval()

    trainer = Trainer(
        accelerator="auto",
        devices=args.devices,
        logger=False,
        enable_checkpointing=False,
        overfit_batches=args.overfit_batches,
    )
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
