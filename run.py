"""Train a model. Run `python run.py --help` for the list of options."""
import os
import sys

import mlflow
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import MLFlowLogger

from dell.cli import add_checkpoint_args, build_model_and_datamodule, build_parser
from dell.utils.setup_mlflow import setup_mlflow

EXPERIMENT_NAMES = {"semantic": "SemanticDELL"}
DEFAULT_EXPERIMENT_NAME = "DELL"


def resolve_gradient_clip_val(model, requested: float) -> float | None:
    """Trainer-level clipping fails on manual-optimization models, so those must clip on their own."""
    gradient_clip_val = requested or None
    if gradient_clip_val is None or model.automatic_optimization:
        return gradient_clip_val

    if not hasattr(model, "gradient_clip_val"):
        print(f"[run] {type(model).__name__} optimizes manually; --gradient-clip-val not applied", file=sys.stderr)
    return None


def build_checkpoint_callback(args, run_name: str) -> ModelCheckpoint | None:
    if args.save_checkpoint == "none":
        return None

    keep_best = args.save_checkpoint == "best"
    return ModelCheckpoint(
        dirpath=f"{args.checkpoint_dir}/{run_name}",
        filename="model",
        save_top_k=1 if keep_best else 0,
        save_last=not keep_best,
        monitor=args.checkpoint_monitor if keep_best else None,
        mode=args.checkpoint_mode,
        enable_version_counter=False,
    )


def upload_checkpoint(checkpoint_callback: ModelCheckpoint, args, run_id: str) -> None:
    checkpoint_path = (checkpoint_callback.best_model_path if args.save_checkpoint == "best"
                       else checkpoint_callback.last_model_path)
    if not checkpoint_path:
        print(f"[run] --save-checkpoint {args.save_checkpoint} was requested but no checkpoint was written "
              f"(--max-epochs 0, or --checkpoint-monitor was never logged)", file=sys.stderr)
        return

    mlflow.log_artifact(checkpoint_path, artifact_path="checkpoints")
    print(f"[run] uploaded checkpoint to MLflow run {run_id}: checkpoints/{os.path.basename(checkpoint_path)}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    add_checkpoint_args(parser)
    args = parser.parse_args(argv)

    if args.save_checkpoint == "best" and args.no_val:
        raise SystemExit("--save-checkpoint best needs validation; use --save-checkpoint last or drop --no-val")

    seed_everything(args.seed, workers=True)

    model, datamodule, model_kwargs, data_kwargs = build_model_and_datamodule(args)

    run_name = args.run_name or f"{args.mode}_{args.task}_{args.dataset}"
    experiment_name = EXPERIMENT_NAMES.get(args.task, DEFAULT_EXPERIMENT_NAME)
    setup_mlflow(experiment_name)
    logger = MLFlowLogger(experiment_name=experiment_name, run_name=run_name, tracking_uri=mlflow.get_tracking_uri())

    gradient_clip_val = resolve_gradient_clip_val(model, args.gradient_clip_val)

    # logger.log_hyperparams only writes on rank zero, unlike mlflow.log_params, which matters under DDP.
    logger.log_hyperparams({
        "mode": args.mode,
        "task": args.task,
        "dataset": args.dataset,
        "run_name": run_name,
        "seed": args.seed,
        "max_epochs": args.max_epochs,
        "devices": args.devices,
        "overfit_batches": args.overfit_batches,
        "no_val": args.no_val,
        "trainer/gradient_clip_val": gradient_clip_val,
        **{f"model/{key}": value for key, value in model_kwargs.items()},
        **{f"data/{key}": value for key, value in data_kwargs.items()},
    })

    checkpoint_callback = build_checkpoint_callback(args, run_name)
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=args.devices,
        logger=logger,
        overfit_batches=args.overfit_batches,
        num_sanity_val_steps=0 if args.no_val else 2,
        gradient_clip_val=gradient_clip_val,
        callbacks=[checkpoint_callback] if checkpoint_callback else [],
    )
    if args.no_val:
        trainer.limit_val_batches = 0

    with mlflow.start_run(run_id=logger.run_id):
        trainer.fit(model, datamodule=datamodule)
        if checkpoint_callback is not None:
            upload_checkpoint(checkpoint_callback, args, logger.run_id)


if __name__ == "__main__":
    main()
