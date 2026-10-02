"""Command-line arguments and model/datamodule construction shared by run.py, infer.py and scripts/."""
import argparse
import inspect
import sys

from pytorch_lightning import LightningDataModule, LightningModule

from dell.registry import DATA_ARGS, DATASETS, MODEL_ARGS, MODULES, TASK_DATASETS
from dell.utils.mlflow_model import download_checkpoint
from dell.utils.setup_mlflow import configure_tracking_uri

DESCRIPTION = """\
Train a spiking U-Net on event-camera data with end-to-end or block-local learning.

Pick a learning mode, a task and a dataset; every other flag is optional.
Flags in the model, logging and data groups are only used when the chosen
model or dataset supports them; otherwise they are ignored with a notice.
"""

EPILOG = """\
examples:
  python run.py --mode e2e     --task optical_flow --dataset dsec
  python run.py --mode local   --task optical_flow --dataset dsec --batch-size 8 --depth 4
  python run.py --mode decolle --task semantic     --dataset m3ed

valid task/dataset pairs:
  optical_flow    dsec
  semantic        dsec, m3ed

All flags are described in docs/options.md.
"""


def accepted_parameter_names(cls: type) -> set[str]:
    """Names accepted by `cls.__init__`, following `**kwargs` forwarding up the class hierarchy."""
    named_kinds = (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    names: set[str] = set()
    for klass in cls.__mro__:
        if "__init__" not in vars(klass):
            continue
        parameters = inspect.signature(klass.__init__).parameters.values()
        names |= {p.name for p in parameters if p.kind in named_kinds}
        if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters):
            break
    return names


def filter_supported_kwargs(cls: type, candidates: dict, notify: bool = True) -> dict:
    """Keep the candidates accepted by `cls.__init__` and, if `notify`, print a notice for the others."""
    accepted = accepted_parameter_names(cls)
    supported = {key: value for key, value in candidates.items() if key in accepted}
    ignored = candidates.keys() - supported.keys()
    if ignored and notify:
        print(f"[run] {cls.__name__} ignores: {', '.join(sorted(ignored))}", file=sys.stderr)
    return supported


def build_parser() -> argparse.ArgumentParser:
    """Arguments describing the model and data. Shared with infer.py, which must rebuild the
    exact architecture a checkpoint was trained with."""
    parser = argparse.ArgumentParser(
        usage="%(prog)s --mode MODE --task TASK --dataset DATASET [options]",
        description=DESCRIPTION, epilog=EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Flags defaulting to argparse.SUPPRESS are left out of the namespace unless passed, so the
    # model's own constructor default applies.
    unset = argparse.SUPPRESS
    boolean = argparse.BooleanOptionalAction

    required = parser.add_argument_group("what to run (required)")
    required.add_argument("--mode", choices=["e2e", "local", "decolle"], required=True,
                          help="e2e: end-to-end backprop | local: one local loss and optimizer per block | "
                               "decolle: like local, with fixed random readouts")
    required.add_argument("--task", choices=["optical_flow", "semantic"], required=True)
    required.add_argument("--dataset", choices=sorted(DATASETS), required=True)

    run = parser.add_argument_group("run")
    run.add_argument("--max-epochs", type=int, default=100, help="number of epochs (default: %(default)s)")
    run.add_argument("--devices", type=int, nargs="+", default=[0], help="GPU ids (default: 0)")
    run.add_argument("--seed", type=int, default=42, help="(default: %(default)s)")
    run.add_argument("--run-name", help="MLflow run name (default: <mode>_<task>_<dataset>)")
    run.add_argument("--overfit-batches", type=int, default=0,
                     help="train and validate on this many batches only, for debugging (default: off)")
    run.add_argument("--no-val", action="store_true", help="skip validation")
    run.add_argument("--gradient-clip-val", type=float, default=1.0,
                     help="gradient-norm clipping, 0 to disable; local/decolle clip each block separately "
                          "(default: %(default)s)")

    model = parser.add_argument_group("model")
    model.add_argument("--base-channels", type=int, default=32, help="channels of the first block (default: %(default)s)")
    model.add_argument("--depth", type=int, default=4, help="number of encoder/decoder levels (default: %(default)s)")
    model.add_argument("--lr", type=float, default=5e-4, help="learning rate (default: %(default)s)")
    model.add_argument("--use-separable-activation", action=boolean, default=unset,
                       help="add a neuron between the depthwise and pointwise convolutions of every "
                            "separable convolution; e2e and local only (default: off)")
    model.add_argument("--use-deep-supervision", action=boolean, default=unset,
                       help="supervise every decoder block, not only the last one "
                            "(default: on for optical_flow, off for semantic)")

    video = parser.add_argument_group("image and video logging")
    video.add_argument("--log-video", action=boolean, default=unset,
                         help="log validation panels and train/val videos (default: on)")
    video.add_argument("--log-last-epoch-only", action=boolean, default=unset,
                         help="log only on the final epoch; --no-log-last-epoch-only logs every "
                              "--log-images-every-n-epochs (default: on)")
    video.add_argument("--log-images-every-n-epochs", type=int, default=unset,
                         help="logging period when --no-log-last-epoch-only is set (default: 10)")
    video.add_argument("--validation-video-fps", type=float, default=unset, help="(default: 8)")
    video.add_argument("--max-video-frames", type=int, default=unset,
                         help="maximum frames kept per video, lower it if you run out of memory (default: 200)")

    data = parser.add_argument_group("data")
    data.add_argument("--data-path", default=unset, help="dataset root directory (default: ~/data)")
    data.add_argument("--batch-size", type=int, default=32, help="(default: %(default)s)")
    data.add_argument("--num-workers", type=int, default=16, help="dataloader workers (default: %(default)s)")
    data.add_argument("--num-steps", type=int, default=16, help="time steps per sample (default: %(default)s)")
    data.add_argument("--target-type", choices=["optical_flow", "semantic"], default=unset,
                      help="dsec only: ground truth to load (default: set by --task)")
    data.add_argument("--window-duration-s", type=float, default=0.1,
                      help="dsec and m3ed: duration of one event window (default: %(default)s)")
    data.add_argument("--target-interval-s", type=float, default=0.1,
                      help="dsec and m3ed: time between two targets (default: %(default)s)")
    data.add_argument("--label-taxonomy", choices=["cityscapes-19", "cityscapes-11"], default=unset,
                      help="dsec and m3ed semantic: label set (default: cityscapes-11)")

    return parser


def add_checkpoint_args(parser: argparse.ArgumentParser) -> None:
    """Training-only flags. They live outside build_parser() so infer.py does not inherit them."""
    checkpoint = parser.add_argument_group("checkpoint")
    checkpoint.add_argument("--save-checkpoint", choices=["none", "last", "best"], default="best",
                            help="weights to upload to MLflow once training ends: last epoch or best "
                                 "--checkpoint-monitor value (default: %(default)s)")
    checkpoint.add_argument("--checkpoint-monitor", default="val/loss",
                            help="metric used to pick the best checkpoint, e.g. val/epe or val/miou; "
                                 "semantic on m3ed logs val_car/... instead of val/... "
                                 "(default: %(default)s)")
    checkpoint.add_argument("--checkpoint-mode", choices=["min", "max"], default="min",
                            help="min if lower --checkpoint-monitor is better, max if higher is "
                                 "(default: %(default)s)")
    checkpoint.add_argument("--checkpoint-dir", default="checkpoints",
                            help="local directory for checkpoint files (default: %(default)s)")


def add_inference_args(parser: argparse.ArgumentParser) -> None:
    """Flags selecting the checkpoint to evaluate."""
    inference = parser.add_argument_group("checkpoint to load (pass exactly one of --checkpoint, --mlflow-run-id)")
    inference.add_argument("--checkpoint", help="path to a local .ckpt file")
    inference.add_argument("--mlflow-run-id", help="download the checkpoint uploaded by this MLflow run")
    inference.add_argument("--mlflow-artifact-path",
                           help="with --mlflow-run-id: file to download if the run has several "
                                "(default: the only one)")


def build_infer_parser() -> argparse.ArgumentParser:
    """Model and data flags plus the checkpoint to load, shared by infer.py and scripts/."""
    parser = build_parser()
    parser.description = (
        "Evaluate a trained checkpoint on the validation set.\n\n"
        "--mode, --task, --dataset and every model flag must match the training run: "
        "checkpoints do not store them."
    )
    parser.epilog = (
        "examples:\n"
        "  python infer.py --mode e2e --task optical_flow --dataset dsec "
        "--checkpoint checkpoints/e2e_optical_flow_dsec/model.ckpt\n"
        "  python infer.py --mode e2e --task optical_flow --dataset dsec --mlflow-run-id <run_id>"
    )
    add_inference_args(parser)
    return parser


def resolve_checkpoint_path(args: argparse.Namespace) -> str:
    """Return the local path given by --checkpoint, or download the one of --mlflow-run-id."""
    if bool(args.checkpoint) == bool(args.mlflow_run_id):
        raise SystemExit("pass exactly one of --checkpoint <path> or --mlflow-run-id <run_id>")
    if args.checkpoint:
        return args.checkpoint

    configure_tracking_uri()
    checkpoint_path = download_checkpoint(args.mlflow_run_id, artifact_path=args.mlflow_artifact_path)
    print(f"downloaded checkpoint to {checkpoint_path}", file=sys.stderr)
    return checkpoint_path


def _build_datamodule(args: argparse.Namespace) -> tuple[LightningDataModule, dict]:
    datamodule_cls = DATASETS[args.dataset]
    given = vars(args)

    candidates = {key: value for key, value in given.items() if key in DATA_ARGS}
    # dsec defaults to optical-flow targets regardless of the task, so a semantic task asks for them.
    if args.task == "semantic" and "target_type" not in candidates and "target_type" in accepted_parameter_names(datamodule_cls):
        candidates["target_type"] = "semantic"

    kwargs = filter_supported_kwargs(datamodule_cls, candidates)
    kwargs["batch_size"] = args.batch_size

    fixed = {"event_view": "left", "rectify": True, "download": True}
    kwargs.update(filter_supported_kwargs(datamodule_cls, fixed, notify=False))

    return datamodule_cls(**kwargs), kwargs


def build_model_and_datamodule(
    args: argparse.Namespace, checkpoint_path: str | None = None,
) -> tuple[LightningModule, LightningDataModule, dict, dict]:
    """Build the model and datamodule selected by `args`, plus the kwargs each was built with.

    With `checkpoint_path`, the model is loaded from that checkpoint. The same flags as the
    training run must be given: checkpoints do not store the constructor arguments.
    """
    if args.dataset not in TASK_DATASETS[args.task]:
        valid = ", ".join(sorted(TASK_DATASETS[args.task]))
        raise SystemExit(f"--task {args.task} has no ground truth for --dataset {args.dataset}; pick one of: {valid}")

    if (args.mode, args.task) not in MODULES:
        valid = ", ".join(sorted(task for mode, task in MODULES if mode == args.mode))
        raise SystemExit(f"--mode {args.mode} has no implementation for --task {args.task}; pick one of: {valid}")

    model_cls = MODULES[(args.mode, args.task)]
    # Built first: the number of classes can depend on the datamodule's arguments.
    datamodule, data_kwargs = _build_datamodule(args)

    model_candidates = {key: value for key, value in vars(args).items() if key in MODEL_ARGS}
    model_kwargs = filter_supported_kwargs(model_cls, model_candidates)
    model_kwargs.update(filter_supported_kwargs(model_cls, {"max_epochs": args.max_epochs}))
    model_kwargs["in_channels"] = type(datamodule).CHANNELS
    if args.task == "semantic":
        model_kwargs["num_classes"] = datamodule.NUM_CLASSES
        model_kwargs["val_stages"] = datamodule.VAL_STAGES
    if args.overfit_batches:
        model_kwargs.update(filter_supported_kwargs(model_cls, {"log_video": False}))

    if checkpoint_path is None:
        model = model_cls(**model_kwargs)
    else:
        model = model_cls.load_from_checkpoint(checkpoint_path, map_location="cpu", **model_kwargs)

    return model, datamodule, model_kwargs, data_kwargs
