"""Download a checkpoint uploaded to MLflow by `run.py --save-checkpoint` as a local .ckpt file.

    python scripts/download_model.py --run-id <run_id>
    python scripts/download_model.py --run-id <run_id> --out my_checkpoints/

Prints only the path of the downloaded file, so it can be used directly:

    ckpt=$(python scripts/download_model.py --run-id <run_id>)
    python infer.py --mode e2e --task optical_flow --dataset dsec --checkpoint "$ckpt"

`infer.py --mlflow-run-id` does the same download in one step; this script is for keeping the file
to reuse it.
"""
import argparse

from dell.utils.mlflow_model import download_checkpoint
from dell.utils.setup_mlflow import configure_tracking_uri


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-id", required=True, help="MLflow run ID, shown in the MLflow UI and printed by run.py")
    parser.add_argument("--artifact-path",
                        help="file to download if the run has several under 'checkpoints/' (default: the only one)")
    parser.add_argument("--out", help="destination directory (default: checkpoints/<run-id>/)")
    args = parser.parse_args(argv)

    configure_tracking_uri()
    local_path = download_checkpoint(args.run_id, artifact_path=args.artifact_path, dst_dir=args.out)
    print(local_path)


if __name__ == "__main__":
    main()
