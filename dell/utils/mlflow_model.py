"""Download checkpoints that run.py uploaded to MLflow.

Checkpoints are plain MLflow artifacts rather than MLflow models: they do not store constructor
arguments, so they only load with the same --mode/--task/--dataset and model flags as the run.
"""
import os

from mlflow import artifacts
from mlflow.tracking import MlflowClient

CHECKPOINT_ARTIFACT_DIR = "checkpoints"


def download_checkpoint(run_id: str, artifact_path: str | None = None, dst_dir: str | None = None) -> str:
    """Download a run's checkpoint and return its local path.

    Without `artifact_path`, the run must have exactly one file under `checkpoints/`.
    """
    if artifact_path is None:
        entries = MlflowClient().list_artifacts(run_id, CHECKPOINT_ARTIFACT_DIR)
        files = [entry.path for entry in entries if not entry.is_dir]
        if not files:
            raise FileNotFoundError(
                f"run {run_id} has no files under '{CHECKPOINT_ARTIFACT_DIR}/'; was it trained with --save-checkpoint?"
            )
        if len(files) > 1:
            raise ValueError(
                f"run {run_id} has several checkpoints under '{CHECKPOINT_ARTIFACT_DIR}/': {files}; "
                f"pass --mlflow-artifact-path to pick one"
            )
        artifact_path = files[0]

    dst_dir = dst_dir or os.path.join(CHECKPOINT_ARTIFACT_DIR, run_id)
    os.makedirs(dst_dir, exist_ok=True)
    return artifacts.download_artifacts(run_id=run_id, artifact_path=artifact_path, dst_path=dst_dir)
