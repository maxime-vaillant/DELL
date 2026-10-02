import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def configure_tracking_uri() -> None:
    """Point MLflow at the tracking server from .env, or at local file tracking if none is set."""
    mlflow.config.enable_async_logging(True)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "local"))


def setup_mlflow(experiment: str) -> None:
    configure_tracking_uri()
    mlflow.set_experiment(experiment)
