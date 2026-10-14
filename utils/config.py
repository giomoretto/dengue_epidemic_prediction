import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# MLflow's standard environment variable takes precedence over the local default
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{PROJECT_ROOT / 'mlflow.db'}"
)
