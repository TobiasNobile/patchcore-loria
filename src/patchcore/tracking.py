import logging
import os
from contextlib import contextmanager
from typing import Any, Dict, Optional

import mlflow

LOGGER = logging.getLogger(__name__)

_DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"


class RunContext:
    """Handle returned by patchcore_run. Use it to log metrics and artifacts."""

    def __init__(self, active_run: mlflow.ActiveRun) -> None:
        self._run = active_run

    @property
    def run_id(self) -> str:
        return self._run.info.run_id

    def log_metrics(self, metrics: Dict[str, float]) -> None:
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))
        LOGGER.info("Logged metrics: %s", {k: f"{v:.4f}" for k, v in metrics.items()})

    def log_artifacts(self, path: str) -> None:
        """Log a file or an entire directory as artifacts."""
        if os.path.isdir(path):
            mlflow.log_artifacts(path)
        elif os.path.isfile(path):
            mlflow.log_artifact(path)
        else:
            LOGGER.warning("Artifact path not found, skipping: %s", path)


@contextmanager
def patchcore_run(
    experiment: str,
    run_name: str,
    params: Dict[str, Any],
    tracking_uri: Optional[str] = None,
):
    """Context manager for a single PatchCore MLflow run.

    Usage:
        with patchcore_run("mvtec", "bottle", config) as run:
            # ... training and evaluation ...
            run.log_metrics({"instance_auroc": 0.98, "full_pixel_auroc": 0.95})
            run.log_artifacts("results/segmentation_images/bottle")  # optional

    Args:
        experiment:   MLflow experiment name (e.g. "mvtec").
        run_name:     Name for this specific run (e.g. the dataset category).
        params:       Flat or nested dict of hyperparameters to log.
        tracking_uri: Override the tracking URI. Defaults to MLFLOW_TRACKING_URI
                      env var, then "sqlite:///mlruns.db".
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params(_flatten_params(params))
        LOGGER.info(
            "MLflow run started: %s (id=%s)", run_name, active_run.info.run_id
        )
        try:
            yield RunContext(active_run)
        except Exception:
            mlflow.set_tag("run_status", "FAILED")
            raise


def _flatten_params(d: Dict[str, Any], prefix: str = "") -> Dict[str, str]:
    """Recursively flatten a nested dict to dot-separated string keys."""
    out: Dict[str, str] = {}
    for k, v in d.items():
        key = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            out.update(_flatten_params(v, key))
        elif isinstance(v, (list, tuple)):
            out[key] = ",".join(str(x) for x in v)
        else:
            out[key] = str(v)
    return out
