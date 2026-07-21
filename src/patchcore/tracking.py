import logging
import os
import socket
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import mlflow

LOGGER = logging.getLogger(__name__)

_DEFAULT_TRACKING_URI = "sqlite:///mlruns.db"


def _environment_tags() -> Dict[str, str]:
    """Tags identifiant où un run a été produit (origin/host/job_id/cluster),
    posés à sa création pour retrouver les runs dans la base unique.

    origin vient de PATCHCORE_ORIGIN (posé par les scripts de run), sinon déduit :
    OAR -> g5k, SLURM -> metz, à défaut local.
    """
    host = socket.gethostname()
    oar_job = os.environ.get("OAR_JOB_ID")
    slurm_job = os.environ.get("SLURM_JOB_ID")

    origin = os.environ.get("PATCHCORE_ORIGIN")
    if not origin:
        origin = "g5k" if oar_job else "metz" if slurm_job else "local"

    tags = {"origin": origin, "host": host}
    job_id = oar_job or slurm_job
    if job_id:
        tags["job_id"] = job_id

    if origin == "g5k":
        # Nœud G5K = "grele-3.nancy.grid5000.fr" : cluster = préfixe avant le tiret.
        cluster = host.split(".")[0].split("-")[0]
        if cluster:
            tags["cluster"] = cluster
    else:
        cluster = os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get(
            "SLURM_JOB_PARTITION"
        )
        if cluster:
            tags["cluster"] = cluster

    return tags


def make_run_name(
    backbone_names: List[str],
    sampler_name: str,
    coreset_pct: float,
    imagesize: int = 224,
) -> str:
    """Nom de run MLflow canonique à partir des hyperparamètres clés.

    Convention : {backbone(s)}-{sampler}-p{pct}-im{size}, p.ex.
    ["wideresnet50"], "approx_greedy_coreset", 0.1, 224
    -> "wideresnet50-approx_greedy_coreset-p10-im224".
    """
    backbone = "+".join(backbone_names)
    pct = f"p{int(round(coreset_pct * 100)):02d}"
    return f"{backbone}-{sampler_name}-{pct}-im{imagesize}"


class RunContext:
    """Objet renvoyé par patchcore_run pour logger metrics et artefacts."""

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
        """Logge un fichier ou tout un dossier comme artefacts, et tague le run
        avec le(s) nom(s) de fichier pour le retrouver sans ouvrir le browser."""
        if os.path.isdir(path):
            mlflow.log_artifacts(path)
            names = sorted(os.listdir(path))
            mlflow.set_tag("output_artifacts", ", ".join(names))
        elif os.path.isfile(path):
            mlflow.log_artifact(path)
            mlflow.set_tag("output_artifacts", os.path.basename(path))
        else:
            LOGGER.warning("Artifact path not found, skipping: %s", path)


@contextmanager
def patchcore_run(
    experiment: str,
    run_name: str,
    params: Dict[str, Any],
    tracking_uri: Optional[str] = None,
):
    """Context manager pour un run MLflow PatchCore.

        with patchcore_run("mvtec", "bottle", config) as run:
            run.log_metrics({"instance_auroc": 0.98})
            run.log_artifacts("results/segmentation_images/bottle")  # optionnel

    tracking_uri surcharge l'URI ; par défaut MLFLOW_TRACKING_URI puis
    sqlite:///mlruns.db.
    """
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI", _DEFAULT_TRACKING_URI)
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tags(_environment_tags())
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
    """Aplatit récursivement un dict imbriqué en clés séparées par des points."""
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
