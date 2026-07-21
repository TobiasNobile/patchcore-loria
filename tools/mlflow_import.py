"""Fusionne les runs d'une base MLflow source dans la base locale canonique.

Raison d'être : on ne peut PAS rsync une base SQLite MLflow par-dessus une autre
(ça écrase au lieu de fusionner, et les artifact_uri sont des chemins absolus
propres à chaque machine). Cet outil lit les runs de la base distante rapatriée
et les RE-CRÉE dans la base locale (params + historique de metrics + tags +
artefacts), pour n'avoir au final qu'UNE seule base — plus de mlruns_remote.db.

Idempotent : chaque run importé est tagué `import.source_run_id`. Relancer
l'import ne recrée pas ce qui est déjà là (on repère les runs déjà importés).

Structure : par défaut chaque run est importé dans une expérience de MÊME nom
que dans la source. Avec --route-by-runname, on classe plutôt par tâche d'après
le préfixe du run_name (inspect_heatmap -> celeba-heatmap, score_histogram ->
celeba-histograms, benchmark -> celeba-benchmark), pour ranger l'historique dans
la même structure "une expérience par tâche" que les nouveaux runs.

Usage typique (appelé par remote_run.sh / grid5000_run.sh après le fetch) :

    python tools/mlflow_import.py \
        --source-db .mlflow_import/g5k/mlruns.db \
        --source-artifacts .mlflow_import/g5k/mlruns \
        --origin g5k --route-by-runname

Migration ponctuelle de l'ancienne base parallèle vers la base unique :

    python tools/mlflow_import.py \
        --source-db mlruns_remote.db --source-artifacts mlruns_remote \
        --origin metz --route-by-runname
"""

import argparse
import logging
import os
from datetime import datetime, timezone

from mlflow.entities import Param
from mlflow.tracking import MlflowClient

LOGGER = logging.getLogger(__name__)

# Préfixe de run_name -> expérience de tâche, pour --route-by-runname.
_TASK_ROUTES = (
    ("inspect_heatmap", "celeba-heatmap"),
    ("score_histogram", "celeba-histograms"),
    ("hist-", "celeba-histograms"),
    ("benchmark", "celeba-benchmark"),
)

# Limites de lot MLflow (log_batch) : <=100 params, <=1000 metrics par appel.
_PARAM_CHUNK = 100
_METRIC_CHUNK = 1000


def _uri(db_or_uri: str) -> str:
    """Accepte un chemin de fichier .db ou une URI sqlite:/// déjà formée."""
    if "://" in db_or_uri:
        return db_or_uri
    return "sqlite:///{}".format(db_or_uri)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _route_experiment(source_exp_name, run_name, route_by_runname):
    if not route_by_runname:
        return source_exp_name
    name = run_name or ""
    for prefix, exp in _TASK_ROUTES:
        if name.startswith(prefix):
            return exp
    return source_exp_name


class _ExperimentCache:
    """Résout un nom d'expérience -> id côté destination, en créant au besoin."""

    def __init__(self, client: MlflowClient):
        self._client = client
        self._by_name = {}

    def id_for(self, name: str) -> str:
        if name not in self._by_name:
            exp = self._client.get_experiment_by_name(name)
            if exp is None:
                exp_id = self._client.create_experiment(name)
            else:
                exp_id = exp.experiment_id
            self._by_name[name] = exp_id
        return self._by_name[name]


def _already_imported(dst: MlflowClient, dest_exp_id: str, source_run_id: str) -> bool:
    hits = dst.search_runs(
        [dest_exp_id],
        filter_string="tags.`import.source_run_id` = '{}'".format(source_run_id),
        max_results=1,
    )
    return len(hits) > 0


def _artifacts_dir(source_artifacts_root, src_exp_id, src_run_id, fallback_uri):
    """Où trouver, sur DISQUE local, les artefacts rapatriés d'un run.

    Le file store MLflow range les artefacts sous <root>/<exp_id>/<run_id>/
    artifacts. Si --source-artifacts n'est pas donné, on retombe sur le chemin
    file:// de la base source (utile quand base et artefacts sont côte à côte).
    """
    if source_artifacts_root:
        cand = os.path.join(source_artifacts_root, src_exp_id, src_run_id, "artifacts")
        if os.path.isdir(cand):
            return cand
    if fallback_uri and fallback_uri.startswith("file:"):
        cand = fallback_uri[len("file://"):]
        if os.path.isdir(cand):
            return cand
    return None


def import_runs(
    source_uri,
    dest_uri,
    source_artifacts_root=None,
    origin=None,
    route_by_runname=False,
):
    src = MlflowClient(tracking_uri=source_uri)
    dst = MlflowClient(tracking_uri=dest_uri)
    dest_cache = _ExperimentCache(dst)
    now_iso = datetime.now(timezone.utc).isoformat()

    imported = skipped = 0
    per_exp = {}

    for src_exp in src.search_experiments():
        src_runs = src.search_runs([src_exp.experiment_id], max_results=50000)
        for run in src_runs:
            src_run_id = run.info.run_id
            run_name = run.data.tags.get("mlflow.runName")
            dest_exp_name = _route_experiment(
                src_exp.name, run_name, route_by_runname
            )
            dest_exp_id = dest_cache.id_for(dest_exp_name)

            if _already_imported(dst, dest_exp_id, src_run_id):
                skipped += 1
                continue

            # Tags : on recopie ceux de l'utilisateur (dont origin si présent),
            # on laisse tomber les internes mlflow.* (regénérés), et on ajoute la
            # traçabilité d'import + l'origine si elle manquait.
            tags = {
                k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")
            }
            if origin and not tags.get("origin"):
                tags["origin"] = origin
            tags["import.source_run_id"] = src_run_id
            tags["import.source_experiment"] = src_exp.name
            tags["import.imported_at"] = now_iso

            new_run = dst.create_run(
                experiment_id=dest_exp_id,
                start_time=run.info.start_time,
                tags=tags,
                run_name=run_name,
            )
            new_run_id = new_run.info.run_id

            params = [Param(k, v) for k, v in run.data.params.items()]
            for chunk in _chunks(params, _PARAM_CHUNK):
                dst.log_batch(new_run_id, params=chunk)

            # Historique complet de chaque metric (pas seulement la dernière
            # valeur) : get_metric_history rend des entités Metric réutilisables
            # telles quelles dans log_batch.
            metrics = []
            for key in run.data.metrics:
                metrics.extend(src.get_metric_history(src_run_id, key))
            for chunk in _chunks(metrics, _METRIC_CHUNK):
                dst.log_batch(new_run_id, metrics=chunk)

            art_dir = _artifacts_dir(
                source_artifacts_root,
                src_exp.experiment_id,
                src_run_id,
                run.info.artifact_uri,
            )
            if art_dir:
                dst.log_artifacts(new_run_id, art_dir)

            dst.set_terminated(
                new_run_id, status=run.info.status, end_time=run.info.end_time
            )

            imported += 1
            per_exp[dest_exp_name] = per_exp.get(dest_exp_name, 0) + 1
            LOGGER.info(
                "Importé %s (%s) -> exp %s", run_name or src_run_id[:8],
                src_run_id[:8], dest_exp_name,
            )

    return imported, skipped, per_exp


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-db", help="Chemin d'une base sqlite MLflow source.")
    ap.add_argument("--source-uri", help="URI MLflow source (alternative à --source-db).")
    ap.add_argument(
        "--source-artifacts",
        help="Racine des artefacts rapatriés (<exp_id>/<run_id>/artifacts dessous).",
    )
    ap.add_argument("--dest-db", default="mlruns.db", help="Base locale cible.")
    ap.add_argument("--dest-uri", help="URI MLflow cible (alternative à --dest-db).")
    ap.add_argument(
        "--origin",
        help="Origine à taguer sur les runs qui n'en ont pas (local|g5k|metz).",
    )
    ap.add_argument(
        "--route-by-runname",
        action="store_true",
        help="Ranger par tâche d'après le préfixe du run_name plutôt que par nom "
        "d'expérience source.",
    )
    args = ap.parse_args()

    if not (args.source_db or args.source_uri):
        ap.error("préciser --source-db ou --source-uri")

    source_uri = args.source_uri or _uri(args.source_db)
    dest_uri = args.dest_uri or _uri(args.dest_db)

    LOGGER.info("Import %s -> %s", source_uri, dest_uri)
    imported, skipped, per_exp = import_runs(
        source_uri=source_uri,
        dest_uri=dest_uri,
        source_artifacts_root=args.source_artifacts,
        origin=args.origin,
        route_by_runname=args.route_by_runname,
    )
    LOGGER.info(
        "Terminé : %d importés, %d déjà présents (ignorés). Répartition : %s",
        imported, skipped, per_exp or "(aucun)",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
