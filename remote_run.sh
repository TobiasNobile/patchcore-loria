#!/usr/bin/env bash
#
# remote_run.sh — Sync le code local vers le serveur distant, exécute un script Python
# là-bas, puis rapatrie les résultats générés.
#
# Usage:
#   ./remote_run.sh                                     # lance le script par défaut
#   ./remote_run.sh bin/fit_memory_bank_celeba.py       # construit la banque mémoire
#   ./remote_run.sh bin/infer_heatmap_celeba.py --image_index 900
#                                                       # arguments transmis au script Python
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
REMOTE_USER="nobile_tob"
REMOTE_HOST="dce.metz.centralesupelec.fr"
REMOTE_DIR="~/patchcore-inspection"
LOCAL_DIR="$HOME/dev/telecom/stage_1a/patchcore-inspection"

# dce.metz.centralesupelec.fr n'est que la passerelle SSH (login node) : elle
# n'a pas de GPU. Les GPU ne sont accessibles que via une partition SLURM
# (cf. https://dce.pages.centralesupelec.fr), donc on passe par `srun` au lieu
# d'exécuter python directement sur la passerelle. gpu_inter est dispo H24
# mais plafonnée à 2h de walltime et 1 job à la fois — largement suffisant
# pour un run d'inspection ; augmenter SLURM_TIME si besoin pour un run plus long.
SLURM_PARTITION="gpu_inter"
SLURM_TIME="02:00:00"

# Script à exécuter côté serveur (par défaut, ou 1er argument), le reste
# des arguments est transmis tel quel au script Python.
SCRIPT="${1:-bin/infer_heatmap_celeba.py}"
if [[ $# -gt 0 ]]; then
  shift
fi
SCRIPT_ARGS=("$@")

# Dossiers/fichiers à ne PAS envoyer (gros fichiers, déjà présents côté serveur,
# ou spécifiques à la machine locale). mlruns.db est exclu car il contient des
# artifact_location absolus propres à cette machine (écraser la base distante
# avec ferait pointer MLflow vers des chemins locaux inexistants sur le serveur).
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude 'models' --exclude 'mlruns' --exclude 'results' --exclude 'mlruns.db' --exclude 'mlflow.db')

# Base MLflow distante rapatriée sous ces noms, séparés de mlruns.db/mlruns
# locaux pour ne jamais les écraser (deux historiques MLflow indépendants).
REMOTE_MLFLOW_DB="mlruns_remote.db"
REMOTE_MLFLOW_ARTIFACTS="mlruns_remote"

# ─── 1. Sync : local → serveur ────────────────────────────────────────────
echo "📤  Envoi du code vers ${REMOTE_HOST}..."
rsync -avz "${EXCLUDES[@]}" \
  "${LOCAL_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# ─── 2. Exécution sur le serveur ──────────────────────────────────────────
QUOTED_ARGS=""
for arg in "${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}"; do
  QUOTED_ARGS+=" $(printf '%q' "$arg")"
done
echo "🚀  Exécution de ${SCRIPT}${QUOTED_ARGS} sur le serveur (partition SLURM ${SLURM_PARTITION}, GPU)..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" \
  "cd ${REMOTE_DIR} && source .venv/bin/activate && srun --partition=${SLURM_PARTITION} --time=${SLURM_TIME} python $(printf '%q' "$SCRIPT")${QUOTED_ARGS}"

# ─── 3. Sync : serveur → local (rapatrie les outputs) ─────────────────────
echo "📥  Récupération des résultats..."
rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results/" \
  "${LOCAL_DIR}/results/" 2>/dev/null || echo "  (pas de dossier results/ à rapatrier)"

rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/mlruns/" \
  "${LOCAL_DIR}/${REMOTE_MLFLOW_ARTIFACTS}/" 2>/dev/null || echo "  (pas de dossier mlruns/ à rapatrier)"

rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/mlruns.db" \
  "${LOCAL_DIR}/${REMOTE_MLFLOW_DB}" 2>/dev/null || echo "  (pas de mlruns.db à rapatrier)"

# La base distante référence des artifact_uri absolus propres au serveur
# (ex: /usr/users/.../patchcore-inspection/mlruns/...). On les réécrit vers
# le chemin local équivalent pour que `mlflow ui` local puisse afficher les
# artefacts (images) sans erreur. Idempotent : sans-effet si déjà réécrit.
if [[ -f "${LOCAL_DIR}/${REMOTE_MLFLOW_DB}" ]]; then
  echo "🔧  Réécriture des chemins d'artefacts (serveur → local)..."
  REMOTE_ABS_DIR=$(ssh "${REMOTE_USER}@${REMOTE_HOST}" "cd ${REMOTE_DIR} && pwd")
  sqlite3 "${LOCAL_DIR}/${REMOTE_MLFLOW_DB}" "
    UPDATE experiments SET artifact_location = REPLACE(artifact_location, '${REMOTE_ABS_DIR}/mlruns', '${LOCAL_DIR}/${REMOTE_MLFLOW_ARTIFACTS}');
    UPDATE runs SET artifact_uri = REPLACE(artifact_uri, '${REMOTE_ABS_DIR}/mlruns', '${LOCAL_DIR}/${REMOTE_MLFLOW_ARTIFACTS}');
  "
fi

echo "✅  Terminé."
echo "   → Compare les runs : mlflow ui --backend-store-uri sqlite:///${REMOTE_MLFLOW_DB}"
