#!/usr/bin/env bash
#
# remote_run.sh — sync le code vers le DCE de Metz, exécute un script sur une
# partition SLURM GPU, rapatrie les résultats.
#
#   ./remote_run.sh bin/celeba/fit/memory_bank.py
#   ./remote_run.sh bin/celeba/infer/heatmap.py --image_index 900
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
REMOTE_USER="nobile_tob"
REMOTE_HOST="dce.metz.centralesupelec.fr"
REMOTE_DIR="~/patchcore-inspection"
LOCAL_DIR="$HOME/dev/telecom/stage_1a/patchcore-inspection"

# La passerelle n'a pas de GPU, d'où `srun`. gpu_inter : dispo H24, plafonnée
# à 2h et 1 job (cf. https://dce.pages.centralesupelec.fr).
SLURM_PARTITION="gpu_inter"
SLURM_TIME="02:00:00"

# Script à exécuter côté serveur (par défaut, ou 1er argument), le reste
# des arguments est transmis tel quel au script Python.
SCRIPT="${1:-bin/celeba/infer/heatmap.py}"
if [[ $# -gt 0 ]]; then
  shift
fi
SCRIPT_ARGS=("$@")

# Exclus : gros fichiers, ou spécifiques à cette machine (mlruns.db porte des
# chemins absolus qui pointeraient dans le vide côté serveur).
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude 'models' --exclude 'mlruns' --exclude 'results'
          --exclude 'mlruns.db' --exclude 'mlflow.db' --exclude 'mlruns.db.bak-*'
          --exclude 'mlruns_remote' --exclude 'mlruns_remote.db' --exclude 'mlruns_array'
          --exclude '.mlflow_import' --exclude '__pycache__' --exclude '.pytest_cache')

# Pas de base MLflow parallèle : la base distante est importée dans mlruns.db
# par tools/mlflow_import.py, taguée "metz".
IMPORT_TMP=".mlflow_import/metz"
IMPORT_ORIGIN="metz"

# Python local (pour l'import) : le venv du projet, sinon le python du PATH.
LOCAL_PYTHON="${LOCAL_DIR}/.venv/bin/python"
[[ -x "${LOCAL_PYTHON}" ]] || LOCAL_PYTHON="python"

# ─── 1. Sync : local → serveur ────────────────────────────────────────────
echo "Envoi du code vers ${REMOTE_HOST}..."
rsync -avz "${EXCLUDES[@]}" \
  "${LOCAL_DIR}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# ─── 2. Exécution sur le serveur ──────────────────────────────────────────
QUOTED_ARGS=""
for arg in "${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}"; do
  QUOTED_ARGS+=" $(printf '%q' "$arg")"
done
echo "Exécution de ${SCRIPT}${QUOTED_ARGS} sur le serveur (partition SLURM ${SLURM_PARTITION}, GPU)..."
# PATCHCORE_ORIGIN=metz : les runs MLflow créés là-bas sont tagués origin=metz
# dès leur création (cf. patchcore.tracking), donc identifiables une fois fusionnés.
ssh "${REMOTE_USER}@${REMOTE_HOST}" \
  "cd ${REMOTE_DIR} && source .venv/bin/activate && PATCHCORE_ORIGIN=metz srun --partition=${SLURM_PARTITION} --time=${SLURM_TIME} python $(printf '%q' "$SCRIPT")${QUOTED_ARGS}"

# ─── 3. Sync : serveur → local (rapatrie les outputs) ─────────────────────
echo "Récupération des résultats..."
rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results/" \
  "${LOCAL_DIR}/results/" 2>/dev/null || echo "  (pas de dossier results/ à rapatrier)"

# ─── 4. MLflow distant -> import dans la base locale unique ───────────────
# Idempotent : pas de doublon au réimport.
echo "Récupération de la base MLflow distante (temporaire ${IMPORT_TMP})..."
mkdir -p "${LOCAL_DIR}/${IMPORT_TMP}/mlruns"
rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/mlruns.db" \
  "${LOCAL_DIR}/${IMPORT_TMP}/mlruns.db" 2>/dev/null || echo "  (pas de mlruns.db distante)"
rsync -avz \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/mlruns/" \
  "${LOCAL_DIR}/${IMPORT_TMP}/mlruns/" 2>/dev/null || echo "  (pas d'artefacts mlruns/ distants)"

if [[ -f "${LOCAL_DIR}/${IMPORT_TMP}/mlruns.db" ]]; then
  echo "Fusion des runs distants dans la base locale unique (mlruns.db)..."
  ( cd "${LOCAL_DIR}" && "${LOCAL_PYTHON}" tools/mlflow_import.py \
      --source-db "${IMPORT_TMP}/mlruns.db" \
      --source-artifacts "${IMPORT_TMP}/mlruns" \
      --origin "${IMPORT_ORIGIN}" --route-by-runname )
fi

echo "Terminé."
echo "  Historique unifié : mlflow ui --backend-store-uri sqlite:///mlruns.db"
