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

# On ne garde PLUS de base MLflow parallèle. La base distante et ses artefacts
# sont rapatriés dans un dossier TEMPORAIRE, puis leurs runs sont IMPORTÉS dans
# la base locale unique (mlruns.db) via tools/mlflow_import.py — origine "metz".
# Résultat : un seul mlruns.db à ouvrir, plus de mlruns_remote.db.
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
# On rapatrie la base + artefacts en temporaire, puis mlflow_import.py re-crée les
# runs dans mlruns.db (chemins régénérés). Idempotent : pas de doublon au réimport.
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
