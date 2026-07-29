#!/usr/bin/env bash
#
# fit_and_score.sh (merged) — Pipeline « personne + couteau » sur la FUSION
# COCO + Open Images V7, en UN job OAR :
#   1. fetch COCO      -> $MERGED/coco   (node-local)
#   2. fetch OIV7      -> $MERGED/oiv7   (FiftyOne, node-local)
#   3. merge manifests -> $MERGED/manifest.json
#   4. fit banque (personne SANS couteau) -> PERSISTÉE dans le home (models/merged/)
#   5. histogramme good vs knife + 30 heatmaps
#
# Réutilise tel quel bin/coco/{fit,infer} (CocoDataset est agnostique : il lit le
# manifest fusionné via COCO_PATH). Mêmes knobs : FIT_LAYERS (2+3 / 2 / 3),
# FIT_TRAIN_SUBSET (20000/40000), 5 %, 3 NN, proj 64, FAISS GPU au scoring.
#
#   REMOTE_ENV="FIT_TRAIN_SUBSET=20000 FIT_LAYERS=layer3" \
#   OAR_RESOURCES='host=1' OAR_WALLTIME=08:00:00 OAR_PROPERTIES="cluster='gres'" \
#   DETACH=true ./grid5000_run.sh bin/merged/fit_and_score.sh
set -euo pipefail

export FIT_SAMPLER="${FIT_SAMPLER:-approx_greedy_coreset}"
export FIT_CORESET_PCT="${FIT_CORESET_PCT:-0.05}"
export FIT_CORESET_PROJ_DIM="${FIT_CORESET_PROJ_DIM:-64}"
export FIT_NUM_NN="${FIT_NUM_NN:-3}"
export FIT_TRAIN_SUBSET="${FIT_TRAIN_SUBSET:-20000}"
export FIT_MODELS_DIR="${FIT_MODELS_DIR:-models/merged}"     # home = banque conservée
export HIST_FAISS_GPU="${HIST_FAISS_GPU:-1}"
export INFER_FAISS_GPU="${INFER_FAISS_GPU:-1}"
MERGED="${MERGED:-/tmp/${USER}/merged}"
export COCO_PATH="${MERGED}"                                 # CocoDataset lit le manifest fusionné
HEATMAPS_PER_CLASS="${HEATMAPS_PER_CLASS:-15}"
NPC="${NPC:-1000}"
SEED=0

# Chaque source fournit ~la moitié du normal (bank + 20% test). Anomalie = toutes.
PER_SRC_NORMAL="${PER_SRC_NORMAL:-$(( FIT_TRAIN_SUBSET * 5 / 8 + 2000 ))}"

# 0) venv fiftyone ISOLÉ (node-local, py3.11) : n'installe RIEN dans le venv
# projet (uv/py3.13) pour ne pas casser mlflow/patchcore. Réutilisé si déjà là
# (pointer FOENV vers un venv persistant évite la réinstallation à chaque job).
UV="${UV:-$HOME/.local/bin/uv}"
FOENV="${FOENV:-/tmp/${USER}/fo_venv}"
FO_PY="${FOENV}/bin/python"
echo "=== 0) venv FiftyOne isolé -> ${FOENV} ==="
if [ ! -x "${FO_PY}" ]; then
  "${UV}" venv "${FOENV}" --python 3.11
  "${UV}" pip install --python "${FO_PY}" fiftyone pillow
fi
"${FO_PY}" -c "import fiftyone; print('fiftyone', fiftyone.__version__)"

echo "=== 1) FETCH COCO -> ${MERGED}/coco (CAP_NORMAL=${PER_SRC_NORMAL}) ==="
DEST="${MERGED}/coco" CAP_NORMAL="${PER_SRC_NORMAL}" CAP_ANOMALY=0 python tools/coco_fetch.py

echo "=== 2) FETCH OIV7 -> ${MERGED}/oiv7 (venv fiftyone, CAP_NORMAL=${PER_SRC_NORMAL}) ==="
DEST="${MERGED}/oiv7" CAP_NORMAL="${PER_SRC_NORMAL}" CAP_ANOMALY=0 "${FO_PY}" tools/oiv7_fetch.py

echo "=== 3) MERGE -> ${MERGED}/manifest.json ==="
python tools/merge_manifests.py "${MERGED}" "${MERGED}/coco:coco" "${MERGED}/oiv7:oiv7"

# tag identique à build_tag() du fit (suffixe layer vide pour le défaut layer2+3).
ts_lc=$(printf '%s' "${FIT_TRAIN_SUBSET}" | tr '[:upper:]' '[:lower:]')
case "${ts_lc}" in ""|none|all) TS="all";; *) TS="${FIT_TRAIN_SUBSET}";; esac
if [ "${FIT_SAMPLER}" = "identity" ]; then SAMP="identity"; else SAMP="${FIT_SAMPLER}_p${FIT_CORESET_PCT}"; fi
case "${FIT_LAYERS:-}" in
  ""|"layer2,layer3") LSUF="";;
  *) LSUF="_$(printf '%s' "${FIT_LAYERS}" | sed 's/layer/l/g; s/,/-/g')";;
esac
TAG="wideresnet50${LSUF}_${SAMP}_ts${TS}_s${SEED}"
BANK_DIR="${FIT_MODELS_DIR}/${TAG}"

echo "=== 4) FIT === ts=${FIT_TRAIN_SUBSET} pct=${FIT_CORESET_PCT} proj=${FIT_CORESET_PROJ_DIM} nn=${FIT_NUM_NN} layers=${FIT_LAYERS:-layer2,layer3}"
echo "    banque (conservée) -> ${BANK_DIR}"
python bin/coco/fit/memory_bank.py

echo "=== 5) HISTOGRAMME + 30 HEATMAPS ==="
export HIST_BANK_DIR="${BANK_DIR}"
export HIST_OUTPUT_PATH="results/merged/histograms/hist_merged${LSUF}_ts${TS}_nn${FIT_NUM_NN}_s${SEED}.png"
export HEATMAP_OUTPUT_PATH="results/merged/heatmaps${LSUF}/overlay_idx{idx}.png"
python bin/coco/infer/histogram.py --n_per_class "${NPC}"
python bin/coco/infer/heatmap.py --n_per_class "${HEATMAPS_PER_CLASS}"

echo "=== Terminé ==="
echo "    Banque   : ${BANK_DIR}"
echo "    Histo    : results/merged/histograms/p${FIT_CORESET_PCT}/"
echo "    Heatmaps : results/merged/heatmaps${LSUF}/ts${TS}_p${FIT_CORESET_PCT}/"
