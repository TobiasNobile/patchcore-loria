#!/usr/bin/env bash
#
# score.sh — Scoring SEUL (histogramme + 30 heatmaps) contre une banque COCO
# DÉJÀ construite et persistée. Pas de fit : on re-fetch juste le subset COCO sur
# le nœud (pour retrouver le même split test, déterministe par seed), on charge la
# banque, et on score en FAISS GPU (la banque ~qq Go tient en VRAM -> rapide).
#
# À utiliser quand un fit_and_score a été coupé par le walltime APRÈS le fit
# (banque sauvée) mais avant/pendant le scoring.
#
#   REMOTE_ENV="FIT_TRAIN_SUBSET=40000" \
#   OAR_RESOURCES='host=1' OAR_WALLTIME=03:00:00 OAR_PROPERTIES="cluster='gres'" \
#   DETACH=true ./grid5000_run.sh bin/coco/score.sh
set -euo pipefail

export FIT_SAMPLER="${FIT_SAMPLER:-approx_greedy_coreset}"
export FIT_CORESET_PCT="${FIT_CORESET_PCT:-0.05}"
export FIT_TRAIN_SUBSET="${FIT_TRAIN_SUBSET:-40000}"
export FIT_MODELS_DIR="${FIT_MODELS_DIR:-models/coco}"
export COCO_PATH="${COCO_PATH:-/tmp/${USER}/coco}"
export HIST_FAISS_GPU=1
export INFER_FAISS_GPU=1
NUM_NN="${FIT_NUM_NN:-3}"
HEATMAPS_PER_CLASS="${HEATMAPS_PER_CLASS:-15}"
NPC="${NPC:-1000}"
SEED=0

# tag identique à build_tag() du fit -> retrouve la banque persistée.
ts_lc=$(printf '%s' "${FIT_TRAIN_SUBSET}" | tr '[:upper:]' '[:lower:]')
case "${ts_lc}" in ""|none|all) TS="all";; *) TS="${FIT_TRAIN_SUBSET}";; esac
if [ "${FIT_SAMPLER}" = "identity" ]; then SAMP="identity"; else SAMP="${FIT_SAMPLER}_p${FIT_CORESET_PCT}"; fi
case "${FIT_LAYERS:-}" in
  ""|"layer2,layer3") LSUF="";;
  *) LSUF="_$(printf '%s' "${FIT_LAYERS}" | sed 's/layer/l/g; s/,/-/g')";;
esac
TAG="wideresnet50${LSUF}_${SAMP}_ts${TS}_s${SEED}"
BANK_DIR="${FIT_MODELS_DIR}/${TAG}"
if [ ! -d "${BANK_DIR}" ]; then
  echo "ERREUR : banque introuvable -> ${BANK_DIR}" >&2
  echo "  (lance d'abord bin/coco/fit_and_score.sh, ou ajuste FIT_TRAIN_SUBSET)" >&2
  exit 1
fi

# Même subset COCO que le fit (déterministe par seed) pour un split test cohérent.
export CAP_NORMAL="${CAP_NORMAL:-$(( FIT_TRAIN_SUBSET * 5 / 4 + 3000 ))}"
export CAP_ANOMALY="${CAP_ANOMALY:-0}"
export DEST="${COCO_PATH}"
echo "=== FETCH COCO (test split) -> ${COCO_PATH} ==="
python tools/coco_fetch.py

echo "=== HISTOGRAMME good vs knife (banque ${TAG}, FAISS GPU) ==="
export HIST_BANK_DIR="${BANK_DIR}"
export HIST_OUTPUT_PATH="results/coco/histograms/hist_coco${LSUF}_ts${TS}_nn${NUM_NN}_s${SEED}.png"
export HEATMAP_OUTPUT_PATH="results/coco/heatmaps${LSUF}/overlay_idx{idx}.png"
python bin/coco/infer/histogram.py --n_per_class "${NPC}"

echo "=== 30 HEATMAPS (${HEATMAPS_PER_CLASS} good + ${HEATMAPS_PER_CLASS} knife) ==="
python bin/coco/infer/heatmap.py --n_per_class "${HEATMAPS_PER_CLASS}"

echo "=== Terminé ==="
echo "    Histo    : results/coco/histograms/p${FIT_CORESET_PCT}/"
echo "    Heatmaps : results/coco/heatmaps/ts${TS}_p${FIT_CORESET_PCT}/"
