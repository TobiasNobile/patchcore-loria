#!/usr/bin/env bash
#
# fit_and_score.sh — Pipeline COCO « personne + couteau » en UN job OAR :
#   1. fetch du sous-ensemble COCO sur le disque local du nœud (node-local, hors quota) ;
#   2. fit de la banque (personne SANS couteau) -> PERSISTÉE dans le home ;
#   3. histogramme good vs knife (test) ;
#   4. 30 heatmaps (15 good + 15 knife).
#
# Les images COCO (~3-6 Go) vivent sur /tmp du nœud et meurent avec le job ; seuls
# la banque (~3-6 Go, models/coco/) + histogramme + heatmaps (results/coco/) sont
# gardés. Rapatriables par grid5000_run.sh --fetch.
#
# Cible par défaut : 20000 images, coreset 5%, proj 64, 3 NN.
#   REMOTE_ENV="FIT_TRAIN_SUBSET=20000" \
#   OAR_RESOURCES='host=1' OAR_WALLTIME=06:00:00 OAR_PROPERTIES="cluster='gres'" \
#   DETACH=true ./grid5000_run.sh bin/coco/fit_and_score.sh
set -euo pipefail

export FIT_SAMPLER="${FIT_SAMPLER:-approx_greedy_coreset}"
export FIT_CORESET_PCT="${FIT_CORESET_PCT:-0.05}"
export FIT_CORESET_PROJ_DIM="${FIT_CORESET_PROJ_DIM:-64}"
export FIT_NUM_NN="${FIT_NUM_NN:-3}"
export FIT_TRAIN_SUBSET="${FIT_TRAIN_SUBSET:-20000}"
export FIT_MODELS_DIR="${FIT_MODELS_DIR:-models/coco}"     # home = banque conservée
export COCO_PATH="${COCO_PATH:-/tmp/${USER}/coco}"          # images node-local
HEATMAPS_PER_CLASS="${HEATMAPS_PER_CLASS:-15}"             # 15+15 = 30
SEED=0

# 1) FETCH — assez de normal pour la banque + le 20% réservé au test.
#    train = 0.8 * CAP_NORMAL, donc CAP_NORMAL = 1.25 * ts + marge.
export CAP_NORMAL="${CAP_NORMAL:-$(( FIT_TRAIN_SUBSET * 5 / 4 + 3000 ))}"
export CAP_ANOMALY="${CAP_ANOMALY:-0}"                      # toutes les images couteau
export DEST="${COCO_PATH}"
echo "=== FETCH COCO -> ${COCO_PATH} (CAP_NORMAL=${CAP_NORMAL}) ==="
python tools/coco_fetch.py

# tag identique à build_tag() du fit
ts_lc=$(printf '%s' "${FIT_TRAIN_SUBSET}" | tr '[:upper:]' '[:lower:]')
case "${ts_lc}" in ""|none|all) TS="all";; *) TS="${FIT_TRAIN_SUBSET}";; esac
if [ "${FIT_SAMPLER}" = "identity" ]; then SAMP="identity"; else SAMP="${FIT_SAMPLER}_p${FIT_CORESET_PCT}"; fi
TAG="wideresnet50_${SAMP}_ts${TS}_s${SEED}"
BANK_DIR="${FIT_MODELS_DIR}/${TAG}"

echo "=== FIT === ts=${FIT_TRAIN_SUBSET} pct=${FIT_CORESET_PCT} proj=${FIT_CORESET_PROJ_DIM} nn=${FIT_NUM_NN}"
echo "    banque (conservée) -> ${BANK_DIR}"
python bin/coco/fit/memory_bank.py

echo "=== HISTOGRAMME good vs knife ==="
export HIST_BANK_DIR="${BANK_DIR}"
export HIST_OUTPUT_PATH="results/coco/histograms/hist_coco_ts${TS}_nn${FIT_NUM_NN}_s${SEED}.png"
python bin/coco/infer/histogram.py

echo "=== 30 HEATMAPS (${HEATMAPS_PER_CLASS} good + ${HEATMAPS_PER_CLASS} knife) ==="
python bin/coco/infer/heatmap.py --n_per_class "${HEATMAPS_PER_CLASS}"

echo "=== Terminé ==="
echo "    Banque   : ${BANK_DIR}"
echo "    Histo    : results/coco/histograms/p${FIT_CORESET_PCT}/"
echo "    Heatmaps : results/coco/heatmaps/ts${TS}_p${FIT_CORESET_PCT}/"
