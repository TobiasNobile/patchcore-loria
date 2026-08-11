#!/usr/bin/env bash
#
# fit_and_score.sh — COCO « personne + couteau » en UN job OAR : fetch sur le
# /tmp du nœud, fit de la banque (personne SANS couteau), histogramme, heatmaps.
# Seuls la banque et results/coco/ survivent au job.
#
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
# La banque (~qq Go à 5%) tient en VRAM -> FAISS GPU au scoring, sinon la
# recherche CPU sur 1,5 M+ vecteurs traîne et dépasse le walltime.
export HIST_FAISS_GPU="${HIST_FAISS_GPU:-1}"
export INFER_FAISS_GPU="${INFER_FAISS_GPU:-1}"
HEATMAPS_PER_CLASS="${HEATMAPS_PER_CLASS:-15}"             # 15+15 = 30
NPC="${NPC:-1000}"                                          # n_per_class histogramme
SEED=0

# 1) FETCH — assez de normal pour la banque + le 20% réservé au test.
#    train = 0.8 * CAP_NORMAL, donc CAP_NORMAL = 1.25 * ts + marge.
export CAP_NORMAL="${CAP_NORMAL:-$(( FIT_TRAIN_SUBSET * 5 / 4 + 3000 ))}"
export CAP_ANOMALY="${CAP_ANOMALY:-0}"                      # toutes les images couteau
export DEST="${COCO_PATH}"
echo "=== FETCH COCO -> ${COCO_PATH} (CAP_NORMAL=${CAP_NORMAL}) ==="
python tools/coco_fetch.py

# tag identique à build_tag() du fit (suffixe layer vide pour le défaut layer2+3).
ts_lc=$(printf '%s' "${FIT_TRAIN_SUBSET}" | tr '[:upper:]' '[:lower:]')
case "${ts_lc}" in ""|none|all) TS="all";; *) TS="${FIT_TRAIN_SUBSET}";; esac
if [ "${FIT_SAMPLER}" = "identity" ]; then SAMP="identity"; else SAMP="${FIT_SAMPLER}_p${FIT_CORESET_PCT}"; fi
case "${FIT_LAYERS:-}" in
  ""|"layer2,layer3") LSUF="";;
  *) LSUF="_$(printf '%s' "${FIT_LAYERS}" | sed 's/layer/l/g; s/,/-/g')";;
esac
# Doit rester aligné sur build_tag() de bin/coco/fit/memory_bank.py.
BB="${FIT_BACKBONE:-wideresnet50}"
case "${FIT_IMAGESIZE:-224}" in 224) IMSUF="";; *) IMSUF="_im${FIT_IMAGESIZE}";; esac
TAG="${BB}${LSUF}${IMSUF}_${SAMP}_ts${TS}_s${SEED}"
BANK_DIR="${FIT_MODELS_DIR}/${TAG}"

echo "=== FIT === ts=${FIT_TRAIN_SUBSET} pct=${FIT_CORESET_PCT} proj=${FIT_CORESET_PROJ_DIM} nn=${FIT_NUM_NN} layers=${FIT_LAYERS:-layer2,layer3} backbone=${BB} imagesize=${FIT_IMAGESIZE:-224}"
echo "    banque (conservée) -> ${BANK_DIR}"
python bin/coco/fit/memory_bank.py

echo "=== HISTOGRAMME good vs knife ==="
export HIST_BANK_DIR="${BANK_DIR}"
export HIST_OUTPUT_PATH="results/coco/histograms/hist_coco_${BB}${LSUF}${IMSUF}_ts${TS}_nn${FIT_NUM_NN}_s${SEED}.png"
export HEATMAP_OUTPUT_PATH="results/coco/heatmaps${LSUF}/overlay_idx{idx}.png"
python bin/coco/infer/histogram.py --n_per_class "${NPC}"

echo "=== 30 HEATMAPS (${HEATMAPS_PER_CLASS} good + ${HEATMAPS_PER_CLASS} knife) ==="
python bin/coco/infer/heatmap.py --n_per_class "${HEATMAPS_PER_CLASS}"

echo "=== Terminé ==="
echo "    Banque   : ${BANK_DIR}"
echo "    Histo    : results/coco/histograms/p${FIT_CORESET_PCT}/"
echo "    Heatmaps : results/coco/heatmaps/ts${TS}_p${FIT_CORESET_PCT}/"
