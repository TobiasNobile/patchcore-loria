#!/usr/bin/env bash
#
# fit_and_score.sh (CelebA) — Fit + histogramme + 30 heatmaps (15 no-hat + 15 hat)
# en UN job OAR. Banque persistée dans le home (models/celeba/<tag>).
#
# Balayage de couches : FIT_LAYERS="layer2" | "layer3" | "layer4" (WideResNet50 =
# ResNet -> layer1..layer4, PAS de layer5). Défaut = layer2+layer3 (suffixe vide).
#
#   REMOTE_ENV="FIT_TRAIN_SUBSET=20000 FIT_LAYERS=layer4" \
#   OAR_RESOURCES='host=1' OAR_WALLTIME=06:00:00 OAR_PROPERTIES="cluster='gres'" \
#   DETACH=true ./grid5000_run.sh bin/celeba/fit_and_score.sh
set -euo pipefail

export FIT_SAMPLER="${FIT_SAMPLER:-approx_greedy_coreset}"
export FIT_CORESET_PCT="${FIT_CORESET_PCT:-0.05}"
export FIT_CORESET_PROJ_DIM="${FIT_CORESET_PROJ_DIM:-64}"
export FIT_NUM_NN="${FIT_NUM_NN:-3}"
export FIT_TRAIN_SUBSET="${FIT_TRAIN_SUBSET:-20000}"
export FIT_MODELS_DIR="${FIT_MODELS_DIR:-models/celeba}"   # home = banque conservée
# La banque (~qq Go à 5%) tient en VRAM -> FAISS GPU au scoring.
export HIST_FAISS_GPU="${HIST_FAISS_GPU:-1}"
export INFER_FAISS_GPU="${INFER_FAISS_GPU:-1}"
HEATMAPS_PER_CLASS="${HEATMAPS_PER_CLASS:-15}"            # 15 + 15 = 30
NPC="${NPC:-1000}"                                         # n_per_class histogramme
SEED=0

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

echo "=== FIT === ts=${FIT_TRAIN_SUBSET} pct=${FIT_CORESET_PCT} proj=${FIT_CORESET_PROJ_DIM} nn=${FIT_NUM_NN} layers=${FIT_LAYERS:-layer2,layer3}"
echo "    banque (conservée) -> ${BANK_DIR}"
python bin/celeba/fit/memory_bank.py

echo "=== HISTOGRAMME no-hat vs hat + 30 HEATMAPS ==="
export HIST_BANK_DIR="${BANK_DIR}"
export HIST_OUTPUT_PATH="results/celeba/histograms/hist_celeba${LSUF}_ts${TS}_nn${FIT_NUM_NN}_s${SEED}.png"
export HEATMAP_OUTPUT_PATH="results/celeba/heatmaps${LSUF}/overlay_idx{idx}.png"
python bin/celeba/infer/histogram.py --n_per_class "${NPC}"
python bin/celeba/infer/heatmap.py --n_per_class "${HEATMAPS_PER_CLASS}"

echo "=== Terminé ==="
echo "    Banque   : ${BANK_DIR}"
echo "    Histo    : results/celeba/histograms/p${FIT_CORESET_PCT}/"
echo "    Heatmaps : results/celeba/heatmaps${LSUF}/ts${TS}_p${FIT_CORESET_PCT}/"
