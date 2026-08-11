#!/usr/bin/env bash
#
# fit_and_score.sh (CelebA) — fit + histogramme + heatmaps en UN job OAR.
# FIT_LAYERS : layer1..layer4 (pas de layer5), défaut layer2+layer3.
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

# Le tag vient de build_tag() : une seule source de vérité avec le fit.
read -r TAG SUFFIX <<<"$(python -c "
from experiments.pipelines import build_tag, fit_settings
cfg = fit_settings('${FIT_MODELS_DIR}', 0.1, 2000)
layers = cfg['layers_to_extract_from']
suffix = '' if layers == ['layer2', 'layer3'] else '_' + '-'.join(
    l.replace('layer', 'l') for l in layers)
print(build_tag(cfg), suffix)
")"
BANK_DIR="${FIT_MODELS_DIR}/${TAG}"

echo "=== FIT === ts=${FIT_TRAIN_SUBSET} pct=${FIT_CORESET_PCT} proj=${FIT_CORESET_PROJ_DIM} nn=${FIT_NUM_NN} layers=${FIT_LAYERS:-layer2,layer3}"
echo "    banque (conservée) -> ${BANK_DIR}"
python bin/celeba/fit/memory_bank.py

echo "=== HISTOGRAMME no-hat vs hat + 30 HEATMAPS ==="
export HIST_BANK_DIR="${BANK_DIR}"
export HIST_OUTPUT_PATH="results/celeba/histograms/hist_celeba${SUFFIX}_ts${FIT_TRAIN_SUBSET}_nn${FIT_NUM_NN}_s${SEED}.png"
export HEATMAP_OUTPUT_PATH="results/celeba/heatmaps${SUFFIX}/overlay_idx{idx}.png"
python bin/celeba/infer/histogram.py --n_per_class "${NPC}"
python bin/celeba/infer/heatmap.py --n_per_class "${HEATMAPS_PER_CLASS}"

echo "=== Terminé ==="
echo "    Banque   : ${BANK_DIR}"
echo "    Histo    : results/celeba/histograms/p${FIT_CORESET_PCT}/"
echo "    Heatmaps : results/celeba/heatmaps${SUFFIX}/ts${FIT_TRAIN_SUBSET}_p${FIT_CORESET_PCT}/"
