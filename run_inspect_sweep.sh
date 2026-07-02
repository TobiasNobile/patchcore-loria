#!/usr/bin/env bash
#
# run_inspect_sweep.sh — Lance bin/inspect_patchcore_celeba.py (via remote_run.sh)
# pour chaque combinaison de sampler / pourcentage de coreset, avec un
# image_index tiré aléatoirement à chaque run.
#
# Combinaisons couvertes :
#   - identity : 1 seul run (percentage n'a aucun effet sur ce sampler,
#     IdentitySampler.run() renvoie les features telles quelles sans les
#     lire — cf. src/patchcore/sampler.py)
#   - approx_greedy_coreset : 1 run par percentage ∈ {0.01, 0.1, 0.2, 0.5}
#   soit 1 + 4 = 5 runs.
#
# greedy_coreset (exact) est volontairement exclu : avec train_subset=2000
# (~784 patches/image en layer2 → N≈1.57M features), il calcule une matrice
# de distances N×N complète, soit ~9.8 To — inatteignable sur CPU comme GPU,
# quel que soit le percentage demandé (celui-ci ne réduit que le nombre de
# points choisis à la fin, pas la taille de la matrice). Seule la variante
# approximative (matrice N × num_starting_points) passe à cette échelle —
# cf. src/patchcore/sampler.py.
#
# Le tracking MLflow est déjà géré côté script Python
# (patchcore.tracking.patchcore_run dans bin/inspect_patchcore_celeba.py) :
# params, anomaly_score et l'image overlay sont loggés automatiquement.
# Ce script se contente de faire varier les paramètres et de nommer les
# images de sortie dans results/.
#
# Usage:
#   ./run_inspect_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"

# Taille du split TEST balancé (839 hat + 839 no-hat, cf.
# bin/inspect_patchcore_celeba.py) : bornes pour l'image_index aléatoire.
TEST_SPLIT_SIZE=1678

SEED=0
TRAIN_SUBSET=2000
BACKBONE_NAME="wideresnet50"
RESIZE=256
IMAGESIZE=224
LOG_PROJECT="CelebA_Results"

CORESET_SAMPLERS=(approx_greedy_coreset)
PERCENTAGES=(0.01 0.1 0.2 0.5)

run_inspect() {
  local sampler="$1" pct="$2" tag="$3"
  local image_index=$(( RANDOM % TEST_SPLIT_SIZE ))
  local output_path="${RESULTS_DIR}/inspect_celeba_hat_${sampler}_${tag}_idx${image_index}.png"
  local log_group="inspect_heatmap_${sampler}_${tag}"

  echo "=================================================================="
  echo "▶ sampler=${sampler} percentage=${pct} image_index=${image_index}"
  echo "=================================================================="

  ./remote_run.sh bin/inspect_patchcore_celeba.py "${output_path}" \
    --gpu 0 \
    --seed "${SEED}" \
    --image_index "${image_index}" \
    --train_subset "${TRAIN_SUBSET}" \
    --backbone_name "${BACKBONE_NAME}" \
    --sampler_name "${sampler}" \
    --percentage "${pct}" \
    --resize "${RESIZE}" \
    --imagesize "${IMAGESIZE}" \
    --log_project "${LOG_PROJECT}" \
    --log_group "${log_group}"
}

# identity : percentage n'a pas de sens ici (ignoré par IdentitySampler),
# un seul run suffit. On passe le défaut du CLI (0.1) juste pour le log MLflow.
run_inspect identity 0.1 "nopct"

for sampler in "${CORESET_SAMPLERS[@]}"; do
  for pct in "${PERCENTAGES[@]}"; do
    run_inspect "${sampler}" "${pct}" "p${pct}"
  done
done

echo "✅  Sweep terminé. Images dans ${RESULTS_DIR}/, runs loggés dans MLflow (projet ${LOG_PROJECT})."
