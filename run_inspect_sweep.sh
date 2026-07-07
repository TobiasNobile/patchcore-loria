#!/usr/bin/env bash
#
# run_inspect_sweep.sh — Lance bin/inspect_patchcore_celeba.py (via remote_run.sh)
# pour CHAQUE combinaison d'hyperparamètres définie dans la section CONFIG
# ci-dessous (produit cartésien complet).
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │  POUR MODIFIER LE SWEEP : édite uniquement les tableaux de la        │
# │  section « CONFIG » plus bas. Chaque hyperparamètre est un tableau ; │
# │  le script exécute toutes les combinaisons possibles.               │
# │                                                                     │
# │  Ex. : SEEDS=(0 1 2)  SAMPLERS=(identity greedy_coreset)            │
# │  PERCENTAGES=(0.1 0.5)  → 3 × 2 × 2 combinaisons (moins les          │
# │  doublons identity, voir la note plus bas).                        │
# └─────────────────────────────────────────────────────────────────────┘
#
# Notes utiles sur les samplers (cf. src/patchcore/sampler.py) :
#   - identity : percentage est ignoré (IdentitySampler renvoie les features
#     telles quelles). Le script ne lance donc QU'UN SEUL run par combinaison,
#     quel que soit le nombre de percentages listés — pas de doublon.
#   - approx_greedy_coreset : 1 run par percentage.
#   - greedy_coreset (exact) : ATTENTION, calcule une matrice de distances
#     N×N complète. Avec train_subset=2000 (~1.57M features) ça fait ~9.8 To,
#     inatteignable sur CPU/GPU. À n'utiliser qu'avec un train_subset TRÈS
#     petit. Il n'est pas exclu automatiquement : à toi de gérer.
#
# Le tracking MLflow est géré côté script Python
# (patchcore.tracking.patchcore_run) : params, anomaly_score et l'image
# overlay sont loggés automatiquement.
#
# Usage:
#   ./run_inspect_sweep.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

RESULTS_DIR="results"
mkdir -p "${RESULTS_DIR}"

# Taille du split TEST balancé (839 hat + 839 no-hat, cf.
# bin/inspect_patchcore_celeba.py) : borne pour l'image_index aléatoire.
TEST_SPLIT_SIZE=1678

# ======================================================================
# CONFIG — édite ces tableaux pour définir les combinaisons à balayer.
# ======================================================================

SEEDS=(0 1 2 3 4 42 104)
TRAIN_SUBSETS=(2000 5000 10000 20000 30000 50000 100000)
BACKBONE_NAMES=(wideresnet50)
SAMPLERS=(identity approx_greedy_coreset)
PERCENTAGES=(0.01 0.02 0.1 0.2 0.3 0.5 0.7)
RESIZES=(256)
IMAGESIZES=(224)

# image_index :
#   - tableau vide ()  → un index ALÉATOIRE tiré à chaque run
#   - liste d'entiers  → chaque index est balayé (ajoute une dimension au sweep)
IMAGE_INDICES=()

LOG_PROJECT="CelebA_Results"

# ======================================================================
# Fin de la CONFIG — normalement rien à modifier en dessous.
# ======================================================================

run_inspect() {
  local seed="$1" image_index="$2" train_subset="$3" backbone="$4" \
        sampler="$5" pct="$6" resize="$7" imagesize="$8"

  local tag
  if [[ "${sampler}" == "identity" ]]; then
    tag="nopct"
  else
    tag="p${pct}"
  fi

  local output_path="${RESULTS_DIR}/inspect_celeba_hat_${sampler}_${tag}_s${seed}_ts${train_subset}_${backbone}_idx${image_index}.png"
  local log_group="inspect_heatmap_${sampler}_${tag}"

  echo "=================================================================="
  echo "▶ seed=${seed} sampler=${sampler} percentage=${pct} train_subset=${train_subset}"
  echo "  backbone=${backbone} resize=${resize} imagesize=${imagesize} image_index=${image_index}"
  echo "=================================================================="

  ./remote_run.sh bin/inspect_patchcore_celeba.py "${output_path}" \
    --gpu 0 \
    --seed "${seed}" \
    --image_index "${image_index}" \
    --train_subset "${train_subset}" \
    --backbone_name "${backbone}" \
    --sampler_name "${sampler}" \
    --percentage "${pct}" \
    --resize "${resize}" \
    --imagesize "${imagesize}" \
    --log_project "${LOG_PROJECT}" \
    --log_group "${log_group}"
}

# Si aucun image_index explicite n'est fourni, on utilise le sentinel "random"
# pour tirer un index aléatoire à CHAQUE run.
index_list=("${IMAGE_INDICES[@]:-}")
if [[ ${#IMAGE_INDICES[@]} -eq 0 ]]; then
  index_list=("random")
fi

n_runs=0
for seed in "${SEEDS[@]}"; do
  for train_subset in "${TRAIN_SUBSETS[@]}"; do
    for backbone in "${BACKBONE_NAMES[@]}"; do
      for resize in "${RESIZES[@]}"; do
        for imagesize in "${IMAGESIZES[@]}"; do
          for sampler in "${SAMPLERS[@]}"; do
            # identity ignore percentage → on ne balaie qu'une valeur bidon
            # pour éviter les runs identiques dupliqués.
            if [[ "${sampler}" == "identity" ]]; then
              pct_list=("0.1")
            else
              pct_list=("${PERCENTAGES[@]}")
            fi
            for pct in "${pct_list[@]}"; do
              for idx in "${index_list[@]}"; do
                if [[ "${idx}" == "random" ]]; then
                  image_index=$(( RANDOM % TEST_SPLIT_SIZE ))
                else
                  image_index="${idx}"
                fi
                run_inspect "${seed}" "${image_index}" "${train_subset}" \
                  "${backbone}" "${sampler}" "${pct}" "${resize}" "${imagesize}"
                n_runs=$(( n_runs + 1 ))
              done
            done
          done
        done
      done
    done
  done
done

echo "✅  Sweep terminé (${n_runs} runs). Images dans ${RESULTS_DIR}/, runs loggés dans MLflow (projet ${LOG_PROJECT})."
