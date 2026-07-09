#!/usr/bin/env bash
#SBATCH --job-name=pc_inspect
#SBATCH --partition=gpu_prod_long
#SBATCH --time=04:00:00
# (pas de --mem : DefMemPerNode=UNLIMITED, la tâche peut utiliser tout le nœud)
#SBATCH --output=slurm_logs/inspect_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# MLflow : store FICHIER (1 dossier par run) → aucune contention de verrou
# entre les tâches parallèles, contrairement au backend sqlite par défaut.
# MLFLOW_ALLOW_FILE_STORE : le file store est en "maintenance mode" sur les
# MLflow récents et lève une exception sans cet opt-out.
export MLFLOW_TRACKING_URI="file:${SLURM_SUBMIT_DIR}/mlruns_array"
export MLFLOW_ALLOW_FILE_STORE=true

# Récupère la combinaison de cette tâche (ligne SLURM_ARRAY_TASK_ID+1 du manifest)
line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs_manifest.txt)
IFS=$'\t' read -r seed ts backbone sampler pct resize imagesize <<< "${line}"

if [[ "${sampler}" == "identity" ]]; then tag="nopct"; else tag="p${pct}"; fi
out="results/idx{idx}/inspect_celeba_hat_${sampler}_${tag}_s${seed}_ts${ts}_${backbone}_idx{idx}.png"

# Idempotence : si les PNG de toutes les images existent déjà, on saute cette
# combinaison. Permet de RELANCER l'array (mêmes tableaux CONFIG) sans refaire
# les fits déjà réussis — seules les combinaisons manquantes seront recalculées.
all_exist=1
for idx in 961 405; do
  f="results/idx${idx}/inspect_celeba_hat_${sampler}_${tag}_s${seed}_ts${ts}_${backbone}_idx${idx}.png"
  [[ -f "${f}" ]] || all_exist=0
done
if [[ "${all_exist}" == "1" ]]; then
  echo "[task ${SLURM_ARRAY_TASK_ID}] SKIP — déjà généré (${sampler} ${tag} ts=${ts})"
  exit 0
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${seed} ts=${ts} sampler=${sampler} pct=${pct}"
python bin/inspect_patchcore_celeba.py "${out}" \
  --gpu 0 --seed "${seed}"  --image_index 961 --image_index 405 \
  --train_subset "${ts}" --backbone_name "${backbone}" \
  --sampler_name "${sampler}" --percentage "${pct}" \
  --resize "${resize}" --imagesize "${imagesize}" \
  --log_project "CelebA_Results" --log_group "inspect_heatmap_${sampler}_${tag}"
