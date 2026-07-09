#!/usr/bin/env bash
#SBATCH --job-name=pc_hist
#SBATCH --partition=gpu_prod_long
#SBATCH --time=02:00:00
# (pas de --mem : la tâche peut utiliser tout le nœud)
#SBATCH --output=slurm_logs/hist_%A_%a.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export MLFLOW_TRACKING_URI="file:${SLURM_SUBMIT_DIR}/mlruns_array"
export MLFLOW_ALLOW_FILE_STORE=true

line=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" jobs_histogram.txt)
IFS=$'\t' read -r seed ts backbone sampler pct resize imagesize <<< "${line}"

if [[ "${sampler}" == "identity" ]]; then tag="nopct"; else tag="p${pct}"; fi
out="results/histograms/hist_celeba_${sampler}_${tag}_s${seed}_ts${ts}_${backbone}.png"

# Idempotence : ne refait pas une figure déjà présente.
if [[ -f "${out}" ]]; then
  echo "[task ${SLURM_ARRAY_TASK_ID}] SKIP — déjà généré (${out})"
  exit 0
fi

echo "[task ${SLURM_ARRAY_TASK_ID}] seed=${seed} ts=${ts} sampler=${sampler} pct=${pct}"
python bin/score_histogram_celeba.py "${out}" \
  --gpu 0 --seed "${seed}" \
  --train_subset "${ts}" --n_per_class 1000 \
  --backbone_name "${backbone}" \
  --sampler_name "${sampler}" --percentage "${pct}" \
  --resize "${resize}" --imagesize "${imagesize}" \
  --log_project "CelebA_Results" --log_group "score_histogram_${sampler}_${tag}"
