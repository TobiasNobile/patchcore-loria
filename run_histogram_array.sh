#!/usr/bin/env bash
#
# run_histogram_array.sh — Génère, en PARALLÈLE via un job array SLURM, un
# histogramme des scores d'anomalie (train held-out vs test) par combinaison
# d'hyperparamètres. Une FIGURE par combo (pas par image ; 405 et 961 sont
# marquées d'une ligne verticale sur chaque figure).
#
# Reprend exactement les hyperparamètres des heatmaps déjà produites :
# seed=42, ts=2000, identity + approx_greedy_coreset p∈{0.01,0.1,0.2,0.5,0.7}.
#
# Script exécuté : bin/score_histogram_celeba.py
#
# Usage:
#   ./run_histogram_array.sh
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# ─── Remote (identique à remote_run.sh) ────────────────────────────────────
REMOTE_USER="nobile_tob"
REMOTE_HOST="dce.metz.centralesupelec.fr"
REMOTE_DIR="~/patchcore-inspection"
LOCAL_DIR="$HOME/dev/telecom/stage_1a/patchcore-inspection"

# ─── SLURM ─────────────────────────────────────────────────────────────────
SLURM_PARTITION="gpu_prod_long"
SLURM_TIME="02:00:00"     # ts=2000 est léger, mais coreset p0.7 ~20min + scoring
SLURM_MEM=""              # vide => pas de --mem (ts=2000 tient large dans 30 Go)
CONCURRENCY=4

# ─── CONFIG (mêmes hyperparamètres que les heatmaps déjà produites) ─────────
SEEDS=(42)
TRAIN_SUBSETS=(2000)
BACKBONE_NAMES=(wideresnet50)
SAMPLERS=(identity approx_greedy_coreset)
PERCENTAGES=(0.01 0.1 0.2 0.5 0.7)
RESIZES=(256)
IMAGESIZES=(224)
MARK_INDICES=(405 961)    # images repérées sur chaque histogramme
N_TRAIN_EVAL=1000         # images de train held-out scorées (distribution bleue)
LOG_PROJECT="CelebA_Results"

# ─── 1. Manifest : 1 ligne = 1 figure (combo) ──────────────────────────────
MANIFEST="jobs_histogram.txt"
: > "${MANIFEST}"
for seed in "${SEEDS[@]}"; do
  for ts in "${TRAIN_SUBSETS[@]}"; do
    for bb in "${BACKBONE_NAMES[@]}"; do
      for rz in "${RESIZES[@]}"; do
        for im in "${IMAGESIZES[@]}"; do
          for sampler in "${SAMPLERS[@]}"; do
            if [[ "${sampler}" == "identity" ]]; then
              pct_list=("0.1")   # identity ignore percentage
            else
              pct_list=("${PERCENTAGES[@]}")
            fi
            for pct in "${pct_list[@]}"; do
              printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
                "${seed}" "${ts}" "${bb}" "${sampler}" "${pct}" "${rz}" "${im}" >> "${MANIFEST}"
            done
          done
        done
      done
    done
  done
done
N=$(wc -l < "${MANIFEST}" | tr -d ' ')
echo "📝  Manifest : ${N} figures → ${MANIFEST}"

# Arguments --mark_index (constants)
MARK_ARGS=""
for i in "${MARK_INDICES[@]}"; do MARK_ARGS+=" --mark_index ${i}"; done

# Directive mémoire optionnelle
if [[ -n "${SLURM_MEM}" ]]; then
  MEM_DIRECTIVE="#SBATCH --mem=${SLURM_MEM}"
else
  MEM_DIRECTIVE="# (pas de --mem : la tâche peut utiliser tout le nœud)"
fi

# ─── 2. Script sbatch ──────────────────────────────────────────────────────
SBATCH="histogram_array.sbatch"
cat > "${SBATCH}" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=pc_hist
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
${MEM_DIRECTIVE}
#SBATCH --output=slurm_logs/hist_%A_%a.out
set -euo pipefail
cd "\${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

export MLFLOW_TRACKING_URI="file:\${SLURM_SUBMIT_DIR}/mlruns_array"
export MLFLOW_ALLOW_FILE_STORE=true

line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" ${MANIFEST})
IFS=\$'\t' read -r seed ts backbone sampler pct resize imagesize <<< "\${line}"

if [[ "\${sampler}" == "identity" ]]; then tag="nopct"; else tag="p\${pct}"; fi
out="results/histograms/hist_celeba_\${sampler}_\${tag}_s\${seed}_ts\${ts}_\${backbone}.png"

# Idempotence : ne refait pas une figure déjà présente.
if [[ -f "\${out}" ]]; then
  echo "[task \${SLURM_ARRAY_TASK_ID}] SKIP — déjà généré (\${out})"
  exit 0
fi

echo "[task \${SLURM_ARRAY_TASK_ID}] seed=\${seed} ts=\${ts} sampler=\${sampler} pct=\${pct}"
python bin/score_histogram_celeba.py "\${out}" \\
  --gpu 0 --seed "\${seed}" \\
  --train_subset "\${ts}" --n_train_eval ${N_TRAIN_EVAL} ${MARK_ARGS} \\
  --backbone_name "\${backbone}" \\
  --sampler_name "\${sampler}" --percentage "\${pct}" \\
  --resize "\${resize}" --imagesize "\${imagesize}" \\
  --log_project "${LOG_PROJECT}" --log_group "score_histogram_\${sampler}_\${tag}"
SB
echo "📄  Script array → ${SBATCH}"

# ─── 3. Sync → serveur ─────────────────────────────────────────────────────
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude 'models' --exclude 'mlruns'
          --exclude 'mlruns_array' --exclude 'results' --exclude 'mlruns.db' --exclude 'mlflow.db')
echo "📤  Envoi du code vers ${REMOTE_HOST}..."
rsync -avz "${EXCLUDES[@]}" "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# ─── 4. Pré-crée l'expérience MLflow puis soumet ──────────────────────────
echo "🚀  Soumission de l'array 0-$((N - 1))%${CONCURRENCY} sur ${SLURM_PARTITION}..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "
  set -e
  cd ${REMOTE_DIR}
  source .venv/bin/activate
  mkdir -p slurm_logs results/histograms
  MLFLOW_TRACKING_URI=\"file:\$PWD/mlruns_array\" MLFLOW_ALLOW_FILE_STORE=true \
    python -c 'import mlflow, os; mlflow.set_tracking_uri(os.environ[\"MLFLOW_TRACKING_URI\"]); mlflow.set_experiment(\"${LOG_PROJECT}\")'
  sbatch --array=0-$((N - 1))%${CONCURRENCY} ${SBATCH}
"

cat <<EOF

✅  Array histogrammes soumis (${N} figures, ${CONCURRENCY} en parallèle max).

   Suivi :
     ssh ${REMOTE_USER}@${REMOTE_HOST} 'squeue -u ${REMOTE_USER}'
     ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ${REMOTE_DIR}/slurm_logs/hist_*_*.out'

   Récupération une fois terminé :
     rsync -avz ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results/histograms/  ${LOCAL_DIR}/results/histograms/
     rsync -avz ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/mlruns_array/         ${LOCAL_DIR}/mlruns_array/
EOF
