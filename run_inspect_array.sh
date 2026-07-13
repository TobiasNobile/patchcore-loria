#!/usr/bin/env bash
#
# run_inspect_array.sh — Sweep d'hyperparamètres en PARALLÈLE (job array SLURM).
#
# Plutôt que d'enchaîner les fits un par un via srun sur gpu_inter (partition
# interactive plafonnée à 1 job), on soumet UN job array SLURM sur la partition
# batch `gpu_prod_long`. Chaque tâche de l'array = 1 fit PatchCore (une
# combinaison seed/ts/sampler/pct), évalué sur toutes les IMAGE_INDICES.
# SLURM répartit les tâches sur autant de GPU que CONCURRENCY (et ton quota)
# l'autorisent → les 12 fits tournent en ~le temps du plus long, pas la somme.
#
# ┌───────────────────────────────────────────────────────────────────────┐
# │  À VÉRIFIER AVANT UN GROS RUN (spécifique au DCE, non testé ici) :     │
# │   1. gpu_prod_long accepte bien le job (sinon essaie gpu_tp_resa).     │
# │   2. Chaque tâche obtient bien UN GPU. La colonne GRES du cluster est  │
# │      (null) et remote_run.sh marche sans --gres, donc on NE met PAS de │
# │      --gres ici. Si 2 tâches se marchent dessus sur le même GPU,       │
# │      ajoute `#SBATCH --gres=gpu:1` (ou --gpus-per-task=1) au sbatch.   │
# │   3. Ton quota de GPU simultanés (sacctmgr) ≥ CONCURRENCY.             │
# │  → Fais d'abord un test avec SEEDS/PERCENTAGES réduits (2-3 tâches).   │
# └───────────────────────────────────────────────────────────────────────┘
#
# Usage:
#   ./run_inspect_array.sh
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
SLURM_PARTITION="gpu_prod_long"   # batch, walltime 2 jours (cf. sinfo)
SLURM_TIME="04:00:00"             # par TÂCHE. ts=10000 est lourd : p=0.01 dépasse
                                  # déjà 1h27, les p élevés frôlent/dépassent 2h.
SLURM_MEM=""                      # Vide => PAS de #SBATCH --mem (DefMemPerNode=UNLIMITED
                                  # sur gpu_prod_long) : la tâche peut utiliser tout le
                                  # nœud (30 Go). Un cap --mem=29G tuait ts=5000 dans le
                                  # pic transitoire fit->subsample (~22Go concat + GPU).
                                  # Mets p.ex. "29G" pour re-réserver la RAM du nœud.
CONCURRENCY=4                     # nb max de tâches (GPU) simultanées — ajuste au quota

# identity garde 100% des features ET un index FAISS complet (~2x la matrice de
# features). À ts=10000 ça ferait ~45 Go > 30 Go/nœud => infaisable ici. On
# n'émet donc PAS de combinaison identity au-delà de ce train_subset.
IDENTITY_MAX_TS=2000

# ─── CONFIG sweep (produit cartésien complet des tableaux ci-dessous) ──────
SEEDS=(42)
TRAIN_SUBSETS=(2000 5000)   # ts=10000 infaisable : ~22,5Go de features > 30Go/nœud
                            # (thrashing/OOM à ~84% de "Computing support features").
BACKBONE_NAMES=(wideresnet50)
SAMPLERS=(identity approx_greedy_coreset)
PERCENTAGES=(0.01 0.1 0.2 0.5 0.7)
RESIZES=(256)
IMAGESIZES=(224)
IMAGE_INDICES=(961 405)           # toutes évaluées sur CHAQUE fit (1 seul fit)
LOG_PROJECT="CelebA_Results"

# ─── 1. Génère le manifest : 1 ligne = 1 fit ───────────────────────────────
MANIFEST="jobs_manifest.txt"
: > "${MANIFEST}"
for seed in "${SEEDS[@]}"; do
  for ts in "${TRAIN_SUBSETS[@]}"; do
    for bb in "${BACKBONE_NAMES[@]}"; do
      for rz in "${RESIZES[@]}"; do
        for im in "${IMAGESIZES[@]}"; do
          for sampler in "${SAMPLERS[@]}"; do
            # identity trop gourmand en RAM au-delà d'IDENTITY_MAX_TS (cf. plus haut)
            if [[ "${sampler}" == "identity" && "${ts}" -gt ${IDENTITY_MAX_TS} ]]; then
              continue
            fi
            # identity ignore percentage → une seule valeur bidon (pas de doublon)
            if [[ "${sampler}" == "identity" ]]; then
              pct_list=("0.1")
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
echo "📝  Manifest : ${N} fits → ${MANIFEST}"

# Arguments --image_index (constants pour tous les fits)
IMG_ARGS=""
for i in "${IMAGE_INDICES[@]}"; do IMG_ARGS+=" --image_index ${i}"; done
IMG_LIST="${IMAGE_INDICES[*]}"   # ex. "961 405" — sert au skip idempotent

# ─── 2. Écrit le script sbatch de l'array ──────────────────────────────────
# Directive mémoire optionnelle : si SLURM_MEM est vide, on n'émet aucun
# #SBATCH --mem (pas de plafond → tout le nœud dispo).
if [[ -n "${SLURM_MEM}" ]]; then
  MEM_DIRECTIVE="#SBATCH --mem=${SLURM_MEM}"
else
  MEM_DIRECTIVE="# (pas de --mem : DefMemPerNode=UNLIMITED, la tâche peut utiliser tout le nœud)"
fi

SBATCH="inspect_array.sbatch"
cat > "${SBATCH}" <<SB
#!/usr/bin/env bash
#SBATCH --job-name=pc_inspect
#SBATCH --partition=${SLURM_PARTITION}
#SBATCH --time=${SLURM_TIME}
${MEM_DIRECTIVE}
#SBATCH --output=slurm_logs/inspect_%A_%a.out
set -euo pipefail
cd "\${SLURM_SUBMIT_DIR}"
source .venv/bin/activate

# MLflow : store FICHIER (1 dossier par run) → aucune contention de verrou
# entre les tâches parallèles, contrairement au backend sqlite par défaut.
# MLFLOW_ALLOW_FILE_STORE : le file store est en "maintenance mode" sur les
# MLflow récents et lève une exception sans cet opt-out.
export MLFLOW_TRACKING_URI="file:\${SLURM_SUBMIT_DIR}/mlruns_array"
export MLFLOW_ALLOW_FILE_STORE=true

# Récupère la combinaison de cette tâche (ligne SLURM_ARRAY_TASK_ID+1 du manifest)
line=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" ${MANIFEST})
IFS=\$'\t' read -r seed ts backbone sampler pct resize imagesize <<< "\${line}"

if [[ "\${sampler}" == "identity" ]]; then tag="nopct"; else tag="p\${pct}"; fi
out="results/idx{idx}/inspect_celeba_hat_\${sampler}_\${tag}_s\${seed}_ts\${ts}_\${backbone}_idx{idx}.png"

# Idempotence : si les PNG de toutes les images existent déjà, on saute cette
# combinaison. Permet de RELANCER l'array (mêmes tableaux CONFIG) sans refaire
# les fits déjà réussis — seules les combinaisons manquantes seront recalculées.
all_exist=1
for idx in ${IMG_LIST}; do
  f="results/idx\${idx}/inspect_celeba_hat_\${sampler}_\${tag}_s\${seed}_ts\${ts}_\${backbone}_idx\${idx}.png"
  [[ -f "\${f}" ]] || all_exist=0
done
if [[ "\${all_exist}" == "1" ]]; then
  echo "[task \${SLURM_ARRAY_TASK_ID}] SKIP — déjà généré (\${sampler} \${tag} ts=\${ts})"
  exit 0
fi

echo "[task \${SLURM_ARRAY_TASK_ID}] seed=\${seed} ts=\${ts} sampler=\${sampler} pct=\${pct}"
python bin/inspect_patchcore_celeba.py "\${out}" \\
  --gpu 0 --seed "\${seed}" ${IMG_ARGS} \\
  --train_subset "\${ts}" --backbone_name "\${backbone}" \\
  --sampler_name "\${sampler}" --percentage "\${pct}" \\
  --resize "\${resize}" --imagesize "\${imagesize}" \\
  --log_project "${LOG_PROJECT}" --log_group "inspect_heatmap_\${sampler}_\${tag}"
SB
echo "📄  Script array → ${SBATCH}"

# ─── 3. Sync code + manifest + sbatch → serveur ────────────────────────────
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude 'models' --exclude 'mlruns'
          --exclude 'mlruns_array' --exclude 'results' --exclude 'mlruns.db' --exclude 'mlflow.db')
echo "📤  Envoi du code vers ${REMOTE_HOST}..."
rsync -avz "${EXCLUDES[@]}" "${LOCAL_DIR}/" "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/"

# ─── 4. Pré-crée l'expérience MLflow (évite la course à la création entre
#         tâches parallèles) puis soumet l'array ────────────────────────────
echo "🚀  Soumission de l'array 0-$((N - 1))%${CONCURRENCY} sur ${SLURM_PARTITION}..."
ssh "${REMOTE_USER}@${REMOTE_HOST}" "
  set -e
  cd ${REMOTE_DIR}
  source .venv/bin/activate
  mkdir -p slurm_logs
  MLFLOW_TRACKING_URI=\"file:\$PWD/mlruns_array\" MLFLOW_ALLOW_FILE_STORE=true \
    python -c 'import mlflow, os; mlflow.set_tracking_uri(os.environ[\"MLFLOW_TRACKING_URI\"]); mlflow.set_experiment(\"${LOG_PROJECT}\")'
  sbatch --array=0-$((N - 1))%${CONCURRENCY} ${SBATCH}
"

cat <<EOF

✅  Array soumis (${N} fits, ${CONCURRENCY} en parallèle max).

   Suivi :
     ssh ${REMOTE_USER}@${REMOTE_HOST} 'squeue -u ${REMOTE_USER}'
     ssh ${REMOTE_USER}@${REMOTE_HOST} 'tail -f ${REMOTE_DIR}/slurm_logs/inspect_*_*.out'

   Récupération une fois terminé :
     rsync -avz ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/results/       ${LOCAL_DIR}/results/
     rsync -avz ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_DIR}/mlruns_array/  ${LOCAL_DIR}/mlruns_array/

   Visualiser :  mlflow ui --backend-store-uri file://${LOCAL_DIR}/mlruns_array
EOF
