#!/usr/bin/env bash
#
# grid5000_run.sh — Équivalent Grid'5000 de remote_run.sh : sync le code local
# vers la frontale, réserve un nœud GPU, exécute un script Python dessus en
# affichant la sortie en direct, puis rapatrie les résultats.
#
# Usage:
#   ./grid5000_run.sh                                       # lance le script par défaut
#   ./grid5000_run.sh bin/fit_memory_bank_celeba.py         # construit la banque mémoire
#   ./grid5000_run.sh bin/infer_heatmap_celeba.py --image_index 900
#   ./grid5000_run.sh bin/score_histogram_celeba.py --n_per_class 200
#
# Prérequis, une seule fois sur la frontale :
#   ssh <login>@access.grid5000.fr
#   ssh nancy
#   cd ~/patchcore-inspection && uv sync
#   uv pip uninstall faiss-cpu && uv pip install faiss-gpu-cu12   # sinon la
#   recherche des plus proches voisins reste sur CPU malgré le GPU réservé.
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
# Login Grid'5000 (différent du login DCE). Surchargeable : G5K_USER=xxx ./grid5000_run.sh
G5K_USER="${G5K_USER:-${USER}}"

# La passerelle SSH publique de Grid'5000. La frontale du site n'est joignable
# qu'à travers elle, d'où le ProxyJump ci-dessous.
G5K_GATEWAY="access.grid5000.fr"
G5K_SITE="nancy"

# À Nancy, la plupart des clusters GPU sont dans la queue `production`.
OAR_QUEUE="production"
OAR_GPU=1
OAR_WALLTIME="02:00:00"

# Filtre optionnel sur les ressources, ex. "gpu_model = 'A100-PCIE-40GB'" pour
# épingler un modèle de GPU, ou "cluster = 'grele'". Vide = n'importe lequel.
OAR_PROPERTIES=""

REMOTE_DIR="patchcore-inspection"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Les banques mémoire pèsent lourd (~4 Ko par vecteur : 640 Mo pour une banque
# à 10% sur 2000 images, 64 Mo à 1%). Elles restent donc côté serveur par
# défaut — l'inférence tourne là-bas de toute façon. Passe à true pour les
# rapatrier et faire des heatmaps en local.
FETCH_BANKS=false

# Script à exécuter côté serveur (par défaut, ou 1er argument), le reste des
# arguments est transmis tel quel au script Python.
SCRIPT="${1:-bin/infer_heatmap_celeba.py}"
if [[ $# -gt 0 ]]; then
  shift
fi
SCRIPT_ARGS=("$@")

# Mêmes exclusions que remote_run.sh : gros fichiers, ou spécifiques à la
# machine locale. `models` est exclu dans les deux sens : les banques se
# construisent et se consomment sur le serveur.
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude 'models' --exclude 'mlruns' --exclude 'results' --exclude 'mlruns.db' --exclude 'mlflow.db')

SSH_FRONTEND=(ssh -J "${G5K_USER}@${G5K_GATEWAY}" "${G5K_USER}@${G5K_SITE}")
RSYNC_SSH="ssh -J ${G5K_USER}@${G5K_GATEWAY}"

# ─── 1. Sync : local → frontale ───────────────────────────────────────────
echo "📤  Envoi du code vers ${G5K_SITE} (via ${G5K_GATEWAY})..."
rsync -avz -e "${RSYNC_SSH}" "${EXCLUDES[@]}" \
  "${LOCAL_DIR}/" \
  "${G5K_USER}@${G5K_SITE}:${REMOTE_DIR}/"

# ─── 2. Réservation d'un nœud GPU + exécution ─────────────────────────────
QUOTED_ARGS=""
for arg in "${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}"; do
  QUOTED_ARGS+=" $(printf '%q' "$arg")"
done

PROPERTY_FLAG=""
if [[ -n "${OAR_PROPERTIES}" ]]; then
  PROPERTY_FLAG="-p \"${OAR_PROPERTIES}\""
fi

echo "🚀  Réservation d'un nœud GPU (queue ${OAR_QUEUE}, gpu=${OAR_GPU}, walltime=${OAR_WALLTIME})..."
echo "    puis exécution de ${SCRIPT}${QUOTED_ARGS}"

# OAR n'a pas d'équivalent bloquant de `srun` : `oarsub -I` ouvre un shell
# interactif et n'accepte pas de commande. On soumet donc un job passif, puis on
# suit sa sortie jusqu'à ce qu'il quitte les états Waiting/Launching/Running —
# ce qui donne le même confort qu'un run bloquant, sortie en direct comprise.
# shellcheck disable=SC2029  # on veut bien l'expansion locale des variables
"${SSH_FRONTEND[@]}" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

JOB_OUT="oar_job.out"
JOB_ID=\$(oarsub -q "${OAR_QUEUE}" \
  -l "gpu=${OAR_GPU},walltime=${OAR_WALLTIME}" ${PROPERTY_FLAG} \
  --stdout="\${JOB_OUT}" --stderr="\${JOB_OUT}" \
  "bash -c 'cd ~/${REMOTE_DIR} && source .venv/bin/activate && python ${SCRIPT}${QUOTED_ARGS}'" \
  | sed -n 's/^OAR_JOB_ID=//p')

if [[ -z "\${JOB_ID}" ]]; then
  echo "❌  oarsub n'a pas rendu de job id (réservation refusée ?)." >&2
  exit 1
fi
echo "🎟️   Job OAR \${JOB_ID} soumis. En attente d'un nœud..."

: > "\${JOB_OUT}"
tail -f "\${JOB_OUT}" &
TAIL_PID=\$!
trap 'kill \${TAIL_PID} 2>/dev/null || true' EXIT

# Tant que le job est dans la file ou tourne, on attend. Les états OAR sont
# Waiting / Launching / Running / Terminated / Error.
while oarstat -s -j "\${JOB_ID}" | grep -qE 'Waiting|Launching|Running|toLaunch|Hold'; do
  sleep 10
done

sleep 2                       # laisse tail -f rattraper les dernières lignes
kill \${TAIL_PID} 2>/dev/null || true

STATE=\$(oarstat -s -j "\${JOB_ID}" | cut -d: -f2 | tr -d ' ')
echo "🏁  Job \${JOB_ID} terminé (état : \${STATE})."
[[ "\${STATE}" == "Error" ]] && exit 1
exit 0
REMOTE_SCRIPT

# ─── 3. Sync : frontale → local (rapatrie les outputs) ────────────────────
echo "📥  Récupération des résultats..."
rsync -avz -e "${RSYNC_SSH}" \
  "${G5K_USER}@${G5K_SITE}:${REMOTE_DIR}/results/" \
  "${LOCAL_DIR}/results/" 2>/dev/null || echo "  (pas de dossier results/ à rapatrier)"

if [[ "${FETCH_BANKS}" == "true" ]]; then
  echo "📥  Récupération des banques mémoire..."
  rsync -avz -e "${RSYNC_SSH}" \
    "${G5K_USER}@${G5K_SITE}:${REMOTE_DIR}/models/celeba/" \
    "${LOCAL_DIR}/models/celeba/" 2>/dev/null || echo "  (pas de banque à rapatrier)"
fi

echo "✅  Terminé."
