#!/usr/bin/env bash
#
# grid5000_run.sh — sync le code vers la frontale Grid'5000, réserve un nœud GPU,
# y exécute un script, rapatrie les résultats. Prérequis : l'alias `<site>.g5k`
# dans ~/.ssh/config (config officielle G5K) et `uv sync` sur la frontale.
#
# Usage:
#   ./grid5000_run.sh bin/celeba/fit/memory_bank.py
#   ./grid5000_run.sh bin/celeba/infer/histogram.py --n_per_class 200
#   DETACH=true ./grid5000_run.sh ...   # soumet et rend la main
#   ./grid5000_run.sh --fetch           # rapatrie un job détaché terminé
#
# Env : OAR_RESOURCES, OAR_WALLTIME, OAR_PROPERTIES (ex. cluster='gres'),
#       REMOTE_ENV (variables passées au script distant),
#       EXTRA_EXCLUDES (chemins à ne pas envoyer, séparés par des espaces).
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
# Le login est dans ~/.ssh/config, qui définit l'alias `<site>.g5k`.
G5K_SITE="nancy"

# Queue des clusters accessibles au groupe 'orpailleur' à Nancy (cf. le message
# d'accueil de la frontale). Ce n'est PAS `production` : Grid'5000 a renommé.
OAR_QUEUE="abaca"
OAR_GPU=1
OAR_WALLTIME="${OAR_WALLTIME:-03:00:00}"

# OAR_RESOURCES="host=1" réserve le nœud entier (toute la RAM), nécessaire
# au-delà de ts=20000.
OAR_RESOURCES="${OAR_RESOURCES:-gpu=${OAR_GPU}}"

# PyTorch exige sm_75+ ; en dessous la 1re convolution meurt sur un message
# trompeur. Comparaison textuelle : fausse dès qu'une majeure passe à 2 chiffres.
OAR_PROPERTIES="${OAR_PROPERTIES:-gpu_compute_capability >= '7.5'}"

REMOTE_DIR="patchcore-inspection"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# Pas de base MLflow parallèle : la base distante est importée dans mlruns.db
# par tools/mlflow_import.py, taguée "g5k". Idempotent.
IMPORT_TMP=".mlflow_import/g5k"
IMPORT_ORIGIN="g5k"
LOCAL_PYTHON="${LOCAL_DIR}/.venv/bin/python"
[[ -x "${LOCAL_PYTHON}" ]] || LOCAL_PYTHON="python"

# Les banques pèsent ~4 Ko par vecteur (640 Mo à 10% sur 2000 images) et
# restent côté serveur, où tourne l'inférence. true = les rapatrier.
FETCH_BANKS="${FETCH_BANKS:-false}"

# Dataset ciblé, pour ne rapatrier que les banques du bon dossier models/<dataset>/
# (celeba, coco...). N'affecte que --fetch / FETCH_BANKS.
DATASET="${DATASET:-celeba}"

# true = soumet le job et rend la main tout de suite, sans suivre la sortie.
# Le job survit à la fermeture du terminal : c'est OAR qui l'exécute, pas toi.
DETACH="${DETACH:-false}"

# Passées AU JOB (l'env local ne le suit pas), guillemets doubles à l'intérieur :
#   REMOTE_ENV='SIZES="1000 2000" N_PER_CLASS=300'
REMOTE_ENV="${REMOTE_ENV:-}"

# Script à exécuter côté serveur (par défaut, ou 1er argument), le reste des
# arguments est transmis tel quel au script Python.
SCRIPT="${1:-bin/celeba/infer/heatmap.py}"
if [[ $# -gt 0 ]]; then
  shift
fi
SCRIPT_ARGS=("$@")

# Mêmes exclusions que remote_run.sh. `models` exclu dans les deux sens : les
# banques se construisent et se consomment sur le serveur. `coresets` aussi, et
# pour une raison de plus : ce sont les banques empaquetées de la page de démo,
# qui ne tourne jamais ici — un seul .pkg y monte à 3 Go, soit un huitième du
# quota du /home poussé à travers la passerelle pour rien.
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude 'models' --exclude 'coresets'
          --exclude 'mlruns' --exclude 'results'
          --exclude 'mlruns.db' --exclude 'mlflow.db' --exclude 'mlruns.db.bak-*'
          --exclude 'mlruns_remote' --exclude 'mlruns_remote.db' --exclude 'mlruns_array'
          --exclude '.mlflow_import' --exclude '__pycache__' --exclude '.pytest_cache')

# `data/` n'est pas exclu par défaut : un fit de scène lit data/scene sur le nœud.
# Mais un job COCO ou CelebA télécharge ses images lui-même, et pousser un dossier
# d'images sur la frontale mange le quota du /home pour rien. D'où ce cran :
#   EXTRA_EXCLUDES="data results/coco" ./grid5000_run.sh ...
for motif in ${EXTRA_EXCLUDES:-}; do
  EXCLUDES+=(--exclude "${motif}")
done

# Alias défini par la config SSH ci-dessus : il traverse la passerelle tout seul.
G5K_HOST="${G5K_SITE}.g5k"
SSH_FRONTEND=(ssh "${G5K_HOST}")

# --partial : un transfert coupé reprend là où il s'était arrêté au lieu de tout
# refaire. La passerelle Grid'5000 coupe volontiers les connexions un peu longues.
RSYNC_OPTS=(-avz --partial)

# ─── Rapatriement (étape 3, ou seule action avec --fetch) ─────────────────
fetch_results() {
  echo "Récupération des résultats..."
  rsync "${RSYNC_OPTS[@]}" \
    "${G5K_HOST}:${REMOTE_DIR}/results/" \
    "${LOCAL_DIR}/results/" 2>/dev/null || echo "  (pas de dossier results/ à rapatrier)"

  # MLflow : rapatriement temporaire puis fusion dans la base locale unique.
  echo "Récupération de la base MLflow (temporaire ${IMPORT_TMP})..."
  mkdir -p "${LOCAL_DIR}/${IMPORT_TMP}/mlruns"
  rsync "${RSYNC_OPTS[@]}" \
    "${G5K_HOST}:${REMOTE_DIR}/mlruns.db" \
    "${LOCAL_DIR}/${IMPORT_TMP}/mlruns.db" 2>/dev/null || echo "  (pas de mlruns.db distante)"
  rsync "${RSYNC_OPTS[@]}" \
    "${G5K_HOST}:${REMOTE_DIR}/mlruns/" \
    "${LOCAL_DIR}/${IMPORT_TMP}/mlruns/" 2>/dev/null || echo "  (pas d'artefacts mlruns/ distants)"
  if [[ -f "${LOCAL_DIR}/${IMPORT_TMP}/mlruns.db" ]]; then
    echo "Fusion des runs Grid'5000 dans la base locale unique (mlruns.db)..."
    ( cd "${LOCAL_DIR}" && "${LOCAL_PYTHON}" tools/mlflow_import.py \
        --source-db "${IMPORT_TMP}/mlruns.db" \
        --source-artifacts "${IMPORT_TMP}/mlruns" \
        --origin "${IMPORT_ORIGIN}" --route-by-runname )
  fi

  if [[ "${FETCH_BANKS}" == "true" ]]; then
    echo "Récupération des banques mémoire (models/${DATASET}/)..."
    mkdir -p "${LOCAL_DIR}/models/${DATASET}/"
    rsync "${RSYNC_OPTS[@]}" \
      "${G5K_HOST}:${REMOTE_DIR}/models/${DATASET}/" \
      "${LOCAL_DIR}/models/${DATASET}/" 2>/dev/null || echo "  (pas de banque à rapatrier)"
  fi
}

if [[ "${SCRIPT}" == "--fetch" ]]; then
  echo "État des jobs OAR :"
  "${SSH_FRONTEND[@]}" "oarstat -u" || true
  fetch_results
  echo "Terminé."
  exit 0
fi

# ─── 1. Sync : local → frontale ───────────────────────────────────────────
echo "Envoi du code vers ${G5K_HOST}..."
rsync "${RSYNC_OPTS[@]}" "${EXCLUDES[@]}" \
  "${LOCAL_DIR}/" \
  "${G5K_HOST}:${REMOTE_DIR}/"

# ─── 2. Réservation d'un nœud GPU + exécution ─────────────────────────────
QUOTED_ARGS=""
for arg in "${SCRIPT_ARGS[@]+"${SCRIPT_ARGS[@]}"}"; do
  QUOTED_ARGS+=" $(printf '%q' "$arg")"
done

PROPERTY_FLAG=""
if [[ -n "${OAR_PROPERTIES}" ]]; then
  PROPERTY_FLAG="-p \"${OAR_PROPERTIES}\""
fi

# Un .sh (ex: sweep_histograms.sh) se lance avec bash, le reste avec python.
if [[ "${SCRIPT}" == *.sh ]]; then
  RUNNER="bash"
else
  RUNNER="python"
fi

echo "Réservation d'un nœud GPU (queue ${OAR_QUEUE}, gpu=${OAR_GPU}, walltime=${OAR_WALLTIME})..."
echo "    puis exécution de ${SCRIPT}${QUOTED_ARGS}"

# Pas d'équivalent bloquant de `srun` sous OAR : job passif, puis suivi de sa
# sortie jusqu'à ce qu'il quitte la file.
# shellcheck disable=SC2029  # on veut bien l'expansion locale des variables
"${SSH_FRONTEND[@]}" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

JOB_OUT="oar_job.out"

# Job écrit dans un script à part : un REMOTE_ENV contenant des guillemets
# (SIZES="1000 2000") casserait l'imbrication des quotes d'un oarsub bash -c.
LAUNCH=".oar_launch.sh"
cat > "\${LAUNCH}" <<'LAUNCHER'
#!/usr/bin/env bash
set -euo pipefail
cd ~/${REMOTE_DIR}
source .venv/bin/activate
export PATCHCORE_ORIGIN=g5k
${REMOTE_ENV} ${RUNNER} ${SCRIPT}${QUOTED_ARGS}
LAUNCHER
chmod +x "\${LAUNCH}"

# On capture la sortie d'oarsub : sous 'set -e' + pipefail, un échec de
# réservation avortait le script sans afficher la moindre raison.
if ! OARSUB_OUT=\$(oarsub -q "${OAR_QUEUE}" \
      -l "${OAR_RESOURCES},walltime=${OAR_WALLTIME}" ${PROPERTY_FLAG} \
      --stdout="\${JOB_OUT}" --stderr="\${JOB_OUT}" \
      "./\${LAUNCH}" 2>&1); then
  echo "\${OARSUB_OUT}" >&2
  echo "oarsub a échoué." >&2
  exit 1
fi
JOB_ID=\$(printf '%s\n' "\${OARSUB_OUT}" | sed -n 's/^OAR_JOB_ID=//p')

if [[ -z "\${JOB_ID}" ]]; then
  echo "\${OARSUB_OUT}" >&2
  echo "oarsub n'a pas rendu de job id (réservation refusée ?)." >&2
  exit 1
fi
echo "Job OAR \${JOB_ID} soumis."

if [[ "${DETACH}" == "true" ]]; then
  echo
  echo "    Le job est maintenant entre les mains d'OAR : il démarrera dès qu'un"
  echo "    nœud se libère, et tournera même si tu fermes ton terminal ou éteins"
  echo "    ton ordinateur. Rien à laisser ouvert."
  echo
  echo "    Suivre :          ssh ${G5K_HOST} 'oarstat -u'"
  echo "    Voir la sortie :  ssh ${G5K_HOST} 'tail -f ${REMOTE_DIR}/\${JOB_OUT}'"
  echo "    Rapatrier :       ./grid5000_run.sh --fetch"
  exit 0
fi

: > "\${JOB_OUT}"
tail -f "\${JOB_OUT}" &
TAIL_PID=\$!
trap 'kill \${TAIL_PID} 2>/dev/null || true' EXIT

echo "    En attente d'un nœud..."
# Tant que le job est dans la file ou tourne, on attend. Les états OAR sont
# Waiting / Launching / Running / Terminated / Error.
while oarstat -s -j "\${JOB_ID}" | grep -qE 'Waiting|Launching|Running|toLaunch|Hold'; do
  sleep 10
done

sleep 2                       # laisse tail -f rattraper les dernières lignes
kill \${TAIL_PID} 2>/dev/null || true

STATE=\$(oarstat -s -j "\${JOB_ID}" | cut -d: -f2 | tr -d ' ')
echo "Job \${JOB_ID} terminé (état : \${STATE})."
[[ "\${STATE}" == "Error" ]] && exit 1
exit 0
REMOTE_SCRIPT

# ─── 3. Sync : frontale → local (rapatrie les outputs) ────────────────────
# Le `exit 0` du mode détaché est dans le heredoc : c'est le shell *distant*
# qu'il arrête, pas celui-ci, qui enchaînait donc sur un rapatriement de
# résultats qu'aucun job n'a encore produits — et sur un import MLflow pour rien.
if [[ "${DETACH}" == "true" ]]; then
  exit 0
fi

fetch_results

echo "Terminé."
