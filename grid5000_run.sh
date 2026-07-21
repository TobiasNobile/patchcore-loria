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
#   DETACH=true ./grid5000_run.sh bin/fit_memory_bank_celeba.py
#       Soumet le job et rend la main IMMÉDIATEMENT (rien à garder ouvert : ni
#       terminal, ni connexion, ni ordinateur allumé). Pour un run long/nocturne.
#   ./grid5000_run.sh --fetch
#       Ne lance rien : rapatrie les résultats d'un job détaché déjà terminé.
#
# Prérequis, une seule fois.
#
# 1) Dans ~/.ssh/config, la config officielle Grid'5000 (sinon la passerelle
#    coupe les transferts un peu longs) :
#
#      Host g5k
#        User <login>
#        Hostname access.grid5000.fr
#        ServerAliveInterval 30
#      Host *.g5k
#        User <login>
#        ProxyCommand ssh g5k -W "$(basename %h .g5k):%p"
#        ServerAliveInterval 30
#
#    Elle donne `ssh nancy.g5k`, ce que ce script utilise.
#
# 2) Sur la frontale : cd ~/patchcore-inspection && uv sync
#
# 3) Toujours sur la frontale, pré-télécharger CelebA (~22 Go) — le /home est
#    partagé en NFS avec les nœuds, donc le job GPU n'aura plus qu'à calculer :
#      uv run python -c "from datasets import load_dataset; load_dataset('flwrlabs/celeba')"
#
# Le fit n'a PAS besoin de faiss-gpu (le coreset tourne sur PyTorch/CUDA).
# `uv pip install faiss-gpu-cu12` ne sert qu'à accélérer l'inférence, plus tard.
#
set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────
# Le login Grid'5000 n'apparaît pas ici : il est dans ~/.ssh/config (cf. les
# prérequis en tête de fichier), qui définit l'alias `<site>.g5k`.
G5K_SITE="nancy"

# Queue des clusters accessibles au groupe 'orpailleur' à Nancy (cf. le message
# d'accueil de la frontale). Ce n'est PAS `production` : Grid'5000 a renommé.
OAR_QUEUE="abaca"
OAR_GPU=1
OAR_WALLTIME="${OAR_WALLTIME:-03:00:00}"

# Le PyTorch installé ne supporte que les compute capabilities sm_75 et plus.
# Sur un GPU plus ancien la première convolution meurt avec un message trompeur,
# "RuntimeError: GET was unable to find an engine to execute this computation".
# On filtre donc sur la capability, ce qui écarte automatiquement les mauvais
# clusters sans avoir à les lister. Clusters GPU de Nancy :
#   grele       GTX 1080 Ti      cc 6.1   INCOMPATIBLE (et c'est le plus dispo)
#   gratouille  Tesla V100       cc 7.0   INCOMPATIBLE
#   graffiti    RTX 2080 Ti      cc 7.5   ok
#   grue        Tesla T4         cc 7.5   ok
#   grat        A100 40 Go       cc 8.0   ok, très demandé
#   grouille    A100 40 Go       cc 8.0   ok
#   gruss       A40              cc 8.6   ok
#   gres        L40S             cc 8.9   ok
# La comparaison est textuelle : correcte tant que la majeure tient en un
# chiffre (un futur GPU en cc 10.x serait écarté à tort, jamais accepté à tort).
OAR_PROPERTIES="${OAR_PROPERTIES:-gpu_compute_capability >= '7.5'}"

REMOTE_DIR="patchcore-inspection"
LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)"

# MLflow : pas de base parallèle. On rapatrie la base + artefacts du serveur dans
# un dossier temporaire, puis on IMPORTE les runs dans la base locale unique
# (mlruns.db) via tools/mlflow_import.py — origine "g5k". Idempotent.
IMPORT_TMP=".mlflow_import/g5k"
IMPORT_ORIGIN="g5k"
LOCAL_PYTHON="${LOCAL_DIR}/.venv/bin/python"
[[ -x "${LOCAL_PYTHON}" ]] || LOCAL_PYTHON="python"

# Les banques mémoire pèsent lourd (~4 Ko par vecteur : 640 Mo pour une banque
# à 10% sur 2000 images, 64 Mo à 1%). Elles restent donc côté serveur par
# défaut — l'inférence tourne là-bas de toute façon. Passe à true pour les
# rapatrier et faire des heatmaps en local.
FETCH_BANKS="${FETCH_BANKS:-false}"

# true = soumet le job et rend la main tout de suite, sans suivre la sortie.
# Le job survit à la fermeture du terminal : c'est OAR qui l'exécute, pas toi.
DETACH="${DETACH:-false}"

# Variables d'env passées AU JOB (les variables locales ne le suivent pas). À
# écrire avec des guillemets doubles, ex :
#   REMOTE_ENV='SIZES="1000 2000" PCTS="0.1" N_PER_CLASS=300'
REMOTE_ENV="${REMOTE_ENV:-}"

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
EXCLUDES=(--exclude '.venv' --exclude '.git' --exclude 'models' --exclude 'mlruns' --exclude 'results'
          --exclude 'mlruns.db' --exclude 'mlflow.db' --exclude 'mlruns.db.bak-*'
          --exclude 'mlruns_remote' --exclude 'mlruns_remote.db' --exclude 'mlruns_array'
          --exclude '.mlflow_import' --exclude '__pycache__' --exclude '.pytest_cache')

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
    echo "Récupération des banques mémoire..."
    rsync "${RSYNC_OPTS[@]}" \
      "${G5K_HOST}:${REMOTE_DIR}/models/celeba/" \
      "${LOCAL_DIR}/models/celeba/" 2>/dev/null || echo "  (pas de banque à rapatrier)"
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

# OAR n'a pas d'équivalent bloquant de `srun` : `oarsub -I` ouvre un shell
# interactif et n'accepte pas de commande. On soumet donc un job passif, puis on
# suit sa sortie jusqu'à ce qu'il quitte les états Waiting/Launching/Running —
# ce qui donne le même confort qu'un run bloquant, sortie en direct comprise.
# shellcheck disable=SC2029  # on veut bien l'expansion locale des variables
"${SSH_FRONTEND[@]}" bash -s <<REMOTE_SCRIPT
set -euo pipefail
cd "${REMOTE_DIR}"

JOB_OUT="oar_job.out"

# Le job est écrit dans un script à part plutôt que passé en ligne à oarsub :
# une valeur de REMOTE_ENV contenant des guillemets (SIZES="1000 2000") casserait
# l'imbrication des quotes d'un oarsub "bash -c '...'" et découperait l'argument.
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
      -l "gpu=${OAR_GPU},walltime=${OAR_WALLTIME}" ${PROPERTY_FLAG} \
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
# En mode détaché on s'est déjà arrêté plus haut : le job n'a rien produit encore.
fetch_results

echo "Terminé."
