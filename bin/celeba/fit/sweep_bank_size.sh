#!/usr/bin/env bash
#
# sweep_bank_size.sh — Trouve le PLAFOND de taille de banque PatchCore sur le
# hardware du nœud, en montant FIT_TRAIN_SUBSET jusqu'à l'OOM.
#
# Par défaut sampler=identity : banque = TOUT le sous-ensemble (le « Total Recall »
# du papier), donc la banque grossit linéairement avec le nombre d'images et c'est
# le vrai test « jusqu'où ». Le nuage de features 1024-dim + l'index FAISS vivent
# en RAM CPU (cf. sampler/common) -> lancer sur un nœud réservé ENTIER (host=1)
# pour disposer de toute la RAM. Les banques ne sont PAS persistées (FIT_NO_SAVE) :
# on ne veut que les mesures (taille, pic RAM, temps).
#
# Via grid5000_run.sh (réserver le nœud entier + gros walltime) :
#   OAR_RESOURCES='host=1' OAR_WALLTIME=06:00:00 \
#     OAR_PROPERTIES="cluster='gres'" \
#     ./grid5000_run.sh bin/celeba/fit/sweep_bank_size.sh
#
# Variables : SIZES (liste), FIT_SAMPLER (identity|approx_greedy_coreset), etc.
set -uo pipefail

SIZES="${SIZES:-10000 20000 40000 60000 80000 100000 120000}"
export FIT_SAMPLER="${FIT_SAMPLER:-identity}"
export FIT_CORESET_PCT="${FIT_CORESET_PCT:-0.1}"     # ignoré si identity
export FIT_NO_SAVE=1
export FIT_FAISS_GPU="${FIT_FAISS_GPU:-}"            # vide = index FAISS en RAM CPU
export FIT_MODELS_DIR="${FIT_MODELS_DIR:-/tmp/${USER}/bank_sweep}"
mkdir -p "${FIT_MODELS_DIR}"

# Résumé durable (rsync par grid5000_run.sh --fetch, contrairement au /tmp du nœud).
SUMMARY="results/celeba/bank_sweep/sweep_$(hostname -s)_${FIT_SAMPLER}.txt"
mkdir -p "$(dirname "${SUMMARY}")"

{
  echo "=== Sweep taille de banque | sampler=${FIT_SAMPLER} | faiss_gpu='${FIT_FAISS_GPU}' | node=$(hostname -s) ==="
  echo "RAM du nœud :"; free -g 2>/dev/null | awk 'NR<=2'
  echo "GPU :"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
  printf '\n%-8s %12s %12s %10s\n' "ts" "bank_Go" "peakRAM_Go" "fit_s"
} | tee "${SUMMARY}"

for N in ${SIZES}; do
  export FIT_TRAIN_SUBSET="${N}"
  LOG="${FIT_MODELS_DIR}/log_ts${N}.txt"
  if python bin/celeba/fit/memory_bank.py > "${LOG}" 2>&1; then
    cfg=$(ls -t "${FIT_MODELS_DIR}"/*/fit_config.json 2>/dev/null | head -1)
    if [ -n "${cfg}" ]; then
      python - "${cfg}" "${N}" <<'PY' | tee -a "${SUMMARY}"
import json, sys
d = json.load(open(sys.argv[1]))
print("%-8s %12.1f %12.1f %10.0f" % (
    sys.argv[2], d.get("bank_gb", 0), d.get("peak_rss_gb", 0), d.get("fit_seconds", 0)))
PY
    fi
  else
    rc=$?
    echo ">>> ÉCHEC à ts=${N} (rc=${rc}$([ ${rc} -eq 137 ] && echo ' = OOM/Killed')) — plafond atteint, arrêt." | tee -a "${SUMMARY}"
    echo "    3 dernières lignes du log :"
    tail -n 3 "${LOG}" | sed 's/^/      /'
    break
  fi
done
echo; echo "=== Sweep terminé. Le dernier ts réussi = ta taille de banque max sur ce nœud. ==="
