#!/usr/bin/env bash
#
# fit_and_score.sh — Fit PatchCore PUIS scoring histogramme dans un SEUL job OAR.
# La banque est PERSISTÉE dans le home (models/celeba/<tag>) pour la garder et la
# réutiliser au scoring plus tard ; le fit et le scoring de ce job l'utilisent
# aussi dans la foulée.
#
# ATTENTION quota : une banque « toutes images, coreset 5% » pèse ~24 Go. Sous le
# hard limit du home (95 Go) mais au-dessus du soft (24 Go) -> déclenche la grâce.
# Pense à supprimer les vieilles banques (models/celeba/) quand tu n'en as plus
# besoin. Pour NE PAS persister (mesure jetable), pointe FIT_MODELS_DIR vers
# /tmp/$USER/... .
#
# L'histogramme (PNG + JSON) atterrit dans results/celeba/histograms/p<pct>/ —
# classé par pct de coreset — rapatriable par grid5000_run.sh --fetch.
#
# Config via env (défauts = la cible « tout le dataset, 5%, 3 NN ») :
#   REMOTE_ENV="FIT_TRAIN_SUBSET=none FIT_CORESET_PCT=0.05 FIT_NUM_NN=3" \
#   OAR_RESOURCES='host=1' OAR_WALLTIME=12:00:00 OAR_PROPERTIES="cluster='gres'" \
#   DETACH=true ./grid5000_run.sh bin/celeba/fit_and_score.sh
set -euo pipefail

export FIT_SAMPLER="${FIT_SAMPLER:-approx_greedy_coreset}"
export FIT_CORESET_PCT="${FIT_CORESET_PCT:-0.05}"
export FIT_NUM_NN="${FIT_NUM_NN:-3}"
export FIT_TRAIN_SUBSET="${FIT_TRAIN_SUBSET:-none}"
export FIT_MODELS_DIR="${FIT_MODELS_DIR:-models/celeba}"   # home = banque conservée
SEED=0

# Reconstruit le <tag> exactement comme build_tag() du fit (sinon le scoring ne
# retrouve pas la banque).
ts_lc=$(printf '%s' "${FIT_TRAIN_SUBSET}" | tr '[:upper:]' '[:lower:]')
case "${ts_lc}" in ""|none|all) TS="all";; *) TS="${FIT_TRAIN_SUBSET}";; esac
if [ "${FIT_SAMPLER}" = "identity" ]; then
  SAMP="identity"
else
  SAMP="${FIT_SAMPLER}_p${FIT_CORESET_PCT}"
fi
TAG="wideresnet50_${SAMP}_ts${TS}_s${SEED}"
BANK_DIR="${FIT_MODELS_DIR}/${TAG}"

echo "=== FIT === sampler=${FIT_SAMPLER} pct=${FIT_CORESET_PCT} num_nn=${FIT_NUM_NN} ts=${FIT_TRAIN_SUBSET}"
echo "    banque (conservée) -> ${BANK_DIR}"
python bin/celeba/fit/memory_bank.py

echo "=== SCORING === histogramme good/hat contre la banque"
export HIST_BANK_DIR="${BANK_DIR}"
# histogram.py insère le sous-dossier p<pct>/ lui-même -> on ne le met pas ici.
export HIST_OUTPUT_PATH="results/celeba/histograms/hist_celeba_ts${TS}_nn${FIT_NUM_NN}_s${SEED}.png"
python bin/celeba/infer/histogram.py

echo "=== Terminé ==="
echo "    Banque conservée : ${BANK_DIR}"
echo "    Histogramme classé par pct : results/celeba/histograms/p${FIT_CORESET_PCT}/ (rapatrier avec --fetch)"
