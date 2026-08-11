#!/usr/bin/env bash
#
# sweep_coreset.sh — enchaîne des fit_and_score.sh COCO à coresets différents
# dans UN SEUL job. Deux DETACH=true coup sur coup s'écraseraient : grid5000_run.sh
# écrit son lanceur sous un nom fixe sur la frontale.
#
#   REMOTE_ENV='PCTS="0.01 0.02" FIT_TRAIN_SUBSET=20000 FIT_LAYERS=layer3,layer4' \
#   OAR_RESOURCES='host=1' OAR_WALLTIME=06:00:00 OAR_PROPERTIES="cluster='gres'" \
#   DETACH=true ./grid5000_run.sh bin/coco/sweep_coreset.sh
set -euo pipefail

PCTS="${PCTS:-0.01 0.02}"

echo "=== SWEEP CORESET : ${PCTS} (ts=${FIT_TRAIN_SUBSET:-20000}, layers=${FIT_LAYERS:-layer2,layer3}) ==="
for pct in ${PCTS}; do
  echo
  echo "############################################################"
  echo "#  CORESET ${pct}"
  echo "############################################################"
  FIT_CORESET_PCT="${pct}" bash bin/coco/fit_and_score.sh
done

echo
echo "=== SWEEP TERMINÉ (${PCTS}) ==="
ls -d models/coco/*/ 2>/dev/null
