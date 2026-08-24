#!/usr/bin/env bash
#
# sweep_coreset.sh — enchaîne des fit_and_score.sh COCO à coresets différents
# dans UN SEUL job. Deux DETACH=true coup sur coup s'écraseraient : grid5000_run.sh
# écrit son lanceur sous un nom fixe sur la frontale.
#
# Chaque entrée est un coreset, éventuellement suivi de « :taille d'image ».
#
#   REMOTE_ENV='BACKBONES="wideresnet50 resnet50" PCTS="0.005:160 0.005:128" \
#                FIT_TRAIN_SUBSET=20000 FIT_LAYERS=layer3,layer4' \
#   OAR_RESOURCES='host=1' OAR_WALLTIME=06:00:00 OAR_PROPERTIES="cluster='gres'" \
#   DETACH=true ./grid5000_run.sh bin/coco/sweep_coreset.sh
set -euo pipefail

PCTS="${PCTS:-0.01 0.02}"
# Un backbone par défaut, plusieurs si on veut comparer : le coreset n'est pas le
# seul axe qui compte pour la cadence, et un job qui les enchaîne tous ne
# retélécharge les images qu'une fois — c'est la moitié du temps de calcul.
BACKBONES="${BACKBONES:-${FIT_BACKBONE:-wideresnet50}}"

echo "=== SWEEP : backbones « ${BACKBONES} » × coresets « ${PCTS} »"
echo "    (ts=${FIT_TRAIN_SUBSET:-20000}, layers=${FIT_LAYERS:-layer2,layer3})"
for backbone in ${BACKBONES}; do
  for entry in ${PCTS}; do
    pct="${entry%%:*}"
    size="${entry#*:}"; [ "${size}" = "${entry}" ] && size="${FIT_IMAGESIZE:-224}"
    echo
    echo "############################################################"
    echo "#  ${backbone}  |  CORESET ${pct}  |  ${size} px"
    echo "############################################################"
    FIT_BACKBONE="${backbone}" FIT_CORESET_PCT="${pct}" FIT_IMAGESIZE="${size}" \
      bash bin/coco/fit_and_score.sh
  done
done

echo
echo "=== SWEEP TERMINÉ (${BACKBONES} × ${PCTS}) ==="
ls -d models/coco/*/ 2>/dev/null
