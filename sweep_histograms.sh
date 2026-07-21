#!/usr/bin/env bash
#
# sweep_histograms.sh — Balaye toutes les combinaisons d'histogrammes no-hat/hat :
#
#     tailles de banque : 1000 2000 5000 10000 20000 50000  (images d'entraînement)
#     plus proches voisins : 3 et 1                          (paramètre de scoring)
#     coreset : 10% et 5%                                    (0.1 et 0.05)
#
# soit 6 x 2 x 2 = 24 histogrammes. Chaque run écrit un PNG + un sidecar JSON avec
# l'indice de Jaccard (cf. histogram_jaccard dans score_histogram_celeba.py).
#
# Le nombre de voisins n'intervient qu'au scoring, pas dans le coreset : la banque
# est identique pour nn=1 et nn=3, donc on ne fitte que 12 banques (une par taille
# x pourcentage) et on score chacune deux fois via HIST_NUM_NN. Fit idempotent.
#
# Usage :
#   ./sweep_histograms.sh                 # dans un env où `python` voit le projet
#   PYTHON="uv run python" ./sweep_histograms.sh
#   N_PER_CLASS=500 ./sweep_histograms.sh # échantillon test plus petit
#
# Sur Grid'5000, à lancer SUR un nœud réservé (les scripts touchent le GPU) :
#   oarsub -q abaca -l gpu=1,walltime=6:00:00 \
#     "bash -c 'cd ~/patchcore-inspection && source .venv/bin/activate && ./sweep_histograms.sh'"
#
set -euo pipefail

cd "$(dirname "$0")"

# Active le venv du projet s'il existe et qu'aucun python n'est déjà imposé.
if [[ -z "${PYTHON:-}" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PYTHON="${PYTHON:-python}"

# Doit correspondre aux constantes du tag dans bin/fit_memory_bank_celeba.py :
# build_tag() = "{backbone}_{sampler}_p{pct:g}_ts{subset}_s{seed}".
BACKBONE="wideresnet50"
SAMPLER="approx_greedy_coreset"
SEED=0
MODELS_DIR="models/celeba"
OUT_DIR="results/histograms/sweep"

SIZES=(1000 2000 5000 10000 20000 50000)
NNS=(3 1)
PCTS=(0.1 0.05)
N_PER_CLASS="${N_PER_CLASS:-1000}"

mkdir -p "${OUT_DIR}"

bank_tag() {  # $1 = pct, $2 = train_subset
  # %g reproduit le formatage Python "{:g}" : 0.1 -> "0.1", 0.05 -> "0.05".
  printf '%s_%s_p%g_ts%s_s%s' "${BACKBONE}" "${SAMPLER}" "$1" "$2" "${SEED}"
}

echo "Balayage : ${#SIZES[@]} tailles x ${#NNS[@]} voisins x ${#PCTS[@]} pourcentages"
echo "  = $(( ${#SIZES[@]} * ${#NNS[@]} * ${#PCTS[@]} )) histogrammes, $(( ${#SIZES[@]} * ${#PCTS[@]} )) banques à (re)construire au plus."

for pct in "${PCTS[@]}"; do
  for size in "${SIZES[@]}"; do
    tag="$(bank_tag "${pct}" "${size}")"
    bank_dir="${MODELS_DIR}/${tag}"

    # Fit (idempotent) : une banque par (taille, pourcentage).
    if [[ -f "${bank_dir}/fit_config.json" ]]; then
      echo "Banque déjà présente, fit sauté : ${bank_dir}"
    else
      echo "Fit banque ts=${size} coreset=${pct} -> ${bank_dir}"
      FIT_TRAIN_SUBSET="${size}" FIT_CORESET_PCT="${pct}" \
        "${PYTHON}" bin/fit_memory_bank_celeba.py
    fi

    # Score : nn=3 puis nn=1 sur la même banque (pas de re-fit).
    for nn in "${NNS[@]}"; do
      out="${OUT_DIR}/hist_ts${size}_p${pct}_nn${nn}.png"
      echo "  Histogramme ts=${size} coreset=${pct} nn=${nn} -> ${out}"
      HIST_BANK_DIR="${bank_dir}" HIST_OUTPUT_PATH="${out}" HIST_NUM_NN="${nn}" \
        "${PYTHON}" bin/score_histogram_celeba.py --n_per_class "${N_PER_CLASS}"
    done
  done
done

# Récapitulatif : indice de Jaccard de chaque config, trié.
echo
echo "Récapitulatif (indice de Jaccard = recouvrement, plus bas = mieux séparé) :"
"${PYTHON}" - "${OUT_DIR}" <<'PY'
import glob, json, os, sys

out_dir = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(out_dir, "*.json")):
    with open(path) as fh:
        d = json.load(fh)
    rows.append((
        int(d.get("train_subset") or 0),
        float(d.get("coreset_pct", float("nan"))),
        int(d.get("num_nn_used", 0)),
        float(d.get("jaccard", float("nan"))),
        float(d.get("w1_normalized", float("nan"))),
        float(d.get("auroc", float("nan"))),
        int(d.get("n_per_class_used", 0)),
    ))

rows.sort(key=lambda r: (r[1], r[0], -r[2]))
hdr = ("ts", "coreset", "nn", "jaccard", "W1_norm", "auroc", "n/classe")
print("{:>7} {:>8} {:>3} {:>8} {:>8} {:>7} {:>9}".format(*hdr))
for ts, pct, nn, jac, w1, auroc, npc in rows:
    print("{:>7} {:>8.2f} {:>3} {:>8.4f} {:>8.3f} {:>7.3f} {:>9}".format(
        ts, pct, nn, jac, w1, auroc, npc))
PY

echo
echo "Terminé — ${OUT_DIR}/hist_ts*_p*_nn*.png (+ sidecars .json)."
