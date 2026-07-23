#!/usr/bin/env bash
#
# sweep_histograms_sohas.sh — Équivalent SOHAS de sweep_histograms.sh. Balaye
# toutes les combinaisons d'histogrammes good/weapon :
#
#     tailles de banque : 250 500 1000 all   (frames SANS arme ; "all" = les 1571)
#     plus proches voisins : 3 et 1          (paramètre de scoring)
#     coreset : 25%, 10%, 5%                 (0.25 0.1 0.05)
#
# Le pool normal de SOHAS ne fait que ~1571 frames : NE PAS mettre de taille
# > 1571 (le fit la ramènerait à 1571 mais garderait un tag ts trompeur, et
# deux tailles clampées produiraient des banques identiques). "all" prend tout.
#
# Le nombre de voisins n'intervient qu'au scoring, pas dans le coreset : la banque
# est identique pour nn=1 et nn=3, donc on ne fitte qu'une banque par (taille x
# pourcentage) et on la score deux fois via HIST_NUM_NN. Fit idempotent.
#
# Usage :
#   SOHAS_PATH=$HOME/sohas_data ./sweep_histograms_sohas.sh
#   SOHAS_PATH=$HOME/sohas_data SIZES="500 all" PCTS="0.1" ./sweep_histograms_sohas.sh
#
# Sur Grid'5000, via grid5000_run.sh (qui lance les .sh avec bash sur le nœud) :
#   REMOTE_ENV='SOHAS_PATH=$HOME/sohas_data SIZES="250 500 1000 all" PCTS="0.25 0.1 0.05" NNS="3 1"' \
#     ./grid5000_run.sh sweep_histograms_sohas.sh
#
set -euo pipefail

cd "$(dirname "$0")"

# Active le venv du projet s'il existe et qu'aucun python n'est déjà imposé.
if [[ -z "${PYTHON:-}" && -f ".venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi
PYTHON="${PYTHON:-python}"

# Racine SOHAS. Indispensable : le fit lit un chemin disque (pas HuggingFace).
if [[ -z "${SOHAS_PATH:-}" ]]; then
  echo "SOHAS_PATH doit pointer vers le dossier Sohas_weapon-Detection." >&2
  exit 1
fi
export SOHAS_PATH

# Doit correspondre aux constantes du tag dans bin/sohas/fit/memory_bank.py :
# build_tag() = "{backbone}_{sampler}_p{pct:g}_ts{subset}_s{seed}".
BACKBONE="wideresnet50"
SAMPLER="approx_greedy_coreset"
SEED=0
# Où écrire les banques. Sur Grid'5000, pointer vers le disque local du nœud
# (MODELS_DIR=/tmp/patchcore-banks) si le quota /home est juste.
MODELS_DIR="${MODELS_DIR:-models/sohas}"

# false = supprime chaque banque une fois scorée, ce qui borne le pic disque à
# une seule banque. Défaut true pour ne jamais effacer des banques persistantes.
KEEP_BANKS="${KEEP_BANKS:-true}"

# Reprise : on saute les configurations dont le sidecar JSON existe déjà.
SKIP_EXISTING="${SKIP_EXISTING:-true}"
OUT_DIR="${OUT_DIR:-results/sohas/histograms/sweep}"

# Surchargeables par env var. Tailles <= 1571 (le pool normal), ou "all".
read -ra SIZES <<< "${SIZES:-250 500 1000 all}"
read -ra NNS   <<< "${NNS:-3 1}"
read -ra PCTS  <<< "${PCTS:-0.25 0.1 0.05}"
N_PER_CLASS="${N_PER_CLASS:-300}"

mkdir -p "${OUT_DIR}" "${MODELS_DIR}"

bank_tag() {  # $1 = pct, $2 = train_subset ("all" ou un entier)
  # %g reproduit le formatage Python "{:g}" : 0.1 -> "0.1", 0.05 -> "0.05".
  printf '%s_%s_p%g_ts%s_s%s' "${BACKBONE}" "${SAMPLER}" "$1" "$2" "${SEED}"
}

echo "Balayage : ${#SIZES[@]} tailles x ${#PCTS[@]} coresets x ${#NNS[@]} voisins = $(( ${#SIZES[@]} * ${#PCTS[@]} * ${#NNS[@]} )) histogrammes."

for size in "${SIZES[@]}"; do
  # "all" -> FIT_TRAIN_SUBSET=none (banque sur les 1571 frames), tag "tsall".
  if [[ "${size}" == "all" ]]; then fit_ts="none"; else fit_ts="${size}"; fi

  for pct in "${PCTS[@]}"; do
    tag="$(bank_tag "${pct}" "${size}")"
    bank_dir="${MODELS_DIR}/${tag}"

    # Ne garder que les voisins dont l'histogramme manque.
    todo=()
    for nn in "${NNS[@]}"; do
      if [[ "${SKIP_EXISTING}" == "true" \
            && -f "${OUT_DIR}/hist_ts${size}_p${pct}_nn${nn}.json" ]]; then
        continue
      fi
      todo+=("${nn}")
    done
    if [[ ${#todo[@]} -eq 0 ]]; then
      echo "ts=${size} coreset=${pct} : déjà complet, sauté."
      continue
    fi

    # Fit (idempotent) : une banque par (taille, pourcentage).
    if [[ -f "${bank_dir}/fit_config.json" ]]; then
      echo "Banque déjà présente, fit sauté : ${bank_dir}"
    else
      echo "Fit banque ts=${size} coreset=${pct} -> ${bank_dir}"
      FIT_TRAIN_SUBSET="${fit_ts}" FIT_CORESET_PCT="${pct}" FIT_MODELS_DIR="${MODELS_DIR}" \
        "${PYTHON}" bin/sohas/fit/memory_bank.py
    fi

    # Score : nn=3 puis nn=1 sur la même banque (pas de re-fit).
    for nn in "${todo[@]}"; do
      out="${OUT_DIR}/hist_ts${size}_p${pct}_nn${nn}.png"
      echo "  Histogramme ts=${size} coreset=${pct} nn=${nn} -> ${out}"
      HIST_BANK_DIR="${bank_dir}" HIST_OUTPUT_PATH="${out}" HIST_NUM_NN="${nn}" \
        "${PYTHON}" bin/sohas/infer/histogram.py --n_per_class "${N_PER_CLASS}"
    done

    if [[ "${KEEP_BANKS}" != "true" ]]; then
      echo "  Banque scorée, suppression (KEEP_BANKS=false) : ${bank_dir}"
      rm -rf "${bank_dir}"
    fi
  done
done

# Récapitulatif : indice de Jaccard + AUROC de chaque config, trié.
echo
echo "Récapitulatif (Jaccard = recouvrement, plus bas = mieux séparé ; AUROC, plus haut = mieux) :"
"${PYTHON}" - "${OUT_DIR}" <<'PY'
import glob, json, os, sys

out_dir = sys.argv[1]
rows = []
for path in glob.glob(os.path.join(out_dir, "*.json")):
    with open(path) as fh:
        d = json.load(fh)
    rows.append((
        str(d.get("train_subset") or "all"),
        float(d.get("coreset_pct", float("nan"))),
        int(d.get("num_nn_used", 0)),
        float(d.get("jaccard", float("nan"))),
        float(d.get("w1_normalized", float("nan"))),
        float(d.get("auroc", float("nan"))),
        int(d.get("n_per_class_used", 0)),
    ))

rows.sort(key=lambda r: (r[1], str(r[0]), -r[2]))
hdr = ("ts", "coreset", "nn", "jaccard", "W1_norm", "auroc", "n/classe")
print("{:>7} {:>8} {:>3} {:>8} {:>8} {:>7} {:>9}".format(*hdr))
for ts, pct, nn, jac, w1, auroc, npc in rows:
    print("{:>7} {:>8.2f} {:>3} {:>8.4f} {:>8.3f} {:>7.3f} {:>9}".format(
        ts, pct, nn, jac, w1, auroc, npc))
PY

echo
echo "Terminé — ${OUT_DIR}/hist_ts*_p*_nn*.png (+ sidecars .json)."
