"""Agrège les sidecars de résultats par configuration, sur l'axe des seeds.

    python tools/aggregate_runs.py results/coco --markdown

Sort moyenne ± écart-type. Une valeur unique ne dit pas si un écart entre deux
lignes est un effet ou du bruit ; `n=1` signale un écart-type inconnu, pas nul.
"""

import glob
import json
import re
import os
import statistics
import sys

import click

DEFAULT_METRICS = ("auroc", "jaccard", "w1_normalized")

# Le nom de banque encode toute la config du fit ; les vieux sidecars n'ont pas les couches.
SEED_SUFFIX = re.compile(r"_s\d+$")


def load_runs(root):
    """Les sidecars plats (ceux qui portent une métrique et un seed)."""
    runs = []
    for path in sorted(glob.glob(os.path.join(root, "**", "*.json"), recursive=True)):
        try:
            with open(path) as fh:
                data = json.load(fh)
        except (OSError, ValueError):
            continue
        # Les sidecars du banc ont une autre forme et pas de métrique de séparation.
        if "seed" not in data or not any(m in data for m in DEFAULT_METRICS):
            continue
        data["_path"] = path
        runs.append(data)
    return runs


def config_key(run):
    """Identité de la configuration : le tag de banque, seed retiré, plus le k
    de la recherche — celui-ci est un paramètre de requête, surchargeable après
    le fit, donc absent du tag."""
    tag = os.path.basename(os.path.normpath(run.get("bank_dir", "?")))
    return (SEED_SUFFIX.sub("", tag), "nn={}".format(run.get("num_nn")))


def summarize(runs, metrics):
    groups = {}
    for run in runs:
        groups.setdefault(config_key(run), []).append(run)

    rows = []
    for key, group in sorted(groups.items()):
        seeds = sorted({r["seed"] for r in group})
        row = {"config": key, "n": len(seeds), "seeds": seeds,
               "runs": len(group), "metrics": {}, "duplicates": False}
        for metric in metrics:
            # Réduire par seed d'abord : deux sidecars d'un même seed sont des doublons, pas une variance.
            per_seed = {}
            for run in group:
                value = run.get(metric)
                if isinstance(value, (int, float)):
                    per_seed.setdefault(run["seed"], []).append(value)
            values = [statistics.fmean(v) for v in per_seed.values()]
            if not values:
                continue
            row["metrics"][metric] = (
                statistics.fmean(values),
                statistics.stdev(values) if len(values) > 1 else None,
                min(values), max(values),
            )
            row["duplicates"] = max(
                (len(v) for v in per_seed.values()), default=1) > 1
        rows.append(row)
    return rows


def render(rows, metrics, markdown):
    single = [r for r in rows if r["n"] < 2]
    if markdown:
        head = ["config", "n"] + list(metrics)
        print("| " + " | ".join(head) + " |")
        print("| " + " | ".join(["---"] * len(head)) + " |")
    for row in rows:
        label = "  ".join(row["config"])
        cells = []
        for metric in metrics:
            stat = row["metrics"].get(metric)
            if stat is None:
                cells.append("—")
            elif stat[1] is None:
                cells.append("{:.4f} (n=1)".format(stat[0]))
            else:
                cells.append("{:.4f} ± {:.4f}".format(stat[0], stat[1]))
        if markdown:
            print("| {} | {} | {} |".format(label, row["n"], " | ".join(cells)))
        else:
            print("{}\n   n={} seeds={}  {}".format(
                label, row["n"], row["seeds"],
                "  ".join("{}={}".format(m, c) for m, c in zip(metrics, cells))))

    if single:
        print("\n{} configuration(s) à un seul seed — écart-type inconnu. "
              "Rejouer avec FIT_SEED pour obtenir une barre d'erreur.".format(len(single)))
    dupes = [r for r in rows if r["duplicates"]]
    if dupes:
        print("{} configuration(s) avec plusieurs sidecars pour un même seed : "
              "moyennés, mais c'est le signe d'un run rejoué sans nettoyer "
              "l'ancien fichier.".format(len(dupes)))


@click.command()
@click.argument("root", default="results", type=click.Path(exists=True))
@click.option("--metric", "metrics", multiple=True,
              help="Métrique à agréger, répétable. Défaut : {}.".format(
                  ", ".join(DEFAULT_METRICS)))
@click.option("--markdown", is_flag=True, help="Sortie en tableau Markdown.")
def main(root, metrics, markdown):
    metrics = metrics or DEFAULT_METRICS
    runs = load_runs(root)
    if not runs:
        raise SystemExit("Aucun sidecar exploitable sous {}.".format(root))
    print("{} runs lus sous {}\n".format(len(runs), root))
    render(summarize(runs, metrics), metrics, markdown)


if __name__ == "__main__":
    main()
