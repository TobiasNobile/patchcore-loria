"""Compare les scores d'anomalie GOOD (personne sans couteau) et KNIFE du split
TEST COCO.

Charge une banque construite par bin/coco/fit/memory_bank.py — aucun fit ici ;
les hyperparamètres du fit sont relus de son fit_config.json.

    COCO_PATH=/tmp/$USER/coco python bin/coco/infer/histogram.py --n_per_class 200
"""

import logging

import click

from experiments.benchmarks import COCO
from experiments.pipelines import run_histogram

BANK_DIR = "models/coco/wideresnet50_approx_greedy_coreset_p0.05_ts20000_s0"
OUTPUT_PATH = "results/coco/histograms/hist_coco.png"


@click.command()
@click.option("--n_per_class", type=int, default=1000, show_default=True,
              help="Échantillon PAR classe, effectifs égaux, plafonné au disponible.")
def main(n_per_class):
    run_histogram(COCO, BANK_DIR, OUTPUT_PATH, "coco-histograms", n_per_class)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
