"""Compare les scores d'anomalie des images NO-HAT et HAT du split TEST.

Charge une banque construite par bin/celeba/fit/memory_bank.py — aucun fit ici.
Les hyperparamètres du fit sont relus du fit_config.json de la banque : une image
encodée autrement que la banque donnerait des distances qui ne veulent rien dire.

    python bin/celeba/infer/histogram.py --n_per_class 200
"""

import logging

import click

from experiments.datasets import CELEBA
from experiments.pipelines import run_histogram

BANK_DIR = "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts2000_s0"
OUTPUT_PATH = "results/celeba/histograms/hist_celeba.png"


@click.command()
@click.option("--n_per_class", type=int, default=1000, show_default=True,
              help="Échantillon PAR classe, effectifs égaux, plafonné au disponible.")
def main(n_per_class):
    run_histogram(CELEBA, BANK_DIR, OUTPUT_PATH, "celeba-histograms", n_per_class)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
