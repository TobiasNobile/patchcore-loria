"""Superpose la heatmap d'anomalie PatchCore sur des images test CelebA.

    python bin/celeba/infer/heatmap.py --image_index 900
    python bin/celeba/infer/heatmap.py --n_per_class 30
"""

import logging

import click

from experiments.benchmarks import CELEBA
from experiments.reports import run_heatmaps

BANK_DIR = "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts10000_s0"
OUTPUT_PATH = "results/celeba/heatmaps/overlay_idx{idx}.png"


@click.command()
@click.option("--image_index", type=int, default=0, show_default=True,
              help="Mode image seule : index dans le split TEST équilibré.")
@click.option("--n_per_class", type=int, default=0, show_default=True,
              help="Si >0, ignore --image_index : n images de chaque classe.")
def main(image_index, n_per_class):
    run_heatmaps(CELEBA, BANK_DIR, OUTPUT_PATH, image_index, n_per_class)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
