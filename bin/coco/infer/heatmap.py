"""Superpose la heatmap d'anomalie PatchCore sur des images test COCO.

    COCO_PATH=/tmp/$USER/coco python bin/coco/infer/heatmap.py --n_per_class 15
"""

import logging

import click

from experiments.benchmarks import COCO
from experiments.reports import run_heatmaps

BANK_DIR = "models/coco/wideresnet50_approx_greedy_coreset_p0.05_ts20000_s0"
OUTPUT_PATH = "results/coco/heatmaps/overlay_idx{idx}.png"


@click.command()
@click.option("--image_index", type=int, default=0, show_default=True,
              help="Mode image seule : index dans le split TEST équilibré.")
@click.option("--n_per_class", type=int, default=0, show_default=True,
              help="Si >0, ignore --image_index : n images de chaque classe.")
def main(image_index, n_per_class):
    run_heatmaps(COCO, BANK_DIR, OUTPUT_PATH, image_index, n_per_class)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
