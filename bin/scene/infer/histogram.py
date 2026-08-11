"""Calibre le seuil : distribution des scores normal vs anomalie sur la scène.

    SCENE_PATH=data/scene python bin/scene/infer/histogram.py

Demande des contre-exemples dans <racine>/anomaly/. Le seuil à reporter dans la
page live se lit entre les deux modes de l'histogramme.
"""

import logging

import click

from experiments.datasets import SCENE
from experiments.pipelines import run_histogram

BANK_DIR = "models/scene/wideresnet50_approx_greedy_coreset_p0.1_tsall_s0"
OUTPUT_PATH = "results/scene/histograms/hist_scene.png"


@click.command()
@click.option("--n_per_class", type=int, default=200, show_default=True,
              help="Échantillon PAR classe, plafonné au disponible.")
def main(n_per_class):
    run_histogram(SCENE, BANK_DIR, OUTPUT_PATH, "scene-histograms", n_per_class)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
