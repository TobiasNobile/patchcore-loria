"""Construit une banque mémoire PatchCore sur les images CelebA sans chapeau.

    python bin/celeba/fit/memory_bank.py

Écrit dans FIT_MODELS_DIR/<tag>/ l'index FAISS, patchcore_params.pkl et
fit_config.json. Réglages surchargeables : cf. experiments.pipelines.fit_settings.
"""

import logging

from experiments.datasets import CELEBA
from experiments.pipelines import run_fit

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_fit(CELEBA, models_dir="models/celeba", coreset_pct=0.1, train_subset=2000)
