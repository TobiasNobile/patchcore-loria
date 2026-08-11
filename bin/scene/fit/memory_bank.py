"""Construit une banque sur les images capturées de la scène de déploiement.

    SCENE_PATH=data/scene python bin/scene/fit/memory_bank.py

Le dossier vient de bin/capture.py (<racine>/normal/). Fitter sur la scène réelle
plutôt que sur un dataset public : le « normal » y est homogène, ce qui est la
condition pour que PatchCore sépare quelque chose.
"""

import logging
import os

from experiments.datasets import SCENE
from experiments.pipelines import run_fit

SOURCE = os.environ.get("SCENE_PATH", "data/scene")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("SCENE_PATH", SOURCE)
    run_fit(SCENE, models_dir="models/scene", coreset_pct=0.1,
            extra_config={"source": SOURCE})
