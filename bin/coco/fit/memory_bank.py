"""Construit une banque mémoire PatchCore sur les images COCO « personne SANS
couteau » (le normal one-class).

    COCO_PATH=/tmp/$USER/coco python bin/coco/fit/memory_bank.py

COCO_PATH est produit par tools/coco_fetch.py (manifest.json + images/). Réglages
surchargeables : cf. experiments.pipelines.fit_settings.
"""

import logging
import os

from experiments.datasets import COCO
from experiments.pipelines import run_fit

SOURCE = os.environ.get(
    "COCO_PATH", os.path.expanduser("/tmp/{}/coco".format(os.environ.get("USER", "u")))
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    os.environ.setdefault("COCO_PATH", SOURCE)
    # `source` va dans le fit_config : l'inférence retrouve les images sans que
    # COCO_PATH soit repositionné.
    run_fit(COCO, models_dir="models/coco", coreset_pct=0.05,
            extra_config={"source": SOURCE})
