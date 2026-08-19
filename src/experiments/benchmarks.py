"""Les datasets publics servant à mesurer PatchCore hors de la scène de démo :
comment les construire, comment nommer leurs deux classes.

Ces specs ne sont utilisées que par bin/celeba/ et bin/coco/ ; la démo, elle,
n'a besoin que de SCENE (experiments/datasets.py).
"""

import os

from experiments.pipelines import Spec
from patchcore.datasets import DatasetSplit
from patchcore.datasets import celeba
from patchcore.datasets import coco


def _celeba(split, resize, imagesize, seed, fit_config=None):
    return celeba.CelebADataset(
        resize=resize, imagesize=imagesize,
        split=DatasetSplit[split.upper()], seed=seed,
    )


def _coco(split, resize, imagesize, seed, fit_config=None):
    # COCO_PATH pointe le dossier produit par tools/coco_fetch.py ; à défaut on
    # relit celui enregistré dans le fit_config de la banque.
    source = os.environ.get("COCO_PATH") or (fit_config or {}).get("source")
    return coco.CocoDataset(
        source=source, resize=resize, imagesize=imagesize,
        split=DatasetSplit[split.upper()], seed=seed,
    )


CELEBA = Spec(name="celeba", build=_celeba,
              normal="No-hat (normal)", anomaly="Hat (anomalie)")
COCO = Spec(name="coco", build=_coco,
            normal="Good (sans couteau)", anomaly="Knife (avec couteau)")
