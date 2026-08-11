"""Les datasets vus par les pipelines : comment les construire, comment nommer
leurs deux classes."""

import os

from experiments.folder import FolderDataset
from experiments.pipelines import Spec
from patchcore.datasets import celeba, coco


def _celeba(split, resize, imagesize, seed, fit_config=None):
    return celeba.CelebADataset(
        resize=resize, imagesize=imagesize,
        split=celeba.DatasetSplit[split.upper()], seed=seed,
    )


def _coco(split, resize, imagesize, seed, fit_config=None):
    # COCO_PATH pointe le dossier produit par tools/coco_fetch.py ; à défaut on
    # relit celui enregistré dans le fit_config de la banque.
    source = os.environ.get("COCO_PATH") or (fit_config or {}).get("source")
    return coco.CocoDataset(
        source=source, resize=resize, imagesize=imagesize,
        split=coco.DatasetSplit[split.upper()], seed=seed,
    )


def _scene(split, resize, imagesize, seed, fit_config=None):
    # SCENE_PATH pointe le dossier de captures ; à défaut celui du fit_config.
    source = os.environ.get("SCENE_PATH") or (fit_config or {}).get("source")
    return FolderDataset(
        source=source, resize=resize, imagesize=imagesize,
        split=coco.DatasetSplit[split.upper()], seed=seed,
    )


CELEBA = Spec(name="celeba", build=_celeba,
              normal="No-hat (normal)", anomaly="Hat (anomalie)")
COCO = Spec(name="coco", build=_coco,
            normal="Good (sans couteau)", anomaly="Knife (avec couteau)")
SCENE = Spec(name="scene", build=_scene, normal="Normal", anomaly="Anomalie")
