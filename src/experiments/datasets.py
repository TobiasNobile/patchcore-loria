"""Le dataset de la démo : une scène filmée sur place, rangée en dossiers.

C'est le seul dont la démo a besoin. Les datasets publics qui servent aux
expériences sont déclarés à part, pour pouvoir être retirés d'un bloc.
"""

import os

from experiments.folder import FolderDataset
from experiments.pipelines import Spec
from patchcore.datasets import DatasetSplit


def _scene(split, resize, imagesize, seed, fit_config=None):
    # SCENE_PATH pointe le dossier de captures ; à défaut celui du fit_config.
    source = os.environ.get("SCENE_PATH") or (fit_config or {}).get("source")
    return FolderDataset(
        source=source, resize=resize, imagesize=imagesize,
        split=DatasetSplit[split.upper()], seed=seed,
    )


SCENE = Spec(name="scene", build=_scene, normal="Normal", anomaly="Anomalie")
