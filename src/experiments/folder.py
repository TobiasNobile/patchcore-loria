"""Dataset « une scène » : les images capturées sur place, rangées en dossiers.

    <racine>/normal/    images sans l'anomalie à détecter (obligatoire)
    <racine>/anomaly/   contre-exemples, seulement pour calibrer un seuil

TRAIN = tout normal/ moins un holdout de 20 %, qui rejoint anomaly/ dans TEST.
"""

import os

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

from patchcore.datasets.coco import DatasetSplit

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
TEST_HOLDOUT = 0.2


def _listdir(path):
    if not os.path.isdir(path):
        return []
    return sorted(
        os.path.join(path, f) for f in os.listdir(path)
        if f.lower().endswith(EXTENSIONS)
    )


class FolderDataset(torch.utils.data.Dataset):
    def __init__(self, source, resize=256, imagesize=224, split=DatasetSplit.TRAIN,
                 seed=0):
        normal = _listdir(os.path.join(source, "normal"))
        anomaly = _listdir(os.path.join(source, "anomaly"))
        if not normal:
            raise FileNotFoundError(
                "Aucune image dans {}/normal — lancer bin/capture.py d'abord."
                .format(source)
            )

        # Le holdout est tiré une fois pour toutes : la même image ne peut pas
        # servir au fit et au test.
        rng = np.random.RandomState(seed)
        held = set(rng.choice(
            len(normal), max(1, int(TEST_HOLDOUT * len(normal))), replace=False
        ).tolist())
        if split == DatasetSplit.TRAIN:
            self.paths = [p for i, p in enumerate(normal) if i not in held]
            self.labels = [0] * len(self.paths)
        else:
            kept = [p for i, p in enumerate(normal) if i in held]
            self.paths = kept + anomaly
            self.labels = [0] * len(kept) + [1] * len(anomaly)

        self.split = split
        self.transform_mean, self.transform_std = IMAGENET_MEAN, IMAGENET_STD
        self.transform_img = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(imagesize),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])
        self.imagesize = (3, imagesize, imagesize)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self.transform_img(PIL.Image.open(self.paths[idx]).convert("RGB"))
        anomaly = self.labels[idx]
        return {
            "image": image,
            "mask": torch.zeros([1, *image.size()[1:]]),
            "classname": "scene",
            "anomaly": "anomaly" if anomaly else "good",
            "is_anomaly": anomaly,
            "image_name": os.path.basename(self.paths[idx]),
            "image_path": self.paths[idx],
        }
