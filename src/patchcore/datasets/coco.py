import json
import logging
import os
import random

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

from patchcore.datasets import DatasetSplit

LOGGER = logging.getLogger(__name__)

# Dataset COCO one-class « personne + couteau ». Alimenté par tools/coco_fetch.py
# qui télécharge le sous-ensemble utile et écrit un manifest.json compact :
#     {"images": [{"file": "images/xxx.jpg", "is_anomaly": 0|1,
#                  "knife_boxes": [[x0,y0,x1,y1],...], "width": W, "height": H}]}
# Toutes les images contiennent une personne (filtré au fetch). is_anomaly=1 <=>
# au moins un couteau. TRAIN = personnes sans couteau (banque) ; TEST = équilibré
# good/knife. Le bbox couteau donne un masque pixel (GT segmentation / heatmap).

_TEST_FRACTION = 0.2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class CocoDataset(torch.utils.data.Dataset):
    """Dataset COCO one-class : personne sans couteau (normal) vs personne avec
    couteau (anomalie). Lit le manifest.json produit par tools/coco_fetch.py.
    Découpage train/test déterministe par `seed` (cf. sohas/weapon)."""

    def __init__(
        self,
        source,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        seed=0,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Dossier contenant manifest.json et images/ (sortie de
                    coco_fetch.py, typiquement le /tmp du nœud).
            resize/imagesize: redimensionnement + center-crop.
            split: TRAIN (personnes sans couteau) ou TEST (équilibré).
            seed: pilote le découpage train/test et l'équilibrage.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.seed = seed

        samples = self._load_manifest()
        self.data = self._split(samples, split, seed)

        self.transform_mean = IMAGENET_MEAN
        self.transform_std = IMAGENET_STD
        self.transform_img = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(imagesize),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        # Même géométrie que l'image, NEAREST (masque binaire aligné).
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(
                    resize, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(imagesize),
            ]
        )
        self.imagesize = (3, imagesize, imagesize)

    def _load_manifest(self):
        manifest_path = os.path.join(self.source, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise RuntimeError(
                "manifest.json introuvable sous {}. Lance d'abord "
                "tools/coco_fetch.py (DEST={}).".format(self.source, self.source)
            )
        with open(manifest_path) as fh:
            images = json.load(fh)["images"]
        samples = []
        for m in images:
            path = os.path.join(self.source, m["file"])
            if not os.path.isfile(path):
                continue
            samples.append(
                {
                    "image_path": path,
                    "is_anomaly": int(m["is_anomaly"]),
                    "knife_boxes": m.get("knife_boxes", []),
                    "width": m.get("width", 0),
                    "height": m.get("height", 0),
                }
            )
        if not samples:
            raise RuntimeError("Manifest vide ou images absentes sous {}.".format(self.source))
        return samples

    @staticmethod
    def _split(samples, split, seed):
        """Personnes sans couteau -> train/test ; toutes les personnes-couteau
        au test ; test équilibré good/knife. Mélange par classe (proportion stable
        quel que soit le seed, cf. sohas/weapon)."""
        rng = random.Random(seed)
        normal = [i for i, s in enumerate(samples) if not s["is_anomaly"]]
        knife = [i for i, s in enumerate(samples) if s["is_anomaly"]]
        rng.shuffle(normal)
        rng.shuffle(knife)

        n_test = int(round(len(normal) * _TEST_FRACTION))
        test_normal, train_normal = normal[:n_test], normal[n_test:]

        if split == DatasetSplit.TRAIN:
            return [samples[i] for i in sorted(train_normal)]

        n = min(len(test_normal), len(knife))
        chosen = sorted(test_normal[:n] + knife[:n])
        return [samples[i] for i in chosen]

    def _knife_mask(self, sample):
        """Masque PIL 'L' à la taille d'origine : 255 dans chaque bbox couteau."""
        arr = np.zeros((sample["height"], sample["width"]), dtype=np.uint8)
        for x0, y0, x1, y1 in sample["knife_boxes"]:
            arr[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
        return PIL.Image.fromarray(arr, mode="L")

    @property
    def labels(self):
        """Labels d'anomalie (0/1), même ordre que __getitem__, sans décoder."""
        return [s["is_anomaly"] for s in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = self.transform_img(PIL.Image.open(sample["image_path"]).convert("RGB"))

        has_mask = (
            self.split == DatasetSplit.TEST
            and sample["is_anomaly"]
            and sample["width"] > 0
            and sample["height"] > 0
            and sample["knife_boxes"]
        )
        if has_mask:
            mask = self.transform_mask(self._knife_mask(sample))
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
            mask = (mask > 0).float()
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": "coco",
            "anomaly": "knife" if sample["is_anomaly"] else "good",
            "is_anomaly": sample["is_anomaly"],
            "image_name": os.path.basename(sample["image_path"]),
            "image_path": sample["image_path"],
        }
