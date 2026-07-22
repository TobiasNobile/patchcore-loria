import logging
import random
from enum import Enum

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

_HF_REPO = "mattmdjaga/human_parsing_dataset"
# ATR : 0 Background, 1 Hat, 2 Hair, 3 Sunglasses, 4 Upper-clothes, 5 Skirt,
# 6 Pants, 7 Dress, 8 Belt, 9 Left-shoe, 10 Right-shoe, 11 Face, 12 Left-leg,
# 13 Right-leg, 14 Left-arm, 15 Right-arm, 16 Bag, 17 Scarf.
_ANOMALY_LABEL = 1
# En dessous de ce nombre de pixels "Hat" l'annotation est du bruit : on ne veut
# pas exclure du train (ni compter comme anormale) une image pour trois pixels.
_MIN_ANOMALY_PIXELS = 64
# Le repo n'expose qu'un split `train` : on découpe nous-mêmes, par seed.
_TEST_FRACTION = 0.2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class AtrDataset(torch.utils.data.Dataset):
    """Dataset ATR one-class : la présence de la classe `Hat` dans le masque de
    segmentation sert de label d'anomalie. TRAIN = images no-hat (pour la banque
    mémoire) ; TEST = échantillon équilibré hat/no-hat (pour l'évaluation).

    Contrairement à CelebA, le repo n'a qu'un split `train` et aucune colonne
    d'attribut : le label vient du masque et le découpage train/test est fait
    ici, de façon déterministe à partir de `seed`. Le masque fournit en prime
    une vérité terrain pixel par pixel, exploitable pour l'AUROC segmentation.
    """

    def __init__(
        self,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        seed=0,
        num_proc=None,
        **kwargs,
    ):
        super().__init__()
        self.split = split
        self.seed = seed

        dataset = load_dataset(_HF_REPO, split=DatasetSplit.TRAIN.value)
        flags = self._hat_flags(dataset, num_proc)
        indices = self._split_indices(flags, split, seed)

        self.dataset = dataset.select(indices)
        self.is_anomaly = [flags[i] for i in indices]

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
        # NEAREST et pas de ToTensor : le masque porte des indices de classe,
        # qu'une interpolation ou une normalisation rendraient ininterprétables.
        self.transform_mask = transforms.Compose(
            [
                transforms.Resize(
                    resize, interpolation=transforms.InterpolationMode.NEAREST
                ),
                transforms.CenterCrop(imagesize),
            ]
        )
        self.imagesize = (3, imagesize, imagesize)

    @staticmethod
    def _hat_flags(dataset, num_proc):
        """Label d'anomalie par image, dérivé du masque. Le `map` ne porte que
        sur la colonne `mask` : le cache HF reste petit et le décodage des
        17k masques n'a lieu qu'une fois."""

        def has_hat(mask):
            pixels = np.asarray(mask)
            return {"is_anomaly": bool((pixels == _ANOMALY_LABEL).sum() >= _MIN_ANOMALY_PIXELS)}

        flags = dataset.select_columns(["mask"]).map(
            has_hat,
            input_columns=["mask"],
            remove_columns=["mask"],
            num_proc=num_proc,
            desc="Détection de la classe Hat dans les masques",
        )
        return flags["is_anomaly"]

    @staticmethod
    def _split_indices(flags, split, seed):
        """Découpe train/test puis, côté test, sous-échantillonne la classe
        majoritaire pour équilibrer hat/no-hat. Le train ne garde que le no-hat.

        Le mélange est fait par classe : la proportion de hat est donc la même
        dans les deux moitiés, quel que soit le seed.
        """
        rng = random.Random(seed)
        test_hat, test_no_hat, train_no_hat = [], [], []

        for is_hat in (True, False):
            indices = [i for i, hat in enumerate(flags) if hat == is_hat]
            rng.shuffle(indices)
            n_test = int(round(len(indices) * _TEST_FRACTION))
            if is_hat:
                test_hat = indices[:n_test]
            else:
                test_no_hat = indices[:n_test]
                train_no_hat = indices[n_test:]

        if split == DatasetSplit.TRAIN:
            return sorted(train_no_hat)

        n = min(len(test_hat), len(test_no_hat))
        return sorted(test_hat[:n] + test_no_hat[:n])

    @property
    def labels(self):
        """Labels d'anomalie (0/1) par image, sans décoder d'image. Même ordre
        que __getitem__."""
        return [int(hat) for hat in self.is_anomaly]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = self.transform_img(row["image"].convert("RGB"))
        is_anomaly = int(self.is_anomaly[idx])
        # Vérité terrain pixel : 1 sur le chapeau, 0 ailleurs.
        classes = np.asarray(self.transform_mask(row["mask"]))
        mask = torch.from_numpy(classes == _ANOMALY_LABEL).float().unsqueeze(0)

        return {
            "image": image,
            "mask": mask,
            "classname": "atr",
            "anomaly": "hat" if is_anomaly else "good",
            "is_anomaly": is_anomaly,
            "image_name": "atr/{}/{}".format(self.split.value, idx),
            "image_path": "",
        }
