import logging
import random
from enum import Enum

import torch
from datasets import load_dataset
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

_HF_REPO = "flwrlabs/celeba"
_ANOMALY_ATTRIBUTE = "Wearing_Hat"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class CelebADataset(torch.utils.data.Dataset):
    """
    One-class PyTorch Dataset for CelebA, using the `Wearing_Hat` attribute
    as the anomaly label:
      - TRAIN: only "normal" (no-hat) images from the CelebA `train` split.
               This is what the PatchCore memory bank is built from.
      - TEST: a class-balanced hat / no-hat sample from the CelebA `test`
              split, for evaluation.
    """

    def __init__(
        self,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        seed=0,
        **kwargs,
    ):
        super().__init__()
        self.split = split
        self.seed = seed

        dataset = load_dataset(_HF_REPO, split=split.value)
        if split == DatasetSplit.TRAIN:
            dataset = dataset.filter(
                lambda hat: not hat, input_columns=[_ANOMALY_ATTRIBUTE]
            )
        else:
            dataset = self._balance(dataset, seed)
        self.dataset = dataset

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
        self.imagesize = (3, imagesize, imagesize)

    @staticmethod
    def _balance(dataset, seed):
        """Subsamples the majority class so hat / no-hat counts match."""
        is_hat = dataset[_ANOMALY_ATTRIBUTE]
        hat_idx = [i for i, hat in enumerate(is_hat) if hat]
        no_hat_idx = [i for i, hat in enumerate(is_hat) if not hat]

        n = min(len(hat_idx), len(no_hat_idx))
        no_hat_idx = random.Random(seed).sample(no_hat_idx, n)

        return dataset.select(sorted(hat_idx + no_hat_idx))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        row = self.dataset[idx]
        image = self.transform_img(row["image"].convert("RGB"))
        is_anomaly = int(row[_ANOMALY_ATTRIBUTE])
        mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": "celeba",
            "anomaly": "hat" if is_anomaly else "good",
            "is_anomaly": is_anomaly,
            "image_name": "celeba/{}/{}".format(self.split.value, idx),
            "image_path": "",
        }
