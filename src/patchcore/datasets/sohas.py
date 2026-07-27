import logging
import os
import random
import xml.etree.ElementTree as ET
from enum import Enum

import numpy as np
import PIL.Image
import torch
from torchvision import transforms

LOGGER = logging.getLogger(__name__)

# Sous-ensemble SOHAS "Detection" : scènes entières (personne tenant un objet) +
# annotations Pascal-VOC. On vise le déploiement webcam (personne de face), donc
# les scènes entières, pas les crops centrés du sous-ensemble Classification.
#
# Arborescence attendue sous `source` :
#     images/            images_test/
#     annotations/xmls/  annotations_test/xmls/
_SUBSETS = [("annotations/xmls", "images"), ("annotations_test/xmls", "images_test")]

# Objets = anomalie. Tout autre objet manipulé (billete, smartphone, monedero,
# tarjeta) est du "normal" : c'est le sens de "train sans arme / test avec arme".
WEAPON_CLASSES = {"knife", "pistol"}

# Le repo ne fournit pas de split normal/anomalie one-class : on découpe nous-
# mêmes les frames sans arme en train/test, par seed (cf. atr.py).
_TEST_FRACTION = 0.2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


class SohasDataset(torch.utils.data.Dataset):
    """Dataset SOHAS one-class : la présence d'un bbox `knife`/`pistol` dans
    l'annotation sert de label d'anomalie. TRAIN = frames sans arme (pour la
    banque mémoire) ; TEST = échantillon équilibré good/weapon (pour l'évaluation).

    Comme MVTec, les images viennent du disque (`source`) ; comme atr, le label
    et le découpage train/test sont dérivés ici, déterministes par `seed`. Le
    bbox de l'arme fournit en prime une vérité terrain pixel (masque rasterisé),
    exploitable pour l'AUROC segmentation et pour valider que la heatmap se pose
    bien sur l'arme.
    """

    def __init__(
        self,
        source,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        seed=0,
        min_weapon_area_frac=None,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Racine du sous-ensemble Sohas_weapon-Detection (le
                    dossier qui contient images/, images_test/, annotations/…).
            resize: [int]. Taille (carrée) de redimensionnement initial.
            imagesize: [int]. Taille (carrée) du center-crop final.
            split: [DatasetSplit]. TRAIN (frames sans arme) ou TEST (équilibré).
                   TEST charge aussi le masque bbox.
            seed: [int]. Pilote le découpage train/test et l'équilibrage.
            min_weapon_area_frac: [float|None]. Ne garde comme `weapon` que les
                   frames où l'arme occupe >= cette fraction de l'aire du cadre.
                   Sert à scoper la tâche au domaine « gros plan » (SOHAS mélange
                   gros plans d'arme et scènes d'action, deux domaines distincts).
                   None -> lit SOHAS_MIN_WEAPON_AREA (défaut 0 = pas de filtre).
                   N'affecte que les frames armes, donc jamais la banque (good).
        """
        super().__init__()
        self.source = source
        self.split = split
        self.seed = seed
        if min_weapon_area_frac is None:
            min_weapon_area_frac = float(os.environ.get("SOHAS_MIN_WEAPON_AREA", "0"))
        self.min_weapon_area_frac = min_weapon_area_frac

        samples = self._index_frames()
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
        # NEAREST et même géométrie (Resize + CenterCrop) que l'image : le masque
        # doit rester aligné, et interpoler des 0/1 les rendrait ininterprétables.
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
    def _parse_xml(xml_path):
        """Label + bbox d'arme d'une frame, lus dans son XML Pascal-VOC. Le label
        d'anomalie vaut 1 dès qu'un objet est dans WEAPON_CLASSES."""
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(float(size.findtext("width"))) if size is not None else 0
        height = int(float(size.findtext("height"))) if size is not None else 0

        weapon_boxes = []
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name in WEAPON_CLASSES:
                box = obj.find("bndbox")
                weapon_boxes.append(
                    (
                        int(float(box.findtext("xmin"))),
                        int(float(box.findtext("ymin"))),
                        int(float(box.findtext("xmax"))),
                        int(float(box.findtext("ymax"))),
                    )
                )

        filename = root.findtext("filename") or (
            os.path.splitext(os.path.basename(xml_path))[0] + ".jpg"
        )
        return {
            "filename": filename,
            "width": width,
            "height": height,
            "weapon_boxes": weapon_boxes,
            "is_anomaly": int(len(weapon_boxes) > 0),
        }

    def _index_frames(self):
        """Parcourt les deux sous-ensembles (train + test du repo, poolés) et
        apparie chaque XML à son image. Les XML sans image sur disque sont
        ignorés : le repo prévient que certains fichiers manquent (anonymisation).
        """
        samples = []
        for ann_rel, img_rel in _SUBSETS:
            ann_dir = os.path.join(self.source, ann_rel)
            img_dir = os.path.join(self.source, img_rel)
            if not os.path.isdir(ann_dir):
                continue
            for xml_name in sorted(os.listdir(ann_dir)):
                if not xml_name.endswith(".xml"):
                    continue
                info = self._parse_xml(os.path.join(ann_dir, xml_name))
                img_path = os.path.join(img_dir, info["filename"])
                if not os.path.isfile(img_path):
                    alt = os.path.join(
                        img_dir, os.path.splitext(xml_name)[0] + ".jpg"
                    )
                    if not os.path.isfile(alt):
                        continue
                    img_path = alt
                samples.append(
                    {
                        "image_path": img_path,
                        "is_anomaly": info["is_anomaly"],
                        "weapon_boxes": info["weapon_boxes"],
                        "width": info["width"],
                        "height": info["height"],
                    }
                )
        if not samples:
            raise RuntimeError(
                "Aucune frame trouvée sous {}. Attendu : images/ + "
                "annotations/xmls/ (Sohas_weapon-Detection).".format(self.source)
            )

        if self.min_weapon_area_frac > 0:
            # On ne garde côté weapon que les frames où l'arme remplit assez le
            # cadre (gros plans) : les scènes d'action (arme minuscule) sont d'un
            # autre domaine et polluent la mesure. Le good n'est jamais touché.
            samples = [
                s for s in samples
                if not s["is_anomaly"]
                or self._weapon_area_frac(s) >= self.min_weapon_area_frac
            ]
        return samples

    @staticmethod
    def _weapon_area_frac(sample):
        """Fraction de l'aire du cadre couverte par le plus gros bbox d'arme."""
        area = sample["width"] * sample["height"]
        if area <= 0 or not sample["weapon_boxes"]:
            return 0.0
        largest = max(
            (x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in sample["weapon_boxes"]
        )
        return largest / area

    @staticmethod
    def _split(samples, split, seed):
        """Découpe les frames sans arme en train/test, envoie toutes les frames
        armes au test, puis équilibre le test (autant de good que de weapon). Le
        train ne garde que du good. Mélange par classe -> même proportion quel
        que soit le seed (cf. atr.py)."""
        rng = random.Random(seed)
        normal = [i for i, s in enumerate(samples) if not s["is_anomaly"]]
        weapon = [i for i, s in enumerate(samples) if s["is_anomaly"]]
        rng.shuffle(normal)
        rng.shuffle(weapon)

        n_test = int(round(len(normal) * _TEST_FRACTION))
        test_normal, train_normal = normal[:n_test], normal[n_test:]

        if split == DatasetSplit.TRAIN:
            return [samples[i] for i in sorted(train_normal)]

        n = min(len(test_normal), len(weapon))
        chosen = sorted(test_normal[:n] + weapon[:n])
        return [samples[i] for i in chosen]

    def _weapon_mask(self, sample):
        """Masque binaire PIL 'L' à la taille d'origine : 255 dans chaque bbox
        d'arme, 0 ailleurs."""
        arr = np.zeros((sample["height"], sample["width"]), dtype=np.uint8)
        for x0, y0, x1, y1 in sample["weapon_boxes"]:
            arr[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = 255
        return PIL.Image.fromarray(arr, mode="L")

    @property
    def labels(self):
        """Labels d'anomalie (0/1) par frame, sans décoder d'image. Même ordre
        que __getitem__."""
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
        )
        if has_mask:
            mask = self.transform_mask(self._weapon_mask(sample))
            mask = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
            mask = (mask > 0).float()
        else:
            mask = torch.zeros([1, *image.size()[1:]])

        return {
            "image": image,
            "mask": mask,
            "classname": "sohas",
            "anomaly": "weapon" if sample["is_anomaly"] else "good",
            "is_anomaly": sample["is_anomaly"],
            "image_name": os.path.basename(sample["image_path"]),
            "image_path": sample["image_path"],
        }
