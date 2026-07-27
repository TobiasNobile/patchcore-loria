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

# Jeu de détection d'armes « en scène » exporté depuis Roboflow au format
# Pascal-VOC (personnes tenant ou non une arme). Contrairement à SOHAS — dont le
# « normal » est un gros plan d'objet (carte, billet…) et non une personne — on
# exige ici la présence d'une PERSONNE dans le normal, pour que la banque
# apprenne « personne sans arme » et non « fond vide ». C'est le use case CelebA
# transposé : fit sur personnes sans arme, test personnes avec/sans arme.
#
# Arborescence attendue sous `source` (export Roboflow « Pascal VOC ») :
#     train/  valid/  test/     # chaque dossier : img.jpg + img.xml côte à côte
# On met en commun les trois splits (le découpage Roboflow est pensé pour la
# détection supervisée, pas pour le one-class) puis on re-découpe nous-mêmes.

# Objet = anomalie. Surchargeable par WEAPON_CLASSES (liste séparée par virgules)
# pour coller aux noms exacts de l'export Roboflow.
_DEFAULT_WEAPON_CLASSES = {"knife", "pistol", "gun", "handgun", "rifle", "weapon"}
# Classe « personne » : ce qui rend une frame sans arme exploitable comme normal.
_DEFAULT_PERSON_CLASSES = {"person", "people", "human", "pedestrian"}

# Le repo ne fournit pas de split normal/anomalie one-class : on découpe nous-
# mêmes les frames sans arme en train/test, par seed (cf. sohas.py / atr.py).
_TEST_FRACTION = 0.2

_IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"


def _parse_class_set(env_name, default):
    """Lit un set de noms de classes depuis une variable d'env (CSV), sinon
    renvoie le défaut. Normalisé en minuscules."""
    raw = os.environ.get(env_name)
    if not raw:
        return set(default)
    return {c.strip().lower() for c in raw.split(",") if c.strip()}


class WeaponDataset(torch.utils.data.Dataset):
    """Dataset armes-en-scène one-class (export Roboflow Pascal-VOC). La présence
    d'un bbox arme (knife/pistol/…) dans l'annotation sert de label d'anomalie.
    TRAIN = frames avec personne SANS arme (pour la banque mémoire) ; TEST =
    échantillon équilibré good/weapon (pour l'évaluation).

    Comme SOHAS, les images viennent du disque et le bbox de l'arme fournit une
    vérité terrain pixel (masque rasterisé) pour l'AUROC segmentation. La
    différence clé : le normal exige une personne, pas seulement l'absence
    d'arme — sinon la banque apprendrait des fonds vides.
    """

    def __init__(
        self,
        source,
        resize=256,
        imagesize=224,
        split=DatasetSplit.TRAIN,
        seed=0,
        require_person=None,
        min_weapon_area_frac=None,
        **kwargs,
    ):
        """
        Args:
            source: [str]. Racine de l'export Roboflow Pascal-VOC (contient les
                    sous-dossiers train/valid/test, ou directement les .xml/.jpg).
            resize: [int]. Taille (carrée) de redimensionnement initial.
            imagesize: [int]. Taille (carrée) du center-crop final.
            split: [DatasetSplit]. TRAIN (personnes sans arme) ou TEST (équilibré).
            seed: [int]. Pilote le découpage train/test et l'équilibrage.
            require_person: [bool|None]. Si True, une frame n'entre dans le normal
                    que si elle contient une personne annotée. None -> lit
                    WEAPON_REQUIRE_PERSON (défaut 1). Désactivé automatiquement si
                    le jeu n'annote aucune personne (sinon le normal serait vide).
            min_weapon_area_frac: [float|None]. Ne garde comme `weapon` que les
                    frames où l'arme occupe >= cette fraction du cadre (scoper au
                    domaine « gros plan »). None -> WEAPON_MIN_AREA (défaut 0).
                    N'affecte que les frames armes, jamais la banque.
        """
        super().__init__()
        self.source = source
        self.split = split
        self.seed = seed

        self.weapon_classes = _parse_class_set("WEAPON_CLASSES", _DEFAULT_WEAPON_CLASSES)
        self.person_classes = _parse_class_set("PERSON_CLASSES", _DEFAULT_PERSON_CLASSES)

        if require_person is None:
            require_person = os.environ.get("WEAPON_REQUIRE_PERSON", "1").lower() not in (
                "0", "false", "no",
            )
        self.require_person = require_person

        if min_weapon_area_frac is None:
            min_weapon_area_frac = float(os.environ.get("WEAPON_MIN_AREA", "0"))
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

    def _iter_xml_dirs(self):
        """Dossiers contenant des .xml : les sous-dossiers de split (train/valid/
        test…) s'ils existent, sinon la racine elle-même (export à plat)."""
        found = []
        for dp, _, fs in os.walk(self.source):
            if any(f.endswith(".xml") for f in fs):
                found.append(dp)
        return sorted(found)

    def _resolve_image(self, xml_dir, xml_name, filename):
        """Trouve l'image associée à un XML dans le même dossier : d'abord le
        <filename> déclaré, sinon le basename du XML avec les extensions usuelles."""
        candidates = []
        if filename:
            candidates.append(filename)
        stem = os.path.splitext(xml_name)[0]
        candidates += [stem + ext for ext in _IMG_EXTS]
        for cand in candidates:
            path = os.path.join(xml_dir, cand)
            if os.path.isfile(path):
                return path
        return None

    def _parse_xml(self, xml_path):
        """Label + bbox d'arme + présence d'une personne, lus dans le XML
        Pascal-VOC. Le label d'anomalie vaut 1 dès qu'un objet est dans
        weapon_classes."""
        root = ET.parse(xml_path).getroot()
        size = root.find("size")
        width = int(float(size.findtext("width"))) if size is not None else 0
        height = int(float(size.findtext("height"))) if size is not None else 0

        weapon_boxes = []
        has_person = False
        for obj in root.findall("object"):
            name = (obj.findtext("name") or "").strip().lower()
            if name in self.person_classes:
                has_person = True
            if name in self.weapon_classes:
                box = obj.find("bndbox")
                if box is None:
                    continue
                weapon_boxes.append(
                    (
                        int(float(box.findtext("xmin"))),
                        int(float(box.findtext("ymin"))),
                        int(float(box.findtext("xmax"))),
                        int(float(box.findtext("ymax"))),
                    )
                )

        filename = root.findtext("filename") or ""
        return {
            "filename": filename,
            "width": width,
            "height": height,
            "weapon_boxes": weapon_boxes,
            "has_person": has_person,
            "is_anomaly": int(len(weapon_boxes) > 0),
        }

    def _index_frames(self):
        """Parcourt tous les splits Roboflow (poolés) et apparie chaque XML à son
        image. Les XML sans image sur disque sont ignorés."""
        samples = []
        any_person = False
        for xml_dir in self._iter_xml_dirs():
            for xml_name in sorted(os.listdir(xml_dir)):
                if not xml_name.endswith(".xml"):
                    continue
                info = self._parse_xml(os.path.join(xml_dir, xml_name))
                img_path = self._resolve_image(xml_dir, xml_name, info["filename"])
                if img_path is None:
                    continue
                any_person = any_person or info["has_person"]
                samples.append(
                    {
                        "image_path": img_path,
                        "is_anomaly": info["is_anomaly"],
                        "weapon_boxes": info["weapon_boxes"],
                        "has_person": info["has_person"],
                        "width": info["width"],
                        "height": info["height"],
                    }
                )
        if not samples:
            raise RuntimeError(
                "Aucune frame trouvée sous {}. Attendu : un export Roboflow "
                "Pascal-VOC (train/valid/test avec .jpg + .xml).".format(self.source)
            )

        # Si le jeu n'annote jamais de personne, exiger une personne viderait le
        # normal : on retombe sur « toute frame sans arme = normal » en prévenant.
        if self.require_person and not any_person:
            LOGGER.warning(
                "require_person=True mais aucune classe personne (%s) trouvée dans "
                "les annotations -> normal = toute frame sans arme.",
                sorted(self.person_classes),
            )
            self.require_person = False

        if self.min_weapon_area_frac > 0:
            # Côté weapon seulement : ne garder que les gros plans d'arme.
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

    def _split(self, samples, split, seed):
        """Découpe les frames normales (personne, sans arme) en train/test, envoie
        toutes les frames armes au test, puis équilibre le test. Le train ne garde
        que du normal. Mélange par classe -> même proportion quel que soit le seed.
        """
        rng = random.Random(seed)
        normal = [
            i for i, s in enumerate(samples)
            if not s["is_anomaly"] and (s["has_person"] or not self.require_person)
        ]
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
            "classname": "weapon",
            "anomaly": "weapon" if sample["is_anomaly"] else "good",
            "is_anomaly": sample["is_anomaly"],
            "image_name": os.path.basename(sample["image_path"]),
            "image_path": sample["image_path"],
        }
