"""Format `.pkg` : une banque mémoire dans un seul fichier partageable.

Un `.pkg` est le dossier de banque zippé, nommé pour que sa configuration se
lise sans l'ouvrir :

    coresets/WideResNet50_DetectionKnife_l3-l4_p0.01_ts20000_s0.pkg

Le nom n'est qu'un résumé ; la référence reste le fit_config.json à l'intérieur.
"""

import json
import logging
import os
import shutil
import tempfile
import zipfile

import patchcore.banks

LOGGER = logging.getLogger(__name__)

SUFFIX = ".pkg"
CORESETS_DIR = "coresets"

# Ce que save_bank écrit ; le reste du dossier est ignoré à l'empaquetage.
MEMBERS = (
    patchcore.banks.CONFIG_FILENAME,
    "patchcore_params.pkl",
    "nnscorer_search_index.faiss",
)

BACKBONE_LABELS = {
    "wideresnet50": "WideResNet50",
    "resnet50": "ResNet50",
    "resnet34": "ResNet34",
    "resnet18": "ResNet18",
}


def backbone_label(name):
    return BACKBONE_LABELS.get(name, name)


def slugify(text, fallback="Tache"):
    """Nom de tâche sûr dans un nom de fichier. `_` et `-` séparent les champs
    du nom de banque, donc aucun ne peut venir d'un champ libre."""
    return "".join(c for c in (text or "").strip() if c.isalnum()) or fallback


def build_name(config, task):
    """`<Backbone>_<Tache>_<couches>[_im<px>]_<coreset>_ts<n>_s<seed>.pkg`.

    Tout y figure, defauts compris : deux variantes d'une même tâche doivent
    cohabiter sans s'écraser, seeds inclus. La taille d'image manquait, et
    trois banques ne différant que par elle — 224, 160, 128 px — tombaient
    toutes sur le même nom. Suffixée seulement hors du défaut, comme dans
    build_tag : les banques déjà empaquetées gardent le leur."""
    layers = "-".join(
        l.replace("layer", "l") for l in config.get("layers_to_extract_from", [])
    ) or "l?"
    sampler = config.get("sampler_name", "identity")
    coreset = ("identity" if sampler == "identity"
               else "p{:g}".format(config.get("coreset_pct", 0)))
    size = "" if config.get("imagesize", 224) == 224 else "_im{}".format(
        config["imagesize"])
    return "{}_{}_{}{}_{}_ts{}_s{}{}".format(
        backbone_label(config.get("backbone_name", "?")), slugify(task), layers,
        size, coreset, config.get("train_subset") or "all",
        config.get("seed", 0), SUFFIX,
    )


def pack(bank_dir, pkg_path):
    """Zippe une banque écrite sur disque. Renvoie pkg_path."""
    missing = [m for m in MEMBERS if not os.path.exists(os.path.join(bank_dir, m))]
    if missing:
        raise FileNotFoundError("Banque incomplète dans {} : il manque {}.".format(
            bank_dir, ", ".join(missing)))
    os.makedirs(os.path.dirname(pkg_path) or ".", exist_ok=True)
    # Écrit à côté puis renommé : un .pkg présent est toujours complet.
    tmp = pkg_path + ".part"
    # ZIP_STORED : l'index faiss est du float32 dense, compresser ne gagne rien.
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for member in MEMBERS:
            zf.write(os.path.join(bank_dir, member), member)
    os.replace(tmp, pkg_path)
    LOGGER.info("Packed %s -> %s", bank_dir, pkg_path)
    return pkg_path


def read_config(pkg_path):
    """Le fit_config.json, sans extraire l'index."""
    with zipfile.ZipFile(pkg_path) as zf:
        with zf.open(patchcore.banks.CONFIG_FILENAME) as fh:
            return json.load(fh)


def extract(pkg_path):
    """Extrait dans un cache et renvoie le dossier de banque.

    Un dossier par banque, réutilisé tant qu'il est plus récent que le .pkg :
    faiss lit un chemin, pas un flux, et réextraire 750 Mo à chaque démarrage
    coûterait pour rien."""
    stem = os.path.basename(pkg_path)[: -len(SUFFIX)]
    target = os.path.join(tempfile.gettempdir(), "patchcore-pkg", stem)
    if os.path.isdir(target) and os.path.getmtime(target) > os.path.getmtime(pkg_path):
        return target
    shutil.rmtree(target, ignore_errors=True)
    with zipfile.ZipFile(pkg_path) as zf:
        for member in MEMBERS:
            zf.extract(member, target)
    os.utime(target)
    LOGGER.info("Extracted %s -> %s", pkg_path, target)
    return target


def load(pkg_path, device, faiss_on_gpu=False, faiss_num_workers=4):
    """Comme banks.load_bank, depuis un .pkg."""
    return patchcore.banks.load_bank(
        extract(pkg_path), device, faiss_on_gpu, faiss_num_workers)


def find(root=CORESETS_DIR):
    """Les .pkg de `root` avec leur config. Un fichier illisible est ignoré :
    un téléchargement à moitié fini ne doit pas empêcher de démarrer."""
    banks = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        if not name.endswith(SUFFIX):
            continue
        path = os.path.join(root, name)
        try:
            config = read_config(path)
        except (OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
            LOGGER.warning("Ignoring %s: %s", path, exc)
            continue
        banks.append({"path": path, "name": name[: -len(SUFFIX)], "config": config})
    return banks
