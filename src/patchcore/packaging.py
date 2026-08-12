"""Le format `.pkg` : une banque mémoire tenant dans un seul fichier partageable.

Une banque produite par le fit est un dossier de trois fichiers (index faiss,
paramètres picklés, fit_config.json). C'est pratique à écrire, pénible à
distribuer. Un `.pkg` est simplement ce dossier zippé, nommé de façon à ce que
la configuration se lise dans le nom :

    coresets/WideResNet50_DetectionKnife_l3-l4_p0.01_ts20000.pkg

Le nom est un résumé ; la vérité reste le fit_config.json à l'intérieur, que
`read_config` relit sans extraire le reste (l'index pèse jusqu'au gigaoctet).
Le chargement, lui, extrait dans un cache temporaire : faiss lit un chemin,
pas un flux.
"""

import hashlib
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

# Ce que save_bank écrit ; tout le reste du dossier est ignoré à l'empaquetage.
MEMBERS = (
    patchcore.banks.CONFIG_FILENAME,
    "patchcore_params.pkl",
    "nnscorer_search_index.faiss",
)

# Noms d'affichage des backbones proposés. Les clés sont celles de
# patchcore.backbones ; la casse du nom de fichier vient d'ici.
BACKBONE_LABELS = {
    "wideresnet50": "WideResNet50",
    "resnet50": "ResNet50",
    "resnet34": "ResNet34",
    "resnet18": "ResNet18",
}


def backbone_label(name):
    return BACKBONE_LABELS.get(name, name)


def slugify(text, fallback="Tache"):
    """Un nom de tâche sûr dans un nom de fichier, sans écraser sa casse.

    Les séparateurs du nom de banque sont `_` et `-` : on ne peut pas les
    laisser passer depuis un champ libre, sinon le nom devient illisible."""
    kept = [c for c in (text or "").strip() if c.isalnum()]
    return "".join(kept) or fallback


def build_name(config, task):
    """`<Backbone>_<Tache>_<couches>_<coreset>_ts<n>.pkg`.

    Tout est dans le nom, y compris ce qui vaut le défaut : un fichier publié se
    lit sans ouvrir son contenu, et deux variantes d'une même tâche doivent
    cohabiter dans le dossier sans s'écraser.
    """
    layers = "-".join(
        l.replace("layer", "l") for l in config.get("layers_to_extract_from", [])
    ) or "l?"
    sampler = config.get("sampler_name", "identity")
    coreset = ("identity" if sampler == "identity"
               else "p{:g}".format(config.get("coreset_pct", 0)))
    subset = config.get("train_subset") or "all"
    return "{}_{}_{}_{}_ts{}{}".format(
        backbone_label(config.get("backbone_name", "?")),
        slugify(task), layers, coreset, subset, SUFFIX,
    )


def pack(bank_dir, pkg_path):
    """Zippe une banque déjà écrite sur disque. Renvoie pkg_path."""
    missing = [m for m in MEMBERS if not os.path.exists(os.path.join(bank_dir, m))]
    if missing:
        raise FileNotFoundError(
            "Banque incomplète dans {} : il manque {}.".format(
                bank_dir, ", ".join(missing)
            )
        )
    os.makedirs(os.path.dirname(pkg_path) or ".", exist_ok=True)
    # Écrit à côté puis renommé : un .pkg présent dans le dossier est toujours
    # complet, même si le processus meurt pendant l'écriture.
    tmp_path = pkg_path + ".part"
    # ZIP_STORED : l'index faiss est du float32 dense, le compresser gagne ~2 %
    # pour plusieurs secondes de CPU par banque.
    with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for member in MEMBERS:
            zf.write(os.path.join(bank_dir, member), member)
    os.replace(tmp_path, pkg_path)
    LOGGER.info("Packed %s -> %s", bank_dir, pkg_path)
    return pkg_path


def read_config(pkg_path):
    """Le fit_config.json d'un .pkg, sans extraire l'index."""
    with zipfile.ZipFile(pkg_path) as zf:
        with zf.open(patchcore.banks.CONFIG_FILENAME) as fh:
            return json.load(fh)


def _cache_dir(pkg_path):
    """Un dossier par (chemin, taille, mtime) : un .pkg réécrit n'hérite pas de
    l'extraction du précédent."""
    stat = os.stat(pkg_path)
    key = "{}:{}:{}".format(os.path.abspath(pkg_path), stat.st_size, stat.st_mtime_ns)
    digest = hashlib.sha1(key.encode()).hexdigest()[:16]
    stem = os.path.basename(pkg_path)[: -len(SUFFIX)]
    return os.path.join(tempfile.gettempdir(), "patchcore-pkg", stem + "-" + digest)


def extract(pkg_path):
    """Extrait le .pkg dans un cache et renvoie le dossier de banque.

    Réutilise l'extraction précédente du même fichier : recharger une banque de
    0,7 Go à chaque démarrage coûterait quelques secondes pour rien.
    """
    target = _cache_dir(pkg_path)
    done = os.path.join(target, ".complete")
    if os.path.exists(done):
        return target
    # Une extraction interrompue laisse un dossier partiel sans témoin : on le
    # jette plutôt que de charger une banque tronquée.
    shutil.rmtree(target, ignore_errors=True)
    os.makedirs(target, exist_ok=True)
    with zipfile.ZipFile(pkg_path) as zf:
        for member in MEMBERS:
            zf.extract(member, target)
    open(done, "w").close()
    LOGGER.info("Extracted %s -> %s", pkg_path, target)
    return target


def load(pkg_path, device, faiss_on_gpu=False, faiss_num_workers=4):
    """Comme banks.load_bank, depuis un .pkg."""
    return patchcore.banks.load_bank(
        extract(pkg_path), device, faiss_on_gpu, faiss_num_workers
    )


def find(root=CORESETS_DIR):
    """Les .pkg de `root`, avec leur config, triés par nom.

    Un .pkg illisible est ignoré plutôt que fatal : un téléchargement à moitié
    fini dans le dossier ne doit pas empêcher de démarrer.
    """
    banks = []
    if not os.path.isdir(root):
        return banks
    for name in sorted(os.listdir(root)):
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
