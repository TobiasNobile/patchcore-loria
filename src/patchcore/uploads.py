"""Dépaquetage d'une archive d'images d'entraînement envoyée par l'interface.

L'archive vient d'un navigateur : on ne sait rien de son contenu, et le zip est
un format qui permet d'écrire hors du dossier d'extraction (chemins absolus ou
remontants — « zip slip »). Chaque membre est donc revalidé ici plutôt que
confié à `ZipFile.extractall`.

Sortie : l'arborescence attendue par experiments.folder.FolderDataset,

    <dest>/normal/   images du fonctionnement nominal
    <dest>/anomaly/  contre-exemples, seulement si l'archive en fournit

Une archive plate (que des images à la racine) est traitée comme du normal :
c'est le cas courant, « voici mes images d'entraînement ».
"""

import logging
import os
import zipfile

LOGGER = logging.getLogger(__name__)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
NORMAL, ANOMALY = "normal", "anomaly"

# Au-delà, l'archive est probablement autre chose qu'un jeu d'images ; refuser
# tôt vaut mieux que remplir le disque puis échouer.
MAX_FILES = 200_000


class UploadError(Exception):
    """Archive inutilisable — le message est montré tel quel dans la page."""


def _classify(member_name):
    """(sous-dossier, nom de fichier) ou None si le membre est à ignorer.

    Le classement suit le premier segment du chemin dans l'archive : un zip
    fait de normal/ + anomaly/ garde sa séparation, tout le reste est du normal.
    """
    # Les zips produits sous Windows utilisent parfois des antislashs, et macOS
    # ajoute un dossier __MACOSX/ de métadonnées qui n'est pas des images.
    parts = [p for p in member_name.replace("\\", "/").split("/") if p]
    if not parts or parts[0] == "__MACOSX":
        return None
    base = parts[-1]
    if base.startswith(".") or not base.lower().endswith(EXTENSIONS):
        return None
    folder = ANOMALY if any(p.lower() == ANOMALY for p in parts[:-1]) else NORMAL
    return folder, base


def _safe_target(dest_dir, folder, base, seen):
    """Chemin de sortie garanti sous dest_dir, et unique.

    `base` est réduit à son nom de fichier par _classify, ce qui neutralise
    déjà « .. » et les chemins absolus ; on le revérifie plutôt que de faire
    reposer la sécurité sur une fonction distante. Deux sous-dossiers de
    l'archive peuvent porter le même nom de fichier, d'où le compteur.
    """
    name = base
    stem, ext = os.path.splitext(base)
    i = 1
    while (folder, name) in seen:
        name = "{}_{}{}".format(stem, i, ext)
        i += 1
    seen.add((folder, name))

    root = os.path.realpath(os.path.join(dest_dir, folder))
    target = os.path.realpath(os.path.join(root, name))
    if os.path.commonpath([root, target]) != root:
        raise UploadError("Chemin refusé dans l'archive : {}".format(base))
    return target


def extract_images(zip_path, dest_dir):
    """Extrait les images de `zip_path` sous `dest_dir`. Renvoie {dossier: n}."""
    try:
        archive = zipfile.ZipFile(zip_path)
    except zipfile.BadZipFile as exc:
        raise UploadError("Archive illisible : {}".format(exc))

    counts = {NORMAL: 0, ANOMALY: 0}
    seen = set()
    with archive:
        members = [m for m in archive.infolist() if not m.is_dir()]
        if len(members) > MAX_FILES:
            raise UploadError(
                "Archive à {} entrées, au-delà de la limite de {}.".format(
                    len(members), MAX_FILES
                )
            )
        for member in members:
            classified = _classify(member.filename)
            if classified is None:
                continue
            folder, base = classified
            target = _safe_target(dest_dir, folder, base, seen)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            with archive.open(member) as src, open(target, "wb") as out:
                # Copie par blocs : un membre décompressé peut être bien plus
                # gros que son entrée dans l'archive.
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            counts[folder] += 1

    if not counts[NORMAL]:
        raise UploadError(
            "Aucune image exploitable dans l'archive (extensions acceptées : "
            "{}).".format(", ".join(EXTENSIONS))
        )
    LOGGER.info(
        "Extracted %d normal / %d anomaly images to %s",
        counts[NORMAL], counts[ANOMALY], dest_dir,
    )
    return counts
