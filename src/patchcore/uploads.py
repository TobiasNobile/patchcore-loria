"""Dépaquetage d'une archive d'images envoyée par l'interface.

Sortie : l'arborescence attendue par experiments.folder.FolderDataset, soit
`<dest>/normal/` et `<dest>/anomaly/` si l'archive en fournit. Une archive plate
est traitée comme du normal.
"""

import logging
import os
import zipfile

LOGGER = logging.getLogger(__name__)

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
NORMAL, ANOMALY = "normal", "anomaly"
MAX_FILES = 200_000


class UploadError(Exception):
    """Archive inutilisable — le message est montré tel quel dans la page."""


def _classify(member_name):
    """(sous-dossier, nom de fichier), ou None si le membre est à ignorer."""
    # Antislashs des zips Windows, et dossier de métadonnées __MACOSX.
    parts = [p for p in member_name.replace("\\", "/").split("/") if p]
    if not parts or parts[0] == "__MACOSX":
        return None
    base = parts[-1]
    if base.startswith(".") or not base.lower().endswith(EXTENSIONS):
        return None
    folder = ANOMALY if any(p.lower() == ANOMALY for p in parts[:-1]) else NORMAL
    return folder, base


def _safe_target(dest_dir, folder, base, seen):
    """Chemin de sortie garanti sous dest_dir, et unique. `base` est déjà réduit à
    un nom de fichier ; on le revérifie plutôt que d'en dépendre.
    """
    stem, ext = os.path.splitext(base)
    name, i = base, 1
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
            raise UploadError("Archive à {} entrées, au-delà de {}.".format(
                len(members), MAX_FILES))
        for member in members:
            classified = _classify(member.filename)
            if classified is None:
                continue
            folder, base = classified
            target = _safe_target(dest_dir, folder, base, seen)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # Par blocs : un membre décompressé peut être bien plus gros que son
            # entrée dans l'archive.
            with archive.open(member) as src, open(target, "wb") as out:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
            counts[folder] += 1

    if not counts[NORMAL]:
        raise UploadError("Aucune image exploitable (extensions : {}).".format(
            ", ".join(EXTENSIONS)))
    LOGGER.info("Extracted %d normal / %d anomaly to %s",
                counts[NORMAL], counts[ANOMALY], dest_dir)
    return counts
