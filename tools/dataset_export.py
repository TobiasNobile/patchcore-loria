"""Zip d'images « good » de MTD ou mini-ShanghaiTech, prêt pour l'interface web.

    python tools/dataset_export.py mtd
    python tools/dataset_export.py stc --source ~/Downloads/archive.zip --scene 01

MTD vient du dépôt GitHub d'origine. mSTC vient de la copie Kaggle, dont les
frames sont déjà extraites ; passer --source évite d'en télécharger les 11,7 Go
fichier par fichier, ce que l'API refuse au-delà de quelques centaines.

Une caméra par zip : mélanger des points de vue rend le normal hétérogène, et
tout se met à scorer haut, le fond compris.
"""

import base64
import io
import json
import logging
import os
import sys
import urllib.error
import urllib.parse
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor

import click

LOGGER = logging.getLogger(__name__)

MTD_URL = ("https://codeload.github.com/abin24/Magnetic-tile-defect-datasets."
           "/zip/refs/heads/master")
MTD_GOOD_DIR = "MT_Free/Imgs/"

STC_DATASET = "nikanvasei/shanghaitech-campus-dataset"
STC_ROOT = "SHANGHAI/SHANGHAI_TRAIN"
STC_INDEX = STC_ROOT + "/SHANGHAI_train.txt"
KAGGLE_FILE_URL = "https://www.kaggle.com/api/v1/datasets/download/{ds}?fileName={f}"


# ─── Magnetic Tile Defects ─────────────────────────────────────────────────

def _download(url, dest, auth=None):
    request = urllib.request.Request(url)
    if auth:
        request.add_header("Authorization", "Basic " + auth)
    with urllib.request.urlopen(request, timeout=600) as response:
        data = response.read()
    if dest:
        with open(dest, "wb") as fh:
            fh.write(data)
    return data


def export_mtd(out, cache, count):
    archive = os.path.join(cache, "mtd_source.zip")
    if not os.path.exists(archive):
        os.makedirs(cache, exist_ok=True)
        LOGGER.info("Téléchargement du dépôt MTD (~53 Mo)…")
        _download(MTD_URL, archive)

    with zipfile.ZipFile(archive) as src:
        # Imgs contient photos (.jpg) et masques (.png) ; seules les photos servent.
        members = [n for n in src.namelist()
                   if MTD_GOOD_DIR in n and n.lower().endswith(".jpg")]
        members.sort()
        if count:
            members = members[:count]
        LOGGER.info("%d images sans défaut retenues.", len(members))
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as dst:
            for name in members:
                dst.writestr("mtd_free/" + os.path.basename(name), src.read(name))
    return len(members)


# ─── mini ShanghaiTech Campus ──────────────────────────────────────────────

def _kaggle_auth():
    path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(path):
        raise SystemExit(
            "~/.kaggle/kaggle.json introuvable — nécessaire pour mSTC. "
            "Le créer depuis kaggle.com > Settings > API > Create New Token.")
    with open(path) as fh:
        creds = json.load(fh)
    raw = "{}:{}".format(creds["username"], creds["key"]).encode()
    return base64.b64encode(raw).decode()


class _RateLimiter:
    """Espace les requêtes, tous threads confondus.
    
    Kaggle répond 404 — pas 429 — dès qu'on pousse : 51 réussites sur 500 avec
    8 workers, contre 11 sur 12 en séquentiel. L'échec est muet.
    """

    def __init__(self, interval):
        self._interval = interval
        self._lock = threading.Lock()
        self._next = 0.0

    def wait(self):
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next - now)
            self._next = max(now, self._next) + self._interval
        if delay:
            time.sleep(delay)


def _stc_frame(auth, clip, index, limiter, retries=3):
    """Une frame par son numéro dans le clip. Un 404 est parfois transitoire, d'où
    les réessais avant de conclure qu'elle manque.
    """
    name = "{}/frames/{}/{:03d}.jpg".format(STC_ROOT, clip, index)
    url = KAGGLE_FILE_URL.format(ds=STC_DATASET, f=urllib.parse.quote(name))
    for attempt in range(1, retries + 1):
        limiter.wait()
        try:
            return _download(url, None, auth)
        except urllib.error.HTTPError:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def _archive_frames(archive, scene):
    """Les frames d'une archive STC non décompressée, groupées par clip."""
    clips = {}
    with zipfile.ZipFile(archive) as zf:
        for name in zf.namelist():
            if "/frames/" not in name or not name.lower().endswith(".jpg"):
                continue
            clip = name.split("/frames/", 1)[1].split("/")[0]
            if scene != "all" and not clip.startswith(scene + "_"):
                continue
            clips.setdefault(clip, []).append(name)
    for names in clips.values():
        names.sort()
    if not clips:
        raise SystemExit("Aucun clip pour la scène {} dans {}.".format(scene, archive))
    return clips


def _local_frames(source, scene):
    """Les frames d'une copie locale, groupées par clip. Trouve `frames/` à
    n'importe quelle profondeur sous la racine donnée.
    """
    root = None
    for base, dirs, _ in os.walk(source):
        if os.path.basename(base) == "frames":
            root = base
            break
    if root is None:
        raise SystemExit("Aucun dossier 'frames/' sous {}.".format(source))

    clips = {}
    for clip in sorted(os.listdir(root)):
        if scene != "all" and not clip.startswith(scene + "_"):
            continue
        path = os.path.join(root, clip)
        if not os.path.isdir(path):
            continue
        images = sorted(f for f in os.listdir(path) if f.lower().endswith(".jpg"))
        if images:
            clips[clip] = [os.path.join(path, f) for f in images]
    if not clips:
        raise SystemExit("Aucun clip pour la scène {} sous {}.".format(scene, root))
    return root, clips


def export_stc_local(out, source, scene, count, step):
    """Construit le zip depuis une copie locale, dossier ou archive. Seul chemin
    tenable au-delà de quelques centaines d'images.
    """
    from_zip = source.lower().endswith(".zip")
    if from_zip:
        clips = _archive_frames(source, scene)
        origin = source
    else:
        origin, clips = _local_frames(source, scene)
    total = sum(len(v) for v in clips.values())
    LOGGER.info("Source %s : %d clips, %d frames.", origin, len(clips), total)

    # Une frame sur `step` dans chaque clip, puis quota réparti : la banque doit voir toute la scène.
    wanted = []
    for clip in sorted(clips):
        wanted.extend(clips[clip][::step])
    if count and count < len(wanted):
        stride = len(wanted) / count
        wanted = [wanted[int(i * stride)] for i in range(count)]
    LOGGER.info("%d frames retenues (1 sur %d, réparties sur les clips).",
                len(wanted), step)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    src = zipfile.ZipFile(source) if from_zip else None
    try:
        with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED, allowZip64=True) as dst:
            for n, path in enumerate(wanted, 1):
                clip = path.split("/frames/", 1)[1].split("/")[0] if from_zip \
                    else os.path.basename(os.path.dirname(path))
                arc = "stc_scene{}/{}_{}".format(scene, clip, os.path.basename(path))
                if from_zip:
                    dst.writestr(arc, src.read(path))
                else:
                    dst.write(path, arc)
                if n % 2000 == 0:
                    LOGGER.info("%d/%d…", n, len(wanted))
    finally:
        if src is not None:
            src.close()
    return len(wanted)


def export_stc(out, cache, scene, count, step, workers, interval):
    auth = _kaggle_auth()
    limiter = _RateLimiter(interval)
    os.makedirs(cache, exist_ok=True)
    index_path = os.path.join(cache, "SHANGHAI_train.txt")
    if not os.path.exists(index_path):
        LOGGER.info("Téléchargement de l'index des clips…")
        _download(KAGGLE_FILE_URL.format(ds=STC_DATASET,
                                         f=urllib.parse.quote(STC_INDEX)),
                  index_path, auth)

    clips = []
    with open(index_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) >= 3:
                clips.append((os.path.basename(parts[0]), int(parts[2])))
    clips = [(c, n) for c, n in clips if c.startswith(scene + "_")]
    if not clips:
        raise SystemExit("Aucun clip pour la scène {}.".format(scene))
    LOGGER.info("Scène %s : %d clips, %d frames disponibles.",
                scene, len(clips), sum(n for _, n in clips))

    # Idem : réparti sur les clips plutôt que de vider le premier.
    wanted = []
    for clip, frames in clips:
        wanted.extend((clip, i) for i in range(0, frames, step))
    if count and count < len(wanted):
        stride = len(wanted) / count
        wanted = [wanted[int(i * stride)] for i in range(count)]
    LOGGER.info("%d frames à récupérer (1 sur %d).", len(wanted), step)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = [0]

    missing = [0]

    def fetch(item):
        clip, i = item
        try:
            return clip, i, _stc_frame(auth, clip, i, limiter)
        except urllib.error.HTTPError:  # frame réellement absente du miroir
            missing[0] += 1
            return clip, i, None

    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as dst:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for clip, i, data in pool.map(fetch, wanted):
                if data is None:
                    continue
                dst.writestr("stc_scene{}/{}_{:03d}.jpg".format(scene, clip, i), data)
                done[0] += 1
                if done[0] % 50 == 0:
                    LOGGER.info("%d/%d…", done[0], len(wanted))
    if missing[0]:
        LOGGER.warning("%d frames introuvables, ignorées.", missing[0])
    return done[0]


# ─── CLI ───────────────────────────────────────────────────────────────────

@click.group()
def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")


@main.command()
@click.option("--out", default="data/benchmarks/mtd_free.zip", show_default=True)
@click.option("--cache", default="data/benchmarks/.cache", show_default=True)
@click.option("--count", default=0, show_default=True,
              help="Nombre d'images ; 0 = toutes (952).")
def mtd(out, cache, count):
    """Magnetic Tile Defects — la classe MT_Free (tuiles sans défaut)."""
    n = export_mtd(out, cache, count)
    click.echo("{} images dans {} ({:.1f} Mo)".format(
        n, out, os.path.getsize(out) / 1e6))


@main.command()
@click.option("--scene", default="01", show_default=True,
              help="Caméra à exporter (01 à 13). Une seule par zip.")
@click.option("--count", default=500, show_default=True,
              help="Nombre de frames ; 0 = toutes celles retenues par --step.")
@click.option("--step", default=5, show_default=True,
              help="Une frame sur N, comme le mSTC de la littérature.")
@click.option("--out", default=None, help="Défaut : data/benchmarks/stc_scene<N>.zip")
@click.option("--cache", default="data/benchmarks/.cache", show_default=True)
@click.option("--source", default=None,
              help="Copie locale de STC (recommandé). Sans elle, les frames sont "
                   "tirées une par une de Kaggle, qui coupe au-delà de quelques "
                   "centaines.")
@click.option("--workers", default=2, show_default=True,
              help="Téléchargements simultanés. Au-delà de 2, Kaggle refuse.")
@click.option("--interval", default=0.8, show_default=True,
              help="Secondes entre deux requêtes, tous workers confondus.")
def stc(scene, count, step, out, cache, source, workers, interval):
    """mini ShanghaiTech Campus — les frames d'entraînement d'une caméra.
    
    `--scene all` traverse les 13 caméras, mais le normal devient hétérogène et
    PatchCore y perd : une caméra à la fois reste le réglage sain.
    """
    out = out or "data/benchmarks/stc_scene{}.zip".format(scene)
    if source:
        n = export_stc_local(out, source, scene, count, step)
    else:
        n = export_stc(out, cache, scene, count, step, workers, interval)
    click.echo("{} images dans {} ({:.1f} Mo)".format(
        n, out, os.path.getsize(out) / 1e6))


if __name__ == "__main__":
    main()
