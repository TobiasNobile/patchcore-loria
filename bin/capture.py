"""Constitue le jeu d'images de la scène de déploiement depuis la caméra.

    python bin/capture.py --out data/scene/normal  --count 400 --every 0.5
    python bin/capture.py --out data/scene/anomaly --count 60

Filmer la scène sans l'anomalie à détecter, sous toutes ses variations : tout ce
qui n'est pas dans la banque sera scoré comme anormal.
"""

import logging
import os
import platform
import time

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click
import cv2

LOGGER = logging.getLogger(__name__)


@click.command()
@click.option("--out", required=True,
              help="Dossier de destination (data/scene/normal ou .../anomaly).")
@click.option("--source", default="0", show_default=True,
              help="Index de webcam, chemin de fichier vidéo ou URL RTSP/HTTP.")
@click.option("--count", default=400, show_default=True, help="Nombre d'images.")
@click.option("--every", default=0.5, show_default=True,
              help="Secondes entre deux prises : laisse le temps de bouger la "
                   "caméra, des images consécutives n'apprennent rien de neuf.")
@click.option("--show/--no-show", default=True, show_default=True,
              help="Aperçu (q pour arrêter). --no-show pour une machine sans écran.")
def main(out, source, count, every, show):
    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit("Impossible d'ouvrir la source '{}'.".format(source))
    os.makedirs(out, exist_ok=True)

    saved, last = 0, 0.0
    try:
        while saved < count:
            ok, frame = capture.read()
            if not ok:
                LOGGER.warning("Fin du flux après %d images.", saved)
                break
            now = time.time()
            if now - last >= every:
                path = os.path.join(out, "cap_{}.jpg".format(int(now * 1000)))
                cv2.imwrite(path, frame)
                saved, last = saved + 1, now
                LOGGER.info("%d/%d -> %s", saved, count, path)
            if show:
                cv2.imshow("capture ({}/{}) - q pour arreter".format(saved, count), frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()
    LOGGER.info("%d images dans %s", saved, out)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
