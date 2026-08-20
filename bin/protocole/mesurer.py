"""Score un clip labellisé, image par image, et range les scores par condition.

    python bin/protocole/mesurer.py --nom scene1

Hors ligne plutôt que dans la page : la page cadence la lecture sur l'horloge et
saute des frames quand l'inférence traîne, ce qui convient à une démo et pas à
une mesure. Ici toutes les frames retenues sont scorées, dans l'ordre.

Une seule passe d'inférence sert les deux références. La normalisation n'est
qu'une arithmétique sur des scores déjà calculés : rejouer les deux échelles sur
les mêmes scores garantit que l'écart mesuré vient de l'échelle, pas du bruit
d'inférence.

Écrit results/protocole/<nom>.jsonl, une ligne par frame scorée.
"""

import json
import logging
import os
import platform
import sys
import time

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import click
import cv2
import numpy as np
import torch  # noqa: F401  avant patchcore/faiss : l'ordre inverse fait abort libomp

from live_camera import build_transform, preprocess, stats_carte  # noqa: E402

import patchcore.banks  # noqa: E402
import patchcore.utils  # noqa: E402
from experiments.datasets import SCENE  # noqa: E402
from experiments.pipelines import run_fit  # noqa: E402
from experiments.runtime import resolve_device, tune_faiss_small_batches  # noqa: E402

LOGGER = logging.getLogger(__name__)


def _banque(nom, dossier_enrolement, coreset_pct):
    """Fitte sur le premier segment du clip, ou réutilise la banque déjà là."""
    racine = os.path.join("models", "protocole", nom)
    if os.path.isdir(racine) and os.listdir(racine):
        tag = sorted(os.listdir(racine))[0]
        LOGGER.info("Banque déjà présente : %s", os.path.join(racine, tag))
        return os.path.join(racine, tag)
    os.environ["SCENE_PATH"] = dossier_enrolement
    return run_fit(SCENE, models_dir=racine, coreset_pct=coreset_pct, num_workers=0,
                   extra_config={"source": dossier_enrolement, "task": "Protocole"})


def _condition(labels, index):
    """Par numéro de frame, jamais par horodatage.

    Une webcam annonce souvent un fps qu'elle ne tient pas : le conteneur est
    alors encodé à la mauvaise cadence et sa timeline ne correspond plus au
    temps réel de la prise. Le numéro de frame, lui, est le même des deux côtés.
    """
    for seg in labels["segments"]:
        if seg["frame0"] <= index < seg["frame1"]:
            return seg["condition"]
    return labels["segments"][-1]["condition"]


@click.command()
@click.option("--nom", required=True)
@click.option("--dossier", default="data/protocole", show_default=True)
@click.option("--stride", default=2, show_default=True,
              help="Ne score qu'une frame sur N. 2 à 30 fps = 15 mesures/s.")
@click.option("--coreset_pct", default=0.1, show_default=True)
@click.option("--device", default="auto", show_default=True)
def main(nom, dossier, stride, coreset_pct, device):
    with open(os.path.join(dossier, nom + ".labels.json")) as fh:
        labels = json.load(fh)

    bank_dir = _banque(nom, os.path.join(dossier, nom + "_enrolement"), coreset_pct)
    tune_faiss_small_batches()
    dev, note = resolve_device(device)
    LOGGER.info("Device : %s (%s)", dev, note)
    instance, fit_config = patchcore.banks.load_bank(bank_dir, dev, False, 1)
    patchcore.utils.fix_seeds(fit_config["seed"])
    transform = build_transform(fit_config)

    capture = cv2.VideoCapture(labels["clip"])
    if not capture.isOpened():
        raise SystemExit("Clip introuvable : " + labels["clip"])

    sortie = os.path.join("results", "protocole", nom + ".jsonl")
    os.makedirs(os.path.dirname(sortie), exist_ok=True)
    fh = open(sortie, "w", buffering=1)

    index, scorees, t0 = 0, 0, time.perf_counter()
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % stride == 0:
            pos_s = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            tensor, _ = preprocess(frame, transform, 1.0)
            scores, cartes = instance.predict(tensor)
            carte = np.asarray(cartes[0], dtype=np.float32)
            mediane, sigma = stats_carte(carte)
            fh.write(json.dumps({
                "frame": index,
                "pos_s": round(pos_s, 3),
                "condition": _condition(labels, index),
                "med": round(mediane, 4),
                "sigma": round(sigma, 4),
                "q99": round(float(np.quantile(carte, 0.99)), 4),
                "max": round(float(carte.max()), 4),
                "score": round(float(scores[0]), 4),
            }) + "\n")
            scorees += 1
            if scorees % 100 == 0:
                LOGGER.info("%d frames scorées (%.0f%%)", scorees,
                            100.0 * index / max(labels["frames"], 1))
        index += 1

    capture.release()
    fh.close()
    LOGGER.info("%d frames scorées en %.0f s -> %s",
                scorees, time.perf_counter() - t0, sortie)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                        datefmt="%H:%M:%S")
    main()
