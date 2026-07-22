"""Score le flux d'une webcam avec une banque mémoire PatchCore, en direct.

Même moitié rapide de PatchCore que bin/infer_heatmap_celeba.py — chargement
d'une banque déjà fittée, aucun fit ici — mais la source est une caméra plutôt
qu'un split de test. Le prétraitement (resize/imagesize) est relu du
fit_config.json de la banque : une requête encodée autrement que la banque
donnerait des distances qui ne veulent rien dire.

    python bin/live_camera.py                          # webcam 0, banque par défaut
    python bin/live_camera.py --bank_dir models/atr/…  # une autre banque
    python bin/live_camera.py --source rtsp://…        # caméra IP
    python bin/live_camera.py --source clip.mp4 --loop # rejouer un fichier

Touches : q/échap quitte, s écrit un instantané dans results/live/, espace met
en pause l'inférence (l'aperçu continue).

Attention au domaine de la banque : une banque CelebA est faite de visages
recadrés serré. Une webcam plein cadre est hors distribution et scorera haut
partout — cadrer un visage, ou utiliser --zoom, pour que les scores veuillent
dire quelque chose.
"""

import logging
import os
import platform
import time
from collections import deque

# Sur macOS, torch et faiss-cpu embarquent chacun leur libomp : la seconde à
# s'initialiser fait abort le processus, et une recherche faiss multi-thread
# segfault même avec KMP_DUPLICATE_LIB_OK. Tolérer le doublon ET forcer faiss
# sur un seul thread (plus bas, via FAISS_NUM_WORKERS) est la seule combinaison
# stable ici. Doit être posé avant l'import de torch.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import patchcore.banks
import patchcore.utils

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIG — surchargeable en ligne de commande.
# --------------------------------------------------------------------------- #
# La petite banque par défaut, et non la meilleure : la recherche faiss est
# exhaustive et linéaire en taille de banque, ce qui décide seul de la cadence.
# Mesuré ici (CPU, faiss mono-thread) : backbone ~55 ms quelle que soit la
# banque, puis 72 ms de recherche à 39 200 features contre 1 475 ms à 784 000,
# soit 8 fps contre 0,6. Le README donne 0,78 d'AUROC à cette taille contre 0,80
# à la plus grosse : deux points d'AUROC payés pour douze fois la cadence, le
# bon change pour du direct. Passer --bank_dir pour arbitrer autrement.
BANK_DIR = "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts500_s0"

GPU = [0]  # [] force le CPU (retombe sur CPU sans CUDA de toute façon).

# Échelle de couleur de la heatmap, en score de patch. Fixe (et non autoscale
# par image) : en direct, une échelle qui bouge d'une frame à l'autre donne une
# heatmap qui clignote et ne se compare pas à celle d'hier.
HEATMAP_VMIN = 0.0
HEATMAP_VMAX = 10.0
HEATMAP_ALPHA = 0.5

# Fenêtre de lissage du score affiché, en frames scorées. Le score PatchCore
# d'une frame isolée est bruité (bougé, autofocus, exposition) ; la médiane
# glissante enlève ce bruit sans traîner comme une moyenne.
SMOOTH_WINDOW = 5

SNAPSHOT_DIR = "results/live"

FAISS_ON_GPU = os.environ.get("INFER_FAISS_GPU", "").lower() in ("1", "true", "yes")
# 1 thread par défaut sur macOS (cf. le commentaire libomp en tête). Une requête
# d'une seule image contre un coreset ne perd de toute façon presque rien à être
# mono-thread : le coût est dans le backbone, pas dans la recherche.
FAISS_NUM_WORKERS = int(
    os.environ.get("INFER_FAISS_THREADS", "1" if platform.system() == "Darwin" else "4")
)
# --------------------------------------------------------------------------- #


def build_transform(fit_config):
    """Le prétraitement exact des datasets (celeba.py / atr.py), relu du fit."""
    return transforms.Compose(
        [
            transforms.Resize(fit_config["resize"]),
            transforms.CenterCrop(fit_config["imagesize"]),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )


def preprocess(frame_bgr, transform, zoom):
    """Frame OpenCV BGR -> (tenseur 1x3xHxW normalisé, vignette RGB affichable).

    La vignette est la frame telle que le réseau la voit (même recadrage) : la
    heatmap se superpose dessus sans réalignement approximatif.
    """
    if zoom > 1.0:
        h, w = frame_bgr.shape[:2]
        ch, cw = int(h / zoom), int(w / zoom)
        top, left = (h - ch) // 2, (w - cw) // 2
        frame_bgr = frame_bgr[top : top + ch, left : left + cw]

    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = transform(image)

    # Le même Resize+CenterCrop, mais sans normalisation, pour l'affichage.
    size = tensor.shape[-1]
    preview = transforms.functional.center_crop(
        transforms.functional.resize(image, transform.transforms[0].size), size
    )
    return tensor.unsqueeze(0), np.asarray(preview)


def render(preview_rgb, heatmap, score, fps, ms, threshold, paused):
    """Superpose la heatmap et l'ATH sur la vignette. Renvoie une image BGR."""
    normalized = np.clip(
        (heatmap - HEATMAP_VMIN) / max(HEATMAP_VMAX - HEATMAP_VMIN, 1e-6), 0, 1
    )
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    frame = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    overlay = cv2.addWeighted(colored, HEATMAP_ALPHA, frame, 1 - HEATMAP_ALPHA, 0)

    # Affichage confortable : la vignette fait 224 px de côté, illisible tel quel.
    overlay = cv2.resize(overlay, (640, 640), interpolation=cv2.INTER_NEAREST)

    if threshold is None:
        verdict, color = "score {:.2f}".format(score), (255, 255, 255)
    elif score >= threshold:
        verdict, color = "ANOMALIE {:.2f}".format(score), (0, 0, 255)
    else:
        verdict, color = "ok {:.2f}".format(score), (0, 200, 0)

    cv2.rectangle(overlay, (0, 0), (640, 40), (0, 0, 0), -1)
    cv2.putText(
        overlay, verdict, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2
    )
    cv2.putText(
        overlay,
        "{:.1f} fps  {:.0f} ms/inf{}".format(fps, ms, "  [pause]" if paused else ""),
        (330, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
    )
    return overlay


@click.command()
@click.option("--bank_dir", default=BANK_DIR, show_default=True,
              help="Dossier de banque écrit par un bin/fit_memory_bank_*.py.")
@click.option("--source", default="0", show_default=True,
              help="Index de webcam (0, 1, …), chemin de fichier vidéo ou URL RTSP/HTTP.")
@click.option("--stride", default=1, show_default=True,
              help="Ne score qu'une frame sur N ; l'aperçu garde la dernière "
                   "heatmap entre deux. Monter si l'affichage saccade.")
@click.option("--zoom", default=1.0, show_default=True,
              help="Recadrage centré avant prétraitement (2.0 = moitié centrale). "
                   "Sert à approcher le cadrage serré d'une banque de visages.")
@click.option("--threshold", type=float, default=None,
              help="Seuil de décision sur le score image. Sans seuil, seul le "
                   "score brut est affiché (cf. bin/score_histogram_celeba.py "
                   "pour en calibrer un).")
@click.option("--flip/--no-flip", default=True, show_default=True,
              help="Effet miroir sur l'aperçu, plus naturel face à une webcam.")
@click.option("--loop/--no-loop", default=False, show_default=True,
              help="Rejouer en boucle une source fichier (sans effet sur une caméra).")
def main(bank_dir, source, stride, zoom, threshold, flip, loop):
    device = patchcore.utils.set_torch_device(GPU)

    patchcore_instance, fit_config = patchcore.banks.load_bank(
        bank_dir, device, FAISS_ON_GPU, FAISS_NUM_WORKERS
    )
    patchcore.utils.fix_seeds(fit_config["seed"])
    transform = build_transform(fit_config)

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit(
            "Impossible d'ouvrir la source '{}'. Pour une webcam sur macOS, "
            "autoriser la caméra pour le terminal dans Réglages Système > "
            "Confidentialité et sécurité > Caméra, puis relancer.".format(source)
        )
    LOGGER.info("Source ouverte : %s. q pour quitter, s pour un instantané.", source)

    scores = deque(maxlen=SMOOTH_WINDOW)
    heatmap = np.zeros((fit_config["imagesize"], fit_config["imagesize"]), np.float32)
    score = float("nan")
    infer_ms = 0.0
    frame_index = 0
    paused = False
    last_tick = time.perf_counter()
    fps = 0.0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                if loop and not source.isdigit():
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                LOGGER.info("Fin du flux.")
                break
            if flip:
                frame = cv2.flip(frame, 1)

            tensor, preview = preprocess(frame, transform, zoom)

            if not paused and frame_index % stride == 0:
                t0 = time.perf_counter()
                batch_scores, batch_masks = patchcore_instance.predict(tensor)
                infer_ms = 1000.0 * (time.perf_counter() - t0)
                scores.append(float(batch_scores[0]))
                # Médiane sur la fenêtre : robuste aux frames aberrantes.
                score = float(np.median(scores))
                heatmap = np.asarray(batch_masks[0])

            now = time.perf_counter()
            # Lissage exponentiel du fps, sinon le chiffre affiché est illisible.
            fps = 0.9 * fps + 0.1 / max(now - last_tick, 1e-6)
            last_tick = now
            frame_index += 1

            overlay = render(preview, heatmap, score, fps, infer_ms, threshold, paused)
            cv2.imshow("PatchCore live", overlay)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                paused = not paused
            if key == ord("s"):
                os.makedirs(SNAPSHOT_DIR, exist_ok=True)
                path = os.path.join(
                    SNAPSHOT_DIR, "live_{}.png".format(time.strftime("%Y%m%d-%H%M%S"))
                )
                cv2.imwrite(path, overlay)
                LOGGER.info("Instantané écrit dans %s", path)
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
