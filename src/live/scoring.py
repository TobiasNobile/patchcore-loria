"""Ce qui se passe sur une frame : prétraitement, rendu, agrégation, faiss.

Séparé de `live.server`, qui orchestre requêtes et threads : ces fonctions-ci ne
connaissent qu'une image et une banque. `bin/live_camera.py` les partage.
"""

import os
import platform

# macOS : torch et faiss embarquent chacun leur libomp, la seconde à
# s'initialiser fait abort. À poser avant l'import de torch, donc avant tout le
# reste — ce module est le premier de `live` à en tirer un.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch  # noqa: F401  avant patchcore/faiss : l'ordre inverse fait abort libomp
from PIL import Image
from torchvision import transforms

# vmax = le plus grand score des images nominales hors banque, mesuré au fit.
# Cette constante n'est que le repli d'une banque qui ne porte pas sa mesure.
HEATMAP_VMAX = 10.0

# Exposant du canal alpha, dans [0, 1] : 1 laisse le fondu linéaire, 0 peint
# tout à l'opacité pleine. Position d'entrée du curseur, réglable en direct.
HEATMAP_ALPHA = 0.5

# Bornes de la rampe de couleur : les deux bouts du jet sont inexploitables.
COLORMAP_LOW = 0.1
COLORMAP_HIGH = 0.9

# Profondeur des agrégations optionnelles, en frames scorées.
SMOOTHING_SECONDS = 1 / 3
SMOOTHING_FRAMES_MAX = 30   # borne mémoire : une carte 224x224 float32 = 200 Ko
SMOOTHING_SECONDS_MAX = 5.0 # au-delà la tache survit si longtemps qu'on la croit figée
DEFAULT_FPS = 30.0          # sources qui ne déclarent rien, webcams surtout
SMOOTHING_MODES = ("none", "mean", "max")

FAISS_ON_GPU = os.environ.get("INFER_FAISS_GPU", "").lower() in ("1", "true", "yes")
# Ramené à 1 sur macOS par FaissNN, où le multi-thread segfault.
FAISS_NUM_WORKERS = int(os.environ.get(
    "INFER_FAISS_THREADS", "1" if platform.system() == "Darwin" else "4"))


def normalize_heatmap(heatmap, vmax):
    """Ce qui dépasse vmax, en fraction de vmax : max(heatmap / vmax - 1, 0)."""
    return np.maximum(
        np.asarray(heatmap, np.float32) / max(float(vmax), 1e-6) - 1.0, 0.0
    )


def overlay_heatmap(preview_rgb, heatmap, vmax, alpha):
    """Vignette + heatmap jet, en BGR. Pur affichage, aucun effet sur les scores.

    Canal alpha = normalisé ** alpha : rien n'est peint sous vmax, et l'exposant
    décide de la vitesse à laquelle la couleur monte au-dessus.
    """
    normalized = normalize_heatmap(heatmap, vmax)
    ramp = np.clip(normalized, COLORMAP_LOW, COLORMAP_HIGH)
    colored = cv2.applyColorMap((ramp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    frame = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    # (H, W, 1) diffusé sur les 3 canaux BGR.
    a = np.clip(normalized ** float(alpha), 0.0, 1.0)[:, :, None]
    return (colored * a + frame * (1 - a)).astype(np.uint8)


def calculer_nb_heatmaps(fps, stride, seconds=SMOOTHING_SECONDS):
    """Combien de cartes agréger pour couvrir `seconds` de scène.

    Une carte tombe toutes les `stride` frames : n = fps * seconds / stride.
    `stride` est le stride *effectif*, sauts compris, pas celui demandé.
    """
    if not fps or fps <= 0 or fps > 240:
        fps = DEFAULT_FPS   # 0 ou aberrant : CAP_PROP_FPS n'est pas fiable partout
    n = round(fps * max(float(seconds), 0.0) / max(1.0, float(stride)))
    return min(max(int(n), 1), SMOOTHING_FRAMES_MAX)


def aggregate(values, mode):
    """Agrège les dernières valeurs scorées ; `none` rend la dernière telle quelle."""
    if not len(values):
        return None
    if mode == "mean":
        return np.mean(values, axis=0)
    if mode == "max":
        return np.max(values, axis=0)
    return values[-1]


def build_transform(fit_config):
    """Le prétraitement exact des datasets (celeba.py / coco.py), relu du fit."""
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

    La vignette est la frame telle que le réseau la voit : la heatmap se
    superpose dessus sans réalignement.
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
