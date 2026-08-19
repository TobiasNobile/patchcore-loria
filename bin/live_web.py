"""Interface web : construire une banque PatchCore, puis scorer la webcam.

À gauche on constitue une banque depuis un zip d'images sans l'anomalie à
détecter, à droite on la charge et on score la caméra. Serveur stdlib, loopback, sans authentification.

Le fit et la boucle de scoring s'excluent : tous deux tournent sur le thread
principal, seul endroit où torch est sûr sur macOS.
"""

import json
import logging
import os
import platform
import shutil
import sys
import tempfile
import threading
import time
import urllib.parse
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

if platform.system() == "Darwin": # equivalent à "est-ce que ça tourne sur MacOS"
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# live_camera d'abord : torch avant faiss, sinon abort libomp.
from live_camera import (  # isort: skip
    FAISS_NUM_WORKERS,
    HEATMAP_VMAX,
    HEATMAP_VMIN,
    SMOOTHING_MODES,
    SMOOTHING_SECONDS,
    SMOOTHING_SECONDS_MAX,
    aggregate,
    calculer_nb_heatmaps,
    build_transform,
    preprocess,
    resolve_device,
    tune_faiss_small_batches,
)

from live_camera import FAISS_ON_GPU  # noqa: E402  défaut de la case à cocher

import patchcore.banks
import patchcore.common
import patchcore.packaging
import patchcore.uploads
import patchcore.utils
from experiments.datasets import SCENE
from experiments.pipelines import run_fit

LOGGER = logging.getLogger(__name__)

# Chemins absolus, déduits de l'emplacement du fichier : la page se sert et les
# banques se trouvent identiquement, quel que soit le répertoire d'où le serveur
# a été lancé.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(_ROOT, "src", "templates")
STATIC_DIR = os.path.join(_ROOT, "src", "static")
CORESETS_DIR = os.environ.get("PATCHCORE_CORESETS_DIR") or os.path.join(
    _ROOT, patchcore.packaging.CORESETS_DIR
)

# Backbones proposés au fit, du plus lourd au plus rapide.
FIT_BACKBONES = ("wideresnet50", "resnet50", "resnet34")
FIT_LAYERS = ("layer1", "layer2", "layer3", "layer4")
FIT_DEFAULT_LAYERS = ("layer2", "layer3")
FIT_DEFAULT_CORESET_PCT = 0.01
# Plafond du nombre d'images retenues pour la banque, quoi qu'on envoie : la
# sélection du coreset est quadratique en nombre de patchs, 20 000 images
# demandent déjà des heures et un GPU. Au-delà, le tirage est aléatoire.
FIT_MAX_IMAGES = 20_000

# Banque présélectionnée à l'ouverture de la page : sans elle, c'est la première
# de coresets/ dans l'ordre alphabétique, qui n'a aucune raison d'être la bonne.
# Comparaison par sous-chaîne sur le nom, pour survivre à un changement de
# coreset ou de ts dans le nom du .pkg.
DEFAULT_BANK = os.environ.get("LIVE_DEFAULT_BANK", "DetectionKnife_l3-l4")


def default_bank_dir(banks):
    """Le .pkg présélectionné, ou le premier si le motif ne correspond à rien."""
    for bank in banks:
        if DEFAULT_BANK.lower() in bank["name"].lower():
            return bank["dir"]
    return banks[0]["dir"] if banks else ""


# Plafond d'un upload : au-delà c'est une erreur de manipulation.
MAX_UPLOAD_BYTES = 8 * 1024 ** 3

# Vidéos envoyées depuis la page, ensuite reprises comme source de scoring. Le
# navigateur ne livre jamais le chemin d'un fichier choisi : sans cet envoi, une
# source locale doit être tapée à la main.
VIDEOS_DIR = os.path.join("data", "videos", "uploads")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v")

# Exposant du canal alpha et son plafond ; la page en déduit la course du curseur.
HEATMAP_ALPHA = 2.0
HEATMAP_ALPHA_MAX = 4.0

# Bornes de la rampe de couleur : les deux bouts du jet sont inexploitables.
COLORMAP_LOW = 0.1
COLORMAP_HIGH = 0.9

# Plafond du mélange : à 1 la couleur cache l'objet qu'on veut voir.
OPACITY_MAX = 0.9

# Plafond du recadrage : le zoom garde le centre sur h/zoom × w/zoom, donc au-delà
# il ne reste qu'une poignée de pixels étirés à 224 — une bouillie floue, sans
# rien qui dise pourquoi. À 8 il reste déjà moins d'un huitième de l'image.
ZOOM_MAX = 8.0

# Rattrapage borné : après une longue pause (onglet en arrière-plan, machine qui
# a ramé), rejouer d'un coup toutes les frames en retard reviendrait à sauter la
# scène entière. On reprend le fil, quitte à décaler l'horloge.
MAX_SKIP_FRAMES = 60


def clamp_alpha(value):
    """Un exposant négatif inverserait la rampe et sortirait l'alpha de [0, 1]."""
    return min(max(float(value), 0.0), HEATMAP_ALPHA_MAX)


def clamp_zoom(value):
    """Sous 1 le recadrage n'a pas de sens, au-delà de ZOOM_MAX on n'y voit rien."""
    return min(max(float(value), 1.0), ZOOM_MAX)


def clamp_seconds(value):
    """Durée de lissage : sous 0 elle n'a pas de sens, au-delà la tache se fige."""
    return min(max(float(value), 0.0), SMOOTHING_SECONDS_MAX)


def clean_dataset(name):
    """Nom du zip ou du dossier choisi pour le fit, gardé pour l'affichage.

    Purement informatif — rien ne le relit — donc on le garde lisible (points,
    espaces, accents) au lieu de le slugifier : seuls le chemin et les caractères
    de contrôle sautent. `source` ne peut pas servir : il pointe le dossier de
    transit, pas ce que l'utilisateur a désigné.
    """
    base = os.path.basename(str(name).replace("\\", "/").rstrip("/"))
    return "".join(c for c in base if c.isprintable())[:80]


def overlay_heatmap(preview_rgb, heatmap, vmin, vmax, alpha):
    """Vignette + heatmap jet, en BGR. Pur affichage, aucun effet sur les scores.
    
    `alpha` est un exposant appliqué par pixel, pas un poids de mélange.
    """
    normalized = np.clip(
        (heatmap - vmin) / max(vmax - vmin, 1e-6), 0, 1
    )
    # Couleur et opacité bornées séparément : l'alpha suit la valeur brute.
    ramp = np.clip(normalized, COLORMAP_LOW, COLORMAP_HIGH)
    colored = cv2.applyColorMap((ramp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    frame = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    # (H, W, 1) diffusé sur les 3 canaux BGR.
    a = (normalized.astype(np.float32) ** alpha)[:, :, None] * OPACITY_MAX
    return (colored * a + frame * (1 - a)).astype(np.uint8)


def find_banks():
    """Les .pkg de coresets/. Seul le fit_config est lu : l'index pèse jusqu'au Go."""
    banks = []
    for entry in patchcore.packaging.find(CORESETS_DIR):
        cfg = entry["config"]
        banks.append({
            "dir": entry["path"],
            "name": entry["name"],
            "task": cfg.get("task", ""),
            "backbone": patchcore.packaging.backbone_label(cfg.get("backbone_name", "?")),
            "layers": "-".join(
                l.replace("layer", "l")
                for l in cfg.get("layers_to_extract_from", [])
            ),
            "coreset": ("identity" if cfg.get("sampler_name") == "identity"
                        else "p{:g}".format(cfg.get("coreset_pct", 0))),
            "imagesize": cfg.get("imagesize", 224),
            "bank_size": cfg.get("memory_bank_size", 0),
            "bank_gb": cfg.get("bank_gb", 0.0),
            "train_images": cfg.get("n_train_images"),
            "dataset": cfg.get("dataset"),
            # Nombre de voisins cherchés au scoring : il vient de la banque, pas
            # de la page, et change l'échelle des scores d'une banque à l'autre.
            "num_nn": cfg.get("anomaly_scorer_num_nn"),
        })
    return banks


class Runner:
    """Les deux travaux longs — fit et scoring — sur le thread principal.
    
    Sur macOS torch abort depuis un thread secondaire, d'où le thread principal
    et le serveur HTTP en fond. C'est aussi ce qui les rend exclusifs.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None  # (kind, params), déposé par un thread HTTP
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._jpeg = None  # dernière vignette encodée (overlay), servie en MJPEG et capturée
        self._bank_meta = None  # {coreset, layer} de la banque, pour ranger les captures
        self._state = {
            "running": False,
            "score": None,
            "fps": 0.0,
            "infer_ms": 0.0,
            "frames": 0,
            "error": None,
            "params": None,
            "device": None,
            "device_note": None,
            # Nombre de cartes agrégées, déduit du fps de la source et du stride.
            # 0 tant que la source n'est pas ouverte : la page n'a rien à annoncer.
            "smoothing_frames": 0,
            "source_fps": 0.0,
        }
        # Avancement du fit. `total` nul = phase sans compteur (le coreset).
        self._fit = {
            "running": False, "phase": None, "done": 0, "total": 0,
            "name": None, "error": None, "seconds": 0.0, "images": None,
            "trained": None,
        }
        # Paramètres modifiables À CHAUD (la page les change sans redémarrer).
        self._live = {
            "zoom": 1.0, "vmin": HEATMAP_VMIN, "vmax": HEATMAP_VMAX,
            "alpha": HEATMAP_ALPHA, "stride": 1, "smoothing": "none",
            "smoothing_seconds": SMOOTHING_SECONDS,
        }

    def state(self):
        with self._lock:
            s = dict(self._state)
            s["live"] = dict(self._live)
            s["fit"] = dict(self._fit)
            return s

    def update_live(self, fields):
        """Zoom / échelle couleur / opacité / stride, ajustables en marche."""
        with self._lock:
            for k in ("vmin", "vmax"):
                if fields.get(k) is not None:
                    self._live[k] = float(fields[k])
            if fields.get("zoom") is not None:
                self._live["zoom"] = clamp_zoom(fields["zoom"])
            if fields.get("alpha") is not None:
                self._live["alpha"] = clamp_alpha(fields["alpha"])
            if fields.get("stride") is not None:
                self._live["stride"] = max(1, int(fields["stride"]))
            if fields.get("smoothing") in SMOOTHING_MODES:
                self._live["smoothing"] = fields["smoothing"]
            if fields.get("smoothing_seconds") is not None:
                self._live["smoothing_seconds"] = clamp_seconds(
                    fields["smoothing_seconds"])

    def snapshot(self):
        """Enregistre l'overlay affiché sous results/<tache>/captures/…
        
        Un dossier définit une série, le nom porte ce qui varie d'une capture à l'autre.
        """
        with self._lock:
            jpeg = self._jpeg  # overlay avec heatmap, déjà encodé
            params = self._state["params"]
            meta = self._bank_meta or {}
            alpha, vmax = self._live["alpha"], self._live["vmax"]
        if jpeg is None:
            return None
        bank = params["bank_dir"] if params else ""
        task = meta.get("task") or os.path.basename(bank).split("_")[0] or "live"
        # Rangé par tâche / couche / coreset, puis par vmax.
        out_dir = os.path.join(
            "results", task, "captures",
            meta.get("layer", "l?"), meta.get("coreset", "p?"),
            "v{:g}".format(vmax),
        )
        os.makedirs(out_dir, exist_ok=True)
        # La page n'envoie que l'exposant : on inverse son mappage.
        slider = (alpha / HEATMAP_ALPHA_MAX) ** 0.5 if HEATMAP_ALPHA_MAX else 0.0
        path = os.path.join(
            out_dir,
            "cap_{}_s{:.2f}_a{:.2f}.jpg".format(
                int(time.time() * 1000), slider, alpha
            ),
        )
        with open(path, "wb") as fh:
            fh.write(jpeg)
        return path

    def _update(self, **fields):
        with self._lock:
            self._state.update(fields)

    def jpeg(self):
        with self._lock:
            return self._jpeg

    def start(self, params):
        """Appelé depuis un thread HTTP : dépose la demande, ne score rien."""
        with self._lock:
            if self._state["running"]:
                return False, "Déjà en cours."
            if self._fit["running"]:
                return False, "Un fit est en cours — attendre qu'il finisse."
            self._stop.clear()
            self._pending = ("live", params)
            self._jpeg = None
            # running dès maintenant : la page fige ses champs sans attendre la banque.
            self._state.update(
                running=True, score=None, fps=0.0, infer_ms=0.0, frames=0,
                error=None, params=params, device_note=None,
                smoothing_frames=0, source_fps=0.0,
            )
            # Valeurs initiales des contrôles à chaud (ensuite pilotés par /api/update).
            self._live = {
                "zoom": clamp_zoom(params.get("zoom", 1.0)),
                "vmin": HEATMAP_VMIN,
                "vmax": float(params.get("vmax", HEATMAP_VMAX)),
                "alpha": float(params.get("alpha", HEATMAP_ALPHA)),
                "stride": max(1, int(params.get("stride", 1))),
                "smoothing": params.get("smoothing", "none"),
                "smoothing_seconds": clamp_seconds(
                    params.get("smoothing_seconds", SMOOTHING_SECONDS)),
            }
        self._wake.set()
        return True, None

    def start_fit(self, params):
        """Dépose une demande de fit. L'archive est déjà sur disque."""
        with self._lock:
            if self._state["running"]:
                return False, "La caméra tourne — l'arrêter avant de fitter."
            if self._fit["running"]:
                return False, "Un fit est déjà en cours."
            self._stop.clear()
            self._pending = ("fit", params)
            self._fit = {
                "running": True, "phase": "préparation", "done": 0, "total": 0,
                "name": None, "error": None, "seconds": 0.0, "images": None,
                "trained": None,
            }
        self._wake.set()
        return True, None

    def stop(self):
        """Arrête la boucle caméra, ou demande l'abandon d'un fit."""
        self._stop.set()

    def _update_fit(self, **fields):
        with self._lock:
            self._fit.update(fields)

    def serve_forever(self):
        """Boucle du thread principal : attend une demande, la joue, recommence."""
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                job, self._pending = self._pending, None
            if job is None:
                continue
            kind, params = job
            if kind == "fit":
                self._fit_bank(params)
            else:
                self._run(params)

    # À partir de là c'est du fit 

    def _fit_bank(self, params):
        """Zip d'images -> banque -> coresets/<nom>.pkg.
        
        Les images extraites restent à côté du .pkg : de quoi refitter sans les renvoyer.
        """
        t0 = time.perf_counter()
        staging = os.path.join(CORESETS_DIR, ".incoming-{}".format(int(t0 * 1000)))
        pkg_path = None
        try:
            self._update_fit(phase="extraction")
            counts = patchcore.uploads.extract_images(params["archive"], staging)
            available = counts[patchcore.uploads.NORMAL]
            # Le nom dit ce qui est vraiment entré : 20 000 demandés sur 400 images donnent ts400.
            # Le plafond s'applique même sans demande — un zip de 50 000 images
            # n'en fitte que FIT_MAX_IMAGES, tirées au hasard (random_subset).
            subset = min(params["train_subset"] or available, available, FIT_MAX_IMAGES)
            # Rien d'écarté : tsall plutôt qu'un ts<n> qui ferait croire à un tirage.
            subset = None if subset >= available else subset
            self._update_fit(images=available)

            def progress(phase, done, total):
                if self._stop.is_set():
                    raise KeyboardInterrupt("Fit interrompu.")
                self._update_fit(
                    phase=phase, done=done, total=total,
                    seconds=time.perf_counter() - t0,
                )

            # SCENE lit SCENE_PATH ; `source` le réinscrit pour l'inférence.
            os.environ["SCENE_PATH"] = staging
            bank_dir = run_fit(
                SCENE, models_dir=staging, coreset_pct=FIT_DEFAULT_CORESET_PCT,
                extra_config={"source": staging, "task": params["task"],
                              "dataset": params["dataset"]},
                progress=progress, random_subset=True,
                # Pas de workers : leur spawn réimporte le module du serveur sur macOS.
                num_workers=0,
                overrides={
                    "backbone_name": params["backbone"],
                    "layers_to_extract_from": params["layers"],
                    "coreset_pct": params["coreset_pct"],
                    "train_subset": subset,
                },
            )

            self._update_fit(phase="empaquetage", done=0, total=0)
            with open(os.path.join(bank_dir, patchcore.banks.CONFIG_FILENAME)) as fh:
                config = json.load(fh)
            name = patchcore.packaging.build_name(config, params["task"])
            pkg_path = patchcore.packaging.pack(bank_dir, os.path.join(CORESETS_DIR, name))

            # Images déplacées après le .pkg : un fit raté ne laisse pas de dossier orphelin.
            images_dir = os.path.join(CORESETS_DIR, name[: -len(patchcore.packaging.SUFFIX)])
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(staging, os.path.basename(bank_dir)),
                          ignore_errors=True)
            os.replace(staging, images_dir)
            staging = None

            # trained < images : FolderDataset garde 20 % hors banque.
            self._update_fit(
                running=False, phase="terminé", name=name,
                trained=config.get("n_train_images"),
                seconds=time.perf_counter() - t0,
            )
            LOGGER.info("Fit terminé : %s", pkg_path)
        except KeyboardInterrupt as exc:
            self._update_fit(running=False, phase=None, error=str(exc))
            LOGGER.info("Fit interrompu à la demande.")
        except Exception as exc:  # remonté tel quel dans la page
            LOGGER.exception("Fit interrompu")
            self._update_fit(running=False, phase=None, error=str(exc))
        finally:
            os.environ.pop("SCENE_PATH", None)
            try:
                os.remove(params["archive"])
            except OSError:
                pass
            if staging is not None:
                shutil.rmtree(staging, ignore_errors=True)

    def _run(self, params):
        capture = None
        try:
            tune_faiss_small_batches()
            device, device_note = resolve_device(params["device"])
            patchcore_instance, fit_config = patchcore.packaging.load(
                params["bank_dir"], device,
                params["faiss_gpu"], params["faiss_threads"],
            )
            patchcore.utils.fix_seeds(fit_config["seed"])
            transform = build_transform(fit_config)
            with self._lock:
                layer = "-".join(
                    l.replace("layer", "l")
                    for l in fit_config.get("layers_to_extract_from", ["layer2", "layer3"])
                )
                # Suffixe seulement si non standard, comme build_tag() du fit.
                backbone = fit_config.get("backbone_name", "wideresnet50")
                if backbone != "wideresnet50":
                    layer = "{}_{}".format(backbone, layer)
                if fit_config.get("imagesize", 224) != 224:
                    layer = "{}_im{}".format(layer, fit_config["imagesize"])
                self._bank_meta = {
                    "coreset": "identity" if fit_config.get("sampler_name") == "identity"
                               else "p{:g}".format(fit_config.get("coreset_pct", 0)),
                    "layer": layer,
                    "task": patchcore.packaging.slugify(fit_config.get("task", ""), ""),
                }

            self._update(device=str(device), device_note=device_note)

            source = params["source"]
            capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
            if not capture.isOpened():
                raise RuntimeError(
                    "Impossible d'ouvrir la source '{}'. Sur macOS, autoriser la "
                    "caméra pour le terminal dans Réglages Système > "
                    "Confidentialité et sécurité > Caméra.".format(source)
                )

            # Le fps déclaré par la source, pas celui du traitement : la fenêtre
            # de lissage se compte en temps de la scène filmée.
            source_fps = capture.get(cv2.CAP_PROP_FPS)
            # Un fichier ne se cadence pas tout seul : sans horloge, il défile à
            # la vitesse du traitement — au ralenti quand le stride est bas,
            # en accéléré quand il est haut. Deux lissages ne sont alors plus
            # comparables, puisque le temps lui-même change avec les réglages.
            # Une caméra, elle, impose déjà son rythme.
            paced = not source.isdigit() and 1.0 <= source_fps <= 240.0
            clock = time.perf_counter()   # origine de l'horloge de lecture
            consumed = 0                  # frames prises à la source depuis clock
            scored_at = 0                 # `consumed` à la dernière inférence
            # Stride effectif : frames de source par inférence, sauts compris.
            # C'est lui, pas le stride demandé, qui donne l'espacement réel des
            # cartes — donc la durée que couvre la fenêtre de lissage.
            effective_stride = float(self.state()["live"]["stride"])
            window = calculer_nb_heatmaps(
                source_fps, effective_stride, self.state()["live"]["smoothing_seconds"])
            scores = deque(maxlen=window)
            heatmap = np.zeros(
                (fit_config["imagesize"], fit_config["imagesize"]), np.float32
            )
            # Cartes scorées récentes et leur agrégat, recalculé à l'inférence seulement.
            heatmaps = deque(maxlen=window)
            self._update(smoothing_frames=window, source_fps=float(source_fps or 0))
            smoothed = heatmap
            smoothed_mode = "none"
            frame_index = 0
            fps = 0.0
            infer_ms = 0.0
            last_tick = time.perf_counter()

            while not self._stop.is_set():
                if paced:
                    # En avance : on attend l'heure de la frame suivante. En
                    # retard : on saute les frames dues avec grab(), qui les
                    # avance sans les décoder. Dans les deux cas la scène défile
                    # à sa vitesse, et c'est le nombre d'inférences qui varie.
                    due = clock + consumed / source_fps
                    ahead = due - time.perf_counter()
                    if ahead > 0:
                        self._stop.wait(min(ahead, 0.25))
                    else:
                        for _ in range(min(int(-ahead * source_fps), MAX_SKIP_FRAMES)):
                            if not capture.grab():
                                break
                            consumed += 1

                ok, frame = capture.read()
                consumed += 1
                if not ok:
                    if params["loop"] and not source.isdigit():
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        # L'horloge repart avec la vidéo, sinon le rattrapage
                        # sauterait aussitôt tout ce qui vient d'être rembobiné.
                        clock, consumed, scored_at = time.perf_counter(), 0, 0
                        continue
                    self._update(error="Fin du flux.")
                    break

                frame = cv2.flip(frame, 1)  # effet miroir, plus naturel face à la webcam

                with self._lock:
                    zoom = self._live["zoom"]
                    vmin, vmax = self._live["vmin"], self._live["vmax"]
                    alpha = self._live["alpha"]
                    stride = self._live["stride"]
                    smoothing = self._live["smoothing"]
                    seconds = self._live["smoothing_seconds"]
                # deque(iterable, maxlen) garde les derniers éléments : le lissage
                # se resserre ou s'élargit sans repartir de zéro.
                want = calculer_nb_heatmaps(
                    source_fps, max(stride, effective_stride), seconds)
                if want != scores.maxlen:
                    scores = deque(scores, maxlen=want)
                    heatmaps = deque(heatmaps, maxlen=want)
                    self._update(smoothing_frames=want)

                tensor, preview = preprocess(frame, transform, zoom)

                if frame_index % stride == 0:
                    # Frames de source réellement consommées depuis la dernière
                    # inférence : c'est l'espacement des cartes. Lissé, sinon un
                    # saut isolé ferait osciller la fenêtre.
                    if scored_at:
                        effective_stride = (0.7 * effective_stride
                                            + 0.3 * (consumed - scored_at))
                    scored_at = consumed
                    t0 = time.perf_counter()
                    batch_scores, batch_masks = patchcore_instance.predict(tensor)
                    infer_ms = 1000.0 * (time.perf_counter() - t0)
                    scores.append(float(batch_scores[0]))
                    score = float(aggregate(list(scores), smoothing))
                    heatmap = np.asarray(batch_masks[0])
                    heatmaps.append(heatmap)
                    smoothed = aggregate(list(heatmaps), smoothing)
                    smoothed_mode = smoothing
                    self._update(score=score, infer_ms=infer_ms)

                # Encodé à chaque frame. Mode changé depuis la dernière inférence : carte brute.
                shown = smoothed if smoothing == smoothed_mode else heatmap
                ok_enc, buf = cv2.imencode(
                    ".jpg", overlay_heatmap(preview, shown, vmin, vmax, alpha),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80],
                )
                if ok_enc:
                    with self._lock:
                        self._jpeg = buf.tobytes()

                now = time.perf_counter()
                # Lissage exponentiel : le fps brut saute trop pour être lu.
                fps = 0.9 * fps + 0.1 / max(now - last_tick, 1e-6)
                last_tick = now
                frame_index += 1
                self._update(fps=fps, frames=frame_index)
        except Exception as exc:  # remonté tel quel dans la page
            LOGGER.exception("Boucle interrompue")
            self._update(error=str(exc))
        finally:
            if capture is not None:
                capture.release()
            self._update(running=False)


RUNNER = Runner()



def _faiss_gpu_unavailable():
    """Message explicite plutôt qu'une AttributeError au chargement de la banque."""
    import faiss

    if not hasattr(faiss, "GpuIndexFlatL2"):
        return ("FAISS sur GPU : le paquet installé est faiss-cpu. "
                "Installer faiss-gpu-cu12, ou décocher.")
    if faiss.get_num_gpus() == 0:
        return "FAISS sur GPU : aucune carte NVIDIA visible. Décocher."
    return None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # le polling noierait la console

    def _send(self, code, body, content_type):
        payload = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _serve_file(self, path, content_type):
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            self._send(404, "not found", "text/plain")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self._serve_file(
                os.path.join(TEMPLATES_DIR, "live.html"), "text/html; charset=utf-8"
            )
        elif self.path.startswith("/static/"):
            name = os.path.basename(self.path.split("?")[0])
            types = {".css": "text/css", ".js": "text/javascript"}
            ctype = types.get(os.path.splitext(name)[1])
            if not ctype:
                self._send(404, "not found", "text/plain")
                return
            self._serve_file(os.path.join(STATIC_DIR, name), ctype + "; charset=utf-8")
        elif self.path == "/api/config":
            banks = find_banks()
            self._json({
                "alpha_max": HEATMAP_ALPHA_MAX,
                "alpha_default": HEATMAP_ALPHA,
                "zoom_max": ZOOM_MAX,
                "faiss_threads": FAISS_NUM_WORKERS,
                "faiss_gpu": FAISS_ON_GPU,
                "backbones": [
                    {"name": b, "label": patchcore.packaging.backbone_label(b)}
                    for b in FIT_BACKBONES
                ],
                "layers": list(FIT_LAYERS),
                "default_layers": list(FIT_DEFAULT_LAYERS),
                "default_coreset_pct": FIT_DEFAULT_CORESET_PCT,
                "max_images": FIT_MAX_IMAGES,
                "smoothing_seconds": SMOOTHING_SECONDS,
                "smoothing_seconds_max": SMOOTHING_SECONDS_MAX,
                "banks": banks,
                "default_bank": default_bank_dir(banks),
            })
        elif self.path == "/api/banks":
            # Relu après un fit, pour peupler le sélecteur sans recharger la page.
            self._json({"banks": find_banks()})
        elif self.path == "/api/state":
            self._json(RUNNER.state())
        elif self.path.startswith("/stream.mjpg"):
            self._stream()
        else:
            self._send(404, "not found", "text/plain")

    def _stream(self):
        """MJPEG tant que la boucle tourne. Pas de Content-Length : la réponse ne finit
        qu'à la coupure, et la page rebranche le <img> au démarrage suivant.
        """
        self.send_response(200)
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        last = None
        try:
            while RUNNER.state()["running"]:
                jpeg = RUNNER.jpeg()
                if jpeg is None or jpeg is last:
                    time.sleep(0.02)  # rien de neuf, ne pas réémettre
                    continue
                last = jpeg
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                self.wfile.write(
                    "Content-Length: {}\r\n\r\n".format(len(jpeg)).encode()
                )
                self.wfile.write(jpeg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # onglet fermé ou rechargé

    def _save_body(self, out, length):
        """Recopie le corps de la requête dans `out`, par blocs.

        Renvoie les octets manquants — non nul si la connexion a été coupée en
        cours d'envoi, ce qui laisse un fichier tronqué à effacer.
        """
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(1024 * 1024, remaining))
            if not chunk:
                break  # connexion coupée en cours d'envoi
            out.write(chunk)
            remaining -= len(chunk)
        return remaining

    def _receive_fit(self):
        """Reçoit l'archive et lance le fit.

        Corps brut plutôt que multipart : il se recopie sur disque par blocs sans
        jamais tenir en mémoire.
        """
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)

        def field(name, default=""):
            return (query.get(name) or [default])[0]

        try:
            params = {
                "task": patchcore.packaging.slugify(field("task")),
                "backbone": field("backbone"),
                "layers": [l for l in field("layers").split(",") if l],
                "coreset_pct": float(field("coreset_pct") or FIT_DEFAULT_CORESET_PCT),
                # 0 ou vide = toutes les images de l'archive.
                "train_subset": int(field("train_subset") or 0),
                # Nom du zip ou du dossier, tel que la page l'a vu.
                "dataset": clean_dataset(field("dataset")),
            }
        except ValueError as exc:
            self._json({"error": "Paramètres invalides : {}".format(exc)}, 400)
            return

        if params["backbone"] not in FIT_BACKBONES:
            self._json({"error": "Backbone inconnu : {}".format(params["backbone"])}, 400)
            return
        unknown = [l for l in params["layers"] if l not in FIT_LAYERS]
        if not params["layers"] or unknown:
            self._json({"error": "Couches invalides : {}".format(
                ", ".join(unknown) or "aucune sélectionnée")}, 400)
            return
        if not 0 < params["coreset_pct"] <= 1:
            self._json({"error": "Le coreset doit être dans ]0, 1]."}, 400)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json({"error": "Aucune archive reçue."}, 400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._json({"error": "Archive de {:.1f} Go, au-delà de la limite de "
                                 "{:.0f} Go.".format(length / 1024 ** 3,
                                                     MAX_UPLOAD_BYTES / 1024 ** 3)}, 413)
            return

        fd, archive = tempfile.mkstemp(prefix="patchcore-upload-", suffix=".zip")
        with os.fdopen(fd, "wb") as out:
            missing = self._save_body(out, length)
        if missing:
            os.remove(archive)
            self._json({"error": "Envoi interrompu ({} octets manquants).".format(
                missing)}, 400)
            return

        params["archive"] = archive
        ok, err = RUNNER.start_fit(params)
        if not ok:
            os.remove(archive)
        self._json({"ok": ok, "error": err}, 200 if ok else 409)

    def _receive_video(self):
        """Range une vidéo envoyée depuis la page et renvoie son chemin.

        La page recopie ce chemin dans Source : rien d'autre ne change au
        démarrage, la boucle ouvre un fichier comme n'importe quel autre.
        """
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        name = os.path.basename((query.get("name") or [""])[0])
        stem, ext = os.path.splitext(name)
        if ext.lower() not in VIDEO_EXTENSIONS:
            self._json({"error": "Format refusé : « {} ». Attendu : {}.".format(
                ext or name or "sans extension", ", ".join(VIDEO_EXTENSIONS))}, 400)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            self._json({"error": "Aucune vidéo reçue."}, 400)
            return
        if length > MAX_UPLOAD_BYTES:
            self._json({"error": "Vidéo de {:.1f} Go, au-delà de la limite de "
                                 "{:.0f} Go.".format(length / 1024 ** 3,
                                                     MAX_UPLOAD_BYTES / 1024 ** 3)}, 413)
            return

        # Nom reconstruit, jamais celui du client : il ne doit ni sortir du
        # dossier, ni écraser un envoi précédent.
        os.makedirs(VIDEOS_DIR, exist_ok=True)
        slug = patchcore.packaging.slugify(stem, "video")
        path = os.path.join(VIDEOS_DIR, slug + ext.lower())
        i = 2
        while os.path.exists(path):
            path = os.path.join(VIDEOS_DIR, "{}_{}{}".format(slug, i, ext.lower()))
            i += 1

        with open(path, "wb") as out:
            missing = self._save_body(out, length)
        if missing:
            os.remove(path)
            self._json({"error": "Envoi interrompu ({} octets manquants).".format(
                missing)}, 400)
            return

        LOGGER.info("Vidéo reçue : %s (%.0f Mo)", path, length / 1024 ** 2)
        # Chemin relatif au dossier de lancement, comme celui qu'on taperait.
        self._json({"ok": True, "path": path})

    def do_POST(self):
        # Avant toute lecture du corps : archive et vidéo pèsent des gigaoctets.
        route = self.path.split("?")[0]
        if route == "/api/fit":
            self._receive_fit()
            return
        if route == "/api/video":
            self._receive_video()
            return
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b"{}"
        if self.path == "/api/start":
            try:
                params = json.loads(raw)
                params = {
                    "bank_dir": str(params["bank_dir"]),
                    "source": str(params.get("source", "0")),
                    "stride": max(1, int(params.get("stride") or 1)),
                    "zoom": clamp_zoom(params.get("zoom") or 1.0),
                    "vmax": float(params.get("vmax") or HEATMAP_VMAX),
                    # `or` interdit : alpha=0 est une valeur voulue (heatmap pleine).
                    "alpha": HEATMAP_ALPHA if params.get("alpha") is None
                             else clamp_alpha(params["alpha"]),
                    "smoothing": (params.get("smoothing")
                                  if params.get("smoothing") in SMOOTHING_MODES
                                  else "none"),
                    "smoothing_seconds": clamp_seconds(
                        params.get("smoothing_seconds") or SMOOTHING_SECONDS),
                    "loop": bool(params.get("loop")),
                    "device": str(params.get("device") or "auto"),
                    "faiss_threads": max(1, int(params.get("faiss_threads") or FAISS_NUM_WORKERS)),
                    "faiss_gpu": bool(params.get("faiss_gpu")),
                }
            except (ValueError, KeyError, TypeError) as exc:
                self._json({"error": "Paramètres invalides : {}".format(exc)}, 400)
                return
            if not params["bank_dir"]:
                self._json({"error": "Aucune banque sélectionnée."}, 400)
                return
            # Page non authentifiée : n'ouvrir que ce que coresets/ contient.
            if params["bank_dir"] not in {b["dir"] for b in find_banks()}:
                self._json({"error": "Banque inconnue dans {}/.".format(CORESETS_DIR)}, 400)
                return
            if params["device"].split(":")[0] not in ("auto", "cpu", "cuda"):
                self._json({"error": "Device inconnu : {}".format(params["device"])}, 400)
                return
            if params["faiss_gpu"]:
                err = _faiss_gpu_unavailable()
                if err:
                    self._json({"error": err}, 400)
                    return
            ok, err = RUNNER.start(params)
            self._json({"ok": ok, "error": err})
        elif self.path == "/api/stop":
            RUNNER.stop()
            self._json({"ok": True})
        elif self.path == "/api/update":
            try:
                p = json.loads(raw)
            except ValueError:
                p = {}
            RUNNER.update_live(p)
            self._json({"ok": True})
        elif self.path == "/api/snapshot":
            path = RUNNER.snapshot()
            if path:
                self._json({"ok": True, "path": path})
            else:
                self._json({"ok": False, "error": "Aucune frame (démarrer d'abord)."}, 400)
        else:
            self._send(404, "not found", "text/plain")


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Loopback par défaut : la page n'a aucune authentification.")
@click.option("--port", default=8000, show_default=True)
def main(host, port):
    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    LOGGER.info("Interface sur http://%s:%d — ctrl-c pour quitter.", host, port)
    try:
        RUNNER.serve_forever()  # le scoring doit rester sur le thread principal
    except KeyboardInterrupt:
        pass
    finally:
        RUNNER.stop()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
