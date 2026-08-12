"""Interface web locale : construire une banque PatchCore, puis scorer la webcam.

Deux moitiés, une par phase. À gauche on constitue une banque — un zip d'images
du nominal, un backbone, des couches, un taux de coreset — et le fit écrit un
`.pkg` dans coresets/. À droite on choisit une banque et on score la caméra en
direct. Serveur stdlib, à n'exposer que sur la loopback (pas d'authentification).

    python main.py                    # puis ouvrir http://127.0.0.1:8000

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

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# live_camera d'abord : il importe torch avant faiss. L'ordre inverse fait
# abort libomp (« OMP: Error #179 ») — ne pas réordonner.
from live_camera import (  # isort: skip
    FAISS_NUM_WORKERS,
    HEATMAP_VMAX,
    HEATMAP_VMIN,
    SMOOTH_WINDOW,
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

CORESETS_DIR = patchcore.packaging.CORESETS_DIR
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

# Les backbones proposés au fit, du plus lourd au plus rapide. Restreint à trois
# ResNet : PatchCore veut des cartes de features convolutives par couche, et ces
# trois-là couvrent le compromis cadence/AUROC mesuré dans le README.
FIT_BACKBONES = ("wideresnet50", "resnet50", "resnet34")
FIT_LAYERS = ("layer1", "layer2", "layer3", "layer4")
FIT_DEFAULT_LAYERS = ("layer2", "layer3")
FIT_DEFAULT_CORESET_PCT = 0.01

# Plafond du corps d'un upload. Un zip d'images de scène pèse quelques centaines
# de Mo ; au-delà c'est une erreur de manipulation, et le refuser tôt évite de
# remplir le disque avant de s'en apercevoir.
MAX_UPLOAD_BYTES = 8 * 1024 ** 3

# Exposant du canal alpha (pas le poids de mélange de live_camera), injecté dans
# la page avec son plafond : elle en déduit la course de son curseur 0→1.
HEATMAP_ALPHA = 2.0
HEATMAP_ALPHA_MAX = 4.0


def clamp_alpha(value):
    """Un exposant négatif inverserait la rampe et sortirait l'alpha de [0, 1]."""
    return min(max(float(value), 0.0), HEATMAP_ALPHA_MAX)


def overlay_heatmap(preview_rgb, heatmap, vmin, vmax, alpha):
    """Vignette + heatmap jet, en BGR. vmin/vmax et alpha sont réglables en direct
    depuis la page (pur affichage, aucun effet sur les scores).

    Le canal alpha vaut normalized ** alpha, par pixel : `alpha` est un exposant,
    pas un poids de mélange. Il tord la rampe en marche — 1 sur l'anomalie, 0 sur
    le normal, qui reste l'image nue."""
    normalized = np.clip(
        (heatmap - vmin) / max(vmax - vmin, 1e-6), 0, 1
    )
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    frame = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    # (H, W, 1) diffusé sur les 3 canaux BGR. Avec normalized ∈ [0, 1] et
    # alpha ≥ 0 la puissance reste dans [0, 1] : pas de clip nécessaire.
    a = (normalized.astype(np.float32) ** alpha)[:, :, None]
    return (colored * a + frame * (1 - a)).astype(np.uint8)


def find_banks():
    """Les .pkg de coresets/, avec de quoi renseigner le bandeau sans les charger.

    Seul le fit_config.json est lu dans l'archive : l'index faiss pèse jusqu'au
    gigaoctet et n'a rien à faire ici."""
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
        })
    return banks


class Runner:
    """Les deux travaux longs — construire une banque, scorer la caméra —
    pilotés par la page.

    Ils tournent sur le thread principal et le serveur HTTP en fond, jamais
    l'inverse : sur macOS torch abort si on le touche depuis un thread
    secondaire. C'est aussi ce qui les rend exclusifs l'un de l'autre, ce qui
    tombe bien — un fit sature déjà la machine. Les handlers HTTP ne font que
    déposer une demande ici, et relisent l'état par polling (un dict sous verrou).
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
            "verdict": None,
            "error": None,
            "params": None,
            "device": None,
            "device_note": None,
        }
        # Avancement du fit. `total` nul = phase de durée inconnue (la sélection
        # du coreset n'expose aucun compteur).
        self._fit = {
            "running": False, "phase": None, "done": 0, "total": 0,
            "name": None, "error": None, "seconds": 0.0, "images": None,
            "trained": None,
        }
        # Paramètres modifiables À CHAUD (la page les change sans redémarrer).
        self._live = {
            "zoom": 1.0, "vmin": HEATMAP_VMIN, "vmax": HEATMAP_VMAX,
            "alpha": HEATMAP_ALPHA, "stride": 1,
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
            for k in ("zoom", "vmin", "vmax"):
                if fields.get(k) is not None:
                    self._live[k] = float(fields[k])
            if fields.get("alpha") is not None:
                self._live["alpha"] = clamp_alpha(fields["alpha"])
            if fields.get("stride") is not None:
                self._live["stride"] = max(1, int(fields["stride"]))

    def snapshot(self):
        """Enregistre la frame courante AVEC heatmap (l'overlay affiché) dans
        results/<tache>/captures/<layer>/<coreset>/v<vmax>/cap_<ts>_s<curseur>_a<exposant>.jpg.

        Ce qui définit une série est un dossier, ce qui varie d'une capture à
        l'autre est dans le nom."""
        with self._lock:
            jpeg = self._jpeg  # overlay avec heatmap, déjà encodé
            params = self._state["params"]
            meta = self._bank_meta or {}
            alpha, vmax = self._live["alpha"], self._live["vmax"]
        if jpeg is None:
            return None
        bank = params["bank_dir"] if params else ""
        task = meta.get("task") or os.path.basename(bank).split("_")[0] or "live"
        # Rangé par tâche / couche / coreset (lus du fit_config de la banque),
        # puis par vmax.
        out_dir = os.path.join(
            "results", task, "captures",
            meta.get("layer", "l?"), meta.get("coreset", "p?"),
            "v{:g}".format(vmax),
        )
        os.makedirs(out_dir, exist_ok=True)
        # La page n'envoie que l'exposant : on inverse son mappage plutôt que de
        # faire circuler deux valeurs qui pourraient se désynchroniser.
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
            # running dès maintenant, pour que la page fige les champs sans
            # attendre que le thread principal ait chargé la banque.
            self._state.update(
                running=True, score=None, fps=0.0, infer_ms=0.0, frames=0,
                verdict=None, error=None, params=params, device_note=None,
            )
            # Valeurs initiales des contrôles à chaud (ensuite pilotés par /api/update).
            self._live = {
                "zoom": float(params.get("zoom", 1.0)),
                "vmin": HEATMAP_VMIN,
                "vmax": float(params.get("vmax", HEATMAP_VMAX)),
                "alpha": float(params.get("alpha", HEATMAP_ALPHA)),
                "stride": max(1, int(params.get("stride", 1))),
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

    # ─── Fit ───────────────────────────────────────────────────────────────

    def _fit_bank(self, params):
        """Zip d'images -> banque -> coresets/<nom>.pkg.

        Les images extraites restent à côté du .pkg : elles disent de quoi la
        banque a été faite, et permettent de la refitter autrement sans avoir à
        les renvoyer.
        """
        t0 = time.perf_counter()
        staging = os.path.join(CORESETS_DIR, ".incoming-{}".format(int(t0 * 1000)))
        pkg_path = None
        try:
            self._update_fit(phase="extraction")
            counts = patchcore.uploads.extract_images(params["archive"], staging)
            available = counts[patchcore.uploads.NORMAL]
            # Le nom de la banque doit dire ce qui est réellement entré dedans :
            # demander 20 000 images d'une archive qui en compte 400 donne une
            # banque à 400, pas un fichier nommé ts20000.
            subset = params["train_subset"]
            subset = None if not subset else min(subset, available)
            self._update_fit(images=available)

            def progress(phase, done, total):
                if self._stop.is_set():
                    raise KeyboardInterrupt("Fit interrompu.")
                self._update_fit(
                    phase=phase, done=done, total=total,
                    seconds=time.perf_counter() - t0,
                )

            # SCENE lit son dossier d'images dans SCENE_PATH ; `source` le
            # réinscrit dans le fit_config pour l'inférence.
            os.environ["SCENE_PATH"] = staging
            bank_dir = run_fit(
                SCENE, models_dir=staging, coreset_pct=FIT_DEFAULT_CORESET_PCT,
                extra_config={"source": staging, "task": params["task"]},
                progress=progress, random_subset=True,
                # Pas de workers : ils démarrent par spawn sur macOS, chacun
                # réimportant le module principal du serveur. Le décodage JPEG
                # pèse ~10 % face au backbone, le jeu n'en vaut pas la chandelle.
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

            # Les images ne sont déplacées qu'une fois le .pkg écrit : un fit
            # raté ne laisse pas un dossier d'images orphelin qui ressemblerait
            # à une banque.
            images_dir = os.path.join(CORESETS_DIR, name[: -len(patchcore.packaging.SUFFIX)])
            shutil.rmtree(images_dir, ignore_errors=True)
            shutil.rmtree(os.path.join(staging, os.path.basename(bank_dir)),
                          ignore_errors=True)
            os.replace(staging, images_dir)
            staging = None

            # `trained` < `images` : FolderDataset réserve 20 % du normal pour
            # un éventuel calibrage de seuil. Affiché, sinon l'écart surprend.
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
                # Sinon deux configurations se mélangeraient dans le même dossier.
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

            scores = deque(maxlen=SMOOTH_WINDOW)
            heatmap = np.zeros(
                (fit_config["imagesize"], fit_config["imagesize"]), np.float32
            )
            threshold = params["threshold"]
            frame_index = 0
            fps = 0.0
            infer_ms = 0.0
            last_tick = time.perf_counter()

            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok:
                    if params["loop"] and not source.isdigit():
                        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    self._update(error="Fin du flux.")
                    break

                frame = cv2.flip(frame, 1)  # effet miroir, plus naturel face à la webcam

                with self._lock:
                    zoom = self._live["zoom"]
                    vmin, vmax = self._live["vmin"], self._live["vmax"]
                    alpha = self._live["alpha"]
                    stride = self._live["stride"]
                tensor, preview = preprocess(frame, transform, zoom)

                if frame_index % stride == 0:
                    t0 = time.perf_counter()
                    batch_scores, batch_masks = patchcore_instance.predict(tensor)
                    infer_ms = 1000.0 * (time.perf_counter() - t0)
                    scores.append(float(batch_scores[0]))
                    # Médiane sur la fenêtre : robuste aux frames aberrantes.
                    score = float(np.median(scores))
                    verdict = (
                        None if threshold is None
                        else ("anomalie" if score >= threshold else "ok")
                    )
                    heatmap = np.asarray(batch_masks[0])
                    self._update(score=score, infer_ms=infer_ms, verdict=verdict)

                # Encodé à chaque frame, même non scorée : l'aperçu reste fluide
                # et garde la dernière heatmap entre deux inférences.
                ok_enc, buf = cv2.imencode(
                    ".jpg", overlay_heatmap(preview, heatmap, vmin, vmax, alpha),
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
            self._json({
                "alpha_max": HEATMAP_ALPHA_MAX,
                "alpha_default": HEATMAP_ALPHA,
                "faiss_threads": FAISS_NUM_WORKERS,
                "faiss_gpu": FAISS_ON_GPU,
                "backbones": [
                    {"name": b, "label": patchcore.packaging.backbone_label(b)}
                    for b in FIT_BACKBONES
                ],
                "layers": list(FIT_LAYERS),
                "default_layers": list(FIT_DEFAULT_LAYERS),
                "default_coreset_pct": FIT_DEFAULT_CORESET_PCT,
                "banks": find_banks(),
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
        """MJPEG : la vignette telle que le réseau la voit, tant que ça tourne.

        Se termine avec la boucle, la page rebranche le <img> au démarrage
        suivant. Pas de Content-Length : la réponse ne finit qu'à la coupure.
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

    def _receive_fit(self):
        """Reçoit l'archive d'images et lance le fit.

        Le zip arrive en corps brut, les réglages en query string, plutôt qu'en
        multipart : un corps brut se recopie sur disque par blocs sans jamais
        tenir en mémoire, là où un multipart de plusieurs gigaoctets devrait
        être découpé à la main.
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
        remaining = length
        with os.fdopen(fd, "wb") as out:
            while remaining > 0:
                chunk = self.rfile.read(min(1024 * 1024, remaining))
                if not chunk:
                    break  # connexion coupée en cours d'envoi
                out.write(chunk)
                remaining -= len(chunk)
        if remaining > 0:
            os.remove(archive)
            self._json({"error": "Envoi interrompu ({} octets manquants).".format(
                remaining)}, 400)
            return

        params["archive"] = archive
        ok, err = RUNNER.start_fit(params)
        if not ok:
            os.remove(archive)
        self._json({"ok": ok, "error": err}, 200 if ok else 409)

    def do_POST(self):
        # Traité avant toute lecture du corps : l'archive peut peser des
        # gigaoctets et ne doit pas passer par la mémoire.
        if self.path.split("?")[0] == "/api/fit":
            self._receive_fit()
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
                    "zoom": float(params.get("zoom") or 1.0),
                    "vmax": float(params.get("vmax") or HEATMAP_VMAX),
                    # `or` interdit : alpha=0 est une valeur voulue (heatmap pleine).
                    "alpha": HEATMAP_ALPHA if params.get("alpha") is None
                             else clamp_alpha(params["alpha"]),
                    "threshold": None if params.get("threshold") is None
                                 else float(params["threshold"]),
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
            # La page n'est pas authentifiée : on n'ouvre que ce que coresets/
            # contient réellement, pas un chemin quelconque venu de la requête.
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
