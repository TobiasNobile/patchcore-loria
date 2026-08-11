"""Interface web locale pour scorer une webcam avec une banque PatchCore.

Même boucle que bin/live_camera.py, sans fenêtre OpenCV : une page servie sur
localhost affiche le seul score d'anomalie, et porte les paramètres comme le
lancement. Serveur stdlib, à n'exposer que sur la loopback (pas d'auth).

    python bin/live_web.py            # puis ouvrir http://127.0.0.1:8000
"""

import json
import logging
import os
import platform
import sys
import threading
import time
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
    select_device,
    tune_faiss_small_batches,
)

from live_camera import FAISS_ON_GPU  # noqa: E402  défaut de la case à cocher

import patchcore.banks
import patchcore.utils

LOGGER = logging.getLogger(__name__)

MODELS_ROOT = "models"
TEMPLATES_DIR = "templates"
STATIC_DIR = "static"

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
    """Les banques de models/, avec de quoi renseigner le bandeau sans les charger."""
    banks = []
    for root, dirs, files in os.walk(MODELS_ROOT):
        if "fit_config.json" not in files:
            continue
        dirs[:] = []  # une banque n'en contient pas une autre
        try:
            with open(os.path.join(root, "fit_config.json")) as fh:
                cfg = json.load(fh)
        except (OSError, ValueError):
            continue
        banks.append({
            "dir": root,
            "dataset": os.path.basename(os.path.dirname(os.path.normpath(root))),
            "backbone": cfg.get("backbone_name", "?"),
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
    return sorted(banks, key=lambda b: b["dir"])


class Runner:
    """Boucle de capture + scoring, pilotée par la page.

    Elle tourne sur le thread principal et le serveur HTTP en fond, jamais
    l'inverse : sur macOS torch abort si on le touche depuis un thread
    secondaire. Les handlers HTTP ne font que déposer une demande ici, et
    relisent l'état par polling (un dict sous verrou).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._pending = None
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._jpeg = None  # dernière vignette encodée (overlay), servie en MJPEG et capturée
        self._fit_meta = None  # {coreset, layer} de la banque, pour ranger les captures
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
        }
        # Paramètres modifiables À CHAUD (la page les change sans redémarrer).
        self._live = {
            "zoom": 1.0, "vmin": HEATMAP_VMIN, "vmax": HEATMAP_VMAX,
            "alpha": HEATMAP_ALPHA,
        }

    def state(self):
        with self._lock:
            s = dict(self._state)
            s["live"] = dict(self._live)
            return s

    def update_live(self, fields):
        """Zoom / échelle couleur / opacité, ajustables pendant que la boucle tourne."""
        with self._lock:
            for k in ("zoom", "vmin", "vmax"):
                if fields.get(k) is not None:
                    self._live[k] = float(fields[k])
            if fields.get("alpha") is not None:
                self._live["alpha"] = clamp_alpha(fields["alpha"])

    def snapshot(self):
        """Enregistre la frame courante AVEC heatmap (l'overlay affiché) dans
        results/<dataset>/captures/<layer>/<coreset>/v<vmax>/cap_<ts>_s<curseur>_a<exposant>.jpg.

        Ce qui définit une série est un dossier, ce qui varie d'une capture à
        l'autre est dans le nom."""
        with self._lock:
            jpeg = self._jpeg  # overlay avec heatmap, déjà encodé
            params = self._state["params"]
            meta = self._fit_meta or {}
            alpha, vmax = self._live["alpha"], self._live["vmax"]
        if jpeg is None:
            return None
        bank = params["bank_dir"] if params else ""
        dataset = os.path.basename(os.path.dirname(os.path.normpath(bank))) or "live"
        # Rangé par dataset / couche / coreset (lus du fit_config de la banque),
        # puis par vmax.
        out_dir = os.path.join(
            "results", dataset, "captures",
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
            self._stop.clear()
            self._pending = params
            self._jpeg = None
            # running dès maintenant, pour que la page fige les champs sans
            # attendre que le thread principal ait chargé la banque.
            self._state.update(
                running=True, score=None, fps=0.0, infer_ms=0.0, frames=0,
                verdict=None, error=None, params=params,
            )
            # Valeurs initiales des contrôles à chaud (ensuite pilotés par /api/update).
            self._live = {
                "zoom": float(params.get("zoom", 1.0)),
                "vmin": HEATMAP_VMIN,
                "vmax": float(params.get("vmax", HEATMAP_VMAX)),
                "alpha": float(params.get("alpha", HEATMAP_ALPHA)),
            }
        self._wake.set()
        return True, None

    def stop(self):
        self._stop.set()

    def serve_forever(self):
        """Boucle du thread principal : attend une demande, la joue, recommence."""
        while True:
            self._wake.wait()
            self._wake.clear()
            with self._lock:
                params, self._pending = self._pending, None
            if params is not None:
                self._run(params)

    def _run(self, params):
        capture = None
        try:
            tune_faiss_small_batches()
            device = select_device(params["device"])
            patchcore_instance, fit_config = patchcore.banks.load_bank(
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
                self._fit_meta = {
                    "coreset": "identity" if fit_config.get("sampler_name") == "identity"
                               else "p{:g}".format(fit_config.get("coreset_pct", 0)),
                    "layer": layer,
                }

            self._update(device=str(device))

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
            stride = params["stride"]
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
                "banks": find_banks(),
            })
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

    def do_POST(self):
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
