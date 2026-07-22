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
    FAISS_ON_GPU,
    GPU,
    HEATMAP_ALPHA,
    HEATMAP_VMAX,
    HEATMAP_VMIN,
    SMOOTH_WINDOW,
    build_transform,
    preprocess,
)

import patchcore.banks
import patchcore.utils

LOGGER = logging.getLogger(__name__)

MODELS_ROOT = "models"


def overlay_heatmap(preview_rgb, heatmap):
    """Vignette + heatmap jet, en BGR — le render() de live_camera sans l'ATH,
    que la page affiche déjà en HTML. Échelle fixe : un autoscale par image
    ferait clignoter la heatmap et interdirait de comparer deux frames."""
    normalized = np.clip(
        (heatmap - HEATMAP_VMIN) / max(HEATMAP_VMAX - HEATMAP_VMIN, 1e-6), 0, 1
    )
    colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    frame = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(colored, HEATMAP_ALPHA, frame, 1 - HEATMAP_ALPHA, 0)


def find_banks():
    """Les dossiers de models/ qui portent un fit_config.json, pour le menu."""
    banks = []
    for root, dirs, files in os.walk(MODELS_ROOT):
        if "fit_config.json" in files:
            banks.append(root)
            dirs[:] = []  # une banque n'en contient pas une autre
    return sorted(banks)


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
        self._jpeg = None  # dernière vignette encodée, servie en MJPEG
        self._state = {
            "running": False,
            "score": None,
            "fps": 0.0,
            "infer_ms": 0.0,
            "frames": 0,
            "verdict": None,
            "error": None,
            "params": None,
        }

    def state(self):
        with self._lock:
            return dict(self._state)

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
            device = patchcore.utils.set_torch_device(GPU)
            patchcore_instance, fit_config = patchcore.banks.load_bank(
                params["bank_dir"], device, FAISS_ON_GPU, FAISS_NUM_WORKERS
            )
            patchcore.utils.fix_seeds(fit_config["seed"])
            transform = build_transform(fit_config)

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

                tensor, preview = preprocess(frame, transform, params["zoom"])

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
                    ".jpg", overlay_heatmap(preview, heatmap),
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

PAGE = """<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PatchCore live</title>
<style>
  body { background:#fff; color:#111; font:14px -apple-system,BlinkMacSystemFont,sans-serif;
         margin:0; padding:48px; display:flex; flex-direction:column; align-items:center; gap:32px; }
  #score { font-size:96px; font-weight:600; font-variant-numeric:tabular-nums; line-height:1; }
  #verdict { font-size:18px; letter-spacing:.08em; text-transform:uppercase; min-height:22px; }
  .ok { color:#0a7a2f; } .anomalie { color:#c01919; } .idle { color:#999; }
  #meta { color:#888; font-size:12px; min-height:16px; }
  form { display:grid; grid-template-columns:auto 1fr; gap:10px 16px; align-items:center;
         width:min(560px,100%); }
  label { color:#555; }
  label small { display:block; color:#aaa; font-size:11px; }
  #cam { width:320px; height:320px; object-fit:cover; border-radius:6px; background:#f5f5f5;
         image-rendering:pixelated; }
  #cam[hidden] { display:none; }
  input, select { font:inherit; padding:6px 8px; border:1px solid #ccc; border-radius:4px;
                  background:#fff; width:100%; box-sizing:border-box; }
  input:disabled, select:disabled { background:#f5f5f5; color:#888; }
  .row { grid-column:1/-1; display:flex; gap:12px; align-items:center; }
  button { font:inherit; padding:8px 20px; border:1px solid #111; border-radius:4px;
           background:#111; color:#fff; cursor:pointer; }
  button.secondary { background:#fff; color:#111; }
  button:disabled { opacity:.35; cursor:default; }
  #error { color:#c01919; font-size:12px; min-height:16px; }
</style>
</head>
<body>
  <img id="cam" hidden alt="">
  <div id="score">—</div>
  <div id="verdict" class="idle">arrêté</div>
  <div id="meta"></div>

  <form id="params" autocomplete="off">
    <label for="bank_dir">Banque<small>images normales de référence</small></label>
    <select id="bank_dir" name="bank_dir">__BANKS__</select>

    <label for="source">Source<small>0 = webcam, ou fichier / URL</small></label>
    <input id="source" name="source" value="0">

    <label for="stride">Stride<small>ne score qu'une frame sur N</small></label>
    <input id="stride" name="stride" type="number" min="1" step="1" value="1">

    <label for="zoom">Zoom<small>recadrage centré (2 = moitié)</small></label>
    <input id="zoom" name="zoom" type="number" min="1" step="0.1" value="1">

    <label for="threshold">Seuil<small>au-delà, verdict anomalie</small></label>
    <input id="threshold" name="threshold" type="number" step="0.1" placeholder="aucun">

    <label for="loop">Boucler<small>rejouer un fichier sans fin</small></label>
    <div><input id="loop" name="loop" type="checkbox" style="width:auto"></div>

    <div class="row">
      <button id="start" type="submit">Démarrer</button>
      <button id="stop" type="button" class="secondary" disabled>Arrêter</button>
      <span id="error"></span>
    </div>
  </form>

<script>
const form = document.getElementById('params');
const fields = [...form.querySelectorAll('input,select')];
const startBtn = document.getElementById('start');
const stopBtn = document.getElementById('stop');

function readParams() {
  const t = document.getElementById('threshold').value;
  return {
    bank_dir: document.getElementById('bank_dir').value,
    source: document.getElementById('source').value,
    stride: parseInt(document.getElementById('stride').value || '1', 10),
    zoom: parseFloat(document.getElementById('zoom').value || '1'),
    threshold: t === '' ? null : parseFloat(t),
    loop: document.getElementById('loop').checked,
  };
}

async function post(path, body) {
  const r = await fetch(path, {method:'POST', headers:{'Content-Type':'application/json'},
                              body: JSON.stringify(body || {})});
  return r.json();
}

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  document.getElementById('error').textContent = '';
  const res = await post('/api/start', readParams());
  if (res.error) document.getElementById('error').textContent = res.error;
  refresh();
});
stopBtn.addEventListener('click', async () => { await post('/api/stop'); refresh(); });

function apply(s) {
  // Le flux MJPEG se termine avec la boucle : on rebranche le <img> à chaque
  // démarrage (URL unique, sinon le navigateur ressert la réponse close).
  const cam = document.getElementById('cam');
  if (s.running && cam.hidden) {
    cam.src = '/stream.mjpg?' + Date.now();
    cam.hidden = false;
  } else if (!s.running && !cam.hidden) {
    cam.hidden = true;
    cam.removeAttribute('src');
  }
  document.getElementById('score').textContent =
    s.score === null || s.score === undefined ? '—' : s.score.toFixed(2);
  const v = document.getElementById('verdict');
  if (!s.running) { v.textContent = 'arrêté'; v.className = 'idle'; }
  else if (s.verdict) { v.textContent = s.verdict; v.className = s.verdict; }
  else { v.textContent = 'en cours'; v.className = 'idle'; }
  document.getElementById('meta').textContent = s.running
    ? `${s.fps.toFixed(1)} fps · ${s.infer_ms.toFixed(0)} ms/inf · ${s.frames} frames` : '';
  document.getElementById('error').textContent = s.error || '';
  // Paramètres figés tant que la boucle tourne : les changer à chaud rendrait
  // les scores incomparables d'une frame à l'autre.
  fields.forEach(f => f.disabled = s.running);
  startBtn.disabled = s.running;
  stopBtn.disabled = !s.running;
}

async function refresh() { apply(await (await fetch('/api/state')).json()); }
refresh();
setInterval(refresh, 300);
</script>
</body>
</html>
"""


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

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            options = "".join(
                '<option value="{0}">{0}</option>'.format(b) for b in find_banks()
            ) or '<option value="">aucune banque dans models/</option>'
            self._send(200, PAGE.replace("__BANKS__", options), "text/html; charset=utf-8")
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
                    "threshold": None if params.get("threshold") is None
                                 else float(params["threshold"]),
                    "loop": bool(params.get("loop")),
                }
            except (ValueError, KeyError, TypeError) as exc:
                self._json({"error": "Paramètres invalides : {}".format(exc)}, 400)
                return
            if not params["bank_dir"]:
                self._json({"error": "Aucune banque sélectionnée."}, 400)
                return
            ok, err = RUNNER.start(params)
            self._json({"ok": ok, "error": err})
        elif self.path == "/api/stop":
            RUNNER.stop()
            self._json({"ok": True})
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
