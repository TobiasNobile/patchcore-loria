"""Coût d'une frame, étape par étape : preprocess, embed, faiss, post, encode.

    python bin/bench_live.py --bank_dir models/coco/<tag>

Écrit un JSON par banque dans results/benchmarks/.
"""

import json
import os
import platform
import subprocess
import sys
import time

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

# live.scoring d'abord : il importe torch avant faiss (cf. live/server.py).
from live.scoring import (  # isort: skip
    FAISS_NUM_WORKERS,
    FAISS_ON_GPU,
    build_transform,
    preprocess,
)

import torch

from experiments.runtime import select_device, tune_faiss_small_batches

import patchcore.banks
import patchcore.utils

REPEAT, WARMUP = 15, 3
COLS = ("preprocess", "embed", "faiss", "post", "encode")


def _median(fn):
    for _ in range(WARMUP):
        fn()
    ts = []
    for _ in range(REPEAT):
        t0 = time.perf_counter()
        fn()
        ts.append(1000 * (time.perf_counter() - t0))
    return float(np.median(ts))


def measure(bank_dir):
    """Une ligne de mesures pour une banque. Renvoie un dict sérialisable."""
    tune_faiss_small_batches()
    device = select_device()
    if os.environ.get("BENCH_TORCH_THREADS"):
        torch.set_num_threads(int(os.environ["BENCH_TORCH_THREADS"]))

    pc, cfg = patchcore.banks.load_bank(
        bank_dir, device, FAISS_ON_GPU, FAISS_NUM_WORKERS
    )
    pc.forward_modules.eval()
    if os.environ.get("BENCH_IMAGESIZE"):
        cfg = dict(cfg, imagesize=int(os.environ["BENCH_IMAGESIZE"]))
    transform = build_transform(cfg)
    size = cfg["imagesize"]

    # Frame BGR synthétique, comme celle que rend cv2.VideoCapture.
    frame = (np.random.rand(480, 640, 3) * 255).astype(np.uint8)
    tensor, preview = preprocess(frame, transform, 1.0)
    tensor = tensor.to(torch.float).to(device)

    def _embed():
        with torch.no_grad():
            return pc._embed(tensor, provide_patch_shapes=True)

    feats, _ = _embed()
    feats = np.asarray(feats)

    row = {
        "coreset": ("identity" if cfg.get("sampler_name") == "identity"
                    else "p{:g}".format(cfg.get("coreset_pct", 0))),
        "layers": ",".join(cfg["layers_to_extract_from"]).replace("layer", "l"),
        "bank": cfg["memory_bank_size"],
        "imagesize": size,
        "patches": int(feats.shape[0]),
        "device": str(device),
        "faiss_threads": FAISS_NUM_WORKERS,
        "faiss_gpu": bool(FAISS_ON_GPU),
        "torch_threads": torch.get_num_threads(),
        "gpu": (torch.cuda.get_device_name(0) if device.type == "cuda" else
                platform.processor() or platform.machine()),
    }
    row["preprocess"] = _median(lambda: preprocess(frame, transform, 1.0))
    row["embed"] = _median(_embed)
    row["faiss"] = _median(lambda: pc.anomaly_scorer.predict([feats]))
    full = _median(lambda: pc.predict(tensor))
    # predict() = embed + faiss + unpatch/score/segmentation ; le reste est le post.
    row["post"] = max(full - row["embed"] - row["faiss"], 0.0)

    import cv2

    heatmap = np.zeros((size, size), np.float32)
    from live.scoring import HEATMAP_ALPHA, overlay_heatmap

    def _encode():
        img = overlay_heatmap(preview, heatmap, 175.0, HEATMAP_ALPHA)
        cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 80])

    row["encode"] = _median(_encode)
    row["scoring"] = row["preprocess"] + row["embed"] + row["faiss"] + row["post"]
    row["total"] = row["scoring"] + row["encode"]
    return row


def _print_table(rows):
    r0 = rows[0]
    print("\ndevice : {} ({})".format(r0["device"], r0["gpu"]))
    print("threads: torch {} | faiss {}{}".format(
        r0["torch_threads"], r0["faiss_threads"],
        " (GPU)" if r0["faiss_gpu"] else ""))
    print("couches: {}\n".format(r0["layers"]))
    head = "{:<9} {:>8} {:>6} {:>7} " + " ".join(["{:>10}"] * len(COLS)) + " {:>9} {:>7} {:>7}"
    print(head.format("coreset", "banque", "taille", "patchs",
                      *COLS, "scoring", "total", "fps"))
    print("-" * 108)
    for r in rows:
        print(head.format(
            r["coreset"], r["bank"],
            "{}px".format(r["imagesize"]), r["patches"],
            *["{:.1f}ms".format(r[c]) for c in COLS],
            "{:.1f}ms".format(r["scoring"]),
            "{:.1f}ms".format(r["total"]),
            "{:.1f}".format(1000 / r["total"])))
    print("\nbudget : 33.3 ms = 30 FPS | 16.7 ms = 60 FPS "
          "(colonne 'scoring' = hors encodage JPEG)")


def main(argv):
    if os.environ.get("BENCH_CHILD"):
        print("@@ROW@@" + json.dumps(measure(argv[0])))
        return
    rows = []
    for bank in argv:
        env = dict(os.environ, BENCH_CHILD="1")
        out = subprocess.run(
            [sys.executable, os.path.abspath(__file__), bank],
            env=env, capture_output=True, text=True,
        )
        line = [l for l in out.stdout.splitlines() if l.startswith("@@ROW@@")]
        if not line:
            print("échec sur {} :\n{}".format(bank, out.stderr[-800:]), file=sys.stderr)
            continue
        rows.append(json.loads(line[0][len("@@ROW@@"):]))
    if rows:
        _print_table(rows)


if __name__ == "__main__":
    main(sys.argv[1:])
