"""Mesure le débit d'inférence de PatchCore : combien de heatmaps par seconde.

Charge une banque et pousse N images test dans l'inférence, en reportant
images/seconde et ms/image pour plusieurs tailles de batch. Pas de fit, pas de
rendu PNG : on ne mesure que la moitié de PatchCore qui tournerait en temps réel.

    python bin/celeba/infer/benchmark.py
    BENCH_BANK_DIR=models/celeba/..._ts250_s0 python bin/celeba/infer/benchmark.py

Chronométré, par image : tensor -> backbone -> recherche FAISS -> heatmap 224x224.

Volontairement NON chronométré : le chargement/décodage des images (décodées une
fois en amont ; en prod les frames viennent d'une caméra), le rendu PNG
(matplotlib ~100 ms, bien plus que l'inférence), et les batches de warm-up
(compilation CUDA/cuDNN, 10-100x plus lents ; exclus mais reportés via
warmup_seconds).

On reporte moyenne ET percentiles : la moyenne (images_per_second) est la
capacité, les percentiles la fiabilité face à une deadline (à 30 fps le budget
est 33 ms/image ; une moyenne à 20 ms avec un p99 à 80 ms rate le budget une
image sur cent). Un JSON par banque, pour tracer le débit vs la taille de banque.
"""

import json
import logging
import os
import platform
import time

# macOS : torch et faiss-cpu embarquent chacun leur libomp, la seconde à
# s'initialiser fait abort. À poser avant l'import de patchcore, qui charge
# faiss. Le mono-thread ci-dessous complète la parade (multi-thread = segfault).
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import patchcore.banks
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)


# Surchargeable par BENCH_BANK_DIR pour balayer plusieurs banques dans un job.
BANK_DIR = "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts2000_s0"
BANK_DIR = os.environ.get("BENCH_BANK_DIR", BANK_DIR)

# Nombre d'images TEST par taille de batch.
N_IMAGES = 500

# batch=1 = le chiffre temps réel : la latence d'une frame traitée seule.
BATCH_SIZES = [1, 8, 32]

# Batches écartés avant de lancer le chrono (ils paient le warm-up CUDA/cuDNN).
WARMUP_BATCHES = 3

OUTPUT_DIR = os.environ.get("BENCH_OUTPUT_DIR", "results/celeba/benchmarks")

# GPU/BENCH_DEVICE place le backbone, BENCH_FAISS_GPU place la recherche.
# "Full GPU" nécessite les deux.

GPU = [0]  # [] force le CPU.
if os.environ.get("BENCH_DEVICE", "").lower() == "cpu":
    GPU = []

FAISS_ON_GPU = False
if os.environ.get("BENCH_FAISS_GPU", "").lower() in ("1", "true", "yes"):
    FAISS_ON_GPU = True

FAISS_NUM_WORKERS = 1 if platform.system() == "Darwin" else 4

def preload_images(dataset, indices):
    """Décode + transforme chaque image une fois, en amont, en un tensor CPU.

    C'est ce qui garde la mesure sur l'inférence : la boucle chronométrée ne fait
    que slicer ce tensor, aucun décodage JPEG entre les chronos.
    """
    return torch.stack([dataset[i]["image"] for i in indices])


def timed_predict(patchcore_instance, batch, device):
    """Un batch dans l'inférence, chronométré par phase. Renvoie (embed, search, post).

    Réimplémente le corps de PatchCore._predict au lieu de l'appeler, car toute la
    question est de savoir quelle phase coûte quoi :

      embed   backbone + patchify + agrégation, features ramenées en numpy. Phase
              qui bouge entre CPU et GPU (celle que vise une comparaison CPU/GPU),
              indépendante de la taille de banque.
      search  recherche FAISS des plus proches voisins ; tourne sur CPU quel que
              soit `device`, et grandit avec la taille de banque.
      post    unpatch + upsampling vers la heatmap 224x224.

    À garder synchro avec src/patchcore/patchcore.py::_predict. Les noyaux CUDA
    sont asynchrones : sans synchronize() autour d'une phase, on mesurerait le
    temps de mettre le travail en file, pas de l'exécuter.
    """
    is_cuda = device.type == "cuda"
    batchsize = batch.shape[0]

    def sync():
        if is_cuda:
            torch.cuda.synchronize(device)

    sync()
    t0 = time.perf_counter()
    images = batch.to(torch.float).to(device)
    with torch.no_grad():
        features, patch_shapes = patchcore_instance._embed(
            images, provide_patch_shapes=True
        )
        features = np.asarray(features)
    sync()
    embed_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    patch_scores = image_scores = patchcore_instance.anomaly_scorer.predict([features])[0]
    search_seconds = time.perf_counter() - t0

    t0 = time.perf_counter()
    with torch.no_grad():
        image_scores = patchcore_instance.patch_maker.unpatch_scores(
            image_scores, batchsize=batchsize
        )
        image_scores = image_scores.reshape(*image_scores.shape[:2], -1)
        patchcore_instance.patch_maker.score(image_scores)
        patch_scores = patchcore_instance.patch_maker.unpatch_scores(
            patch_scores, batchsize=batchsize
        )
        scales = patch_shapes[0]
        patch_scores = patch_scores.reshape(batchsize, scales[0], scales[1])
        patchcore_instance.anomaly_segmentor.convert_to_segmentation(patch_scores)
    sync()
    post_seconds = time.perf_counter() - t0

    return embed_seconds, search_seconds, post_seconds


def time_batches(patchcore_instance, images, batch_size, device, max_batches=None):
    """Run inference batch by batch, returning per-batch (embed, search, post)."""
    rows = []
    for start in range(0, images.shape[0], batch_size):
        batch = images[start : start + batch_size]
        rows.append(timed_predict(patchcore_instance, batch, device))
        if max_batches is not None and len(rows) >= max_batches:
            break
    return rows


def measure(patchcore_instance, images, batch_size, device):
    """Warm-up, puis chronomètre N images à cette taille de batch. Renvoie un dict."""
    warmup = time_batches(
        patchcore_instance, images, batch_size, device, max_batches=WARMUP_BATCHES
    )
    rows = time_batches(patchcore_instance, images, batch_size, device)

    n_images = int(images.shape[0])
    embed = np.array([r[0] for r in rows])
    search = np.array([r[1] for r in rows])
    post = np.array([r[2] for r in rows])
    times = embed + search + post
    total_seconds = float(np.sum(times))

    # Temps par batch / taille. Seul batch=1 donne une mesure par image, donc
    # de vraies latences ; au-delà chaque valeur est une moyenne intra-batch.
    # Percentiles à lire à batch=1, débit à batch 8/32.
    per_image_ms = []
    for i, t in enumerate(times):
        this_batch = min(batch_size, n_images - i * batch_size)
        per_image_ms.append(1000.0 * t / this_batch)
    per_image_ms = np.asarray(per_image_ms)

    return {
        "batch_size": batch_size,
        "n_images": n_images,
        "n_batches": len(times),
        "total_seconds": total_seconds, # somme des secondes de chaque batch = le temps total
        "images_per_second": n_images / total_seconds,
        "images_per_minute": 60.0 * n_images / total_seconds,
        "ms_per_image_mean": 1000.0 * total_seconds / n_images,
        "ms_per_image_embed": 1000.0 * float(embed.sum()) / n_images, # embed = backbone, patchify, interpolation, agrégation, rappartriement des features -> indé de bank size
        "ms_per_image_search": 1000.0 * float(search.sum()) / n_images, # recherche FAISS = plus proche voisin, pareil GPU et CPU, grandit avec la banque
        "ms_per_image_postprocess": 1000.0 * float(post.sum()) / n_images, # interpolation bilinéaire (224x224) + flou gaussien
        # percentiles : une valeur par batch = (temps, taille de batch)
        # pour size > 1, temps = moyenne des temps d'images du batch
        "ms_per_image_p50": float(np.percentile(per_image_ms, 50)),
        "ms_per_image_p90": float(np.percentile(per_image_ms, 90)), # 90% des images prennent moins ou autant de temps
        "ms_per_image_p99": float(np.percentile(per_image_ms, 99)),
        "warmup_batches": len(warmup), # nombre de batches écartés avant le chrono
        "warmup_seconds": float(np.sum([sum(r) for r in warmup])), # temps total du warmup = premiers passages dans modèle où cuDNN teste plusieurs algos, compilation des noyaux CUDA, allocation mémoire, ...
    }


def main():
    device = patchcore.utils.set_torch_device(GPU)

    patchcore_instance, fit_config = patchcore.banks.load_bank(
        BANK_DIR, device, FAISS_ON_GPU, FAISS_NUM_WORKERS
    )
    seed = fit_config["seed"]
    patchcore.utils.fix_seeds(seed)

    test_dataset = CelebADataset(
        resize=fit_config["resize"],
        imagesize=fit_config["imagesize"],
        split=DatasetSplit.TEST,
        seed=seed,
    )
    n = min(N_IMAGES, len(test_dataset))
    rng = np.random.RandomState(seed)
    indices = rng.choice(len(test_dataset), n, replace=False).tolist()

    LOGGER.info("Preloading %d test images (not timed)...", n)
    images = preload_images(test_dataset, indices)

    device_is_cuda = device.type == "cuda"
    context = {
        "bank_dir": BANK_DIR,
        "bank_tag": os.path.basename(os.path.normpath(BANK_DIR)),
        "memory_bank_size": fit_config["memory_bank_size"],
        "train_subset": fit_config["train_subset"],
        "coreset_pct": fit_config["coreset_pct"],
        "backbone_name": fit_config["backbone_name"],
        "imagesize": fit_config["imagesize"],
        "n_images": n,
        "faiss_on_gpu": FAISS_ON_GPU,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device_is_cuda else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "cpu",
    }

    results = []
    for batch_size in BATCH_SIZES:
        LOGGER.info("Benchmarking batch_size=%d ...", batch_size)
        results.append(measure(patchcore_instance, images, batch_size, device))

    LOGGER.info(
        "Bank %s: %d patch features, on %s",
        context["bank_tag"], context["memory_bank_size"], context["gpu_name"],
    )
    LOGGER.info(
        "%-6s %10s %10s %9s %9s %9s %9s %8s",
        "batch", "img/s", "img/min", "ms/img", "embed", "search", "post", "p99 ms",
    )
    for r in results:
        LOGGER.info(
            "%-6d %10.1f %10.0f %9.1f %9.1f %9.1f %9.1f %8.1f",
            r["batch_size"], r["images_per_second"], r["images_per_minute"],
            r["ms_per_image_mean"], r["ms_per_image_embed"],
            r["ms_per_image_search"], r["ms_per_image_postprocess"],
            r["ms_per_image_p99"],
        )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # Les deux switches dans le nom de fichier : sinon collision entre runs, et
    # "_cuda" seul laisserait croire à un full-GPU avec recherche sur CPU.
    out_path = os.path.join(
        OUTPUT_DIR,
        "bench_{}_{}_{}.json".format(
            context["bank_tag"],
            device.type,
            "faissgpu" if FAISS_ON_GPU else "faisscpu",
        ),
    )
    with open(out_path, "w") as fh:
        json.dump({**context, "results": results}, fh, indent=2)
    LOGGER.info("Saved benchmark to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
