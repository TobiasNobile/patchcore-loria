"""Measure PatchCore inference throughput: how many heatmaps per second.

Loads a memory bank built by bin/fit_memory_bank_celeba.py and pushes N test
images through inference, reporting images/second and ms/image for several batch
sizes. No fitting, no PNG rendering -- this measures the half of PatchCore that
would have to run in real time.

    python bin/benchmark_inference_celeba.py
    BENCH_BANK_DIR=models/celeba/..._ts250_s0 python bin/benchmark_inference_celeba.py

What is timed, per image:

    tensor -> backbone (feature extraction) -> FAISS search -> 224x224 heatmap

What is deliberately NOT timed, and why:

  - Image loading/decoding. Images are decoded and transformed once, up front,
    into a tensor. A real deployment gets frames from a camera, not from a
    Hugging Face dataset, so that cost is not representative.
  - PNG rendering. matplotlib takes ~100 ms per figure, far more than the
    inference itself; timing it would measure matplotlib, not PatchCore. A real
    deployment consumes the heatmap array directly.
  - Warm-up batches. The first batches pay CUDA kernel compilation and cuDNN
    autotuning and are 10-100x slower than steady state. They are excluded from
    the numbers but still measured and reported (warmup_seconds), so the choice
    is auditable rather than hidden.

Mean and percentiles answer different questions, and a real-time claim needs
both. The mean (and images_per_second) is capacity: how many images fit in a
second. The percentiles are deadline reliability: at 30 fps the budget is 33
ms/image, and a 20 ms mean with an 80 ms p99 still blows that budget on one
image in a hundred -- a dropped frame every ~3 seconds that the mean alone hides.
A p50 and p99 far apart mean jitter (GPU scheduling, allocation, thermal
throttling) rather than a slow model.

Writes a JSON per bank so runs over several banks can be aggregated into a
throughput-vs-bank-size curve -- bank size is what inference cost scales with.
"""

import json
import logging
import os
import time

import numpy as np
import torch

import patchcore.banks
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)


# Overridable by env var (BENCH_BANK_DIR) so one job can sweep several banks.
BANK_DIR = "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts2000_s0"
BANK_DIR = os.environ.get("BENCH_BANK_DIR", BANK_DIR)

# Number of TEST images per batch size
N_IMAGES = 500

# batch=1 is the real-time number: the latency of one frame handled on its own.
BATCH_SIZES = [1, 8, 32]

# Batches discarded before the clock starts (they pay CUDA/cuDNN warm-up).
# not take in account for inference measuring
WARMUP_BATCHES = 3

OUTPUT_DIR = os.environ.get("BENCH_OUTPUT_DIR", "results/benchmarks")

# GPU/BENCH_DEVICE decides where the *backbone* runs (the embed phase),
# BENCH_FAISS_GPU decides where the *search* runs. "Full GPU" needs both. The

GPU = [0]  # [] forces CPU.
if os.environ.get("BENCH_DEVICE", "").lower() == "cpu":
    GPU = []

FAISS_ON_GPU = False
if os.environ.get("BENCH_FAISS_GPU", "").lower() in ("1", "true", "yes"):
    FAISS_ON_GPU = True

FAISS_NUM_WORKERS = 4

def preload_images(dataset, indices):
    """Decode + transform every image once, up front, into one CPU tensor.

    This is what keeps the measurement about inference: the timed loop then only
    slices this tensor, so no JPEG decoding happens between the clocks.
    """
    return torch.stack([dataset[i]["image"] for i in indices])


def timed_predict(patchcore_instance, batch, device):
    """One batch through inference, timed per phase. Returns (embed, search, post).

    This re-implements the body of PatchCore._predict instead of calling it,
    because the whole question is which phase costs what:

      embed   backbone forward + patchify + aggregation, ending with the
              features pulled back to numpy. This is the phase that moves
              between CPU and GPU, so it is the one a CPU-vs-GPU comparison is
              about. It never touches the memory bank, so it is independent of
              bank size.
      search  the FAISS nearest-neighbour lookup. Runs on CPU with faiss-cpu
              whatever `device` says, and is the phase that grows with bank size.
      post    unpatching plus the upsampling to the 224x224 heatmap.

    Keep in sync with src/patchcore/patchcore.py::_predict.

    CUDA kernels are asynchronous: without synchronize() around a phase we would
    time how long it takes to *queue* the work, not to run it.
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
    """Warm up, then time N images at this batch size. Returns a metrics dict."""
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

    # Per-batch time divided by that batch's size. Only at batch=1 is one
    # measurement one image, which is what makes the percentiles below true
    # latencies there: above batch=1 each value is a within-batch average, so a
    # single slow image is diluted by its neighbours and the tail is smoothed
    # away. Read the percentiles at batch=1, the throughput at batch 8/32.
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
        "ms_per_image_search": 1000.0 * float(search.sum()) / n_images, # FAISS search = nearest neighbor, same on GPU and CPU, grows with bank size
        "ms_per_image_postprocess": 1000.0 * float(post.sum()) / n_images, # interpolation bilinéaire (224x224) + flou gaussien
        # percentiles : one value per batch = (time, batch size)
        # for size > 1 -> time = mean of 'images times' in the batch
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
    # Both switches go in the filename. Runs of the same bank would otherwise
    # collide and the later one would silently destroy the number the earlier was
    # measured for -- and "_cuda" alone would suggest a full-GPU result even when
    # the search ran on CPU, which is exactly the claim not to make.
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
