"""Les pipelines de mesure : relire une banque, en tirer un histogramme ou des
heatmaps. Rien ici ne sert à la page live, qui ne fait que fitter et scorer.

Chaque script de bin/<dataset>/infer/ déclare un `Spec` (experiments/benchmarks.py
ou experiments/datasets.py) et appelle la fonction correspondante.
"""

import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.banks
import patchcore.metrics
import patchcore.tracking
import patchcore.utils
from experiments.metrics import histogram_jaccard, normalized_wasserstein, t_test_scores
from experiments.pipelines import BATCH_SIZE
from experiments.pipelines import NUM_WORKERS
from experiments.pipelines import faiss_settings
from experiments.runtime import select_device

LOGGER = logging.getLogger(__name__)

COLOR_NORMAL = "#5B8FB9"
COLOR_ANOMALY = "#E8A33D"

BINS = 50

def _load_bank(bank_dir, prefix):
    device = select_device()
    on_gpu, threads = faiss_settings(prefix)
    instance, fit_config = patchcore.banks.load_bank(bank_dir, device, on_gpu, threads)
    patchcore.utils.fix_seeds(fit_config["seed"])
    return device, instance, fit_config


def _balanced_subset(dataset, n_per_class, seed, spec):
    """Index tirés par classe avant de scorer : les labels sont connus sans inférence."""
    labels = np.asarray(dataset.labels, dtype=int)
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]
    n = min(n_per_class, len(normal_idx), len(anomaly_idx))
    if n < n_per_class:
        LOGGER.warning(
            "n_per_class=%d demandé mais %d %s / %d %s disponibles => n=%d.",
            n_per_class, len(normal_idx), spec.normal, len(anomaly_idx), spec.anomaly, n,
        )
    rng = np.random.RandomState(seed)
    selected = np.concatenate([
        rng.choice(normal_idx, n, replace=False),
        rng.choice(anomaly_idx, n, replace=False),
    ])
    return selected, n, len(normal_idx), len(anomaly_idx)


def _timed_predict(instance, dataloader, device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    out = instance.predict(dataloader)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return out, time.perf_counter() - t0
# ─── Histogramme ───────────────────────────────────────────────────────────

def _pct_subdir(output_path, fit_config):
    """Range les figures par taux de compression. Idempotent."""
    subdir = (
        "identity" if fit_config.get("sampler_name") == "identity"
        else "p{:g}".format(fit_config.get("coreset_pct"))
    )
    head, base = os.path.split(output_path)
    if os.path.basename(head) == subdir:
        return output_path
    return os.path.join(head, subdir, base)


def _override_num_nn(instance, fit_config):
    """k est un paramètre de requête, pas de construction : surchargeable sans
    re-fitter. imagelevel_nn a capturé l'ancien k dans sa closure -> rebind."""
    num_nn = int(fit_config.get("anomaly_scorer_num_nn", 1))
    env = os.environ.get("HIST_NUM_NN")
    if env:
        num_nn = int(env)
        scorer = instance.anomaly_scorer
        scorer.n_nearest_neighbours = num_nn
        scorer.imagelevel_nn = lambda q, s=scorer, k=num_nn: s.nn_method.run(k, q)
        LOGGER.info("num_nn surchargé à %d (HIST_NUM_NN).", num_nn)
    return num_nn


def run_histogram(spec, bank_dir, output_path, log_project, n_per_class):
    """Score le split TEST équilibré et superpose les deux distributions."""
    bank_dir = os.environ.get("HIST_BANK_DIR", bank_dir)
    device, instance, fit_config = _load_bank(bank_dir, "HIST")
    output_path = _pct_subdir(os.environ.get("HIST_OUTPUT_PATH", output_path), fit_config)
    num_nn = _override_num_nn(instance, fit_config)
    seed = fit_config["seed"]

    dataset = spec.build(
        split="test", resize=fit_config["resize"],
        imagesize=fit_config["imagesize"], seed=seed, fit_config=fit_config,
    )
    selected, n, n_normal, n_anomaly = _balanced_subset(dataset, n_per_class, seed, spec)
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selected.tolist()), batch_size=BATCH_SIZE,
        shuffle=False, num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
    )

    params = {
        "bank_dir": bank_dir,
        "num_nn": num_nn,
        "n_per_class_requested": n_per_class,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "cpu",
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        # Sans les couches, deux runs ne différant que par elles sont indistinguables.
        **{k: fit_config[k] for k in (
            "seed", "train_subset", "backbone_name", "layers_to_extract_from",
            "sampler_name", "coreset_pct", "resize", "imagesize",
            "memory_bank_size", "patchsize",
        )},
        **{k: fit_config[k] for k in ("node", "cluster") if k in fit_config},
    }

    LOGGER.info("Scoring %d test images (n=%d par classe)...", 2 * n, n)
    (scores, _, labels_gt, _), scoring_seconds = _timed_predict(
        instance, dataloader, device
    )

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels_gt, dtype=int)
    normal_scores, anomaly_scores = scores[labels == 0], scores[labels == 1]
    LOGGER.info("Histogramme sur n=%d par classe (%s vs %s).", n, spec.normal, spec.anomaly)

    sampled = np.concatenate([normal_scores, anomaly_scores])
    auroc = float("nan")
    if n > 0:
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            sampled, np.concatenate([np.zeros(n, int), np.ones(n, int)])
        )["auroc"]
    wass = normalized_wasserstein(normal_scores, anomaly_scores)
    p_value, _ = t_test_scores(normal_scores, anomaly_scores)

    lo, hi = float(sampled.min()), float(sampled.max())
    edges = np.linspace(lo, max(hi, lo + 1e-6), BINS + 1)
    jac = histogram_jaccard(normal_scores, anomaly_scores, edges)

    tag = fit_config["sampler_name"]
    if tag != "identity":
        tag = "{} p={}".format(tag, fit_config["coreset_pct"])
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores, bins=edges, alpha=0.65, color=COLOR_NORMAL,
             label="{}  n={}".format(spec.normal, n))
    plt.hist(anomaly_scores, bins=edges, alpha=0.65, color=COLOR_ANOMALY,
             label="{}  n={}".format(spec.anomaly, n))
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Nombre d'instances")
    plt.title("{}  |  ts={}  |  nn={}  |  AUROC={:.3f}  |  W1n={:.3f}  |  J={:.3f}".format(
        tag, fit_config["train_subset"], num_nn,
        auroc, wass["w1_normalized"], jac["jaccard"],
    ))
    plt.legend()
    plt.grid(True, alpha=0.2)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close()
    LOGGER.info("Saved histogram to %s", output_path)

    metrics = {
        "jaccard": jac["jaccard"],
        "jaccard_intersection": jac["intersection"],
        "jaccard_union": jac["union"],
        "num_nn_used": num_nn,
        "bins": BINS,
        "w1_normalized": wass["w1_normalized"],
        "w1": wass["w1"],
        "p_value": p_value,
        "pooled_std": wass["pooled_std"],
        "auroc": auroc,
        "scoring_seconds": scoring_seconds,
        "n_per_class_used": int(n),
        "normal_score_mean": float(np.mean(normal_scores)),
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "n_normal_available": int(n_normal),
        "n_anomaly_available": int(n_anomaly),
    }
    sidecar = os.path.splitext(output_path)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump({**params, **metrics}, fh, indent=2)
    LOGGER.info("Saved metrics to %s", sidecar)

    run_name = "hist-ts{}-p{:g}-nn{}".format(
        fit_config["train_subset"], fit_config["coreset_pct"], num_nn
    )
    # Sidecar garanti, MLflow best-effort : son file store NFS lâche en parallèle.
    try:
        with patchcore.tracking.patchcore_run(
            experiment=log_project, run_name=run_name, params=params
        ) as run:
            run.log_metrics(metrics)
            run.log_artifacts(output_path)
    except Exception as exc:  # noqa: BLE001 - MLflow ne doit jamais tuer le run
        LOGGER.warning(
            "Logging MLflow échoué (%s) — figure et métriques déjà sauvées (%s, %s).",
            exc, output_path, sidecar,
        )


# ─── Heatmaps ──────────────────────────────────────────────────────────────

def _save_overlay(image, heatmap, score, anomaly, ms, out_path, vmin, vmax):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)
    plt.colorbar(label="Score d'anomalie (patch)")
    plt.axis("off")
    plt.title("anomaly={}  score={:.3f}  |  {:.0f} ms".format(anomaly, score, ms))
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    LOGGER.info("Saved overlay to %s", out_path)


def run_heatmaps(spec, bank_dir, output_path, image_index, n_per_class):
    """Superpose la heatmap sur une image (image_index) ou sur n par classe."""
    bank_dir = os.environ.get("HIST_BANK_DIR", bank_dir)
    output_path = os.environ.get("HEATMAP_OUTPUT_PATH", output_path)
    # None = autoscale par image ; deux bornes = échelle commune sur un lot.
    vmin = float(os.environ.get("HEATMAP_VMIN", "0"))
    vmax = float(os.environ.get("HEATMAP_VMAX", "10"))

    device, instance, fit_config = _load_bank(bank_dir, "INFER")
    seed = fit_config["seed"]
    dataset = spec.build(
        split="test", resize=fit_config["resize"],
        imagesize=fit_config["imagesize"], seed=seed, fit_config=fit_config,
    )
    mean = np.array(dataset.transform_mean).reshape(-1, 1, 1)
    std = np.array(dataset.transform_std).reshape(-1, 1, 1)

    out_dir = None
    if n_per_class > 0:
        selected = _balanced_subset(dataset, n_per_class, seed, spec)[0].tolist()
        out_dir = os.path.join(
            os.path.dirname(output_path) or "results/{}/heatmaps".format(spec.name),
            "ts{}_p{}".format(fit_config["train_subset"], fit_config["coreset_pct"]),
        )
    else:
        selected = [image_index]

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, selected), batch_size=BATCH_SIZE
    )
    (scores, segmentations, _, _), seconds = _timed_predict(instance, dataloader, device)
    ms = 1000.0 * seconds / max(len(selected), 1)
    LOGGER.info("Scored %d image(s), %.1f ms/image.", len(selected), ms)

    for pos, idx in enumerate(selected):
        sample = dataset[idx]
        image = np.clip(
            (sample["image"].numpy() * std + mean) * 255, 0, 255
        ).astype(np.uint8).transpose(1, 2, 0)
        out_path = (
            os.path.join(out_dir, "{}_idx{}.png".format(sample["anomaly"], idx))
            if out_dir else output_path.format(idx=idx)
        )
        _save_overlay(
            image, np.array(segmentations[pos]), scores[pos], sample["anomaly"],
            ms, out_path, vmin, vmax,
        )
