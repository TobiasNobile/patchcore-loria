"""Les trois pipelines communs aux datasets : fit, histogramme, heatmaps.

Chaque script de bin/<dataset>/ déclare un `Spec` (comment construire le dataset,
comment nommer ses deux classes) et appelle la fonction correspondante.
"""

import json
import logging
import os
import platform
import resource
import time
from dataclasses import dataclass
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.backbones
import patchcore.banks
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.tracking
import patchcore.utils
from experiments.metrics import histogram_jaccard, normalized_wasserstein, t_test_scores
from experiments.runtime import select_device

LOGGER = logging.getLogger(__name__)

COLOR_NORMAL = "#5B8FB9"
COLOR_ANOMALY = "#E8A33D"

BATCH_SIZE = 8
NUM_WORKERS = 8
BINS = 50


@dataclass
class Spec:
    """Ce qui distingue un dataset d'un autre dans ces pipelines.

    `build` reçoit le fit_config quand il existe (l'inférence y relit le chemin
    des images) ; il vaut None au fit."""

    name: str
    build: Callable
    normal: str
    anomaly: str


def _env_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def _faiss(prefix):
    threads = "1" if platform.system() == "Darwin" else "4"
    return (
        _env_flag(prefix + "_FAISS_GPU"),
        int(os.environ.get(prefix + "_FAISS_THREADS", threads)),
    )


def _load_bank(bank_dir, prefix):
    device = select_device()
    on_gpu, threads = _faiss(prefix)
    instance, fit_config = patchcore.banks.load_bank(bank_dir, device, on_gpu, threads)
    patchcore.utils.fix_seeds(fit_config["seed"])
    return device, instance, fit_config


def _balanced_subset(dataset, n_per_class, seed, spec):
    """Index tirés par classe AVANT de scorer : les labels sont connus sans
    inférence, inutile de scorer ce qu'on ne tracera pas."""
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


# ─── Fit ───────────────────────────────────────────────────────────────────

def _sampler(name, pct, device, proj_dim):
    if name == "identity":
        return patchcore.sampler.IdentitySampler()
    kinds = {
        "greedy_coreset": patchcore.sampler.GreedyCoresetSampler,
        "approx_greedy_coreset": patchcore.sampler.ApproximateGreedyCoresetSampler,
    }
    if name not in kinds:
        raise ValueError("Unknown FIT_SAMPLER: {}".format(name))
    return kinds[name](pct, device, dimension_to_project_features_to=proj_dim)


def fit_settings(models_dir, coreset_pct, train_subset, overrides=None):
    """Les réglages du fit, tous surchargeables par l'environnement.

    `overrides` l'emporte sur l'environnement : il porte des valeurs saisies
    explicitement (l'interface web), là où les variables d'environnement sont
    le réglage ambiant des scripts. Les clés valant None sont ignorées.
    """
    subset = os.environ.get("FIT_TRAIN_SUBSET")
    if subset:
        train_subset = None if subset.lower() in ("none", "all") else int(subset)
    layers = os.environ.get("FIT_LAYERS")
    imagesize = int(os.environ.get("FIT_IMAGESIZE", "224"))
    on_gpu, threads = _faiss("FIT")
    settings = {
        "seed": 0,
        "train_subset": train_subset,
        "backbone_name": os.environ.get("FIT_BACKBONE", "wideresnet50"),
        "layers_to_extract_from": (
            [l.strip() for l in layers.split(",") if l.strip()]
            if layers else ["layer2", "layer3"]
        ),
        "pretrain_embed_dimension": 1024,
        "target_embed_dimension": 1024,
        "patchsize": 3,
        "anomaly_scorer_num_nn": int(os.environ.get("FIT_NUM_NN", "1")),
        "sampler_name": os.environ.get("FIT_SAMPLER", "approx_greedy_coreset"),
        "coreset_pct": float(os.environ.get("FIT_CORESET_PCT", coreset_pct)),
        "coreset_proj_dim": int(os.environ.get("FIT_CORESET_PROJ_DIM", "128")),
        "imagesize": imagesize,
        # Le resize garde le rapport 256/224, donc le même cadrage.
        "resize": int(os.environ.get("FIT_RESIZE", str(round(imagesize * 256 / 224)))),
        "faiss_on_gpu": on_gpu,
        "_faiss_threads": threads,
        "_models_dir": os.environ.get("FIT_MODELS_DIR", models_dir),
    }
    settings.update({k: v for k, v in (overrides or {}).items() if v is not None})
    # `resize` est dérivé d'imagesize : le laisser au défaut après une surcharge
    # d'imagesize donnerait un cadrage incohérent avec la banque.
    if overrides and "imagesize" in overrides and "resize" not in overrides:
        settings["resize"] = round(settings["imagesize"] * 256 / 224)
    return settings


def build_tag(cfg):
    """Suffixes de couche et de taille seulement s'ils sortent du défaut, pour
    que les banques déjà construites gardent leur nom."""
    sampler = cfg["sampler_name"]
    if sampler != "identity":
        sampler = "{}_p{:g}".format(sampler, cfg["coreset_pct"])
    layers = cfg["layers_to_extract_from"]
    layers = "" if layers == ["layer2", "layer3"] else "_" + "-".join(
        l.replace("layer", "l") for l in layers
    )
    size = "" if cfg["imagesize"] == 224 else "_im{}".format(cfg["imagesize"])
    return "{}{}{}_{}_ts{}_s{}".format(
        cfg["backbone_name"], layers, size, sampler,
        cfg["train_subset"] or "all", cfg["seed"],
    )


class _ProgressLoader:
    """Proxy d'itération sur le DataLoader, pour suivre l'avancement du fit.

    PatchCore.fit() parcourt son argument sous tqdm et n'offre aucun rappel :
    on s'insère ici plutôt que dans patchcore/, laissé conforme à l'amont. tqdm
    appelle len() sur ce qu'il enveloppe, d'où le __len__.
    """

    def __init__(self, loader, n_images, callback):
        self._loader = loader
        self._n_images = n_images
        self._callback = callback

    def __len__(self):
        return len(self._loader)

    def __iter__(self):
        batches = len(self._loader)
        for i, batch in enumerate(self._loader, 1):
            yield batch
            # En images plutôt qu'en lots : c'est l'unité que l'utilisateur a
            # saisie. Le dernier lot est incomplet, d'où le plafond.
            self._callback(min(i * BATCH_SIZE, self._n_images), self._n_images)


def _subset_indices(n_total, n_wanted, seed, random_subset):
    """Les indices retenus pour le fit. Triés dans les deux cas : l'ordre de
    parcours ne change rien au résultat, et un ordre croissant lit le disque
    séquentiellement."""
    if n_wanted is None or n_wanted >= n_total:
        return None
    if not random_subset:
        return range(n_wanted)
    rng = np.random.RandomState(seed)
    return sorted(rng.choice(n_total, n_wanted, replace=False).tolist())


def run_fit(spec, models_dir, coreset_pct, train_subset=None, extra_config=None,
            progress=None, random_subset=False, overrides=None,
            num_workers=NUM_WORKERS):
    """Construit la banque sur le split TRAIN et l'écrit dans <models_dir>/<tag>.

    `progress(phase, done, total)` suit l'avancement (total nul = indéterminé).
    `random_subset` tire les images au hasard plutôt que de prendre les
    premières — sous le même seed, donc reproductible. `num_workers` à 0 charge
    les images sur le thread appelant, sans processus fils. Renvoie le dossier
    écrit.
    """
    cfg = fit_settings(models_dir, coreset_pct, train_subset, overrides)
    threads, save_root = cfg.pop("_faiss_threads"), cfg.pop("_models_dir")
    seed, imagesize = cfg["seed"], cfg["imagesize"]
    notify = progress or (lambda *a: None)

    device = select_device()
    patchcore.utils.fix_seeds(seed)

    dataset = spec.build(split="train", resize=cfg["resize"], imagesize=imagesize, seed=seed)
    indices = _subset_indices(len(dataset), cfg["train_subset"], seed, random_subset)
    if indices is not None:
        dataset = torch.utils.data.Subset(dataset, indices)
    LOGGER.info("Fitting on %d %s train images.", len(dataset), spec.normal)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    backbone = patchcore.backbones.load(cfg["backbone_name"])
    backbone.name, backbone.seed = cfg["backbone_name"], None

    instance = patchcore.patchcore.PatchCore(device)
    instance.load(
        backbone=backbone,
        layers_to_extract_from=cfg["layers_to_extract_from"],
        device=device,
        input_shape=(3, imagesize, imagesize),
        pretrain_embed_dimension=cfg["pretrain_embed_dimension"],
        target_embed_dimension=cfg["target_embed_dimension"],
        patchsize=cfg["patchsize"],
        featuresampler=_sampler(
            cfg["sampler_name"], cfg["coreset_pct"], device, cfg["coreset_proj_dim"]
        ),
        anomaly_scorer_num_nn=cfg["anomaly_scorer_num_nn"],
        nn_method=patchcore.common.FaissNN(cfg["faiss_on_gpu"], threads),
    )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    # La sélection du coreset suit l'extraction, dans le même appel et sans
    # rappel possible : on bascule de phase au dernier lot. Total nul = barre
    # indéterminée, faute de savoir combien de temps elle prendra.
    n_images = len(dataset)

    def _features_done(done, total):
        if done >= total:
            notify("coreset", 0, 0)
        else:
            notify("features", done, total)

    notify("features", 0, n_images)
    instance.fit(_ProgressLoader(dataloader, n_images, _features_done))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - t0

    bank = np.asarray(instance.anomaly_scorer.detection_features)
    # ru_maxrss : Ko sur Linux, octets sur macOS.
    maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gb = maxrss / (1024 ** 2 if platform.system() == "Linux" else 1024 ** 3)
    LOGGER.info(
        "Fit took %.1f s. Memory bank holds %d patch features. Peak RSS %.1f Go.",
        fit_seconds, len(bank), peak_rss_gb,
    )

    config = {
        **(extra_config or {}),
        **cfg,
        "device": str(device),
        "node": platform.node(),
        "cluster": platform.node().split(".")[0].split("-")[0],
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "n_train_images": len(dataset),
        "memory_bank_size": int(len(bank)),
        "feature_dim": int(bank.shape[1]),
        "bank_gb": len(bank) * int(bank.shape[1]) * 4 / 1024 ** 3,
        "peak_rss_gb": peak_rss_gb,
        "fit_seconds": fit_seconds,
    }
    notify("sauvegarde", 0, 0)
    save_dir = os.path.join(save_root, build_tag(cfg))
    if _env_flag("FIT_NO_SAVE"):
        # Mesures seules : une banque identity peut peser des centaines de Go.
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "fit_config.json"), "w") as fh:
            json.dump(config, fh, indent=2)
        LOGGER.info("FIT_NO_SAVE : banque non persistée, mesures dans %s", save_dir)
    else:
        patchcore.banks.save_bank(instance, save_dir, config)
    return save_dir


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
        **{k: fit_config[k] for k in (
            "seed", "train_subset", "backbone_name", "sampler_name",
            "coreset_pct", "resize", "imagesize", "memory_bank_size",
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
    # Sidecar garanti, MLflow best-effort : son file store sur NFS lève un
    # "Stale file handle" quand des tâches parallèles se disputent le verrou.
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
    # None = autoscale par image (incomparable d'une image à l'autre) ; deux
    # bornes fixes = échelle commune sur un lot.
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
