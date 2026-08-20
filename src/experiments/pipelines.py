"""Le fit : construire une banque mémoire à partir d'un `Spec`.

Un `Spec` dit comment construire le dataset et comment nommer ses deux classes.
Les pipelines de mesure qui relisent une banque (histogramme, heatmaps) ne
servent pas à la page live et vivent à part.
"""

import json
import logging
import os
import platform
import resource
import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch

import patchcore.backbones
import patchcore.banks
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from experiments.runtime import select_device

LOGGER = logging.getLogger(__name__)

BATCH_SIZE = 8
NUM_WORKERS = 8


@dataclass
class Spec:
    """Ce qui distingue un dataset d'un autre. `build` reçoit le fit_config à
    l'inférence, None au fit.
    """

    name: str
    build: Callable
    normal: str
    anomaly: str


def _env_flag(name):
    return os.environ.get(name, "").lower() in ("1", "true", "yes")


def faiss_settings(prefix):
    threads = "1" if platform.system() == "Darwin" else "4"
    return (
        _env_flag(prefix + "_FAISS_GPU"),
        int(os.environ.get(prefix + "_FAISS_THREADS", threads)),
    )



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
    """Les réglages du fit, surchargeables par l'environnement.
    
    `overrides` l'emporte sur l'environnement : valeurs saisies explicitement
    (l'interface) contre réglage ambiant des scripts. Les None sont ignorés.
    """
    subset = os.environ.get("FIT_TRAIN_SUBSET")
    if subset:
        train_subset = None if subset.lower() in ("none", "all") else int(subset)
    layers = os.environ.get("FIT_LAYERS")
    imagesize = int(os.environ.get("FIT_IMAGESIZE", "224"))
    on_gpu, threads = faiss_settings("FIT")
    settings = {
        # Surchargeable, à la différence de l'amont : sans ça, pas de barre d'erreur.
        "seed": int(os.environ.get("FIT_SEED", "0")),
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
    # resize dérive d'imagesize : le figer donnerait un cadrage incohérent.
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
    
    PatchCore.fit() n'offre aucun rappel ; on s'insère ici plutôt que dans
    patchcore/, laissé conforme à l'amont. tqdm appelle len(), d'où __len__.
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
            # En images, l'unité saisie par l'utilisateur ; dernier lot incomplet d'où le plafond.
            self._callback(min(i * BATCH_SIZE, self._n_images), self._n_images)


def _subset_indices(n_total, n_wanted, seed, random_subset):
    """Les indices retenus pour le fit, triés : l'ordre ne change pas le résultat
    et un ordre croissant lit le disque séquentiellement.
    """
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
    
    `progress(phase, done, total)` suit l'avancement, total nul = indéterminé.
    `random_subset` tire au hasard sous le même seed. Renvoie le dossier écrit.
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
    # Le coreset suit l'extraction sans rappel possible : on bascule au dernier lot.
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
