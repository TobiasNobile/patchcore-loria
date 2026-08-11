"""Construit une banque mémoire PatchCore sur les images no-hat de CelebA.

Pas de CLI : on édite le bloc CONFIG puis on lance. Le fit (extraction des
features + coreset) est la moitié coûteuse et hors-ligne de PatchCore ; les
scripts de scoring rechargent ensuite la banque.

    python bin/celeba/fit/memory_bank.py

Écrit dans MODELS_DIR/<tag>/ l'index FAISS, patchcore_params.pkl et
fit_config.json. Le <tag> vient de la CONFIG, donc deux configs différentes ne
s'écrasent pas et relancer une config identique est inutile.
"""

import json
import logging
import os
import platform
import resource
import time

# macOS : torch et faiss embarquent chacun leur libomp, la seconde à s'initialiser
# fait abort. À poser avant l'import de patchcore, qui charge faiss.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

import patchcore.backbones
import patchcore.banks
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# ─── CONFIG ────────────────────────────────────────────────────────────────
SEED = 0
GPU = [0]  # [] force le CPU (retombe sur CPU sans CUDA de toute façon).

# Images no-hat du train pour la banque. None = tout le split. Coût croissant
# (~784 features par image, coreset séquentiel). Env : FIT_TRAIN_SUBSET.
TRAIN_SUBSET = 2000
_env_ts = os.environ.get("FIT_TRAIN_SUBSET")
if _env_ts:
    TRAIN_SUBSET = None if _env_ts.lower() in ("none", "all") else int(_env_ts)

BACKBONE_NAME = "wideresnet50"
# Env FIT_LAYERS (CSV) : layer1..layer4, pas de layer5. La résolution des patches
# est celle de la 1re couche listée.
_env_layers = os.environ.get("FIT_LAYERS")
LAYERS_TO_EXTRACT_FROM = (
    [l.strip() for l in _env_layers.split(",") if l.strip()]
    if _env_layers else ["layer2", "layer3"]
)
PRETRAIN_EMBED_DIMENSION = 1024
TARGET_EMBED_DIMENSION = 1024
PATCHSIZE = 3
# Paramètre de requête : n'affecte ni la banque ni la RAM du fit, juste le score.
# Env FIT_NUM_NN, re-surchargeable au scoring par HIST_NUM_NN.
ANOMALY_SCORER_NUM_NN = int(os.environ.get("FIT_NUM_NN", "1"))

# identity = pas de coreset : fit rapide, grosse banque, inférence lente.
# Env FIT_SAMPLER : identity | greedy_coreset | approx_greedy_coreset.
SAMPLER_NAME = os.environ.get("FIT_SAMPLER", "approx_greedy_coreset")
# Fraction des features gardée par le coreset. Surchargeable par FIT_CORESET_PCT.
PERCENTAGE = 0.1
_env_pct = os.environ.get("FIT_CORESET_PCT")
if _env_pct:
    PERCENTAGE = float(_env_pct)

RESIZE = 256
IMAGESIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 8

FAISS_ON_GPU = os.environ.get("FIT_FAISS_GPU", "").lower() in ("1", "true", "yes")
FAISS_NUM_WORKERS = int(os.environ.get("FIT_FAISS_THREADS", "1" if platform.system() == "Darwin" else "4"))

# Env : FIT_MODELS_DIR. Sur Grid'5000, pointer le disque local du nœud pour ne
# pas saturer le quota /home.
MODELS_DIR = os.environ.get("FIT_MODELS_DIR", "models/celeba")
# --------------------------------------------------------------------------- #


def _layers_suffix():
    """Suffixe de couche pour le tag. Vide pour le défaut layer2+layer3 (compat
    banques existantes) ; sinon _l2 / _l3 / _l4 pour ne pas écraser."""
    if LAYERS_TO_EXTRACT_FROM == ["layer2", "layer3"]:
        return ""
    return "_" + "-".join(l.replace("layer", "l") for l in LAYERS_TO_EXTRACT_FROM)


def build_tag():
    """Tag de fichier identifiant la banque produite par cette CONFIG."""
    sampler = (
        "identity"
        if SAMPLER_NAME == "identity"
        else "{}_p{:g}".format(SAMPLER_NAME, PERCENTAGE)
    )
    return "{}{}_{}_ts{}_s{}".format(
        BACKBONE_NAME, _layers_suffix(), sampler, TRAIN_SUBSET or "all", SEED
    )


# Projection Johnson-Lindenstrauss du coreset : c'est elle qui fixe la VRAM du
# fit (N_patches x DIM x 4 o, en un bloc sur le GPU). Env FIT_CORESET_PROJ_DIM.
CORESET_PROJ_DIM = int(os.environ.get("FIT_CORESET_PROJ_DIM", "128"))


def build_sampler(device):
    if SAMPLER_NAME == "identity":
        return patchcore.sampler.IdentitySampler()
    if SAMPLER_NAME == "greedy_coreset":
        return patchcore.sampler.GreedyCoresetSampler(
            PERCENTAGE, device, dimension_to_project_features_to=CORESET_PROJ_DIM
        )
    if SAMPLER_NAME == "approx_greedy_coreset":
        return patchcore.sampler.ApproximateGreedyCoresetSampler(
            PERCENTAGE, device, dimension_to_project_features_to=CORESET_PROJ_DIM
        )
    raise ValueError("Unknown SAMPLER_NAME: {}".format(SAMPLER_NAME))


def main():
    device = patchcore.utils.set_torch_device(GPU)
    patchcore.utils.fix_seeds(SEED)

    train_dataset = CelebADataset(
        resize=RESIZE, imagesize=IMAGESIZE, split=DatasetSplit.TRAIN, seed=SEED
    )
    if TRAIN_SUBSET is not None:
        train_dataset = torch.utils.data.Subset(train_dataset, range(TRAIN_SUBSET))
    LOGGER.info("Fitting on %d no-hat train images.", len(train_dataset))

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=device.type == "cuda",
    )

    backbone = patchcore.backbones.load(BACKBONE_NAME)
    backbone.name, backbone.seed = BACKBONE_NAME, None

    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load(
        backbone=backbone,
        layers_to_extract_from=LAYERS_TO_EXTRACT_FROM,
        device=device,
        input_shape=(3, IMAGESIZE, IMAGESIZE),
        pretrain_embed_dimension=PRETRAIN_EMBED_DIMENSION,
        target_embed_dimension=TARGET_EMBED_DIMENSION,
        patchsize=PATCHSIZE,
        featuresampler=build_sampler(device),
        anomaly_scorer_num_nn=ANOMALY_SCORER_NUM_NN,
        nn_method=patchcore.common.FaissNN(FAISS_ON_GPU, FAISS_NUM_WORKERS),
    )

    device_is_cuda = device.type == "cuda"
    if device_is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    patchcore_instance.fit(train_dataloader)
    if device_is_cuda:
        torch.cuda.synchronize(device)
    fit_seconds = time.perf_counter() - t0

    bank = patchcore_instance.anomaly_scorer.detection_features
    bank_size = int(len(bank))
    # ru_maxrss : Ko sur Linux, octets sur macOS.
    _maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gb = _maxrss / (1024 ** 2 if platform.system() == "Linux" else 1024 ** 3)
    LOGGER.info(
        "Fit took %.1f s. Memory bank holds %d patch features. Peak RSS %.1f Go.",
        fit_seconds,
        bank_size,
        peak_rss_gb,
    )

    config = {
        "seed": SEED,
        "train_subset": TRAIN_SUBSET,
        "backbone_name": BACKBONE_NAME,
        "layers_to_extract_from": LAYERS_TO_EXTRACT_FROM,
        "pretrain_embed_dimension": PRETRAIN_EMBED_DIMENSION,
        "target_embed_dimension": TARGET_EMBED_DIMENSION,
        "patchsize": PATCHSIZE,
        "anomaly_scorer_num_nn": ANOMALY_SCORER_NUM_NN,
        "sampler_name": SAMPLER_NAME,
        "coreset_pct": PERCENTAGE,
        "coreset_proj_dim": CORESET_PROJ_DIM,
        "resize": RESIZE,
        "imagesize": IMAGESIZE,
        "faiss_on_gpu": FAISS_ON_GPU,
        "device": str(device),
        # Trace du hardware : nom complet du nœud + cluster g5k (gres-3.nancy... -> gres).
        "node": platform.node(),
        "cluster": platform.node().split(".")[0].split("-")[0],
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "n_train_images": len(train_dataset),
        "memory_bank_size": bank_size,
        "feature_dim": int(np.asarray(bank).shape[1]),
        "bank_gb": bank_size * int(np.asarray(bank).shape[1]) * 4 / 1024 ** 3,
        "peak_rss_gb": peak_rss_gb,
        "fit_seconds": fit_seconds,
    }
    save_dir = os.path.join(MODELS_DIR, build_tag())
    if os.environ.get("FIT_NO_SAVE", "").lower() in ("1", "true", "yes"):
        # Mesures seules (une banque identity peut peser des centaines de Go) :
        # fit_config.json, ni index FAISS ni params.
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "fit_config.json"), "w") as fh:
            json.dump(config, fh, indent=2)
        LOGGER.info("FIT_NO_SAVE : banque non persistée, mesures dans %s", save_dir)
    else:
        patchcore.banks.save_bank(patchcore_instance, save_dir, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
