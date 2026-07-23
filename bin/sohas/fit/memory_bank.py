"""Construit une banque mémoire PatchCore sur les frames SANS ARME de SOHAS.

Pas de CLI : on édite le bloc CONFIG puis on lance. Le fit (extraction des
features + coreset) est la moitié coûteuse et hors-ligne de PatchCore ; les
scripts de scoring rechargent ensuite la banque.

    SOHAS_PATH=/chemin/vers/Sohas_weapon-Detection python bin/sohas/fit/memory_bank.py

Écrit dans MODELS_DIR/<tag>/ l'index FAISS, patchcore_params.pkl et
fit_config.json. Le <tag> vient de la CONFIG, donc deux configs différentes ne
s'écrasent pas et relancer une config identique est inutile.
"""

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

import patchcore.backbones
import patchcore.banks
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from patchcore.datasets.sohas import SohasDataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIG — à éditer avant de lancer.
# --------------------------------------------------------------------------- #
SEED = 0
GPU = [0]  # [] force le CPU (retombe sur CPU sans CUDA de toute façon).

# Racine du sous-ensemble Sohas_weapon-Detection (contient images/, annotations/…).
SOURCE = os.environ.get(
    "SOHAS_PATH",
    os.path.expanduser("~/dev/telecom/stage_1a/data/Sohas_weapon-Detection"),
)

# Frames sans arme du train pour la banque. None = tout le split (le set normal
# de SOHAS est modeste, on le prend en entier par défaut). Env : FIT_TRAIN_SUBSET.
TRAIN_SUBSET = None
_env_ts = os.environ.get("FIT_TRAIN_SUBSET")
if _env_ts:
    TRAIN_SUBSET = None if _env_ts.lower() in ("none", "all") else int(_env_ts)

BACKBONE_NAME = "wideresnet50"
LAYERS_TO_EXTRACT_FROM = ["layer2", "layer3"]
PRETRAIN_EMBED_DIMENSION = 1024
TARGET_EMBED_DIMENSION = 1024
PATCHSIZE = 3
ANOMALY_SCORER_NUM_NN = 1

# identity = pas de coreset : fit rapide, grosse banque, inférence lente.
# approx_greedy_coreset compresse la banque à PERCENTAGE des features.
SAMPLER_NAME = "approx_greedy_coreset"
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
MODELS_DIR = os.environ.get("FIT_MODELS_DIR", "models/sohas")
# --------------------------------------------------------------------------- #


def build_tag():
    """Tag de fichier identifiant la banque produite par cette CONFIG."""
    sampler = (
        "identity"
        if SAMPLER_NAME == "identity"
        else "{}_p{:g}".format(SAMPLER_NAME, PERCENTAGE)
    )
    return "{}_{}_ts{}_s{}".format(
        BACKBONE_NAME, sampler, TRAIN_SUBSET or "all", SEED
    )


def build_sampler(device):
    if SAMPLER_NAME == "identity":
        return patchcore.sampler.IdentitySampler()
    if SAMPLER_NAME == "greedy_coreset":
        return patchcore.sampler.GreedyCoresetSampler(PERCENTAGE, device)
    if SAMPLER_NAME == "approx_greedy_coreset":
        return patchcore.sampler.ApproximateGreedyCoresetSampler(PERCENTAGE, device)
    raise ValueError("Unknown SAMPLER_NAME: {}".format(SAMPLER_NAME))


def main():
    device = patchcore.utils.set_torch_device(GPU)
    patchcore.utils.fix_seeds(SEED)

    train_dataset = SohasDataset(
        source=SOURCE,
        resize=RESIZE,
        imagesize=IMAGESIZE,
        split=DatasetSplit.TRAIN,
        seed=SEED,
    )
    if TRAIN_SUBSET is not None:
        n = min(TRAIN_SUBSET, len(train_dataset))
        train_dataset = torch.utils.data.Subset(train_dataset, range(n))
    LOGGER.info("Fitting on %d weapon-free train frames.", len(train_dataset))

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
    LOGGER.info(
        "Fit took %.1f s. Memory bank holds %d patch features.",
        fit_seconds,
        bank_size,
    )

    config = {
        "seed": SEED,
        "source": SOURCE,
        "train_subset": TRAIN_SUBSET,
        "backbone_name": BACKBONE_NAME,
        "layers_to_extract_from": LAYERS_TO_EXTRACT_FROM,
        "pretrain_embed_dimension": PRETRAIN_EMBED_DIMENSION,
        "target_embed_dimension": TARGET_EMBED_DIMENSION,
        "patchsize": PATCHSIZE,
        "anomaly_scorer_num_nn": ANOMALY_SCORER_NUM_NN,
        "sampler_name": SAMPLER_NAME,
        "coreset_pct": PERCENTAGE,
        "resize": RESIZE,
        "imagesize": IMAGESIZE,
        "faiss_on_gpu": FAISS_ON_GPU,
        "device": str(device),
        "torch_version": torch.__version__,
        "n_train_images": len(train_dataset),
        "memory_bank_size": bank_size,
        "feature_dim": int(np.asarray(bank).shape[1]),
        "fit_seconds": fit_seconds,
    }
    patchcore.banks.save_bank(
        patchcore_instance, os.path.join(MODELS_DIR, build_tag()), config
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
