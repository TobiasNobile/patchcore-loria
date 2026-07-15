"""Fit a PatchCore memory bank on CelebA no-hat images and save it to disk.

Standalone: no CLI, edit the CONFIG block below and run the script. Building the
bank (feature extraction + coreset) is the expensive, offline half of PatchCore;
scoring scripts can then load the saved bank and go straight to inference.

    python bin/fit_memory_bank_celeba.py

Writes to MODELS_DIR/<tag>/:
    nnscorer_search_index.faiss   the memory bank (FAISS index)
    patchcore_params.pkl          backbone / layers / dims, for load_from_path
    fit_config.json               this CONFIG + fit stats, for provenance

<tag> is derived from the CONFIG, so two different configs never overwrite each
other and re-running an identical config is a no-op you can skip.
"""

import logging
import os
import time

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

# --------------------------------------------------------------------------- #
# CONFIG -- edit these, then run the script.
# --------------------------------------------------------------------------- #
SEED = 0
GPU = [0]  # [] forces CPU. On a CUDA-less box this falls back to CPU anyway.

# Number of no-hat TRAIN images the bank is built from. None = the whole split.
# Cost scales with this: each image contributes ~784 patch features, and the
# greedy coreset needs PERCENTAGE x that many sequential iterations.
# Overridable by env var (FIT_TRAIN_SUBSET=10000) so one job can sweep several
# sizes without editing this file; unset, it uses the value below.
TRAIN_SUBSET = 2000
_env_ts = os.environ.get("FIT_TRAIN_SUBSET")
if _env_ts:
    TRAIN_SUBSET = None if _env_ts.lower() in ("none", "all") else int(_env_ts)

BACKBONE_NAME = "wideresnet50"
LAYERS_TO_EXTRACT_FROM = ["layer2", "layer3"]
PRETRAIN_EMBED_DIMENSION = 1024
TARGET_EMBED_DIMENSION = 1024
PATCHSIZE = 3
ANOMALY_SCORER_NUM_NN = 1

# "identity" keeps every feature (no coreset computed at all -- fastest fit,
# biggest bank, slowest inference). "approx_greedy_coreset" compresses the bank
# to PERCENTAGE of its features: slow to build, but this is the lever that makes
# inference fast.
SAMPLER_NAME = "approx_greedy_coreset"
PERCENTAGE = 0.1

RESIZE = 256
IMAGESIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 8

FAISS_ON_GPU = False
FAISS_NUM_WORKERS = 4

MODELS_DIR = "models/celeba"
# --------------------------------------------------------------------------- #


def build_tag():
    """Filesystem tag identifying the bank this CONFIG produces."""
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
    LOGGER.info(
        "Fit took %.1f s. Memory bank holds %d patch features.",
        fit_seconds,
        bank_size,
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
