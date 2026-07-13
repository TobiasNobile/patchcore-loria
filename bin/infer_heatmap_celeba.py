"""Overlay the PatchCore anomaly heatmap on one CelebA test image.

Standalone: no CLI, edit the CONFIG block below and run the script. Loads a
memory bank built by bin/fit_memory_bank_celeba.py -- no fitting, no coreset, so
this is the fast half of PatchCore and the one that has to run in real time.

    python bin/fit_memory_bank_celeba.py   # once, offline
    python bin/infer_heatmap_celeba.py     # as often as you like

Preprocessing (resize / imagesize) is read back from the bank's fit_config.json
rather than restated here: a query embedded differently from the bank it is
searched against would give meaningless distances.
"""

import json
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.common
import patchcore.patchcore
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIG -- edit these, then run the script.
# --------------------------------------------------------------------------- #
# Directory written by bin/fit_memory_bank_celeba.py (MODELS_DIR/<tag>).
BANK_DIR = "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts2000_s0"

# Index into the balanced CelebA TEST split (839 hat + 839 no-hat).
IMAGE_INDEX = 0

OUTPUT_PATH = "results/heatmaps/overlay_idx{idx}.png"

GPU = [0]  # [] forces CPU. Falls back to CPU on a CUDA-less box anyway.
SEED = 0  # Only picks the TEST split's balanced subsample; use the bank's seed.

# Colour scale of the overlay. None = autoscale to this image's own min/max,
# which always shows structure but is not comparable across images. Fix both
# (e.g. 0 and 10) to compare heatmaps on a shared scale.
HEATMAP_VMIN = 0
HEATMAP_VMAX = 10
HEATMAP_ALPHA = 0.5

FAISS_ON_GPU = True
FAISS_NUM_WORKERS = 4
# --------------------------------------------------------------------------- #


def load_bank(device):
    """Rebuild a PatchCore from a saved bank, plus the config it was fit with."""
    with open(os.path.join(BANK_DIR, "fit_config.json")) as fh:
        fit_config = json.load(fh)

    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load_from_path(
        BANK_DIR, device, patchcore.common.FaissNN(FAISS_ON_GPU, FAISS_NUM_WORKERS)
    )
    LOGGER.info(
        "Loaded bank from %s (%d patch features, %s p=%s, fit on %d images).",
        BANK_DIR,
        fit_config["memory_bank_size"],
        fit_config["sampler_name"],
        fit_config["coreset_pct"],
        fit_config["n_train_images"],
    )
    return patchcore_instance, fit_config


def main():
    device = patchcore.utils.set_torch_device(GPU)
    patchcore.utils.fix_seeds(SEED, device)

    patchcore_instance, fit_config = load_bank(device)

    test_dataset = CelebADataset(
        resize=fit_config["resize"],
        imagesize=fit_config["imagesize"],
        split=DatasetSplit.TEST,
        seed=fit_config["seed"],
    )
    sample = test_dataset[IMAGE_INDEX]
    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, [IMAGE_INDEX]), batch_size=1
    )

    device_is_cuda = device.type == "cuda"
    if device_is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    scores, segmentations, _, _ = patchcore_instance.predict(dataloader)
    if device_is_cuda:
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - t0

    score, heatmap = scores[0], np.array(segmentations[0])
    LOGGER.info(
        "Image #%d (anomaly=%s): score=%.3f, inference=%.1f ms.",
        IMAGE_INDEX,
        sample["anomaly"],
        score,
        1000.0 * inference_seconds,
    )

    mean = np.array(test_dataset.transform_mean).reshape(-1, 1, 1)
    std = np.array(test_dataset.transform_std).reshape(-1, 1, 1)
    image = np.clip((sample["image"].numpy() * std + mean) * 255, 0, 255).astype(
        np.uint8
    )
    image = image.transpose(1, 2, 0)

    out_path = OUTPUT_PATH.format(idx=IMAGE_INDEX)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    plt.imshow(image)
    plt.imshow(
        heatmap,
        cmap="jet",
        alpha=HEATMAP_ALPHA,
        vmin=HEATMAP_VMIN,
        vmax=HEATMAP_VMAX,
    )
    plt.colorbar(label="Score d'anomalie (patch)")
    plt.axis("off")
    plt.title(
        "anomaly={}  score={:.3f}  |  {:.0f} ms".format(
            sample["anomaly"], score, 1000.0 * inference_seconds
        )
    )
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    LOGGER.info("Saved overlay to %s", out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
