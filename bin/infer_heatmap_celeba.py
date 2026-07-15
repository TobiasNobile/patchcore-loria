"""Overlay the PatchCore anomaly heatmap on CelebA test images.

Loads a memory bank built by bin/fit_memory_bank_celeba.py -- no fitting, no
coreset, so this is the fast half of PatchCore. Preprocessing (resize /
imagesize) and the split seed are read back from the bank's fit_config.json
rather than restated here: a query embedded differently from the bank it is
searched against would give meaningless distances.

    python bin/fit_memory_bank_celeba.py                    # once, offline
    python bin/infer_heatmap_celeba.py                      # default single image
    python bin/infer_heatmap_celeba.py --image_index 900    # any other single one
    python bin/infer_heatmap_celeba.py --n_per_class 30     # 30 hat + 30 no-hat
"""
import logging
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.banks
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIG -- edit these, then run the script.
# --------------------------------------------------------------------------- #
# Directory written by bin/fit_memory_bank_celeba.py (MODELS_DIR/<tag>).
BANK_DIR = "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts2000_s0"

# Single-image mode: index into the balanced CelebA TEST split (839 hat + 839
# no-hat). Ignored when --n_per_class > 0.
IMAGE_INDEX_DEFAULT = 0

# Overlays land here. In single-image mode the {idx} placeholder is filled in.
# In --n_per_class mode a per-bank subfolder is created next to it and each file
# is named <hat|good>_idx<N>.png.
OUTPUT_PATH = "results/heatmaps/overlay_idx{idx}.png"

GPU = [0]  # [] forces CPU. Falls back to CPU on a CUDA-less box anyway.

# Colour scale of the overlay. None = autoscale to each image's own min/max,
# which always shows structure but is not comparable across images. Fix both
# (e.g. 0 and 10) to compare heatmaps on a shared scale -- what you want for a
# batch of images meant to be looked at side by side.
HEATMAP_VMIN = 0
HEATMAP_VMAX = 10
HEATMAP_ALPHA = 0.5

BATCH_SIZE = 8
FAISS_ON_GPU = False
FAISS_NUM_WORKERS = 4
# --------------------------------------------------------------------------- #


def _denormalize(sample_image, mean, std):
    """Undo the ImageNet normalization to get a uint8 HxWx3 image for display."""
    image = np.clip((sample_image.numpy() * std + mean) * 255, 0, 255).astype(np.uint8)
    return image.transpose(1, 2, 0)


def _save_overlay(image, heatmap, score, anomaly, ms, out_path):
    """Write one image + heatmap overlay to out_path."""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.imshow(image)
    plt.imshow(
        heatmap, cmap="jet", alpha=HEATMAP_ALPHA, vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX
    )
    plt.colorbar(label="Score d'anomalie (patch)")
    plt.axis("off")
    plt.title("anomaly={}  score={:.3f}  |  {:.0f} ms".format(anomaly, score, ms))
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close()
    LOGGER.info("Saved overlay to %s", out_path)


@click.command()
@click.option(
    "--image_index",
    type=int,
    default=IMAGE_INDEX_DEFAULT,
    show_default=True,
    help="Single-image mode: index into the balanced CelebA TEST split.",
)
@click.option(
    "--n_per_class",
    type=int,
    default=0,
    show_default=True,
    help="If >0, ignore --image_index: draw n hat and n no-hat TEST images at "
    "random (equal counts, capped at ~839 available per class) and overlay each.",
)
def main(image_index, n_per_class):
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
    mean = np.array(test_dataset.transform_mean).reshape(-1, 1, 1)
    std = np.array(test_dataset.transform_std).reshape(-1, 1, 1)

    # Pick which test images to overlay. Labels are known without inference, so
    # the hat / no-hat draw happens up front (same approach as the histogram).
    if n_per_class > 0:
        labels = np.asarray(test_dataset.labels, dtype=int)
        normal_idx = np.where(labels == 0)[0]
        anomaly_idx = np.where(labels == 1)[0]
        n = min(n_per_class, len(normal_idx), len(anomaly_idx))
        if n < n_per_class:
            LOGGER.warning(
                "n_per_class=%d requested but only %d no-hat / %d hat available "
                "=> using n=%d per class.",
                n_per_class, len(normal_idx), len(anomaly_idx), n,
            )
        rng = np.random.RandomState(seed)
        selected = np.concatenate([
            rng.choice(normal_idx, n, replace=False),
            rng.choice(anomaly_idx, n, replace=False),
        ]).tolist()
        out_dir = os.path.join(
            os.path.dirname(OUTPUT_PATH) or "results/heatmaps",
            "ts{}_p{}".format(fit_config["train_subset"], fit_config["coreset_pct"]),
        )
    else:
        selected = [image_index]
        out_dir = None

    dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, selected), batch_size=BATCH_SIZE
    )

    device_is_cuda = device.type == "cuda"
    if device_is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    scores, segmentations, _, _ = patchcore_instance.predict(dataloader)
    if device_is_cuda:
        torch.cuda.synchronize(device)
    ms = 1000.0 * (time.perf_counter() - t0) / max(len(selected), 1)
    LOGGER.info("Scored %d image(s), %.1f ms/image.", len(selected), ms)

    for pos, idx in enumerate(selected):
        sample = test_dataset[idx]
        image = _denormalize(sample["image"], mean, std)
        if out_dir is not None:
            out_path = os.path.join(
                out_dir, "{}_idx{}.png".format(sample["anomaly"], idx)
            )
        else:
            out_path = OUTPUT_PATH.format(idx=idx)
        _save_overlay(
            image, np.array(segmentations[pos]), scores[pos], sample["anomaly"],
            ms, out_path,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
