"""Superpose la heatmap d'anomalie PatchCore sur des images test CelebA.

Charge une banque construite par bin/celeba/fit/memory_bank.py — pas de fit ni de
coreset, c'est la moitié rapide de PatchCore. Le prétraitement et le seed sont
relus du fit_config.json de la banque.

    python bin/celeba/infer/heatmap.py                      # une image (défaut)
    python bin/celeba/infer/heatmap.py --image_index 900    # une autre image
    python bin/celeba/infer/heatmap.py --n_per_class 30     # 30 hat + 30 no-hat
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

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.banks
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIG — à éditer avant de lancer.
# --------------------------------------------------------------------------- #
# Dossier écrit par bin/celeba/fit/memory_bank.py (MODELS_DIR/<tag>).
BANK_DIR = os.environ.get(
    "HIST_BANK_DIR", "models/celeba/wideresnet50_approx_greedy_coreset_p0.1_ts10000_s0"
)

# Mode image seule : index dans le split TEST équilibré (839 hat + 839 no-hat).
# Ignoré si --n_per_class > 0.
IMAGE_INDEX_DEFAULT = 0

# Sorties ici. En mode image seule, {idx} est remplacé. En mode --n_per_class,
# un sous-dossier par banque est créé, fichiers <hat|good>_idx<N>.png.
OUTPUT_PATH = os.environ.get("HEATMAP_OUTPUT_PATH", "results/celeba/heatmaps/overlay_idx{idx}.png")

GPU = [0]  # [] force le CPU (retombe sur CPU sans CUDA de toute façon).

# None = autoscale par image : montre la structure, incomparable entre images.
# Deux bornes fixes (p.ex. 0 et 10) = échelle commune pour un lot d'images.
# Bornes de couleur du heatmap (scores = distances L2, échelle dépendante de la
# couche : ~7 en layer3, ~15 en layer2, ~190 en layer4). Surchargeables par env
# HEATMAP_VMIN / HEATMAP_VMAX pour recaler par couche.
HEATMAP_VMIN = float(os.environ.get("HEATMAP_VMIN", "0"))
HEATMAP_VMAX = float(os.environ.get("HEATMAP_VMAX", "10"))
HEATMAP_ALPHA = 0.5

BATCH_SIZE = 8
FAISS_ON_GPU = os.environ.get("INFER_FAISS_GPU", "").lower() in ("1", "true", "yes")
FAISS_NUM_WORKERS = int(os.environ.get("INFER_FAISS_THREADS", "1" if platform.system() == "Darwin" else "4"))
# --------------------------------------------------------------------------- #


def _denormalize(sample_image, mean, std):
    """Annule la normalisation ImageNet -> image uint8 HxWx3 pour l'affichage."""
    image = np.clip((sample_image.numpy() * std + mean) * 255, 0, 255).astype(np.uint8)
    return image.transpose(1, 2, 0)


def _save_overlay(image, heatmap, score, anomaly, ms, out_path):
    """Écrit une image + sa heatmap superposée dans out_path."""
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
    help="Mode image seule : index dans le split TEST équilibré de CelebA.",
)
@click.option(
    "--n_per_class",
    type=int,
    default=0,
    show_default=True,
    help="Si >0, ignore --image_index : tire n images hat et n no-hat au hasard "
    "(effectifs égaux, plafonné à ~839 par classe) et superpose chacune.",
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

    # Labels connus sans inférence : le tirage hat/no-hat se fait en amont.
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
            os.path.dirname(OUTPUT_PATH) or "results/celeba/heatmaps",
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
