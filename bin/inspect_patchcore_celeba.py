import logging
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.tracking
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)


def _resolve_output_path(output_path, idx, n_images):
    """Resolve the overlay path for one image_index and ensure its dir exists.

    - If ``output_path`` contains ``{idx}`` it is substituted (lets the caller
      route each image into its own folder, e.g. ``results/idx{idx}/...``).
    - Else, when several images share one fit, ``_idx{idx}`` is inserted before
      the extension so they don't overwrite each other.
    - Else (single image, no placeholder) the path is used as-is.
    """
    if "{idx}" in output_path:
        path = output_path.format(idx=idx)
    elif n_images > 1:
        base, ext = os.path.splitext(output_path)
        path = "{}_idx{}{}".format(base, idx, ext)
    else:
        path = output_path
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    return path


@click.command()
@click.argument("output_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--image_index",
    type=int,
    default=(0,),
    multiple=True,
    show_default=True,
    help="Index into the balanced CelebA TEST split (839 hat + 839 no-hat). "
    "Repeatable: pass several --image_index to evaluate multiple images from a "
    "single fit (the memory bank is identical for all of them).",
)
@click.option(
    "--n_per_class",
    type=int,
    default=0,
    show_default=True,
    help="Instead of --image_index, draw n hat and n no-hat images at random from "
    "the TEST split (equal counts) and overlay a heatmap on each. Capped at the "
    "number available per class (~839).",
)
@click.option(
    "--train_subset",
    type=int,
    default=2000,
    show_default=True,
    help="Only fit on this many no-hat train images (qualitative check, not a full run).",
)
@click.option("--backbone_name", "-b", type=str, default="wideresnet50", show_default=True)
@click.option(
    "--sampler_name",
    type=click.Choice(["identity", "greedy_coreset", "approx_greedy_coreset"]),
    default="identity",
    show_default=True,
    help="'identity' (no coreset subsampling) is fastest and enough for a sanity check.",
)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
@click.option("--resize", type=int, default=256, show_default=True)
@click.option("--imagesize", type=int, default=224, show_default=True)
@click.option("--num_workers", type=int, default=8, show_default=True)
@click.option("--log_project", type=str, default="CelebA_Results", show_default=True)
@click.option("--log_group", type=str, default="inspect_heatmap", show_default=True)
def main(
    output_path,
    gpu,
    seed,
    image_index,
    n_per_class,
    train_subset,
    backbone_name,
    sampler_name,
    percentage,
    resize,
    imagesize,
    num_workers,
    log_project,
    log_group,
):
    """Fit PatchCore on a subset of CelebA no-hat images, then overlay the
    anomaly heatmap on one CelebA TEST image to eyeball whether it lights up
    on the hat. Logged to MLflow (params + overlay artifact + score)."""
    device = patchcore.utils.set_torch_device(gpu)
    patchcore.utils.fix_seeds(seed)

    train_dataset = CelebADataset(
        resize=resize, imagesize=imagesize, split=DatasetSplit.TRAIN, seed=seed
    )
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_subset))
    train_dataset.imagesize = (3, imagesize, imagesize)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_dataset = CelebADataset(
        resize=resize, imagesize=imagesize, split=DatasetSplit.TEST, seed=seed
    )

    if n_per_class:
        # Labels are known without inference, so the hat / no-hat indices are
        # drawn up front rather than by scoring images and discarding some.
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
        image_index = [
            int(i)
            for i in np.concatenate(
                [
                    rng.choice(normal_idx, n, replace=False),
                    rng.choice(anomaly_idx, n, replace=False),
                ]
            )
        ]

    backbone = patchcore.backbones.load(backbone_name)
    backbone.name, backbone.seed = backbone_name, None
    nn_method = patchcore.common.FaissNN(False, 4)
    if sampler_name == "identity":
        sampler = patchcore.sampler.IdentitySampler()
    elif sampler_name == "greedy_coreset":
        sampler = patchcore.sampler.GreedyCoresetSampler(percentage, device)
    else:
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load(
        backbone=backbone,
        layers_to_extract_from=["layer2", "layer3"],
        device=device,
        input_shape=(3, imagesize, imagesize),
        pretrain_embed_dimension=1024,
        target_embed_dimension=1024,
        patchsize=3,
        featuresampler=sampler,
        anomaly_scorer_num_nn=1,
        nn_method=nn_method,
    )

    device_is_cuda = device.type == "cuda"

    # Fit ONCE. The memory bank is identical for every test image, so we build
    # it a single time and then evaluate each requested image_index against it
    # instead of refitting per image.
    LOGGER.info("Fitting on %d no-hat images...", len(train_dataset))
    patchcore_instance.fit(train_dataloader)

    mean = np.array(train_dataset.dataset.transform_mean).reshape(-1, 1, 1)
    std = np.array(train_dataset.dataset.transform_std).reshape(-1, 1, 1)

    for idx in image_index:
        sample = test_dataset[idx]
        test_dataloader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(test_dataset, [idx]), batch_size=1
        )
        out_path = _resolve_output_path(output_path, idx, len(image_index))

        # Log the exact torch/CUDA stack so runs from this machine vs. the
        # server can be told apart (and version mismatches diagnosed) in MLflow.
        mlflow_params = {
            "seed": seed,
            "image_index": idx,
            "image_anomaly_label": sample["anomaly"],
            "train_subset": train_subset,
            "backbone_name": backbone_name,
            "sampler_name": sampler_name,
            "coreset_pct": percentage,
            "resize": resize,
            "imagesize": imagesize,
            "device": str(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda or "cpu",
            "gpu_name": torch.cuda.get_device_name(device) if device_is_cuda else "cpu",
        }

        with patchcore.tracking.patchcore_run(
            experiment=log_project, run_name=log_group, params=mlflow_params
        ) as mlflow_run:
            LOGGER.info(
                "Predicting on test image #%d (anomaly=%s)...", idx, sample["anomaly"]
            )
            # Time inference. CUDA kernels are async, so synchronize on both
            # sides to measure real GPU time -- but only when actually on CUDA,
            # otherwise torch.cuda.synchronize() raises on a CPU/MPS-only build.
            if device_is_cuda:
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            scores, segmentations, _, _ = patchcore_instance.predict(test_dataloader)
            if device_is_cuda:
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - t0

            n_images = len(test_dataloader.dataset)
            LOGGER.info(
                "Inference on %d image(s) took %.3f s (%.1f ms/image).",
                n_images,
                inference_seconds,
                1000.0 * inference_seconds / max(n_images, 1),
            )

            score, heatmap = scores[0], np.array(segmentations[0])

            image = np.clip((sample["image"].numpy() * std + mean) * 255, 0, 255).astype(
                np.uint8
            )
            image = image.transpose(1, 2, 0)

            plt.imshow(image)
            plt.imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=10)
            plt.axis("off")
            plt.title("anomaly={} score={:.3f}".format(sample["anomaly"], score))
            plt.savefig(out_path, bbox_inches="tight")
            plt.close()
            LOGGER.info("Saved overlay to %s", out_path)

            mlflow_run.log_metrics(
                {
                    "anomaly_score": score,
                    "inference_seconds": inference_seconds,
                    "inference_ms_per_image": 1000.0 * inference_seconds / max(n_images, 1),
                }
            )
            mlflow_run.log_artifacts(out_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
