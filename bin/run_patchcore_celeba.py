import contextlib
import logging
import os
import sys

import click
import torch

import patchcore.backbones
import patchcore.common
import patchcore.patchcore
import patchcore.sampler
import patchcore.tracking
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default=None)
@click.option("--log_project", type=str, default="CelebA_Results", show_default=True)
@click.option("--save_patchcore_model", is_flag=True)
# Backbone.
@click.option("--backbone_name", "-b", type=str, default="wideresnet50", show_default=True)
@click.option(
    "--layers_to_extract_from",
    "-le",
    type=str,
    multiple=True,
    default=["layer2", "layer3"],
    show_default=True,
)
@click.option("--pretrain_embed_dimension", type=int, default=1024, show_default=True)
@click.option("--target_embed_dimension", type=int, default=1024, show_default=True)
@click.option("--anomaly_scorer_num_nn", type=int, default=1, show_default=True)
@click.option("--patchsize", type=int, default=3, show_default=True)
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
# Coreset sampler.
@click.option(
    "--sampler_name",
    type=click.Choice(["identity", "greedy_coreset", "approx_greedy_coreset"]),
    default="approx_greedy_coreset",
    show_default=True,
)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
# Dataset / dataloader.
@click.option("--resize", type=int, default=256, show_default=True)
@click.option("--imagesize", type=int, default=224, show_default=True)
@click.option("--batch_size", type=int, default=8, show_default=True)
@click.option("--num_workers", type=int, default=8, show_default=True)
def main(
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_patchcore_model,
    backbone_name,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    anomaly_scorer_num_nn,
    patchsize,
    faiss_on_gpu,
    faiss_num_workers,
    sampler_name,
    percentage,
    resize,
    imagesize,
    batch_size,
    num_workers,
):
    """Fit a PatchCore memory bank on CelebA "no-hat" images. No evaluation."""
    layers_to_extract_from = list(layers_to_extract_from)
    log_group = log_group or patchcore.tracking.make_run_name(
        [backbone_name], sampler_name, percentage, imagesize
    )

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )
    patchcore.utils.fix_seeds(seed, device)

    train_dataset = CelebADataset(
        resize=resize, imagesize=imagesize, split=DatasetSplit.TRAIN, seed=seed
    )
    LOGGER.info("CelebA train (no-hat only): %d images.", len(train_dataset))
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    train_dataloader.name = "celeba_no_hat"

    backbone = patchcore.backbones.load(backbone_name)
    backbone.name, backbone.seed = backbone_name, None
    nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

    if sampler_name == "identity":
        sampler = patchcore.sampler.IdentitySampler()
    elif sampler_name == "greedy_coreset":
        sampler = patchcore.sampler.GreedyCoresetSampler(percentage, device)
    else:
        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load(
        backbone=backbone,
        layers_to_extract_from=layers_to_extract_from,
        device=device,
        input_shape=train_dataset.imagesize,
        pretrain_embed_dimension=pretrain_embed_dimension,
        target_embed_dimension=target_embed_dimension,
        patchsize=patchsize,
        featuresampler=sampler,
        anomaly_scorer_num_nn=anomaly_scorer_num_nn,
        nn_method=nn_method,
    )

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    mlflow_params = {
        "seed": seed,
        "dataset": "celeba_no_hat",
        "backbone_name": backbone_name,
        "layers_to_extract_from": layers_to_extract_from,
        "pretrain_embed_dimension": pretrain_embed_dimension,
        "target_embed_dimension": target_embed_dimension,
        "anomaly_scorer_num_nn": anomaly_scorer_num_nn,
        "patchsize": patchsize,
        "faiss_on_gpu": faiss_on_gpu,
        "sampler_name": sampler_name,
        "coreset_pct": percentage,
        "resize": resize,
        "imagesize": imagesize,
    }

    with patchcore.tracking.patchcore_run(
        experiment=log_project, run_name=log_group, params=mlflow_params
    ) as mlflow_run:
        with device_context:
            torch.cuda.empty_cache()
            LOGGER.info("Fitting PatchCore memory bank (no evaluation)...")
            patchcore_instance.fit(train_dataloader)
            LOGGER.info(
                "Done. Memory bank holds %d patch features.",
                len(patchcore_instance.anomaly_scorer.detection_features),
            )

        if save_patchcore_model:
            patchcore_save_path = os.path.join(
                run_save_path, "models", train_dataloader.name
            )
            os.makedirs(patchcore_save_path, exist_ok=True)
            patchcore_instance.save_to_path(patchcore_save_path)
            LOGGER.info("Saved PatchCore model to %s", patchcore_save_path)
            mlflow_run.log_artifacts(patchcore_save_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
