import logging

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


@click.command()
@click.argument("output_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--image_index",
    type=int,
    default=0,
    show_default=True,
    help="Index into the balanced CelebA TEST split (839 hat + 839 no-hat).",
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
    patchcore.utils.fix_seeds(seed, device)

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
    sample = test_dataset[image_index]
    test_dataloader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(test_dataset, [image_index]), batch_size=1
    )

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

    mlflow_params = {
        "seed": seed,
        "image_index": image_index,
        "image_anomaly_label": sample["anomaly"],
        "train_subset": train_subset,
        "backbone_name": backbone_name,
        "sampler_name": sampler_name,
        "coreset_pct": percentage,
        "resize": resize,
        "imagesize": imagesize,
    }

    with patchcore.tracking.patchcore_run(
        experiment=log_project, run_name=log_group, params=mlflow_params
    ) as mlflow_run:
        LOGGER.info("Fitting on %d no-hat images...", len(train_dataset))
        patchcore_instance.fit(train_dataloader)

        LOGGER.info(
            "Predicting on test image #%d (anomaly=%s)...", image_index, sample["anomaly"]
        )
        scores, segmentations, _, _ = patchcore_instance.predict(test_dataloader)
        score, heatmap = scores[0], np.array(segmentations[0])

        mean = np.array(train_dataset.dataset.transform_mean).reshape(-1, 1, 1)
        std = np.array(train_dataset.dataset.transform_std).reshape(-1, 1, 1)
        image = np.clip((sample["image"].numpy() * std + mean) * 255, 0, 255).astype(np.uint8)
        image = image.transpose(1, 2, 0)


        plt.imshow(image)
        plt.imshow(heatmap, cmap="jet", alpha=0.5, vmin=0, vmax=1)
        plt.axis("off")
        plt.title("anomaly={} score={:.3f}".format(sample["anomaly"], score))
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        LOGGER.info("Saved overlay to %s", output_path)

        mlflow_run.log_metrics({"anomaly_score": score})
        mlflow_run.log_artifacts(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
