import logging
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.tracking
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# Couleurs reprises de l'exemple (bleu = train / in-dist, orange = test).
COLOR_TRAIN = "#5B8FB9"
COLOR_TEST = "#E8A33D"


@click.command()
@click.argument("output_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=42, show_default=True)
@click.option(
    "--train_subset",
    type=int,
    default=2000,
    show_default=True,
    help="Nombre d'images no-hat du split TRAIN servant à construire la banque.",
)
@click.option(
    "--n_train_eval",
    type=int,
    default=1000,
    show_default=True,
    help="Nombre d'images no-hat de train HELD-OUT (au-delà du fit) scorées pour "
    "la distribution 'train' (bleu). Elles ne sont PAS dans la banque.",
)
@click.option(
    "--mark_index",
    type=int,
    default=(405, 961),
    multiple=True,
    show_default=True,
    help="Index d'images du split TEST à marquer d'une ligne verticale (leur score).",
)
@click.option("--backbone_name", "-b", type=str, default="wideresnet50", show_default=True)
@click.option(
    "--sampler_name",
    type=click.Choice(["identity", "greedy_coreset", "approx_greedy_coreset"]),
    default="approx_greedy_coreset",
    show_default=True,
)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
@click.option("--resize", type=int, default=256, show_default=True)
@click.option("--imagesize", type=int, default=224, show_default=True)
@click.option("--num_workers", type=int, default=8, show_default=True)
@click.option("--test_batch_size", type=int, default=8, show_default=True)
@click.option("--bins", type=int, default=50, show_default=True)
@click.option("--log_project", type=str, default="CelebA_Results", show_default=True)
@click.option("--log_group", type=str, default="score_histogram", show_default=True)
def main(
    output_path,
    gpu,
    seed,
    train_subset,
    n_train_eval,
    mark_index,
    backbone_name,
    sampler_name,
    percentage,
    resize,
    imagesize,
    num_workers,
    test_batch_size,
    bins,
    log_project,
    log_group,
):
    """Fit PatchCore sur `train_subset` images no-hat, puis score deux ensembles
    d'images et superpose leurs distributions de score d'anomalie (niveau image) :
      - BLEU  : images no-hat de train HELD-OUT (hors banque) = 'in-distribution'.
      - ORANGE: images du split test (hat + no-hat).
    Les images `mark_index` (par défaut 405 et 961) sont repérées par une ligne
    verticale. Loggé dans MLflow (params + AUROC test + means + figure)."""
    device = patchcore.utils.set_torch_device(gpu)
    patchcore.utils.fix_seeds(seed, device)

    # --- Datasets ---------------------------------------------------------
    train_full = CelebADataset(
        resize=resize, imagesize=imagesize, split=DatasetSplit.TRAIN, seed=seed
    )
    n_train = len(train_full)
    fit_dataset = torch.utils.data.Subset(train_full, range(min(train_subset, n_train)))
    fit_dataset.imagesize = (3, imagesize, imagesize)

    # Held-out : images de train APRÈS celles du fit (jamais mises dans la banque).
    heldout_hi = min(train_subset + n_train_eval, n_train)
    heldout_range = range(train_subset, heldout_hi)
    if len(heldout_range) == 0:
        raise click.ClickException(
            "Pas d'images de train held-out disponibles (train_subset={} >= {}).".format(
                train_subset, n_train
            )
        )
    heldout_dataset = torch.utils.data.Subset(train_full, list(heldout_range))

    test_dataset = CelebADataset(
        resize=resize, imagesize=imagesize, split=DatasetSplit.TEST, seed=seed
    )

    def _loader(ds):
        return torch.utils.data.DataLoader(
            ds, batch_size=test_batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

    fit_dataloader = torch.utils.data.DataLoader(
        fit_dataset, batch_size=8, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # --- PatchCore --------------------------------------------------------
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
    mlflow_params = {
        "seed": seed,
        "train_subset": train_subset,
        "n_train_eval": len(heldout_dataset),
        "n_test_eval": len(test_dataset),
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
        LOGGER.info("Fitting on %d no-hat images...", len(fit_dataset))
        patchcore_instance.fit(fit_dataloader)

        LOGGER.info("Scoring %d held-out train (blue) images...", len(heldout_dataset))
        if device_is_cuda:
            torch.cuda.synchronize(device)
        t0 = time.perf_counter()
        train_scores, _, _, _ = patchcore_instance.predict(_loader(heldout_dataset))

        LOGGER.info("Scoring %d test (orange) images...", len(test_dataset))
        test_scores, _, test_labels, _ = patchcore_instance.predict(_loader(test_dataset))
        if device_is_cuda:
            torch.cuda.synchronize(device)
        scoring_seconds = time.perf_counter() - t0

        train_scores = np.asarray(train_scores, dtype=float)
        test_scores = np.asarray(test_scores, dtype=float)
        test_labels = np.asarray(test_labels, dtype=int)

        # AUROC anomalie (hat vs no-hat) DANS le test (le test est balancé).
        auroc = float("nan")
        if len(np.unique(test_labels)) == 2:
            auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
                test_scores, test_labels
            )["auroc"]

        # --- Figure : histogrammes superposés, bins partagés ---------------
        all_scores = np.concatenate([train_scores, test_scores])
        lo, hi = float(all_scores.min()), float(all_scores.max())
        if hi <= lo:
            hi = lo + 1e-6
        edges = np.linspace(lo, hi, bins + 1)

        plt.figure(figsize=(8, 5))
        plt.hist(
            train_scores, bins=edges, alpha=0.65, color=COLOR_TRAIN,
            label="Train no-hat held-out  n={}".format(len(train_scores)),
        )
        plt.hist(
            test_scores, bins=edges, alpha=0.65, color=COLOR_TEST,
            label="Test (hat+no-hat)  n={}".format(len(test_scores)),
        )

        # Lignes verticales pour les images marquées (405, 961).
        mark_colors = ["#B4436C", "#3B7A57", "#7D3AC1", "#C1440E"]
        for i, idx in enumerate(mark_index):
            if 0 <= idx < len(test_scores):
                s = test_scores[idx]
                lab = "hat" if test_labels[idx] == 1 else "no-hat"
                plt.axvline(
                    s, color=mark_colors[i % len(mark_colors)], linestyle="--", linewidth=1.8,
                    label="idx {} ({})  score={:.2f}".format(idx, lab, s),
                )

        plt.xlabel("Score d'anomalie")
        plt.ylabel("Nombre d'instances")
        tag = "identity" if sampler_name == "identity" else "{} p={}".format(
            sampler_name, percentage
        )
        plt.title("{}  |  ts={}  |  AUROC test={:.3f}".format(tag, train_subset, auroc))
        plt.legend(fontsize=8)
        plt.grid(True, alpha=0.2)

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", dpi=120)
        plt.close()
        LOGGER.info("Saved histogram to %s", output_path)

        metrics = {
            "auroc_test": auroc,
            "scoring_seconds": scoring_seconds,
            "train_score_mean": float(np.mean(train_scores)),
            "test_score_mean": float(np.mean(test_scores)),
            "n_train_eval": len(train_scores),
            "n_test_eval": len(test_scores),
        }
        for idx in mark_index:
            if 0 <= idx < len(test_scores):
                metrics["score_idx{}".format(idx)] = float(test_scores[idx])
        mlflow_run.log_metrics(metrics)
        mlflow_run.log_artifacts(output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
