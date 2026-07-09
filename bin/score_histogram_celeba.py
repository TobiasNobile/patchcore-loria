import json
import logging
import os
import time

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import wasserstein_distance

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.tracking
import patchcore.utils
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# Couleurs reprises de l'exemple (bleu = normal / in-dist, orange = anomalie / OOD).
COLOR_NORMAL = "#5B8FB9"
COLOR_ANOMALY = "#E8A33D"


def normalized_wasserstein(scores_a, scores_b):
    """Distance de Wasserstein-1 entre deux échantillons 1D, normalisée par
    l'écart-type regroupé.

    W1 seule a les unités du score d'anomalie (échelle arbitraire, qui varie
    d'une config d'hyperparamètres à l'autre) -> on la divise par sigma_pooled
    pour obtenir une grandeur SANS dimension, comparable entre configs. C'est
    une taille d'effet, qui généralise le d de Cohen (auquel elle est égale si
    les deux distributions sont gaussiennes de même variance).

        W1           = aire entre les CDF empiriques (scipy.wasserstein_distance)
        sigma_pooled = sqrt((var_a + var_b) / 2)          [écarts-types échantillon]
        W1_normalise = W1 / sigma_pooled

    Fonction pure : renvoie {w1, w1_normalized, pooled_std}.
    """
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {"w1": float("nan"), "w1_normalized": float("nan"), "pooled_std": float("nan")}
    w1 = float(wasserstein_distance(scores_a, scores_b))
    pooled_std = float(
        np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2.0)
    )
    w1_normalized = w1 / pooled_std if pooled_std > 0 else float("nan")
    return {"w1": w1, "w1_normalized": w1_normalized, "pooled_std": pooled_std}


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
    "--n_per_class",
    type=int,
    default=1000,
    show_default=True,
    help="Taille d'échantillon PAR classe (no-hat et hat), effectifs égaux. "
    "Plafonné au nombre d'images disponibles par classe dans le test "
    "(la classe hat n'a qu'~839 images).",
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
    n_per_class,
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
    """Fit PatchCore sur `train_subset` images no-hat, score le split TEST, puis
    superpose la distribution des scores d'anomalie des images NO-HAT (normal,
    bleu) et HAT (anomalie, orange) — avec le MÊME effectif par classe
    (`n_per_class`, échantillonné aléatoirement, plafonné au dispo). Les deux
    groupes sont des données de TEST. Loggé dans MLflow (AUROC + means + figure)."""
    device = patchcore.utils.set_torch_device(gpu)
    patchcore.utils.fix_seeds(seed, device)

    train_dataset = CelebADataset(
        resize=resize, imagesize=imagesize, split=DatasetSplit.TRAIN, seed=seed
    )
    train_dataset = torch.utils.data.Subset(train_dataset, range(train_subset))
    train_dataset.imagesize = (3, imagesize, imagesize)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=8, shuffle=False,
        num_workers=num_workers, pin_memory=True,   
    )

    test_dataset = CelebADataset(
        resize=resize, imagesize=imagesize, split=DatasetSplit.TEST, seed=seed
    )

    # On connaît les labels (hat/no-hat) SANS inférence -> on tire n index par
    # classe en amont et on ne score QUE ces images (pas de compute gaspillé sur
    # des images qu'on jetterait ensuite). Effectifs égaux = min(demandé, dispo).
    test_labels = np.asarray(test_dataset.labels, dtype=int)
    normal_idx = np.where(test_labels == 0)[0]
    anomaly_idx = np.where(test_labels == 1)[0]
    n = min(n_per_class, len(normal_idx), len(anomaly_idx))
    if n < n_per_class:
        LOGGER.warning(
            "n_per_class=%d demandé mais seulement %d no-hat / %d hat "
            "disponibles => on utilise n=%d par classe.",
            n_per_class, len(normal_idx), len(anomaly_idx), n,
        )
    rng = np.random.RandomState(seed)
    selected_idx = np.concatenate([
        rng.choice(normal_idx, n, replace=False),
        rng.choice(anomaly_idx, n, replace=False),
    ])
    test_subset = torch.utils.data.Subset(test_dataset, selected_idx.tolist())
    test_subset.imagesize = (3, imagesize, imagesize)
    test_dataloader = torch.utils.data.DataLoader(
        test_subset, batch_size=test_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
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
        anomaly_scorer_num_nn=1, # 1 seul plus proche voisin
        nn_method=nn_method,
    )

    device_is_cuda = device.type == "cuda"
    mlflow_params = {
        "seed": seed,
        "train_subset": train_subset,
        "n_per_class_requested": n_per_class,
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

    LOGGER.info("Fitting on %d no-hat images...", len(train_dataset))
    patchcore_instance.fit(train_dataloader)

    LOGGER.info("Scoring %d test images (n=%d par classe)...", 2 * n, n)
    if device_is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    scores, _, labels_gt, _ = patchcore_instance.predict(test_dataloader)
    if device_is_cuda:
        torch.cuda.synchronize(device)
    scoring_seconds = time.perf_counter() - t0

    # Les images scorées sont déjà l'échantillon équilibré (n par classe) :
    # il suffit de séparer par label retourné, plus aucun tirage à faire ici.
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels_gt, dtype=int)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    LOGGER.info("Histogramme sur n=%d par classe (no-hat vs hat).", n)

    # AUROC sur l'échantillon équilibré tracé.
    sampled_scores = np.concatenate([normal_scores, anomaly_scores])
    sampled_labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    auroc = float("nan")
    if n > 0:
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            sampled_scores, sampled_labels
        )["auroc"]

    # W1 normalisée entre no-hat et hat : l'écart d'amplitude, sans échelle.
    wass = normalized_wasserstein(normal_scores, anomaly_scores)

    # Histogramme superposé, bins PARTAGÉS.
    lo, hi = float(sampled_scores.min()), float(sampled_scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)

    plt.figure(figsize=(8, 5))
    plt.hist(
        normal_scores, bins=edges, alpha=0.65, color=COLOR_NORMAL,
        label="No-hat (normal)  n={}".format(n),
    )
    plt.hist(
        anomaly_scores, bins=edges, alpha=0.65, color=COLOR_ANOMALY,
        label="Hat (anomalie)  n={}".format(n),
    )
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Nombre d'instances")
    tag = "identity" if sampler_name == "identity" else "{} p={}".format(
        sampler_name, percentage
    )
    plt.title("{}  |  ts={}  |  W1 norm={:.3f}".format(
        tag, train_subset, wass["w1_normalized"]
    ))
    plt.legend()
    plt.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, bbox_inches="tight", dpi=120)
    plt.close()
    LOGGER.info("Saved histogram to %s", output_path)

    # Métriques : sidecar JSON GARANTI (à côté du PNG) + logging MLflow
    # BEST-EFFORT. Le file store MLflow sur NFS lève un "Stale file handle"
    # quand des tâches parallèles se disputent le verrou -> on ne laisse jamais
    # ça détruire un résultat déjà calculé (figure + JSON écrits avant).
    metrics = {
        "w1_normalized": wass["w1_normalized"],
        "w1": wass["w1"],
        "pooled_std": wass["pooled_std"],
        "auroc": auroc,
        "scoring_seconds": scoring_seconds,
        "n_per_class_used": int(n),
        "normal_score_mean": float(np.mean(normal_scores)),
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "n_normal_available": int(len(normal_idx)),
        "n_anomaly_available": int(len(anomaly_idx)),
    }
    sidecar = os.path.splitext(output_path)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump({**mlflow_params, **metrics}, fh, indent=2)
    LOGGER.info("Saved metrics to %s", sidecar)

    try:
        with patchcore.tracking.patchcore_run(
            experiment=log_project, run_name=log_group, params=mlflow_params
        ) as mlflow_run:
            mlflow_run.log_metrics(metrics)
            mlflow_run.log_artifacts(output_path)
    except Exception as exc:  # noqa: BLE001 - MLflow ne doit jamais tuer le run
        LOGGER.warning(
            "Logging MLflow échoué (%s) — figure et métriques déjà sauvées (%s, %s).",
            exc, output_path, sidecar,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
