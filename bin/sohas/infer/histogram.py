"""Compare les scores d'anomalie des frames GOOD (sans arme) et WEAPON du TEST.

Charge une banque mémoire construite par bin/sohas/fit/memory_bank.py — aucun
fit ici. Les hyperparamètres du fit (backbone, sampler, percentage, train_subset,
resize/imagesize, seed) sont relus dans le fit_config.json de la banque : une
image encodée autrement que la banque contre laquelle on la compare donnerait des
distances qui ne veulent rien dire, et le ferait silencieusement.

    SOHAS_PATH=/chemin/Sohas_weapon-Detection python bin/sohas/fit/memory_bank.py  # une fois
    python bin/sohas/infer/histogram.py                     # histogramme + stats
    python bin/sohas/infer/histogram.py --n_per_class 200   # échantillon plus petit
"""

import json
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
from scipy.stats import wasserstein_distance, ttest_ind

import patchcore.banks
import patchcore.metrics
import patchcore.tracking
import patchcore.utils
from patchcore.datasets.sohas import SohasDataset, DatasetSplit

LOGGER = logging.getLogger(__name__)

# Couleurs reprises de l'exemple (bleu = normal / in-dist, orange = anomalie / OOD).
COLOR_NORMAL = "#5B8FB9"
COLOR_ANOMALY = "#E8A33D"

# --------------------------------------------------------------------------- #
# CONFIG — édite ici, puis lance le script.
# --------------------------------------------------------------------------- #
# Dossier écrit par bin/sohas/fit/memory_bank.py (MODELS_DIR/<tag>).
# BANK_DIR et OUTPUT_PATH surchargeables par HIST_BANK_DIR / HIST_OUTPUT_PATH,
# pour scorer plusieurs banques dans un même job.
BANK_DIR = "models/sohas/wideresnet50_approx_greedy_coreset_p0.1_tsall_s0"

OUTPUT_PATH = "results/sohas/histograms/hist_sohas.png"

BANK_DIR = os.environ.get("HIST_BANK_DIR", BANK_DIR)
OUTPUT_PATH = os.environ.get("HIST_OUTPUT_PATH", OUTPUT_PATH)

# Racine SOHAS. Par défaut relu du fit_config de la banque, surchargeable ici/env.
SOURCE = os.environ.get("SOHAS_PATH")

# Échantillon PAR classe, effectifs égaux. Plafonné au disponible.
N_PER_CLASS_DEFAULT = 1000

GPU = [0]  # [] force le CPU. Retombe sur CPU sur une machine sans CUDA de toute façon.
BINS = 50
TEST_BATCH_SIZE = 8
NUM_WORKERS = 8

# La recherche FAISS domine le scoring et croît avec la banque. HIST_FAISS_GPU=1
# la bascule sur GPU, index alors entièrement en mémoire GPU (4 Ko par vecteur).
FAISS_ON_GPU = os.environ.get("HIST_FAISS_GPU", "").lower() in ("1", "true", "yes")
FAISS_NUM_WORKERS = int(os.environ.get("HIST_FAISS_THREADS", "1" if platform.system() == "Darwin" else "4"))

# Une expérience MLflow par tâche. run_name encode la config, l'origine est
# taguée par patchcore.tracking.
LOG_PROJECT = "sohas-histograms"
# --------------------------------------------------------------------------- #


def normalized_wasserstein(scores_a, scores_b):
    """Wasserstein-1 entre deux échantillons 1D, normalisée par l'écart-type
    regroupé : W1 / sqrt((var_a + var_b) / 2).

    W1 seule a les unités du score (échelle arbitraire selon la config) ; la
    normaliser donne une taille d'effet sans dimension, comparable entre configs
    (= d de Cohen si les deux distributions sont gaussiennes de même variance).
    Renvoie {w1, w1_normalized, pooled_std}.
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


def histogram_jaccard(scores_a, scores_b, edges):
    """Indice de Jaccard du recouvrement des deux distributions, sur les mêmes
    bins que l'histogramme tracé : J = somme min(n_i, a_i) / somme max(n_i, a_i).

    J=0 séparables, J=1 identiques. Renvoie {jaccard, intersection, union}.
    """
    n = np.histogram(np.asarray(scores_a, dtype=float), bins=edges)[0]
    a = np.histogram(np.asarray(scores_b, dtype=float), bins=edges)[0]
    inter = int(np.minimum(n, a).sum())
    union = int(np.maximum(n, a).sum())
    return {
        "jaccard": inter / union if union > 0 else float("nan"),
        "intersection": inter,
        "union": union,
    }


def t_test_scores(scores_good, scores_anomaly):
    """Test t unilatéral de Welch entre scores good et weapon : la moyenne des
    scores d'anomalie est-elle significativement plus haute ? Seuil 5%.
    Renvoie (p_value, True si p < 0.05).
    """
    res = ttest_ind(scores_good, scores_anomaly, alternative="less", equal_var=False)
    p_value = float(res.pvalue)
    return p_value, p_value < 0.05


@click.command()
@click.option(
    "--n_per_class",
    type=int,
    default=N_PER_CLASS_DEFAULT,
    show_default=True,
    help="Taille d'échantillon PAR classe (good et weapon), effectifs égaux. "
    "Plafonné au nombre de frames disponibles par classe dans le test.",
)
def main(n_per_class):
    """Score le split TEST contre une banque déjà construite, puis superpose la
    distribution des scores d'anomalie des frames GOOD (sans arme, bleu) et
    WEAPON (avec arme, orange) — avec le MÊME effectif par classe. Loggé dans
    MLflow (AUROC + means + figure)."""
    device = patchcore.utils.set_torch_device(GPU)

    patchcore_instance, fit_config = patchcore.banks.load_bank(
        BANK_DIR, device, FAISS_ON_GPU, FAISS_NUM_WORKERS
    )

    # k est un paramètre de scoring, pas de construction : surchargeable sans
    # re-fitter. imagelevel_nn a capturé l'ancien k dans sa closure -> rebind.
    num_nn_used = int(fit_config.get("anomaly_scorer_num_nn", 1))
    _env_num_nn = os.environ.get("HIST_NUM_NN")
    if _env_num_nn:
        num_nn_used = int(_env_num_nn)
        scorer = patchcore_instance.anomaly_scorer
        scorer.n_nearest_neighbours = num_nn_used
        scorer.imagelevel_nn = (
            lambda q, s=scorer, k=num_nn_used: s.nn_method.run(k, q)
        )
        LOGGER.info("num_nn surchargé à %d (HIST_NUM_NN).", num_nn_used)
    # Le seed de la banque pilote aussi le split/équilibrage du TEST : mêmes
    # frames d'une analyse à l'autre.
    seed = fit_config["seed"]
    patchcore.utils.fix_seeds(seed)

    source = SOURCE or fit_config.get("source")
    test_dataset = SohasDataset(
        source=source,
        resize=fit_config["resize"],
        imagesize=fit_config["imagesize"],
        split=DatasetSplit.TEST,
        seed=seed,
    )

    # Labels connus sans inférence -> n index tirés par classe en amont, on ne
    # score que ceux-là. Effectifs égaux = min(demandé, dispo).
    test_labels = np.asarray(test_dataset.labels, dtype=int)
    normal_idx = np.where(test_labels == 0)[0]
    anomaly_idx = np.where(test_labels == 1)[0]
    n = min(n_per_class, len(normal_idx), len(anomaly_idx))
    if n < n_per_class:
        LOGGER.warning(
            "n_per_class=%d demandé mais seulement %d good / %d weapon "
            "disponibles => on utilise n=%d par classe.",
            n_per_class, len(normal_idx), len(anomaly_idx), n,
        )
    rng = np.random.RandomState(seed)
    selected_idx = np.concatenate([
        rng.choice(normal_idx, n, replace=False),
        rng.choice(anomaly_idx, n, replace=False),
    ])
    test_subset = torch.utils.data.Subset(test_dataset, selected_idx.tolist())
    test_dataloader = torch.utils.data.DataLoader(
        test_subset, batch_size=TEST_BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=device.type == "cuda",
    )

    device_is_cuda = device.type == "cuda"
    mlflow_params = {
        "bank_dir": BANK_DIR,
        "num_nn": num_nn_used,
        "n_per_class_requested": n_per_class,
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda or "cpu",
        "gpu_name": torch.cuda.get_device_name(device) if device_is_cuda else "cpu",
        **{k: fit_config[k] for k in (
            "seed", "train_subset", "backbone_name", "sampler_name",
            "coreset_pct", "resize", "imagesize", "memory_bank_size",
        )},
    }

    LOGGER.info("Scoring %d test frames (n=%d par classe)...", 2 * n, n)
    if device_is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    scores, _, labels_gt, _ = patchcore_instance.predict(test_dataloader)
    if device_is_cuda:
        torch.cuda.synchronize(device)
    scoring_seconds = time.perf_counter() - t0

    # Échantillon déjà équilibré : séparer par label suffit, plus de tirage.
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels_gt, dtype=int)
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    LOGGER.info("Histogramme sur n=%d par classe (good vs weapon).", n)

    # AUROC sur l'échantillon équilibré tracé.
    sampled_scores = np.concatenate([normal_scores, anomaly_scores])
    sampled_labels = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    auroc = float("nan")
    if n > 0:
        auroc = patchcore.metrics.compute_imagewise_retrieval_metrics(
            sampled_scores, sampled_labels
        )["auroc"]

    # W1 normalisée entre good et weapon : l'écart d'amplitude, sans échelle.
    wass = normalized_wasserstein(normal_scores, anomaly_scores)
    p_value, _ = t_test_scores(normal_scores, anomaly_scores)

    # Histogramme superposé, bins PARTAGÉS.
    lo, hi = float(sampled_scores.min()), float(sampled_scores.max())
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, BINS + 1)

    # Recouvrement des deux distributions sur ces mêmes bins (FP/FN / union).
    jac = histogram_jaccard(normal_scores, anomaly_scores, edges)

    plt.figure(figsize=(8, 5))
    plt.hist(
        normal_scores, bins=edges, alpha=0.65, color=COLOR_NORMAL,
        label="Good (sans arme)  n={}".format(n),
    )
    plt.hist(
        anomaly_scores, bins=edges, alpha=0.65, color=COLOR_ANOMALY,
        label="Weapon (avec arme)  n={}".format(n),
    )
    plt.xlabel("Score d'anomalie")
    plt.ylabel("Nombre d'instances")
    sampler_name, percentage = fit_config["sampler_name"], fit_config["coreset_pct"]
    tag = "identity" if sampler_name == "identity" else "{} p={}".format(
        sampler_name, percentage
    )
    plt.title("{}  |  ts={}  |  nn={}  |  AUROC={:.3f}  |  W1n={:.3f}".format(
        tag, fit_config["train_subset"], num_nn_used, auroc,
        wass["w1_normalized"],
    ))
    plt.legend()
    plt.grid(True, alpha=0.2)

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    plt.savefig(OUTPUT_PATH, bbox_inches="tight", dpi=120)
    plt.close()
    LOGGER.info("Saved histogram to %s", OUTPUT_PATH)

    # Sidecar JSON garanti, MLflow best-effort : le file store sur NFS lève un
    # "Stale file handle" quand des tâches parallèles se disputent le verrou, et
    # ne doit pas détruire un résultat déjà écrit.
    metrics = {
        "jaccard": jac["jaccard"],
        "jaccard_intersection": jac["intersection"],
        "jaccard_union": jac["union"],
        "num_nn_used": num_nn_used,
        "bins": BINS,
        "w1_normalized": wass["w1_normalized"],
        "w1": wass["w1"],
        "p_value": p_value,
        "pooled_std": wass["pooled_std"],
        "auroc": auroc,
        "scoring_seconds": scoring_seconds,
        "n_per_class_used": int(n),
        "normal_score_mean": float(np.mean(normal_scores)),
        "anomaly_score_mean": float(np.mean(anomaly_scores)),
        "n_normal_available": int(len(normal_idx)),
        "n_anomaly_available": int(len(anomaly_idx)),
    }
    sidecar = os.path.splitext(OUTPUT_PATH)[0] + ".json"
    with open(sidecar, "w") as fh:
        json.dump({**mlflow_params, **metrics}, fh, indent=2)
    LOGGER.info("Saved metrics to %s", sidecar)

    run_name = "hist-ts{}-p{:g}-nn{}".format(
        fit_config["train_subset"], fit_config["coreset_pct"], num_nn_used
    )
    try:
        with patchcore.tracking.patchcore_run(
            experiment=LOG_PROJECT, run_name=run_name, params=mlflow_params
        ) as mlflow_run:
            mlflow_run.log_metrics(metrics)
            mlflow_run.log_artifacts(OUTPUT_PATH)
    except Exception as exc:  # noqa: BLE001 - MLflow ne doit jamais tuer le run
        LOGGER.warning(
            "Logging MLflow échoué (%s) — figure et métriques déjà sauvées (%s, %s).",
            exc, OUTPUT_PATH, sidecar,
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
