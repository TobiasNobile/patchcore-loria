"""Métriques de comparaison de deux distributions de scores (normal vs anomalie)."""

import numpy as np
from scipy.stats import ttest_ind, wasserstein_distance


def normalized_wasserstein(scores_a, scores_b):
    """W1 / écart-type regroupé : une taille d'effet sans dimension, comparable
    entre configurations là où W1 seule a les unités du score."""
    scores_a = np.asarray(scores_a, dtype=float)
    scores_b = np.asarray(scores_b, dtype=float)
    nan = float("nan")
    if len(scores_a) < 2 or len(scores_b) < 2:
        return {"w1": nan, "w1_normalized": nan, "pooled_std": nan}
    w1 = float(wasserstein_distance(scores_a, scores_b))
    pooled_std = float(
        np.sqrt((np.var(scores_a, ddof=1) + np.var(scores_b, ddof=1)) / 2.0)
    )
    return {
        "w1": w1,
        "w1_normalized": w1 / pooled_std if pooled_std > 0 else nan,
        "pooled_std": pooled_std,
    }


def histogram_jaccard(scores_a, scores_b, edges):
    """Recouvrement des deux distributions sur les bins de l'histogramme tracé.

    L'intersection compte bin par bin la classe minoritaire, soit les images
    qu'un classifieur par seuil se tromperait forcément. J=0 séparables, J=1
    identiques."""
    n = np.histogram(np.asarray(scores_a, dtype=float), bins=edges)[0]
    a = np.histogram(np.asarray(scores_b, dtype=float), bins=edges)[0]
    inter, union = int(np.minimum(n, a).sum()), int(np.maximum(n, a).sum())
    return {
        "jaccard": inter / union if union > 0 else float("nan"),
        "intersection": inter,
        "union": union,
    }


def t_test_scores(scores_normal, scores_anomaly):
    """Test t unilatéral de Welch : la moyenne des scores d'anomalie est-elle
    significativement plus haute ? Renvoie (p_value, p < 0.05)."""
    p_value = float(
        ttest_ind(scores_normal, scores_anomaly, alternative="less", equal_var=False).pvalue
    )
    return p_value, p_value < 0.05
