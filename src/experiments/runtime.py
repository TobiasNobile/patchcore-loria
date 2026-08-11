"""Réglages d'exécution partagés : choix du device, ajustement de faiss."""

import logging
import os

import torch

LOGGER = logging.getLogger(__name__)


def select_device(preferred=None):
    """Device d'inférence : auto | cpu | cuda[:N].

    `preferred` (choisi dans la page) l'emporte sur PATCHCORE_DEVICE, qui
    l'emporte sur `auto`. Un device indisponible retombe sur cpu.
    """
    choice = (preferred or os.environ.get("PATCHCORE_DEVICE") or "auto").strip().lower()
    if choice == "auto":
        choice = "cuda" if torch.cuda.is_available() else "cpu"
    if choice.startswith("mps"):
        raise ValueError(
            "MPS non supporté : l'agrégation de patchs de PatchCore utilise un "
            "pooling adaptatif que Metal n'implémente pas, et il y est plus lent "
            "que le CPU. Utiliser cpu ou cuda."
        )

    kind = choice.split(":")[0]
    available = {"cpu": True, "cuda": torch.cuda.is_available()}
    if kind not in available:
        raise ValueError("PATCHCORE_DEVICE inconnu : {}".format(choice))
    if not available[kind]:
        LOGGER.warning("Device %s indisponible, repli sur cpu.", choice)
        choice = "cpu"
    return torch.device(choice)


def tune_faiss_small_batches():
    """Sous 128 requêtes, faiss quitte le chemin BLAS pour une boucle bien plus
    lente par requête — cas de layer4 seul ou d'une image sous 224 px."""
    import faiss

    faiss.cvar.distance_compute_blas_threshold = 20
