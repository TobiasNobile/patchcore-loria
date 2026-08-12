"""Réglages d'exécution partagés : choix du device, ajustement de faiss."""

import logging
import os

import torch

# Importé pour son effet de bord : y sont enregistrés les ResNet que _BACKBONES
# amont ne déclare pas (resnet18, resnet34), dont le fit a besoin ici.
import patchcore.banks  # noqa: F401

LOGGER = logging.getLogger(__name__)


def select_device(preferred=None):
    """Device d'inférence. Voir resolve_device pour les règles."""
    return resolve_device(preferred)[0]


def resolve_device(preferred=None):
    """(device, avertissement) — l'avertissement est non nul en cas de repli.

    `preferred` (choisi dans la page) l'emporte sur PATCHCORE_DEVICE, qui
    l'emporte sur `auto`. Un device indisponible retombe sur cpu ; l'appelant
    doit le signaler, sinon on croit tourner sur GPU sans y être.
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
    if available[kind]:
        return torch.device(choice), None
    note = "{} indisponible sur cette machine — calcul sur CPU.".format(choice)
    LOGGER.warning(note)
    return torch.device("cpu"), note


def tune_faiss_small_batches():
    """Sous 128 requêtes, faiss quitte le chemin BLAS pour une boucle bien plus
    lente par requête — cas de layer4 seul ou d'une image sous 224 px."""
    import faiss

    faiss.cvar.distance_compute_blas_threshold = 20
