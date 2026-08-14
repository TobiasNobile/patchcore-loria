"""Sauvegarde et chargement des banques mémoire, avec la config du fit.

Le fit_config.json à côté de la banque porte le prétraitement et le seed : une
requête encodée autrement que la banque donnerait des distances silencieusement
fausses.
"""

import json
import logging
import os

import patchcore.backbones
import patchcore.common
import patchcore.patchcore

LOGGER = logging.getLogger(__name__)

CONFIG_FILENAME = "fit_config.json"

# torchvision expose plus de ResNet que _BACKBONES amont : enregistrés ici, dans le module qui charge les banques.
for _name in ("resnet18", "resnet34"):
    patchcore.backbones._BACKBONES.setdefault(
        _name, "models.{}(pretrained=True)".format(_name)
    )


def save_bank(patchcore_instance, bank_dir, config):
    """Écrit un PatchCore entraîné et la config qui l'a produit dans bank_dir."""
    os.makedirs(bank_dir, exist_ok=True)
    patchcore_instance.save_to_path(bank_dir)
    with open(os.path.join(bank_dir, CONFIG_FILENAME), "w") as fh:
        json.dump(config, fh, indent=2)
    LOGGER.info("Saved memory bank to %s", bank_dir)


def load_bank(bank_dir, device, faiss_on_gpu=False, faiss_num_workers=4):
    """Reconstruit un PatchCore depuis bank_dir. Renvoie (patchcore, fit_config)."""
    config_path = os.path.join(bank_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        raise SystemExit(
            "No memory bank at {}. Build one first with "
            "`python bin/<dataset>/fit/memory_bank.py`, or point BANK_DIR at "
            "an existing bank.".format(bank_dir)
        )
    with open(config_path) as fh:
        fit_config = json.load(fh)

    patchcore_instance = patchcore.patchcore.PatchCore(device)
    patchcore_instance.load_from_path(
        bank_dir, device, patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)
    )
    LOGGER.info(
        "Loaded bank from %s (%d patch features, %s p=%s, fit on %d images).",
        bank_dir,
        fit_config["memory_bank_size"],
        fit_config["sampler_name"],
        fit_config["coreset_pct"],
        fit_config["n_train_images"],
    )
    return patchcore_instance, fit_config
