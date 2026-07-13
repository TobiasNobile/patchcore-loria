"""Saving and loading PatchCore memory banks, with the config they were fit with.

A bank on disk is a directory holding what ``PatchCore.save_to_path`` writes plus
a ``fit_config.json`` describing how it was built. That sidecar is what lets a
scoring script reuse a bank safely: preprocessing (``resize`` / ``imagesize``)
and the split ``seed`` are read back from it rather than restated at the call
site, since a query embedded differently from the bank it is searched against
would give meaningless distances -- and would do so silently.
"""

import json
import logging
import os

import patchcore.common
import patchcore.patchcore

LOGGER = logging.getLogger(__name__)

CONFIG_FILENAME = "fit_config.json"


def save_bank(patchcore_instance, bank_dir, config):
    """Write a fitted PatchCore plus the config that produced it to ``bank_dir``."""
    os.makedirs(bank_dir, exist_ok=True)
    patchcore_instance.save_to_path(bank_dir)
    with open(os.path.join(bank_dir, CONFIG_FILENAME), "w") as fh:
        json.dump(config, fh, indent=2)
    LOGGER.info("Saved memory bank to %s", bank_dir)


def load_bank(bank_dir, device, faiss_on_gpu=False, faiss_num_workers=4):
    """Rebuild a PatchCore from ``bank_dir``. Returns (patchcore, fit_config)."""
    config_path = os.path.join(bank_dir, CONFIG_FILENAME)
    if not os.path.exists(config_path):
        raise SystemExit(
            "No memory bank at {}. Build one first with "
            "`python bin/fit_memory_bank_celeba.py`, or point BANK_DIR at an "
            "existing bank.".format(bank_dir)
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
