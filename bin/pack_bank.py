"""Empaquette une banque déjà construite (models/<dataset>/<tag>/) en .pkg.

Les banques produites par bin/<dataset>/fit/memory_bank.py sont des dossiers ;
l'interface web ne lit que des .pkg dans coresets/. Ce script fait le pont, sans
refitter — il ne fait que zipper et renommer.

    python bin/pack_bank.py models/coco/wideresnet50_l3-l4_approx_greedy_coreset_p0.01_ts20000_s0 \
        --task DetectionKnife

Le nom de sortie est déduit du fit_config : <Backbone>_<Tache>_<couches>_<coreset>_ts<n>.pkg
"""

import json
import logging
import os
import platform
import sys

# macOS : torch et faiss embarquent chacun leur libomp, la seconde à s'initialiser
# fait abort. Même parade qu'ailleurs dans le dépôt, à poser avant les imports qui
# tirent l'un ou l'autre — ce script n'en fait qu'un zip, mais patchcore.packaging
# passe par patchcore.banks, donc par les deux.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

import patchcore.banks  # noqa: E402
import patchcore.packaging  # noqa: E402

LOGGER = logging.getLogger(__name__)


@click.command()
@click.argument("bank_dir", type=click.Path(exists=True, file_okay=False))
@click.option("--task", required=True,
              help="Ce que la banque détecte, ex. DetectionKnife. Va dans le nom.")
@click.option("--out", default=patchcore.packaging.CORESETS_DIR, show_default=True,
              help="Dossier de destination.")
@click.option("--force", is_flag=True,
              help="Écraser un .pkg de même nom (sinon on s'arrête).")
def main(bank_dir, task, out, force):
    config_path = os.path.join(bank_dir, patchcore.banks.CONFIG_FILENAME)
    if not os.path.exists(config_path):
        raise SystemExit("Pas de {} dans {}.".format(
            patchcore.banks.CONFIG_FILENAME, bank_dir))
    with open(config_path) as fh:
        config = json.load(fh)

    # La tâche n'existait pas au fit : on la réinjecte dans la config empaquetée,
    # sinon le bandeau de la page et le rangement des captures l'ignorent.
    if config.get("task") != task:
        config["task"] = task
        with open(config_path, "w") as fh:
            json.dump(config, fh, indent=2)

    name = patchcore.packaging.build_name(config, task)
    target = os.path.join(out, name)
    if os.path.exists(target) and not force:
        raise SystemExit("{} existe déjà — relancer avec --force.".format(target))

    patchcore.packaging.pack(bank_dir, target)
    click.echo("{}  ({:.0f} Mo)".format(target, os.path.getsize(target) / 1e6))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
