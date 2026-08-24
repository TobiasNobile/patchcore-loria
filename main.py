"""Le seul exécutable du dépôt : l'interface web de fit et de scoring live.

    python main.py                # puis ouvrir http://127.0.0.1:8000
    python main.py fetch-bank     # installe la banque de démonstration

Sans authentification : à garder sur la loopback. Le code vit dans `src/live/`,
avec le reste des sources ; ce fichier ne fait que le lancer.
"""

import hashlib
import os
import sys
import urllib.request

import click

# L'install éditable pose déjà src/ sur le sys.path, mais un clone où seules les
# dépendances sont installées, non : le dépôt doit rester lançable tel quel.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

# La banque de démonstration pèse 153 Mo, au-delà des 100 Mio par fichier que
# GitHub accepte au push — et un blob de cette taille resterait dans l'historique
# de tous les clones, même supprimé ensuite. Elle est donc publiée comme asset de
# release, à côté du dépôt et hors de son historique.
BANK_REPO = os.environ.get("BANK_REPO", "TobiasNobile/patchcore-loria")
BANK_VERSION = os.environ.get("BANK_VERSION", "v0.1")
BANK_NAME = "WideResNet50_DetectionKnife_l3-l4_p0.01_ts20000.pkg"
BANK_SHA256 = "542bd27165b0f1a7d7b1e341c231726cc940aba25ad9294261233fb84fd9fa0a"


@click.group(invoke_without_command=True)
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Loopback par défaut : la page n'a aucune authentification.")
@click.option("--port", default=8000, show_default=True)
@click.pass_context
def cli(ctx, host, port):
    """Sans sous-commande, sert la page."""
    if ctx.invoked_subcommand is not None:
        return
    # Importé ici et non en tête : la sous-commande fetch-bank ne charge alors ni
    # torch ni faiss, soit une dizaine de secondes épargnées pour un téléchargement.
    from live.server import serve

    serve(host, port)


@cli.command("fetch-bank")
@click.option("--version", default=BANK_VERSION, show_default=True,
              help="Release d'où tirer la banque.")
@click.option("--force", is_flag=True, help="Retélécharger même si elle est là.")
def fetch_bank(version, force):
    """Installe la banque de démonstration dans coresets/."""
    dest = os.path.join(_ROOT, "coresets", BANK_NAME)
    if os.path.exists(dest) and not force:
        click.echo("Déjà installée : {}".format(dest))
        return

    url = "https://github.com/{}/releases/download/{}/{}".format(
        BANK_REPO, version, BANK_NAME)
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".part"
    click.echo("Téléchargement de {} (153 Mo, release {})…".format(BANK_NAME, version))

    # Écrit à côté puis renommé, somme calculée au passage : un .pkg présent est
    # toujours complet et vérifié, jamais un téléchargement à moitié fini.
    digest = hashlib.sha256()
    try:
        with urllib.request.urlopen(url) as reponse, open(tmp, "wb") as sortie:
            total = int(reponse.headers.get("Content-Length") or 0)
            recu = 0
            with click.progressbar(length=total or None, label="  ") as barre:
                while True:
                    bloc = reponse.read(1 << 20)
                    if not bloc:
                        break
                    sortie.write(bloc)
                    digest.update(bloc)
                    recu += len(bloc)
                    if total:
                        barre.update(len(bloc))
        obtenu = digest.hexdigest()
        if obtenu != BANK_SHA256:
            raise click.ClickException(
                "Somme de contrôle inattendue — rien n'a été installé.\n"
                "  attendu : {}\n  obtenu  : {}".format(BANK_SHA256, obtenu))
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)

    click.echo("Installée : {}".format(dest))
    click.echo("Lancer « python main.py » : la page la présélectionne à l'ouverture.")


if __name__ == "__main__":
    cli()
