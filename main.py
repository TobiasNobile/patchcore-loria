"""Le seul exécutable du dépôt : l'interface web de fit et de scoring live.

    python main.py            # puis ouvrir http://127.0.0.1:8000

Sans authentification : à garder sur la loopback. Le code vit dans `src/live/`,
avec le reste des sources ; ce fichier ne fait que le lancer.
"""

import os
import sys

import click

# L'install éditable pose déjà src/ sur le sys.path, mais un clone où seules les
# dépendances sont installées, non : le dépôt doit rester lançable tel quel.
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from live.server import serve  # noqa: E402  sys.path doit être posé d'abord


@click.command()
@click.option("--host", default="127.0.0.1", show_default=True,
              help="Loopback par défaut : la page n'a aucune authentification.")
@click.option("--port", default=8000, show_default=True)
def main(host, port):
    """Sert la page de fit et de scoring live."""
    serve(host, port)


if __name__ == "__main__":
    main()
