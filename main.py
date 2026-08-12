"""Point d'entrée : l'interface web de construction de banque et de scoring live.

    python main.py                    # puis ouvrir http://127.0.0.1:8000
    python main.py --port 9000
    python main.py --host 0.0.0.0     # à n'utiliser que sur un réseau de confiance

La page n'a pas d'authentification et le fit accepte une archive arbitraire :
elle est faite pour tourner sur la loopback de la machine qui filme.

Le code de l'application est dans bin/live_web.py ; ce fichier n'existe que
pour donner au dépôt une entrée évidente.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"))

from live_web import main  # noqa: E402  sys.path doit être posé d'abord

if __name__ == "__main__":
    main()
