"""Point d'entrée : l'interface web de fit et de scoring live.

    python main.py            # puis ouvrir http://127.0.0.1:8000

Sans authentification : à garder sur la loopback. Le code est dans
bin/live_web.py ; ce fichier ne sert qu'à donner au dépôt une entrée évidente.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "bin"))

from live_web import main  # noqa: E402  sys.path doit être posé d'abord

if __name__ == "__main__":
    main()
