"""La démonstration : une page web pour construire une banque et scorer une
source vidéo avec.

`main.py`, à la racine, est le seul exécutable du dépôt ; il n'appelle que
`live.server.main`. Le module `live.scoring` tient ce qui touche à une frame —
prétraitement, agrégation, réglages faiss — séparé du serveur qui l'orchestre.
"""
