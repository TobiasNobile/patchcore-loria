"""Code partagé par les expériences de bin/ (celeba, coco).

Séparé de `patchcore/` (amont, non modifié) : ici vivent les métriques de
comparaison et les trois pipelines — fit, histogramme, heatmaps — dont les
scripts de bin/ ne sont plus que la configuration.
"""

import os
import platform

# macOS : torch et faiss embarquent chacun leur libomp, la seconde à s'initialiser
# fait abort. Posé ici, donc avant que les sous-modules n'importent l'un ou l'autre.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
