# PatchCore — fork LORIA / Telecom Nancy

Fork de [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection),
l'implémentation de `PatchCore` (Roth et al., 2021, <https://arxiv.org/abs/2106.08265>).
Code amont sous Apache 2.0, historique git conservé depuis le commit initial.

**Retiré de l'amont** : les scripts MVTec (`bin/mvtec/`), les modèles pré-entraînés
et les exemples `sample_*.sh`. `src/patchcore/` n'est pas modifié.

**Ajouté** : les expériences one-class sur CelebA (chapeau) et COCO (couteau) dans
`bin/<dataset>/`, leurs pipelines partagés dans `src/experiments/`, le scoring
webcam en direct (`bin/live_camera.py`, `bin/live_web.py`), le banc de mesure
d'une frame (`bin/bench_live.py`), et les lanceurs Grid'5000 / DCE Metz.

Contact amont pour PatchCore lui-même : karsten.rh1@gmail.com.

---

### Citing

If you use the code in this repository, please cite

```
@misc{roth2021total,
      title={Towards Total Recall in Industrial Anomaly Detection},
      author={Karsten Roth and Latha Pemula and Joaquin Zepeda and Bernhard Schölkopf and Thomas Brox and Peter Gehler},
      year={2021},
      eprint={2106.08265},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Organisation des scripts et des sorties

Deux axes : le dataset, puis la phase. Le fit est la moitié coûteuse et hors-ligne
(features + coreset) et n'écrit que des banques ; l'inférence recharge une banque
et n'écrit que des figures et des mesures.

```
bin/
  celeba/  fit/memory_bank.py  infer/{heatmap,histogram}.py  fit_and_score.sh
  coco/    fit/memory_bank.py  infer/{heatmap,histogram}.py  fit_and_score.sh
           sweep_coreset.sh
  live_camera.py            # fenêtre OpenCV      | agnostiques : le dataset
  live_web.py               # page sur localhost  | vient de --bank_dir
  bench_live.py             # coût d'une frame, étape par étape

src/experiments/            # métriques et pipelines partagés par les datasets
tools/                      # coco_fetch.py, mlflow_import.py
models/<dataset>/<tag>/     # banques mémoire (gitignoré)
results/<dataset>/<sortie>/ # figures et mesures (gitignoré)
```

Les scripts live déduisent de `--bank_dir` où écrire leurs captures
(`models/coco/…` -> `results/coco/captures/<couche>/<coreset>/v<vmax>/`).

## CelebA — résumé des résultats

Extension de PatchCore à CelebA, l'attribut `Wearing_Hat` servant de label
d'anomalie : la banque mémoire est construite sur des visages sans chapeau, et
l'on compare les distributions de scores no-hat vs hat sur un échantillon test
équilibré (839 images par classe).

L'indice de Jaccard mesure le recouvrement des deux distributions sur les bins
de l'histogramme (intersection / union), c'est-à-dire la proportion de faux
positifs et faux négatifs incompressibles. Plus bas = mieux séparé.

Balayage à coreset fixé à 5 %, backbone `wideresnet50` :

| Images d'entraînement | Banque | Voisins | Jaccard | AUROC | W1 normalisée |
| --- | --- | --- | --- | --- | --- |
| 1 000  | 39 200  | 1 | 0,3856 | 0,7808 | 0,983 |
| 1 000  | 39 200  | 3 | 0,3856 | 0,7772 | 0,966 |
| 2 000  | 78 400  | 1 | 0,3732 | 0,7868 | 1,016 |
| 2 000  | 78 400  | 3 | 0,3799 | 0,7822 | 0,992 |
| 5 000  | 196 000 | 1 | 0,3777 | 0,7951 | 1,063 |
| 5 000  | 196 000 | 3 | 0,3788 | 0,7904 | 1,042 |
| 10 000 | 392 000 | 1 | 0,3565 | 0,8035 | 1,103 |
| 10 000 | 392 000 | 3 | 0,3642 | 0,7994 | 1,082 |

L'AUROC et la distance de Wasserstein normalisée progressent régulièrement avec
la taille de banque, tandis que le Jaccard reste dans un mouchoir jusqu'à 5 000
avant de baisser nettement à 10 000. La configuration `ts=20000` reste à produire.

Le coreset est quadratique en nombre de features : au-delà de 20 000 images
d'entraînement, le fit dépasse la journée de calcul et la mémoire d'un nœud
ordinaire (le nuage brut atteint 60 Go, doublé pendant sa concaténation).

Reproduction : `FIT_TRAIN_SUBSET=<n> FIT_CORESET_PCT=<p> bash
bin/celeba/fit_and_score.sh`, ou via `./grid5000_run.sh`.

### Balayage de couches (single-layer, ts=20000, coreset 5 %, 3 NN)

Une seule couche du backbone à la fois. WideResNet50 est un ResNet : couches
disponibles `layer1`..`layer4` (pas de layer5) ; la résolution des patches est
celle de la couche extraite.

| Couche | Résolution | AUROC | Jaccard |
| --- | --- | --- | --- |
| layer2 | 28×28 | 0,760 | 0,431 |
| layer3 | 14×14 | **0,859** | **0,270** |
| layer4 | 7×7   | 0,694 | 0,492 |

Courbe en cloche : `layer3` (features mid-level) sépare le mieux no-hat / hat ;
`layer2` est trop fin (sensible à la texture), `layer4` trop grossier (49 patches,
perd la localisation du chapeau). Reproduction :
`FIT_LAYERS=layer3 FIT_TRAIN_SUBSET=20000 bash bin/celeba/fit_and_score.sh`.

## COCO — détection « personne + couteau »

Même protocole one-class (banque sur personnes sans couteau, test personne avec
couteau), coreset 5 %, 3 NN, backbone WideResNet50. Balayage couches × taille de
banque (AUROC) :

| Couches | 20 000 | 40 000 | 50 000 |
| --- | --- | --- | --- |
| layer2 seul | 0,612 | 0,621 | 0,605 |
| layer3 seul | 0,636 | 0,837\* | 0,627 |
| layer2 + layer3 | 0,715 | 0,595 | 0,588 |

Les scores restent faibles (~0,6) : les scènes COCO sont très hétérogènes et
PatchCore score surtout la nouveauté de scène plutôt que le petit couteau. Le
0,837 (\*layer3, 40k) est un point non-monotone (variance : layer3 retombe à ~0,63
en 20k et 50k), à ne pas sur-interpréter. Contraste avec CelebA (visages alignés,
AUROC 0,86) : PatchCore exige un « normal » homogène, que COCO n'offre pas.

## Cadence live — coût d'une frame (`bin/bench_live.py`)

COCO l3-l4, ts=20000. Budgets : 33,3 ms = 30 FPS, 16,7 ms = 60 FPS. `scoring`
exclut l'encodage JPEG.

CPU (Apple M-series, torch 4 threads / faiss 1) :

| backbone | taille | coreset | banque | preprocess | embed | faiss | post | encode | scoring | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 | 224 px | p0.01 | 39 200 | 2,3 ms | 54,2 ms | 26,8 ms | 3,9 ms | 1,0 ms | 87,2 ms | 11,3 | 0,6375 |
| wideresnet50 | 224 px | p0.02 | 78 400 | 2,5 ms | 58,6 ms | 54,2 ms | 0,0 ms | 0,9 ms | 115,3 ms | 8,6 | 0,6395 |
| wideresnet50 | 224 px | p0.05 | 196 000 | 2,4 ms | 52,7 ms | 133,4 ms | 2,4 ms | 1,0 ms | 190,9 ms | 5,2 | 0,6406 |
| resnet50 | 128 px | p0.01 | 12 800 | 1,9 ms | 14,7 ms | 4,7 ms | 1,7 ms | 0,4 ms | 23,1 ms | 42,6 | 0,5989 |
| resnet18 | 128 px | p0.01 | 12 800 | 1,9 ms | 6,7 ms | 4,5 ms | 1,1 ms | 0,4 ms | 14,2 ms | 68,4 | — |
| resnet18 | 160 px | p0.01 | 20 000 | 2,1 ms | 8,4 ms | 10,2 ms | 0,9 ms | 0,6 ms | 21,4 ms | 45,5 | — |
| resnet18 | 192 px | p0.01 | 28 800 | 2,2 ms | 10,5 ms | 16,5 ms | 1,3 ms | 0,7 ms | 30,4 ms | 32,1 | — |
| resnet18 | 224 px | p0.01 | 39 200 | 2,3 ms | 12,8 ms | 27,4 ms | 1,4 ms | 0,9 ms | 43,9 ms | 22,3 | — |
| resnet18 | 224 px | p0.005 | 19 600 | 2,3 ms | 12,8 ms | 13,6 ms | 1,4 ms | 0,9 ms | 30,2 ms | 32,0 | — |
| resnet34 | 128 px | p0.01 | 12 800 | 1,9 ms | 11,3 ms | 4,6 ms | 1,2 ms | 0,4 ms | 19,0 ms | 51,4 | — |
| resnet34 | 160 px | p0.01 | 20 000 | 2,1 ms | 13,9 ms | 9,9 ms | 1,1 ms | 0,6 ms | 27,0 ms | 36,3 | — |
| resnet34 | 192 px | p0.01 | 28 800 | 2,2 ms | 16,8 ms | 16,3 ms | 1,7 ms | 0,7 ms | 37,1 ms | 26,4 | — |
| resnet50 | 160 px | p0.01 | 20 000 | 2,1 ms | 19,0 ms | 9,8 ms | 0,9 ms | 0,6 ms | 31,9 ms | 30,8 | — |
| resnet50 | 192 px | p0.01 | 28 800 | 2,2 ms | 23,9 ms | 16,3 ms | 1,3 ms | 0,7 ms | 43,7 ms | 22,5 | — |
| resnet50 | 224 px | p0.01 | 39 200 | 2,3 ms | 29,6 ms | 27,2 ms | 1,2 ms | 0,9 ms | 60,4 ms | 16,3 | — |
| wideresnet50 | 128 px | p0.01 | 12 800 | 2,3 ms | 28,7 ms | 4,6 ms | 0,8 ms | 0,4 ms | 36,5 ms | 27,1 | — |
| wideresnet50 | 160 px | p0.01 | 20 000 | 2,1 ms | 34,8 ms | 9,9 ms | 2,1 ms | 0,6 ms | 48,9 ms | 20,2 | — |
| wideresnet50 | 192 px | p0.01 | 28 800 | 2,2 ms | 45,2 ms | 16,4 ms | 1,3 ms | 0,7 ms | 65,2 ms | 15,2 | — |

AUROC « — » : temps mesurés avec une banque synthétique de la taille exacte
qu'aurait le fit correspondant (le coût d'un `IndexFlatL2` ne dépend que du
nombre de vecteurs et de requêtes). Recoupé sur les deux banques réelles :
42,6 vs 43,7 FPS et 11,3 vs 11,4 FPS. Seul l'AUROC demande un vrai fit.

Device : `PATCHCORE_DEVICE` = `auto` (cuda sinon cpu) | `cpu` | `cuda[:N]` | `mps`.
MPS est exclu de l'automatique — PatchCore y échoue sur le pooling adaptatif, et
s'y révèle plus lent que le CPU (embed 29,8 contre 15,2 ms à 128 px).

GPU (Grid'5000, NVIDIA L40S, `INFER_FAISS_GPU=1`) :

| backbone | taille | coreset | banque | preprocess | embed | faiss | post | encode | scoring | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 | 224 px | p0.01 | 39 200 | 3,9 ms | 8,2 ms | 0,9 ms | 1,4 ms | 0,7 ms | 14,3 ms | 66,6 | 0,6375 |
| wideresnet50 | 224 px | p0.02 | 78 400 | 3,9 ms | 8,2 ms | 1,8 ms | 1,4 ms | 0,7 ms | 15,2 ms | 62,7 | 0,6395 |
| wideresnet50 | 224 px | p0.05 | 196 000 | 4,0 ms | 8,2 ms | 4,1 ms | 1,4 ms | 0,7 ms | 17,7 ms | 54,5 | 0,6406 |

## Déploiement sur une scène réelle

Les banques COCO/CelebA scorent surtout la nouveauté de scène. Sur un robot qui
filme toujours le même environnement, fitter sur *cette* scène :

```shell
python bin/capture.py --out data/scene/normal  --count 400 --every 0.5
python bin/capture.py --out data/scene/anomaly --count 60     # pour le seuil
SCENE_PATH=data/scene python bin/scene/fit/memory_bank.py
SCENE_PATH=data/scene python bin/scene/infer/histogram.py     # seuil à lire entre les modes
python bin/live_web.py                                        # puis choisir la banque
```

Filmer le nominal sous toutes ses variations : tout ce qui n'est pas dans la
banque sera scoré comme anormal.

## MLflow

Tous les runs vivent dans une base unique, y compris ceux rapatriés des serveurs
distants. Chaque run porte un tag `origin` (`local`, `g5k`, `metz`) et une
expérience par tâche (`celeba-histograms`, `celeba-heatmap`) :

```shell
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Les runs distants sont fusionnés dans cette base au moment du rapatriement
(`./grid5000_run.sh --fetch`), via `tools/mlflow_import.py`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
