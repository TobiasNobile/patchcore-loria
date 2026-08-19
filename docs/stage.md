# La version complète — expériences, calcul distant, résultats

Le [README](../README.md) documente la démo et le cœur de PatchCore, publiés
seuls sur la branche `main`. Ce fichier documente ce que cette branche ajoute
par-dessus : les expériences one-class sur datasets publics, les scripts de
calcul distant, et les résultats mesurés avec.

## Ce que cette branche ajoute

```
bin/
  celeba/  fit/memory_bank.py  infer/{heatmap,histogram}.py  fit_and_score.sh
  coco/    fit/memory_bank.py  infer/{heatmap,histogram}.py  fit_and_score.sh
           sweep_coreset.sh
  scene/   fit/memory_bank.py  infer/histogram.py    # les mêmes, sur une scène filmée
  capture.py                # intervallomètre webcam -> data/scene/normal
  pack_bank.py              # models/<tag>/ -> coresets/<nom>.pkg
src/
  experiments/benchmarks.py       # les specs CELEBA et COCO
  experiments/reports.py          # histogramme et heatmaps : relire une banque
  experiments/metrics.py          # Jaccard, Wasserstein, test de Student
  patchcore/metrics.py            # AUROC, amont
  patchcore/tracking.py           # le run MLflow
  patchcore/datasets/celeba.py    # CelebA one-class (attribut Wearing_Hat)
  patchcore/datasets/coco.py      # COCO one-class (personne ± couteau)
  patchcore/datasets/mvtec.py     # MVTec, le jeu de référence de l'amont
tools/
  coco_fetch.py             # télécharge les seules images COCO nécessaires
  dataset_export.py         # zips « good » de MTD et mini-ShanghaiTech
  aggregate_runs.py         # moyenne ± écart-type par configuration
  mlflow_import.py          # rapatrie les runs distants dans mlruns.db
test/                       # la suite pytest du cœur et des pipelines
grid5000_run.sh             # exécute un script sur un nœud GPU de Grid'5000
remote_run.sh               # idem sur le DCE de CentraleSupélec Metz
```

Les tests couvrent `sampler.py` et `patchcore.py`, les deux fichiers modifiés
par rapport à l'amont : `test_coreset_sampling_on_same_samples` est ce qui
soutient la ligne « sélection numériquement identique » du README.

Le dépôt publiable, lui, ne garde que ce qui fait tourner la page : le fit
(`experiments/pipelines.py`), le scoring live, et rien d'autre. Toute mesure —
histogramme, heatmap, AUROC, MLflow — vit ici.

Ces scripts installés :

```shell
uv sync --extra experiments        # datasets (HuggingFace) et mlflow
```

## Comment les deux versions restent alignées

`main` est le sous-ensemble publiable, `stage` le contient tout entier. **Le code
partagé se modifie sur `main`, et `stage` le reprend par merge — jamais
l'inverse.** Ce qui est propre aux expériences n'existe que sur `stage`, et
n'ajoute que des fichiers : aucun fichier de `main` n'est modifié ici, ce qui est
la raison pour laquelle les merges ne conflictent pas.

```shell
git worktree add ../patchcore-publi main   # les deux versions côte à côte
git checkout stage && git merge main       # récupérer ce qui a bougé côté démo
```

Corriger le cœur depuis cette branche est donc à éviter : la correction se fait
dans le worktree `main`, puis revient ici par merge. Si l'ordre a été inversé par
inadvertance, `git cherry-pick` vers `main` puis `git rebase main stage` remet
les choses d'aplomb — `stage` peut être réécrite librement, rien n'en dépend.

## Calcul distant

```shell
./grid5000_run.sh bin/celeba/fit/memory_bank.py
DETACH=true OAR_RESOURCES='host=1' OAR_PROPERTIES="cluster='gres'" \
    ./grid5000_run.sh bin/coco/fit_and_score.sh
./grid5000_run.sh --fetch                  # rapatrier un job détaché terminé
./remote_run.sh bin/celeba/infer/heatmap.py --image_index 900
```

Réserver le nœud entier (`OAR_RESOURCES='host=1'`) : le pic de mémoire est le
nuage de features complet, avant coreset, et il tient dans la RAM CPU du nœud —
pas dans la part qu'une réservation partielle accorde.

## Reproductibilité : balayages et agrégation

```shell
SEEDS="0 1 2" bash bin/coco/sweep_seeds.sh    # une config, plusieurs tirages
python tools/aggregate_runs.py results/coco --markdown
```

`aggregate_runs.py` regroupe les sidecars par configuration — identité lue dans
le nom de la banque, seed retiré — et sort moyenne ± écart-type. Il réduit
d'abord par seed : deux sidecars d'un même seed sont des doublons, et leur
dispersion n'est pas une variance de tirage. Une configuration à un seul seed est
marquée `n=1`, son écart-type étant inconnu et non nul.

**Ce que les résultats ci-dessous ne disent pas encore.** Tous les runs existants
sont en seed 0, sans réplication : les écarts entre lignes n'ont pas de barre
d'erreur, et la non-monotonie COCO (0,837 à 40 k, 0,627 à 50 k) est pour l'instant
indissociable du bruit de tirage. Les sidecars antérieurs à août 2026
n'enregistraient pas `layers_to_extract_from` — la couche ne s'y lit que par le
nom du dossier de banque, ce que l'agrégateur exploite pour rester utilisable sur
l'historique.

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
bin/celeba/fit_and_score.sh`.

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

## MLflow

Tous les runs vivent dans une base unique, y compris ceux rapatriés des serveurs
distants. Chaque run porte un tag `origin` (`local`, `g5k`, `metz`) et une
expérience par tâche (`celeba-histograms`, `celeba-heatmap`) :

```shell
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Les runs distants sont fusionnés dans cette base via `tools/mlflow_import.py`.
MLflow reste du confort : la source de vérité est le sidecar JSON écrit à côté de
chaque figure, que `tools/aggregate_runs.py` relit.

