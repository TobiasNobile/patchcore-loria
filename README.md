# PatchCore — fork LORIA / Telecom Nancy

Fork de [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection),
l'implémentation de `PatchCore` (Roth et al., 2021, <https://arxiv.org/abs/2106.08265>).
Code amont sous Apache 2.0, historique git conservé depuis le commit initial.

**Retiré de l'amont** : les scripts MVTec (`bin/mvtec/`), les modèles pré-entraînés
et les exemples `sample_*.sh`.

**Modifié dans l'amont** — quatre fichiers, aucun ne change les résultats
(`git diff upstream/main -- src/patchcore/` pour les relire) :

| fichier | modification | pourquoi |
| --- | --- | --- |
| `sampler.py` | projection par blocs ; boucle du coreset réécrite sans matrice `(N, k)` ni recalcul des normes | le nuage non projeté de 20 000 images pèse 60 Go, plus que le GPU. Sélection **numériquement identique** : mêmes indices que l'amont à ancres égales |
| `patchcore.py` | `_fill_memory_bank` préalloue le tableau final au lieu de `np.concatenate` | `concatenate` fait coexister la liste et son résultat, soit 120 Go pour 20 000 images |
| `utils.py` | `set_torch_device` vérifie `cuda.is_available()` et retombe sur CPU en le signalant | l'amont renvoyait un device cuda inexistant et échouait plus loin, sans rapport apparent |
| `datasets/mvtec.py` | ajoute `transform_mean` / `transform_std` | purement additif ; les heatmaps en ont besoin pour dénormaliser l'image |

`common.py` est identique à l'amont. `banks.py`, `packaging.py`, `uploads.py`,
`tracking.py` et `datasets/{celeba,coco}.py` sont des ajouts.

**Ajouté** : l'interface web `main.py` (construire une banque depuis un zip
d'images, puis scorer la webcam), le format de banque `coresets/*.pkg`, les
expériences one-class sur CelebA (chapeau) et COCO (couteau) dans
`bin/<dataset>/`, leurs pipelines partagés dans `src/experiments/`, le scoring
webcam en direct (`bin/live_camera.py`, `bin/live_web.py`), le banc de mesure
d'une frame (`bin/bench_live.py`).

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

## Démarrage rapide

```shell
python main.py            # puis ouvrir http://127.0.0.1:8000
```

Une page, deux moitiés. **À gauche**, on construit une banque : un nom de tâche,
un backbone, les couches, le taux de coreset, un zip d'images sans l'anomalie à
détecter, et « Fitter ». **À droite**, on choisit une banque et on score la caméra en direct
— source, stride, échelle couleur et alpha se règlent pendant que ça tourne.

Le fit et le scoring s'excluent : les deux occupent le thread principal, seul
endroit où torch est sûr sur macOS.

### Les banques : `coresets/*.pkg`

Une banque tient dans un fichier, et son nom dit sa configuration :

```
coresets/WideResNet50_DetectionKnife_l3-l4_p0.01_ts20000.pkg
         backbone     tâche          couches coreset images de fit
```

C'est un zip de ce qu'écrit le fit (index faiss, paramètres, `fit_config.json`).
Le fichier de config reste la référence : le nom n'en est qu'un résumé, lisible
sans ouvrir l'archive. À côté du `.pkg`, un dossier de même nom garde les images
qui ont servi — de quoi refitter autrement sans les renvoyer.

Les deux sont gitignorés : un `.pkg` pèse de 80 Mo à 750 Mo, à distribuer en
release plutôt qu'à committer (`git add -f` pour forcer une banque de démo).

Une banque construite par les scripts de recherche s'y convertit sans refit :

```shell
python bin/pack_bank.py models/coco/wideresnet50_l3-l4_..._ts20000_s0 --task DetectionKnife
```

### Le zip d'images

Des images à plat = toutes sont sans l'anomalie à détecter. Un zip contenant
`normal/` et `anomaly/` garde la séparation, les contre-exemples ne servant qu'à
calibrer un seuil. 20 % de ces images est réservé hors banque pour ce calibrage :
envoyer 400 images en met 320 dans la banque, et la page affiche les deux
chiffres.

Le coût du fit est dominé par la sélection du coreset, quadratique en nombre de
patchs : quelques centaines d'images passent en secondes sur un portable, 20 000
demandent des heures et un GPU. C'est aussi le plafond : au-delà de 20 000 images
utilisables, la page (dossier) ou le serveur (zip) en tire 20 000 au hasard, un
sous-dossier `anomaly/` restant gardé entier. Pour une démo filmée sur place, quelques
centaines d'images du décor réel valent mieux que des milliers d'images
génériques (cf. « Déploiement sur une scène réelle » plus bas).

## Organisation des scripts et des sorties

Deux axes : le dataset, puis la phase. Le fit est la moitié coûteuse et hors-ligne
(features + coreset) et n'écrit que des banques ; l'inférence recharge une banque
et n'écrit que des figures et des mesures.

```
main.py                     # l'interface web : fit + scoring live
coresets/<nom>.pkg          # banques empaquetées (gitignoré)
coresets/<nom>/normal/      # les images qui ont servi au fit (gitignoré)

bin/
  celeba/  fit/memory_bank.py  infer/{heatmap,histogram}.py  fit_and_score.sh
  coco/    fit/memory_bank.py  infer/{heatmap,histogram}.py  fit_and_score.sh
           sweep_coreset.sh
  live_camera.py            # fenêtre OpenCV      | agnostiques : le dataset
  live_web.py               # l'app servie par main.py
  bench_live.py             # coût d'une frame, étape par étape
  pack_bank.py              # models/<tag>/ -> coresets/<nom>.pkg

src/experiments/            # métriques et pipelines partagés par les datasets
tools/
  coco_fetch.py             # télécharge les seules images COCO nécessaires
  dataset_export.py         # zips « good » de MTD et mini-ShanghaiTech
  aggregate_runs.py         # moyenne ± écart-type par configuration
  mlflow_import.py          # rapatrie les runs distants dans mlruns.db
models/<dataset>/<tag>/     # banques mémoire des scripts de recherche (gitignoré)
results/<tâche>/<sortie>/   # figures et mesures (gitignoré)
```

Les scripts live déduisent de la banque où écrire leurs captures
(`results/<tâche>/captures/<couche>/<coreset>/v<vmax>/`).

## Reproductibilité et seeds

`FIT_SEED` fixe le tirage de bout en bout : sous-ensemble d'images
d'entraînement, initialisation de la projection du coreset, échantillon de test
équilibré. Deux seeds donnent donc deux banques et deux mesures indépendantes.
Il apparaît dans le nom du dossier de banque et dans celui du `.pkg`, si bien
qu'un même réglage rejoué n'écrase pas le précédent.

```shell
SEEDS="0 1 2" bash bin/coco/sweep_seeds.sh    # une config, plusieurs tirages
python tools/aggregate_runs.py results/coco --markdown
```

`aggregate_runs.py` regroupe les sidecars par configuration — identité lue dans
le nom de la banque, seed retiré — et sort moyenne ± écart-type. Il réduit
d'abord par seed : deux sidecars d'un même seed sont des doublons, et leur
dispersion n'est pas une variance de tirage. Une configuration à un seul seed est
marquée `n=1`, son écart-type étant inconnu et non nul.

**Ce que les résultats publiés plus bas ne disent pas encore.** Tous les runs
existants sont en seed 0, sans réplication : les écarts entre lignes n'ont pas de
barre d'erreur, et la non-monotonie COCO (0,837 à 40 k, 0,627 à 50 k) est pour
l'instant indissociable du bruit de tirage. Les sidecars antérieurs à août 2026
n'enregistraient pas `layers_to_extract_from` — la couche ne s'y lit que par le
nom du dossier de banque, ce que l'agrégateur exploite pour rester utilisable sur
l'historique.

**Deux biais de protocole à connaître.** Le prétraitement est un `Resize` suivi
d'un `CenterCrop` : les défauts hors du centre sont invisibles et les faux
positifs artificiellement réduits. Et il n'existe pas de split de validation —
seuils et échelles de couleur sont choisis en regardant le test.

## Rendu de la heatmap

Trois constantes de `bin/live_web.py` gouvernent l'affichage, et donc les
captures. Aucune ne touche aux scores.

| constante | rôle |
| --- | --- |
| `COLORMAP_LOW` / `COLORMAP_HIGH` = 0,1 / 0,9 | écrêtent l'indice dans la rampe jet. Au-delà de 0,9 elle vire au bordeaux, où deux distances très différentes rendent la même couleur ; sous 0,1 elle plonge dans le bleu nuit |
| `OPACITY_MAX` = 0,9 | plafonne le mélange, pour que l'objet reste visible sous la tache même à très grande distance |
| `SMOOTHING_FRAMES` = 10 | profondeur des agrégations `moyenne` / `maximum` proposées dans la page |

L'écrêtage porte sur la **couleur seule** ; l'opacité suit la valeur brute, ce
qui laisse le fond normal parfaitement intact — un score nul rend l'image nue,
pas un voile bleu.

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
| resnet18 | 160 px | p0.01 | 20 000 | 2,2 ms | 8,8 ms | 10,4 ms | 0,8 ms | 0,6 ms | 22,2 ms | 43,9 | 0,5388 |
| resnet18 | 224 px | p0.005 | 19 600 | 2,4 ms | 12,9 ms | 13,6 ms | 2,0 ms | 1,0 ms | 30,9 ms | 31,4 | 0,4855 |

Alléger le backbone achète des FPS et coûte de l'AUROC, jusqu'à tomber au niveau
du hasard : resnet18 à 224 px est à 0,4855, soit sous 0,5. Les seules configurations
utiles au-dessus de 30 FPS restent resnet50 · 128 px (0,5989) et resnet18 · 160 px
(0,5388) — encore loin des 0,6406 de wideresnet50.

Device : `PATCHCORE_DEVICE` = `auto` (cuda sinon cpu) | `cpu` | `cuda[:N]` | `mps`.
MPS est exclu de l'automatique — PatchCore y échoue sur le pooling adaptatif, et
s'y révèle plus lent que le CPU (embed 29,8 contre 15,2 ms à 128 px).

GPU (NVIDIA L40S, `INFER_FAISS_GPU=1`) :

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

Filmer la scène sans l'anomalie à détecter, sous toutes ses variations : tout ce
qui n'est pas dans la banque sera scoré comme anormal.

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

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
