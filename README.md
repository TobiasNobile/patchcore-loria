# PatchCore — fork LORIA / Telecom Nancy

Fork de [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection),
l'implémentation de `PatchCore` (Roth et al., 2021, <https://arxiv.org/abs/2106.08265>).
Code amont sous Apache 2.0, historique git conservé depuis le commit initial.

**Retiré de l'amont** : les scripts MVTec (`bin/mvtec/`) et le dataset qui allait
avec, les modèles pré-entraînés, les exemples `sample_*.sh`, et tout ce que plus
rien n'appelait — `utils.py` se réduit à `fix_seeds`.

**Modifié dans l'amont** — deux fichiers, aucun ne change les résultats
(`git diff upstream/main -- src/patchcore/` pour les relire) :

| fichier | modification | pourquoi |
| --- | --- | --- |
| `sampler.py` | projection par blocs ; boucle du coreset réécrite sans matrice `(N, k)` ni recalcul des normes | le nuage non projeté de 20 000 images pèse 60 Go, plus que le GPU. Sélection **numériquement identique** : mêmes indices que l'amont à ancres égales |
| `patchcore.py` | `_fill_memory_bank` préalloue le tableau final au lieu de `np.concatenate` | `concatenate` fait coexister la liste et son résultat, soit 120 Go pour 20 000 images |

`common.py` est identique à l'amont. `banks.py`, `packaging.py` et `uploads.py`
sont des ajouts.

**Ajouté** : l'interface web `main.py` (construire une banque depuis un zip
d'images, puis scorer la webcam), dont le code vit dans `src/live/`, le format de
banque `coresets/*.pkg`, et les pipelines de fit et d'inférence partagés dans
`src/experiments/`. Le dépôt n'a qu'un exécutable : tout passe par `main.py`. Les
scripts d'expérience — fit sur scène filmée, pipelines CelebA et COCO, banc de
mesure d'une frame — vivent dans `bin/` sur la branche `stage`.

**Deux versions.** Ceci est la version minimale : la démo et le cœur de
PatchCore, sans dataset à télécharger ni serveur de calcul. La version complète
— expériences one-class sur CelebA et COCO, scripts de calcul distant
(Grid'5000, DCE Metz), suivi MLflow et résultats mesurés — vit sur la branche
`stage`, dont celle-ci est le sous-ensemble. Le code partagé se modifie ici et
`stage` le reprend par merge, jamais l'inverse : c'est ce qui garde les deux
versions alignées sans divergence à réconcilier.

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

Le dépôt livre deux banques dans `coresets/`, prêtes à scorer : la même tâche
« personne + couteau » sur deux backbones, WideResNet50 et ResNet50, à
configuration identique par ailleurs — de quoi voir en direct ce que coûte un
backbone plus léger. La page présélectionne la première. Toute autre banque
déposée dans `coresets/` est trouvée au démarrage suivant.

Une page, deux moitiés. **À gauche**, on construit une banque : un nom de tâche,
un backbone, les couches, le taux de coreset, un zip d'images sans l'anomalie à
détecter, et « Fitter ». **À droite**, on choisit une banque et on score la caméra en direct
— source, stride, échelle couleur et alpha se règlent pendant que ça tourne.

Le fit et le scoring s'excluent : les deux occupent le thread principal, seul
endroit où torch est sûr sur macOS.

Un fichier vidéo est lu **à sa vitesse réelle** : la boucle se cale sur l'horloge
de la source, attend si elle est en avance et saute des frames si elle est en
retard. Sans ça le clip défilerait au rythme du traitement — au ralenti à stride
bas, en accéléré à stride haut — et deux réglages de lissage ne seraient plus
comparables. Le prix est visible dans la page : à stride 1 sur un clip 60 fps,
seule une frame sur dix environ est scorée si l'inférence prend 170 ms.

### Les banques : `coresets/*.pkg`

Une banque tient dans un fichier, et son nom dit sa configuration :

```
coresets/WideResNet50_DetectionKnife_l3-l4_p0.005_ts20000_s0.pkg
         backbone     tâche          couches coreset images  seed
```

Une taille d'image non standard s'y glisse aussi, après les couches : `_im128`.
Rien pour 224 px, le défaut — les banques déjà nommées gardent leur nom.

**D'où viennent ces images.** Rien n'a été filmé pour cette banque : elle est
fittée sur **COCO 2017**, le jeu de photos annotées de Microsoft, récupéré depuis
`images.cocodataset.org`. Le normal, c'est toute image contenant une personne et
**aucun** couteau ; les images de personne **avec** couteau ne rentrent pas dans
la banque, elles ne servent qu'à mesurer ce qui la sépare du reste. Cette
banque-ci a pris 20 000 images des splits `train2017` et `val2017`, fittées sur
un nœud Grid5000 — d'où le `ts20000` du nom. Sa jumelle ResNet50 est fittée sur
les mêmes images, avec les mêmes couches, le même coreset et le même seed : seul
le backbone change.

Une seconde banque, `DetectionKnifeVal2017_l3-l4_p0.005`, tient la même tâche à
partir du seul `val2017` : 1 200 images de personne sans couteau, dont 960 dans
la banque et 240 gardées hors banque pour le seuil, plus les 99 images à couteau
du split. Elle se fitte en deux minutes sur un portable, là où les 20 000
demandent un nœud — le nuage de features avant coreset ne tient pas en 16 Go.

C'est un zip de ce qu'écrit le fit (index faiss, paramètres, `fit_config.json`).
Le fichier de config reste la référence : le nom n'en est qu'un résumé, lisible
sans ouvrir l'archive. À côté du `.pkg`, un dossier de même nom garde les images
qui ont servi — de quoi refitter autrement sans les renvoyer.

Les deux sont gitignorés. Un `.pkg` pèse de 80 Mo à 3 Go, quand GitHub refuse
au push tout fichier au-delà de 100 Mio — et un blob de cette taille resterait
dans l'historique de chaque clone même après suppression. Les banques se
distribuent donc en **asset de release** : un fichier attaché à une version
publiée, hébergé à côté du dépôt et non dedans, que `git clone` ne rapatrie pas.
Une petite banque peut toujours être committée pour de bon avec `git add -f`.

La page empaquette elle-même à la fin d'un fit : le `.pkg` apparaît dans
`coresets/` et dans le sélecteur, sans rien à convertir à la main.

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

Un biais à connaître : le prétraitement est un `Resize` suivi d'un `CenterCrop`.
Ce qui sort du centre du cadre n'est ni appris ni scoré — cadrer la scène en
conséquence. `FIT_SEED` fixe le tirage de bout en bout, et apparaît dans le nom
de la banque : un même réglage rejoué n'écrase pas le précédent.

## Organisation des scripts et des sorties

Deux axes : le dataset, puis la phase. Le fit est la moitié coûteuse et hors-ligne
(features + coreset) et n'écrit que des banques ; l'inférence recharge une banque
et n'écrit que des figures et des mesures.

```
main.py                     # le seul exécutable : sert la page
coresets/<nom>.pkg          # banques empaquetées (gitignoré)
coresets/<nom>/normal/      # les images qui ont servi au fit (gitignoré)

src/live/server.py          # l'app servie par main.py : fit + scoring live
src/live/scoring.py         # une frame : prétraitement, agrégation, faiss
src/patchcore/              # le cœur : backbone, coreset, banque, scoring
src/experiments/            # le fit, du Spec au .pkg
src/templates/live.html     # la page servie par src/live/server.py
src/static/live.{css,js}    # sa feuille de style et son script
```

Les scripts live déduisent de la banque où écrire leurs captures
(`results/<tâche>/captures/<couche>/<coreset>/v<vmax>/`).

## Rendu de la heatmap

Trois constantes de `src/live/server.py` gouvernent l'affichage, et donc les
captures. Aucune ne touche aux scores.

| constante | rôle |
| --- | --- |
| `COLORMAP_LOW` / `COLORMAP_HIGH` = 0,1 / 0,9 | écrêtent l'indice dans la rampe jet. Au-delà de 0,9 elle vire au bordeaux, où deux distances très différentes rendent la même couleur ; sous 0,1 elle plonge dans le bleu nuit |
| `OPACITY_MAX` = 0,9 | plafonne le mélange, pour que l'objet reste visible sous la tache même à très grande distance |
| `SMOOTHING_SECONDS` = 1/3 | durée de vidéo couverte par la case « Lissage », réglable en direct dans la page. Le nombre de cartes en découle, via `calculer_nb_heatmaps(fps, stride, seconds)` de `src/live/scoring.py`, à partir du stride **effectif** — celui que la lecture en temps réel impose, sauts compris |

L'écrêtage porte sur la **couleur seule** ; l'opacité suit la valeur brute, ce
qui laisse le fond normal parfaitement intact — un score nul rend l'image nue,
pas un voile bleu.

## Cadence live — coût d'une frame

Banque « personne + couteau » (COCO, layer3 + layer4, 20 000 images de fit —
construite dans la version complète). Budgets : 33,3 ms = 30 FPS, 16,7 ms = 60 FPS. `scoring`
exclut l'encodage JPEG. Mesuré avec le banc `bin/bench_live.py`, qui vit sur
`stage` avec le reste des outils de mesure.

CPU (Apple M-series, torch 4 threads / faiss 1) :

| backbone | taille | coreset | banque | preprocess | embed | faiss | post | encode | scoring | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 | 224 px | p0.005 | 19 600 | 2,5 ms | 54,5 ms | 16,4 ms | 2,7 ms | 1,0 ms | 76,2 ms | 12,9 | 0,6321 |
| wideresnet50 | 224 px | p0.01 | 39 200 | 2,3 ms | 54,2 ms | 26,8 ms | 3,9 ms | 1,0 ms | 87,2 ms | 11,3 | 0,6375 |
| wideresnet50 | 224 px | p0.02 | 78 400 | 2,5 ms | 58,6 ms | 54,2 ms | 0,0 ms | 0,9 ms | 115,3 ms | 8,6 | 0,6395 |
| wideresnet50 | 224 px | p0.05 | 196 000 | 2,4 ms | 52,7 ms | 133,4 ms | 2,4 ms | 1,0 ms | 190,9 ms | 5,2 | 0,6406 |
| wideresnet50 | 160 px | p0.005 | 10 000 | 2,2 ms | 34,4 ms | 5,5 ms | 1,8 ms | 0,6 ms | 43,9 ms | 22,5 | 0,6309 |
| wideresnet50 | 128 px | p0.005 | 6 400 | 2,0 ms | 27,6 ms | 2,5 ms | 1,5 ms | 0,4 ms | 33,5 ms | 29,4 | 0,6355 |
| resnet50 | 160 px | p0.005 | 10 000 | 2,2 ms | 18,4 ms | 5,5 ms | 1,4 ms | 0,6 ms | 27,5 ms | 35,6 | 0,5634 |
| resnet50 | 128 px | p0.005 | 6 400 | 2,0 ms | 14,9 ms | 2,4 ms | 0,8 ms | 0,4 ms | 20,2 ms | 48,5 | 0,5851 |
| resnet50 | 128 px | p0.01 | 12 800 | 1,9 ms | 14,7 ms | 4,7 ms | 1,7 ms | 0,4 ms | 23,1 ms | 42,6 | 0,5989 |
| resnet18 | 160 px | p0.01 | 20 000 | 2,2 ms | 8,8 ms | 10,4 ms | 0,8 ms | 0,6 ms | 22,2 ms | 43,9 | 0,5388 |
| resnet18 | 224 px | p0.005 | 19 600 | 2,4 ms | 12,9 ms | 13,6 ms | 2,0 ms | 1,0 ms | 30,9 ms | 31,4 | 0,4855 |

Le coreset butte vite : diviser la banque par cinq (p0.05 → p0.01) rend 6,1 FPS,
la diviser encore par deux n'en rend plus que 1,6. Le temps faiss n'est pas tout
à fait proportionnel — 16,4 ms à 19 600 vecteurs contre 26,8 à 39 200 — il porte
un coût fixe d'environ 6 ms. Et surtout le backbone, lui, ne bouge pas : ses
54 ms font désormais les trois quarts du budget. À 224 px sur wideresnet50, même
une banque vide plafonnerait vers 17 FPS.

**C'est la résolution qui débloque, et elle est gratuite.** À coreset et couches
constants, wideresnet50 passe de 12,9 à 22,5 puis 29,4 FPS en descendant de 224 à
160 puis 128 px — et l'AUROC ne baisse pas : 0,6321, 0,6309, **0,6355**. La plus
petite est même la meilleure des trois, à 26 Mo de banque contre 77. Deux raisons
se cumulent : le backbone traite quatre fois moins de pixels (54 → 28 ms) et la
banque compte trois fois moins de vecteurs, puisque le coreset est une fraction
d'un nombre de patchs qui suit la surface.

Changer de backbone, lui, se paie toujours : resnet50 double encore la cadence
(48,5 FPS à 128 px) mais tombe à 0,5851 d'AUROC. Autrement dit, descendre en
résolution avant d'alléger le backbone.

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

Une banque construite sur un dataset public score surtout la nouveauté de scène,
et non l'anomalie cherchée. Sur un robot qui filme toujours le même
environnement, fitter sur *cette* scène : filmer le décor sans l'anomalie à
détecter, sous toutes ses variations, puis envoyer les images dans la moitié
gauche de la page — un dossier `normal/`, et un `anomaly/` si l'on veut de quoi
calibrer un seuil. Tout ce qui n'est pas dans la banque sera scoré comme
anormal, le décor compris s'il a changé.

Quelques centaines d'images espacées valent mieux que des milliers de frames
consécutives : à 30 images par seconde, deux voisines n'apprennent rien de neuf
à la banque.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
