# PatchCore — fork LORIA / Telecom Nancy

Fork de [amazon-science/patchcore-inspection](https://github.com/amazon-science/patchcore-inspection),
l'implémentation de `PatchCore` (Roth et al., 2021, <https://arxiv.org/abs/2106.08265>).
Code amont sous Apache 2.0, historique git conservé depuis le commit initial.

**Ajouté** : l'interface web `main.py` — construire une banque depuis un zip
d'images, puis scorer la webcam —, dont le code vit dans `src/live/`, le format
de banque `coresets/*.pkg`, et les pipelines de fit et d'inférence partagés dans
`src/experiments/`. Le dépôt n'a qu'un exécutable : tout passe par `main.py`.

**Retiré de l'amont** : les scripts MVTec (`bin/mvtec/`) et le dataset qui allait
avec, les modèles pré-entraînés, les exemples `sample_*.sh`, et tout ce que plus
rien n'appelait — `utils.py` se réduit à `fix_seeds`.

**Modifié dans l'amont** — deux fichiers, aucun ne change les résultats
(`git diff upstream/main -- src/patchcore/` pour les relire) :

| fichier | modification | pourquoi |
| --- | --- | --- |
| `sampler.py` | projection par blocs ; boucle du coreset réécrite sans matrice `(N, k)` ni recalcul des normes | le nuage non projeté de 20 000 images pèse 60 Go, plus que le GPU. Sélection **numériquement identique** : mêmes indices que l'amont à ancres égales |
| `patchcore.py` | `_fill_memory_bank` préalloue le tableau final au lieu de `np.concatenate` | `concatenate` fait coexister la liste et son résultat, soit 120 Go pour 20 000 images |

`common.py` est identique à l'amont ; `banks.py`, `packaging.py` et `uploads.py`
sont des ajouts.

**Deux versions.** La version minimale — la démo et le cœur de PatchCore, sans
dataset à télécharger ni serveur de calcul — vit sur `main`. La version complète
y ajoute les expériences one-class CelebA et COCO, les scripts de calcul distant
(Grid'5000, DCE Metz), le suivi MLflow et les résultats mesurés : c'est la
branche `stage`, dont `main` est le sous-ensemble. Le code partagé se modifie sur
`main` et `stage` le reprend par merge, jamais l'inverse.

## Démarrage rapide

```shell
python main.py            # puis ouvrir http://127.0.0.1:8000
```

Le dépôt livre quatre banques dans `coresets/`, prêtes à scorer : la même tâche
« personne + couteau » sur quatre backbones — WideResNet50, ResNet50, ResNet34,
ResNet18 — à configuration identique par ailleurs (224 px, l3-l4, coreset 0,005,
20 000 images, seed 0) : de quoi voir en direct ce que coûte un backbone plus
léger. La page présélectionne WideResNet50, le seul des quatre au-dessus de 0,6
d'AUROC. Toute autre banque déposée dans `coresets/` est trouvée au démarrage
suivant.

Une page, deux moitiés. **À gauche**, on construit une banque : un nom de tâche,
un backbone, les couches, le taux de coreset, un zip d'images sans l'anomalie à
détecter, et « Fitter ». **À droite**, on choisit une banque et on score la
caméra en direct — source, stride, échelle couleur et opacité se règlent pendant
que ça tourne.

Le fit et le scoring s'excluent : les deux occupent le thread principal, seul
endroit où torch est sûr sur macOS.

Un fichier vidéo est lu **à sa vitesse réelle** : la boucle se cale sur l'horloge
de la source, attend si elle est en avance et saute des frames si elle est en
retard. Sans ça le clip défilerait au rythme du traitement — au ralenti à stride
bas, en accéléré à stride haut — et deux réglages de lissage ne seraient plus
comparables. Le prix est visible dans la page : à stride 1 sur un clip 60 fps,
seule une frame sur dix environ est scorée si l'inférence prend 170 ms.

## Les banques : `coresets/*.pkg`

Une banque tient dans un fichier, et son nom dit sa configuration :

```
coresets/WideResNet50_DetectionKnife_l3-l4_p0.005_ts20000_s0.pkg
         backbone     tâche          couches coreset images  seed
```

Une taille d'image non standard s'y glisse aussi, après les couches : `_im128`.
Rien pour 224 px, le défaut — les banques déjà nommées gardent leur nom.

C'est un zip de ce qu'écrit le fit (index faiss, paramètres, `fit_config.json`).
Le fichier de config reste la référence : le nom n'en est qu'un résumé, lisible
sans ouvrir l'archive. À côté du `.pkg`, un dossier de même nom garde les images
qui ont servi — de quoi refitter autrement sans les renvoyer. La page empaquette
elle-même à la fin d'un fit : le `.pkg` apparaît dans `coresets/` et dans le
sélecteur, sans rien à convertir à la main.

**D'où viennent ces images.** Rien n'a été filmé pour ces banques : elles sont
fittées sur **COCO 2017**, le jeu de photos annotées de Microsoft. Le normal,
c'est toute image contenant une personne et **aucun** couteau ; les images de
personne **avec** couteau ne rentrent pas dans la banque, elles ne servent qu'à
mesurer ce qui la sépare du reste. 20 000 images des splits `train2017` et
`val2017`, fittées sur un nœud Grid5000 — d'où le `ts20000` du nom.

Les `.pkg` sont gitignorés : un fichier pèse de 80 Mo à 3 Go, quand GitHub refuse
au push tout fichier au-delà de 100 Mio — et un blob de cette taille resterait
dans l'historique de chaque clone même après suppression. Les banques se
distribuent donc en **asset de release** : un fichier attaché à une version
publiée, hébergé à côté du dépôt et non dedans, que `git clone` ne rapatrie pas.
Une petite banque peut toujours être committée pour de bon avec `git add -f`.

### Le zip d'images

Des images à plat = toutes sont sans l'anomalie à détecter. Un zip contenant
`normal/` et `anomaly/` garde la séparation, les contre-exemples ne servant qu'à
calibrer un seuil. 20 % des images normales est réservé hors banque pour ce
calibrage : envoyer 400 images en met 320 dans la banque, et la page affiche les
deux chiffres.

Le coût du fit est dominé par la sélection du coreset, quadratique en nombre de
patchs : quelques centaines d'images passent en secondes sur un portable, 20 000
demandent des heures et un GPU. C'est aussi le plafond : au-delà de 20 000 images
utilisables, la page (dossier) ou le serveur (zip) en tire 20 000 au hasard, un
sous-dossier `anomaly/` restant gardé entier.

Un biais à connaître : le prétraitement est un `Resize` suivi d'un `CenterCrop`.
Ce qui sort du centre du cadre n'est ni appris ni scoré — cadrer la scène en
conséquence. `FIT_SEED` fixe le tirage de bout en bout, et apparaît dans le nom
de la banque : un même réglage rejoué n'écrase pas le précédent.

## Organisation

```
main.py                     # le seul exécutable : sert la page
coresets/<nom>.pkg          # banques empaquetées (gitignoré)
coresets/<nom>/normal/      # les images qui ont servi au fit (gitignoré)

src/live/server.py          # l'app servie par main.py : fit + scoring live
src/live/scoring.py         # une frame : prétraitement, agrégation, faiss, rendu
src/patchcore/              # le cœur : backbone, coreset, banque, scoring
src/experiments/pipelines.py# le fit, du Spec au dossier de banque
src/templates/live.html     # la page servie par src/live/server.py
src/static/live.{css,js}    # sa feuille de style et son script
```

Les captures prises depuis la page se rangent d'après la banque chargée, sous
`results/<tâche>/captures/<couche>/<coreset>/v<vmax>/`.

## Rendu de la heatmap

`overlay_heatmap()` de `src/live/scoring.py` fait tout le rendu, partagé par la
page et par `bin/live_camera.py`. La *vignette*, c'est la frame telle que le
réseau la voit — même `Resize` + `CenterCrop`, sans la normalisation — pour que
la heatmap se superpose dessus sans réalignement. Aucun de ces réglages ne touche
aux scores :

```
normalisé = max(heatmap / vmax − 1, 0)      # nul sous vmax
couleur   = jet(clip(normalisé, 0,1 – 0,9))
a         = clip(normalisé ** α, 0, 1)
sortie    = couleur × a + vignette × (1 − a)
```

| constante | rôle |
| --- | --- |
| `COLORMAP_LOW` / `COLORMAP_HIGH` = 0,1 / 0,9 | écrêtent l'indice dans la rampe jet. Au-delà de 0,9 elle vire au bordeaux, où deux distances très différentes rendent la même couleur ; sous 0,1 elle plonge dans le bleu nuit |
| `HEATMAP_ALPHA` = 0,5 | position d'entrée du curseur d'opacité, l'exposant α ci-dessus, dans [0, 1] : à 1 le fondu est linéaire, à 0 tout est peint. Réglable en direct |
| `SMOOTHING_SECONDS` = 1/3 | durée de vidéo couverte par la case « Lissage », réglable en direct dans la page. Le nombre de cartes en découle, via `calculer_nb_heatmaps(fps, stride, seconds)`, à partir du stride **effectif** — celui que la lecture en temps réel impose, sauts compris |

C'est `vmax` qui décide de ce qui s'affiche : rien n'est peint en dessous, et la
couleur monte avec ce qui dépasse. L'exposant ne règle que la vitesse de cette
montée, l'écrêtage que la **couleur**.

### Pourquoi `sortie = couleur × a + vignette × (1 − a)`

Les deux poids somment à 1 : chaque pixel de sortie est un point du segment entre
la couleur du jet et le pixel filmé. C'est une combinaison convexe, et c'est tout
ce qu'on lui demande.

- **Rien ne déborde.** Une simple somme `couleur + vignette` sortirait de
  [0, 255] et écrêterait vers le blanc — deux zones franchement différentes
  rendraient le même blanc cramé. Une combinaison convexe reste entre ses deux
  bornes, donc dans l'octet, sans clip et sans virage de teinte.
- **Un seul curseur, continu, d'un bout à l'autre.** `a` = 0 rend la vignette au
  pixel près, `a` = 1 la couleur seule, et la transition entre les deux est
  continue : pas de seuil où l'image saute.
- **`a` varie par pixel.** C'est ce qui sépare la formule d'un
  `cv2.addWeighted`, qui applique un `a` global et voile la scène entière, fond
  calme compris. Ici `a` vaut 0 partout où le score ne dépasse pas `vmax` : le
  décor traverse intact, seule la tache se teinte, et on voit du même coup *ce
  qui* est signalé et *où c'est* dans la scène. Un masque binaire — peindre ou ne
  pas peindre — découperait la tache au ciseau et perdrait son intensité ; le
  fondu la fait porter par la couleur.

`a` est de forme (H, W, 1), diffusé sur les trois canaux : le même poids pour B,
G et R. Un `a` par canal doserait les couleurs les unes contre les autres au lieu
de doser le mélange.

### `vmax`, mesuré au fit (branche `stage`)

`vmax` est **le plus grand score du train en distribution** : le fit repasse à
travers la banque les 20 % d'images normales gardées hors banque et retient le
plus grand de leurs scores. C'est le pire nominal observé, donc le niveau
au-dessus duquel ce qui passe devant la caméra ne ressemble plus à rien de connu
de la scène — et c'est exactement là que la couleur commence.

La valeur part dans `fit_config.json` (`vmax_holdout`), donc dans le `.pkg` ; la
page la reprend dans le champ en disant sur combien d'images elle a été mesurée,
et le champ reste réglable en direct.

La case **Self-calibrating VMax**, à côté du champ, commute les deux origines
possibles : cochée, le vmax est cette échelle mesurée ; décochée, c'est la table
par couche de `live.js` (`l2: 20`, `l3: 10`, `l4: 260`, `l3-l4: 150–200`), qui
suit les cases cochées à gauche et ne dépend d'aucun fit. C'est aussi le repli
d'une banque construite avant la calibration, qui n'en porte aucune. La mesure
reste affichée dans l'aide dans les deux cas : elle informe même décochée.

La passe coûte une inférence par image de holdout, plafonnée à 200
(`FIT_CALIB_IMAGES`, 0 pour couper). Réserve à connaître en mode « Filmer
maintenant » : le holdout y est fait de frames voisines de celles de la banque,
l'échelle obtenue est un plancher optimiste. Détail et relevés dans
[docs/vmax.md](docs/vmax.md).

## Cadence live — coût d'une frame

Banque « personne + couteau » (COCO, layer3 + layer4, 20 000 images de fit,
224 px). Budgets : 33,3 ms = 30 FPS, 16,7 ms = 60 FPS. `scoring` exclut
l'encodage JPEG. Mesuré avec le banc `bin/bench_live.py`, qui vit sur `stage`
avec le reste des outils de mesure.

CPU (Apple M-series, torch 4 threads / faiss 1) :

| backbone | coreset | vecteurs en banque | preprocess | embed | faiss | post | encode | scoring | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 | p0.005 | 19 600 | 2,5 ms | 54,5 ms | 16,4 ms | 2,7 ms | 1,0 ms | 76,2 ms | 12,9 | 0,6321 |
| wideresnet50 | p0.01 | 39 200 | 2,3 ms | 54,2 ms | 26,8 ms | 3,9 ms | 1,0 ms | 87,2 ms | 11,3 | 0,6375 |
| wideresnet50 | p0.02 | 78 400 | 2,5 ms | 58,6 ms | 54,2 ms | 0,0 ms | 0,9 ms | 115,3 ms | 8,6 | 0,6395 |
| wideresnet50 | p0.05 | 196 000 | 2,4 ms | 52,7 ms | 133,4 ms | 2,4 ms | 1,0 ms | 190,9 ms | 5,2 | 0,6406 |
| resnet50 | p0.005 | 19 600 | 2,5 ms | 31,3 ms | 16,5 ms | 1,4 ms | 1,0 ms | 51,7 ms | 19,0 | 0,5560 |
| resnet34 | p0.005 | 19 600 | 2,5 ms | 20,1 ms | 16,3 ms | 2,1 ms | 1,0 ms | 40,9 ms | 23,9 | 0,5507 |
| resnet18 | p0.005 | 19 600 | 2,4 ms | 12,9 ms | 13,6 ms | 2,0 ms | 1,0 ms | 30,9 ms | 31,4 | 0,4855 |

GPU (NVIDIA L40S, `INFER_FAISS_GPU=1`) :

| backbone | coreset | vecteurs en banque | preprocess | embed | faiss | post | encode | scoring | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 | p0.01 | 39 200 | 3,9 ms | 8,2 ms | 0,9 ms | 1,4 ms | 0,7 ms | 14,3 ms | 66,6 | 0,6375 |
| wideresnet50 | p0.02 | 78 400 | 3,9 ms | 8,2 ms | 1,8 ms | 1,4 ms | 0,7 ms | 15,2 ms | 62,7 | 0,6395 |
| wideresnet50 | p0.05 | 196 000 | 4,0 ms | 8,2 ms | 4,1 ms | 1,4 ms | 0,7 ms | 17,7 ms | 54,5 | 0,6406 |

- **Le coreset ne débloque plus rien.** Diviser la banque par cinq
  (p0.05 → p0.01) rend 6,1 FPS, la diviser encore par deux n'en rend plus que
  1,6 : le temps faiss suit strictement le nombre de vecteurs, mais le backbone,
  lui, ne bouge pas — ses 54 ms font les trois quarts du budget, et même une
  banque vide plafonnerait vers 17 FPS à 224 px sur wideresnet50.
- **Changer de backbone achète de la cadence et coûte de l'AUROC.** À coreset
  identique, wideresnet50 rend 12,9 FPS pour 0,6321 quand resnet50 en rend 19,0
  pour 0,5560 — la moitié de cadence en plus, un dixième d'AUROC en moins ;
  resnet34 tient encore le niveau de resnet50 (0,5507 pour 23,9 FPS).
- **resnet18 est le plancher.** 0,4855, soit sous 0,5 : les FPS qu'il achète ne
  valent plus rien, et les 0,6406 de wideresnet50 restent hors de portée des
  trois autres.

Le relevé par étape des quatre banques livrées — backbone contre pooling, part de
chaque étape, coût de chargement d'un `.pkg` — est dans
[docs/cadence.md](docs/cadence.md).

Device : `PATCHCORE_DEVICE` = `auto` (cuda sinon cpu) | `cpu` | `cuda[:N]` | `mps`.
MPS est exclu de l'automatique — PatchCore y échoue sur le pooling adaptatif, et
s'y révèle plus lent que le CPU.

## Déploiement sur une scène réelle

Une banque construite sur un dataset public score surtout la nouveauté de scène,
et non l'anomalie cherchée. Sur un robot qui filme toujours le même
environnement, fitter sur *cette* scène : filmer le décor sans l'anomalie à
détecter, sous toutes ses variations, puis envoyer les images dans la moitié
gauche de la page — un dossier `normal/`, et un `anomaly/` si l'on veut de quoi
calibrer un seuil. Tout ce qui n'est pas dans la banque sera scoré comme anormal,
le décor compris s'il a changé.

Quelques centaines d'images espacées valent mieux que des milliers de frames
consécutives : à 30 images par seconde, deux voisines n'apprennent rien de neuf
à la banque.

## Citing

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

Contact amont pour PatchCore lui-même : karsten.rh1@gmail.com.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
