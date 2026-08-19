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
d'images, puis scorer la webcam), le format de banque `coresets/*.pkg`, les
pipelines de fit et d'inférence partagés dans `src/experiments/`, le fit sur une
scène filmée sur place (`bin/capture.py`, `bin/scene/`), le scoring webcam en
direct (`bin/live_camera.py`, `bin/live_web.py`), le banc de mesure d'une frame
(`bin/bench_live.py`).

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
bash bin/fetch_bank.sh    # la banque de démonstration, 153 Mo, hors du dépôt
python main.py            # puis ouvrir http://127.0.0.1:8000
```

Le premier script est facultatif : sans lui la page s'ouvre quand même, mais il
faut fitter une banque avant de pouvoir scorer quoi que ce soit. Avec, la banque
« personne + couteau » est présélectionnée à l'ouverture.

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
coresets/WideResNet50_DetectionKnife_l3-l4_p0.01_ts20000.pkg
         backbone     tâche          couches coreset images de fit
```

C'est un zip de ce qu'écrit le fit (index faiss, paramètres, `fit_config.json`).
Le fichier de config reste la référence : le nom n'en est qu'un résumé, lisible
sans ouvrir l'archive. À côté du `.pkg`, un dossier de même nom garde les images
qui ont servi — de quoi refitter autrement sans les renvoyer.

Les deux sont gitignorés. Un `.pkg` pèse de 80 Mo à 3 Go, quand GitHub refuse
au push tout fichier au-delà de 100 Mio — et un blob de cette taille resterait
dans l'historique de chaque clone même après suppression. Les banques se
distribuent donc en **asset de release** : un fichier attaché à une version
publiée, hébergé à côté du dépôt et non dedans, que `git clone` ne rapatrie pas.
C'est ce que va chercher `bin/fetch_bank.sh`, somme de contrôle à l'appui. Une
petite banque peut toujours être committée pour de bon avec `git add -f`.

Une banque construite en ligne de commande s'y convertit sans refit :

```shell
python bin/pack_bank.py models/scene/wideresnet50_l3-l4_..._tsall_s0 --task PointStage
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
  fetch_bank.sh             # installe la banque de démonstration (release)
  capture.py                # filme la scène de déploiement -> data/scene/
  scene/   fit/memory_bank.py  infer/histogram.py
  live_camera.py            # fenêtre OpenCV      | agnostiques : le dataset
  live_web.py               # l'app servie par main.py
  bench_live.py             # coût d'une frame, étape par étape
  pack_bank.py              # models/<tag>/ -> coresets/<nom>.pkg

src/experiments/            # métriques et pipelines partagés par les datasets
src/templates/live.html     # la page servie par live_web.py
src/static/live.{css,js}    # sa feuille de style et son script
models/<dataset>/<tag>/     # banques mémoire des scripts de fit (gitignoré)
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

Chaque figure est doublée d'un sidecar JSON portant la configuration et les
métriques : c'est lui la source de vérité, la figure n'en est que la lecture.
Les balayages de seeds et leur agrégation vivent dans la version complète.

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
| `SMOOTHING_SECONDS` = 1/3 | durée de vidéo couverte par la case « Lissage », réglable en direct dans la page. Le nombre de cartes en découle, via `calculer_nb_heatmaps(fps, stride, seconds)` de `bin/live_camera.py`, à partir du stride **effectif** — celui que la lecture en temps réel impose, sauts compris |

L'écrêtage porte sur la **couleur seule** ; l'opacité suit la valeur brute, ce
qui laisse le fond normal parfaitement intact — un score nul rend l'image nue,
pas un voile bleu.

## Cadence live — coût d'une frame (`bin/bench_live.py`)

Banque « personne + couteau » (COCO, layer3 + layer4, 20 000 images de fit —
construite dans la version complète). Budgets : 33,3 ms = 30 FPS, 16,7 ms = 60 FPS. `scoring`
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

Une banque construite sur un dataset public score surtout la nouveauté de scène,
et non l'anomalie cherchée. Sur un robot qui filme toujours le même
environnement, fitter sur *cette* scène :

```shell
python bin/capture.py --out data/scene/normal  --count 400 --every 0.5
python bin/capture.py --out data/scene/anomaly --count 60     # pour le seuil
SCENE_PATH=data/scene python bin/scene/fit/memory_bank.py
SCENE_PATH=data/scene python bin/scene/infer/histogram.py     # seuil à lire entre les modes
python bin/live_web.py                                        # puis choisir la banque
```

Filmer la scène sans l'anomalie à détecter, sous toutes ses variations : tout ce
qui n'est pas dans la banque sera scoré comme anormal.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
