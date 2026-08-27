# Échelle du score : note de cadrage

Note complète, mise en page : <https://claude.ai/code/artifact/eb9fddcd-f474-41ef-a769-23ab64fc07fa>

Ce fichier garde les chiffres, pour qu'ils ne dépendent pas d'un lien externe.

## Le problème

Le score de PatchCore est une distance à la banque : sans unité. Son ordre de
grandeur dépend de la couche, du backbone et de la scène. La page embarquait une
table de valeurs devinées (`l2: 20`, `l3: 10`, `l4: 260`) — facteur 26 entre
deux couches voisines — et il fallait remonter `vmax` à la main à chaque
changement de scène.

Que la table ne suffise pas se voit sur un cas : une banque `resnet18` en
`l2-l3` sur 28 images de scène score ses images hors banque à **73**, là où la
table annonce 10 pour `l2-l3`. Elle avait été relevée sur `wideresnet50` — le
backbone change l'échelle autant que la couche.

## Ce qui est implémenté (branche `stage` uniquement)

**`vmax` = le plus grand score des images normales gardées hors banque.**

`folder.py` écarte déjà 20 % des images `normal/` du fit, pour qu'une image ne
serve pas à la fois à construire la banque et à la juger — une image de la
banque est son propre plus proche voisin et score presque zéro. `holdout_vmax()`
(`src/experiments/pipelines.py`) les repasse à travers la banque juste après le
fit et garde le maximum de leurs scores. Il part dans `fit_config.json` sous
`vmax_holdout`, donc dans le `.pkg` :

```json
"vmax_holdout": {
  "vmax": 73.06, "n_images": 7, "score_median": 59.51, "heatmap_max": 70.55
}
```

- Le **maximum** plutôt qu'un quantile : sur une vingtaine d'images, un quantile
  haut n'a aucun point pour lui. C'est le pire nominal observé, donc le niveau
  au-dessus duquel ce qui passe ne ressemble plus à rien de connu de la scène.
- Le **score d'image**, qui est déjà le max de ses patchs, et non le pic de la
  heatmap — flouté et rééchantillonné, donc plus bas (70,55 contre 73,06 sur le
  relevé ci-dessus). Le nombre affiché sous la caméra et la borne de couleur
  sont alors la même grandeur. Le pic de carte est gardé pour situer l'écart.
- **Aucune marge par défaut** : la saturation commence pile au-dessus du pire
  normal. Le coefficient plus bas la déplace, sans toucher à la mesure.
- Plafonné à 200 images (`FIT_CALIB_IMAGES`, 0 pour couper) et enveloppé dans un
  `try` : un dataset sans nominal hors banque ne fait pas échouer le fit.

Côté page : le champ `vmax` est pré-rempli avec la valeur de la banque
sélectionnée, l'aide dit sur combien d'images elle a été mesurée, une puce du
bandeau la porte, et le champ reste réglable en direct. Une banque d'avant
retombe sur la table par couche, **en le disant** (« banque non calibrée »).
En mode « Filmer maintenant », la banque n'entre pas dans le sélecteur et le
scoring enchaîne : c'est l'état du fit qui livre l'échelle, que le serveur
applique et que la page écrit dans le champ. `bin/live_camera.py` prend la même
valeur par défaut, `--vmax` la remplace.

## Le mode « Self-calibrating VMax »

Une case du mode « Filmer maintenant », qui intercale une phase entre la banque
et la démo :

```
filmer le normal (20 s) → banque → filmer l'ANOMALIE → [Terminer le test] → démo
                                   ↑ scores collectés, p90 gardé
```

VMax devient alors le **p90 des scores du test** (`CALIB_PERCENTILE`). Pas le
maximum : une seule frame prise au bon angle placerait la saturation là où
l'anomalie n'est presque jamais, et elle resterait orange tout le reste du temps.
Le p90 laisse franchement saturer le dixième le plus marqué.

La phase se clôt au bouton et non au chronomètre — personne ne sait d'avance
combien de temps il faut pour présenter une anomalie sous un angle qui la montre.
Pendant le test, le bouton porte les deux nombres (« garder 121,5 · p90 de 212
scores · pic 137,2 ») et la ligne d'état rappelle le plafond du normal mesuré au
fit. Trois lectures immédiates :

- pic **et** p90 loin au-dessus du max normal : l'anomalie a été montrée assez
  longtemps, l'échelle est bonne ;
- pic haut mais p90 proche du max normal : elle n'a été vue qu'un instant, le
  p90 porte sur *tout* le test et retombe dans le normal — refaire en la
  montrant plus longtemps, ou garder l'échelle du fit ;
- pic lui-même sous le max normal : rien d'anormal n'a été filmé.

Les deux échelles ne disent pas la même chose, et c'est pourquoi les deux
existent :

| | mesure | ce qu'on lit ensuite |
| --- | --- | --- |
| holdout (défaut) | max des scores du **normal** hors banque | tout ce qui dépasse le pire nominal sature — sensible, sans rien avoir à montrer |
| test (la case) | p90 des scores de l'**anomalie jouée** | la scène occupe la rampe et l'anomalie sature vraiment — mais le p90 dépend de la part du test où elle était visible |

Côté implémentation, la phase de test **est** la boucle de scoring, avec un
drapeau `calibrer` : même cadence, même lissage, même overlay. Deux sorties pour
une boucle — `stop` abandonne tout sans rien garder (`Test interrompu.`),
`end_test` la clôt en gardant sa mesure. Ce qui a été réglé à chaud pendant le
test (zoom, alpha, stride, lissage) suit dans la démo : sans ça, l'image
changerait entre la calibration et ce qu'elle est censée calibrer.

## Le coefficient

Les deux échelles sont des mesures ; ce qu'on en fait reste un réglage
d'affichage. D'où un coefficient à côté du champ, `vmax = coefficient × mesure`,
1 par défaut. Sous 1, le pic mesuré passe au-dessus de la borne et sature
franchement ; à 1, il arrive pile dessus. Il ne multiplie **qu'une mesure** : un
vmax tapé à la main ou repris de la table lui échappe, et le calcul écrit sous le
champ disparaît alors plutôt que d'afficher une égalité fausse.

```
0,80 × 73,1 (mesure hors banque) = 58,4
```

Un coefficient à part, et non le curseur alpha, malgré la tentation : alpha est
un **exposant** d'opacité, `opacité = (score / vmax)^α × 0,9`. Les deux réglages
sont couplés — à α = 2, un pixel atteint la demi-opacité à 0,71 × vmax, à α = 4 à
0,84 × vmax — mais par une puissance, pas par un produit. Confier la marge au
curseur ferait bouger d'un seul geste ce qu'on voit *et* le niveau où ça sature,
sans plus pouvoir régler l'un sans l'autre.

Le serveur applique la même marge de son côté dans les deux cas où la mesure
arrive après le démarrage — fit online, fin de test — et garde la mesure brute
dans son état : c'est elle que la page multiplie pour afficher le calcul, donc
les deux parlent toujours du même nombre.

## Ce que ça ne dit pas

Sur une banque filmée d'un coup, le holdout est fait des **frames voisines** de
celles de la banque — des quasi-doublons. L'échelle obtenue est alors un
plancher optimiste, pas une borne du normal en général : dès que la scène varie
plus qu'à la prise, le nominal dépasse ce `vmax` et la carte rougit. La page le
dit sous les réglages du mode online. Pour un zip d'images vraiment distinctes,
l'objection tombe.

## Ce qui a été essayé, puis retiré (août 2026)

Une première version (`abff2aa`, `65fe1d0`) mesurait médiane, MAD et quantiles
du holdout, exprimait le score en écarts robustes `z = (s − médiane) / σ` et
proposait trois références : distance brute, échelle du fit, fenêtre glissante
de 20 s. Retirée en `9553aec`.

Trois relevés en sont restés. Distribution des scores nominaux (banque jouet,
5 images hors banque, 250 880 patchs) :

| quantile | mesuré | attendu si gaussien |
| --- | --- | --- |
| q90 | 2,4 σ | 1,3 σ |
| q99 | 8,7 σ | 2,3 σ |
| q999 | 11,4 σ | 3,1 σ |

Vidéo hors domaine :

| référence | score global | carte saturée |
| --- | --- | --- |
| enrôlement | 27 σ | 43 – 52 % |
| fenêtre 20 s | 4,2 σ | 0 % |

Contraste intra-image (pic au-dessus du fond de la même frame) : 3,9 σ avec
l'échelle du fit, 3,4 σ en fenêtre glissante. La fenêtre déplace le niveau de
référence, elle n'écrase pas le pic.

Ce qui l'a fait retirer, et qui vaut toujours : la queue de distribution est très
lourde, donc « k sigmas » n'est pas transférable — le quantile visé l'est un peu
plus, mais il s'estime mal sur vingt images non indépendantes. Et une échelle
réestimée sur les dernières secondes suppose ces secondes normales : un objet
présent en continu perdait son pic en douze secondes.

La version actuelle ne rouvre ni l'un ni l'autre : pas de quantile, pas de
réestimation en marche. Un seul nombre, mesuré une fois, affiché et modifiable.

## Ce que ça ne prouve pas

Banque de 25 images pour les relevés ci-dessus, une seule vidéo, aucune dérive
contrôlée, aucune captation aérienne. Le protocole qui en ferait un résultat est
décrit dans la note.
