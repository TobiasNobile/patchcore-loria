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

## La case « Self-calibrating VMax »

Le commutateur des deux origines possibles du vmax, à côté du champ :

| case | vmax | ce que ça vaut |
| --- | --- | --- |
| décochée (défaut) | la table par couche de `live.js`, suivant les cases cochées à gauche | ne dépend d'aucun fit, mais ne sait rien de la scène ni du backbone |
| cochée | l'échelle mesurée au fit sur les images tirées hors banque | connaît la scène, et n'existe que si la banque porte la mesure |

Elle agit en marche comme à l'arrêt : la basculer réécrit le champ et prévient le
serveur. Le coefficient ne s'applique qu'au premier cas — il multiplie une
mesure, pas une valeur de table — et la mesure reste affichée dans l'aide même
décochée, parce qu'elle informe même quand elle n'est pas appliquée.

Il n'y a **pas** de phase de test à jouer devant la caméra : les images qui
mesurent l'échelle sont tirées au hasard du même filmage que la banque, en une
seule prise. C'est tout l'intérêt — rien à présenter, rien à arrêter.

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

Le coefficient place le **haut** de la rampe ; le curseur sous la caméra en coupe
le **bas**. Ce curseur est un seuil, en fraction de vmax : sous `seuil × vmax`,
rien n'est dessiné, l'image reste nue — une découpe, pas un fondu. Le score en
dessous duquel plus rien n'apparaît est donc `seuil × coefficient × mesure`, et
c'est bien un produit des deux, chacun réglant un bout de la rampe.

C'était auparavant un exposant d'opacité, `(score / vmax)^α`, qui ne rendait
jamais rien tout à fait invisible : le fond nominal gardait un voile, et il
fallait pousser α à son maximum pour l'effacer — ce qui écrasait du même coup
tout ce qui n'était pas au pic. Le seuil sépare les deux questions : jusqu'où va
la couleur, et à partir d'où on montre quelque chose.

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
