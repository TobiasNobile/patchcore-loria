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
- **Aucune marge** : la couleur commence pile au-dessus du pire normal, et c'est
  la mesure elle-même qui décide de ce qui s'affiche.
- Plafonné à 200 images (`FIT_CALIB_IMAGES`, 0 pour couper) et enveloppé dans un
  `try` : un dataset sans nominal hors banque ne fait pas échouer le fit.

Côté page : le champ `vmax` est pré-rempli avec la valeur de la banque
sélectionnée, l'aide dit sur combien d'images elle a été mesurée, une puce du
bandeau la porte, et le champ reste réglable en direct. En mode « Filmer
maintenant », la banque n'entre pas dans le sélecteur et le scoring enchaîne :
c'est l'état du fit qui livre l'échelle, que le serveur applique et que la page
écrit dans le champ. `bin/live_camera.py` prend la même valeur par défaut,
`--vmax` la remplace.

## La case « Self-calibrating VMax »

Le commutateur des deux origines possibles du vmax, à côté du champ :

| case | vmax | ce que ça vaut |
| --- | --- | --- |
| décochée (défaut) | la table par couche de `live.js`, suivant les cases cochées à gauche | ne dépend d'aucun fit, mais ne sait rien de la scène ni du backbone |
| cochée | l'échelle mesurée au fit sur les images tirées hors banque | connaît la scène, et n'existe que si la banque porte la mesure |

Elle agit en marche comme à l'arrêt : la basculer réécrit le champ et prévient le
serveur. La mesure reste affichée dans l'aide même décochée, parce qu'elle
informe même quand elle n'est pas appliquée. Une banque d'avant la calibration
n'en porte aucune : c'est la table qui sert, et l'aide affiche alors son repère
(« 150–200 pour l3-l4 ») au lieu d'une mesure.

Il n'y a **pas** de phase de test à jouer devant la caméra : les images qui
mesurent l'échelle sont tirées au hasard du même filmage que la banque, en une
seule prise. C'est tout l'intérêt — rien à présenter, rien à arrêter.

## Ce que l'affichage en fait

Le vmax est le pied de la couleur, pas son sommet :

```
normalisé = max(heatmap / vmax − 1, 0)
alpha     = clip(normalisé ** α, 0, 1)
```

Sous vmax, `normalisé` est nul et rien n'est peint — l'image reste nue, sans le
voile que laissait l'ancien `(score / vmax)^α` mesuré depuis zéro. Au-dessus, la
couleur monte avec l'écart relatif, et le curseur sous la caméra ne règle que
l'exposant α, dans [0, 1] : à 1 le fondu est linéaire, plus bas il monte vite,
à 0 tout est peint. Une découpe à seuil a été essayée entre les deux, puis
retirée : elle re-tranchait ce que le vmax tranche déjà.

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
