# Échelle du score : note de cadrage

Note complète, mise en page : <https://claude.ai/code/artifact/eb9fddcd-f474-41ef-a769-23ab64fc07fa>

Ce fichier garde les chiffres, pour qu'ils ne dépendent pas d'un lien externe.

## Le problème

Le score de PatchCore est une distance à la banque : sans unité. Son ordre de
grandeur dépend de la couche, du backbone et de la scène. La page embarque une
table de valeurs devinées (`l2: 20`, `l3: 10`, `l4: 260`) — facteur 26 entre
deux couches voisines — et il fallait remonter `vmax` à la main à chaque
changement de scène.

## Ce qui est implémenté (branche `stage` uniquement)

- `run_fit` score le holdout — les 20 % d'images normales écartées de la banque
  — et écrit médiane, MAD et quantiles dans `fit_config.json`.
- Le scoring live lit le score en écarts robustes, `z = (s − médiane) / σ`, avec
  trois références sélectionnables : distance brute, échelle du fit, ou fenêtre
  glissante de 20 s.
- Un journal JSONL par session sous `results/live/` : statistiques de la frame,
  échelle en vigueur, score, fraction saturée.

## Les trois relevés

Distribution des scores nominaux (banque jouet, 5 images hors banque,
250 880 patchs) :

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

## Deux défauts connus

- Le score global est normalisé avec les statistiques **patch**, alors qu'il est
  le maximum sur les patchs. Une image nominale se lit 8,8 σ au lieu de 0. Les
  statistiques image sont déjà mesurées, elles sont inutilisées.
- Une banque sans `nominal_scores` fait retomber le mode « échelle du fit » sur
  la distance brute, sans le dire. La banque couteau de la release est dans ce
  cas.

## Ce que ça ne prouve pas

Banque de 25 images, une seule vidéo, aucune dérive contrôlée, aucune captation
aérienne. Le protocole qui en ferait un résultat est décrit dans la note.
