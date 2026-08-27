# Cadence live — le détail par étape

Le [README](../README.md) garde les deux tables qui répondent aux deux
questions — quel backbone, quel coreset. Ce fichier garde le relevé fin, dont
les conclusions sont déjà résumées là-bas.

Même banc (`bin/bench_live.py`), découpé plus fin et restreint aux banques que le
dépôt livre : 25 mesures par étape après 5 rodages, un processus par banque,
frame source 640×480, CPU (torch 4 threads / faiss 1). `backbone` est la passe du
réseau seule, `pooling` ce que `_embed` ajoute derrière — patchify,
interpolation, agrégation — et `post` le déballage des patchs et la
segmentation. Les écarts avec la table de balayage du README — au plus 13 %, sur
resnet18 — tiennent à deux campagnes distinctes sur une machine partagée ; les
médianes ci-dessous sont les plus récentes.

| banque | preprocess | backbone | pooling | embed | faiss | post | scoring | encode | total | FPS | AUROC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 p0.01 | 2,5 ms | 46,9 ms | 7,5 ms | 54,4 ms | 33,1 ms | 2,1 ms | 92,1 ms | 1,0 ms | 93,1 ms | 10,7 | 0,6375 |
| wideresnet50 p0.005 | 2,5 ms | 47,4 ms | 5,9 ms | 53,3 ms | 16,4 ms | 4,5 ms | 76,6 ms | 1,0 ms | 77,6 ms | 12,9 | 0,6321 |
| resnet50 p0.005 | 2,5 ms | 20,7 ms | 8,4 ms | 29,1 ms | 16,4 ms | 2,3 ms | 50,3 ms | 1,0 ms | 51,2 ms | 19,5 | 0,5560 |
| resnet34 p0.005 | 2,5 ms | 17,4 ms | 2,6 ms | 20,1 ms | 16,3 ms | 2,1 ms | 40,9 ms | 1,0 ms | 41,9 ms | 23,9 | 0,5507 |
| resnet18 p0.005 | 2,5 ms | 10,3 ms | 2,4 ms | 12,7 ms | 16,6 ms | 3,8 ms | 35,6 ms | 1,0 ms | 36,5 ms | 27,4 | 0,4855 |

Le backbone domine partout, sauf sur resnet18 où faiss passe devant :

| banque | preprocess | backbone | pooling | faiss | post | encode |
| --- | --- | --- | --- | --- | --- | --- |
| wideresnet50 p0.01 | 2,7 % | 50,4 % | 8,1 % | 35,5 % | 2,2 % | 1,1 % |
| wideresnet50 p0.005 | 3,2 % | 61,1 % | 7,7 % | 21,1 % | 5,8 % | 1,3 % |
| resnet50 p0.005 | 4,9 % | 40,3 % | 16,4 % | 32,0 % | 4,5 % | 1,9 % |
| resnet34 p0.005 | 5,9 % | 41,6 % | 6,3 % | 38,8 % | 5,0 % | 2,3 % |
| resnet18 p0.005 | 6,8 % | 28,1 % | 6,7 % | 45,4 % | 10,3 % | 2,7 % |

Et faiss suit strictement la taille de la banque — 16,4 ms à 19 600 vecteurs,
33,1 à 39 200, soit ×2,02 pour ×2 — sans coût fixe mesurable. Alléger la banque
ne rend donc quelque chose qu'une fois le backbone déjà léger : entre les deux
WideResNet50, diviser la banque par deux rend 2,2 FPS, quand passer à resnet34
en rend 11,0.

Du `.pkg` au PatchCore prêt à scorer :

| banque | extraction (cache froid) | extraction (cache chaud) | `load_bank` | RSS |
| --- | --- | --- | --- | --- |
| wideresnet50 p0.01 | 365,2 ms | 0,04 ms | 976,9 ms | 0,92 Go |
| wideresnet50 p0.005 | 109,5 ms | 0,03 ms | 955,4 ms | 0,84 Go |
| resnet50 p0.005 | 171,1 ms | 0,04 ms | 437,3 ms | 0,66 Go |
| resnet34 p0.005 | 210,0 ms | 0,04 ms | 410,8 ms | 0,61 Go |
| resnet18 p0.005 | 130,2 ms | 0,04 ms | 282,4 ms | 0,54 Go |

L'extraction est mise en cache dans le dossier temporaire du système et n'est
repayée qu'au premier démarrage suivant l'écriture du `.pkg`. Les écarts-types
restent sous 0,3 ms sur faiss et sous 2,7 ms sur le backbone ; la seule étape
franchement dispersée est le `predict` de resnet18 (écart-type 4,2 ms pour une
médiane de 33,1).
