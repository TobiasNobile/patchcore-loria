# Towards Total Recall in Industrial Anomaly Detection

This repository contains the implementation for `PatchCore` as proposed in Roth et al. (2021), <https://arxiv.org/abs/2106.08265>.

It also provides various pretrained models that can achieve up to _99.6%_ image-level anomaly
detection AUROC, _98.4%_ pixel-level anomaly localization AUROC and _>95%_ PRO score (although the
later metric is not included for license reasons).

![defect_segmentation](images/patchcore_defect_segmentation.png)

_For questions & feedback, please reach out to karsten.rh1@gmail.com!_

---

## Quick Guide

First, clone this repository and set the `PYTHONPATH` environment variable with `env PYTHONPATH=src python bin/mvtec/run_patchcore.py`.
To train PatchCore on MVTec AD (as described below), run

```
datapath=/path_to_mvtec_folder/mvtec datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))


python bin/mvtec/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model \
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_online --log_project MVTecAD_Results results \
patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 --anomaly_scorer_num_nn 1 --patchsize 3 \
sampler -p 0.1 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath
```

which runs PatchCore on MVTec images of sizes 224x224 using a WideResNet50-backbone pretrained on
ImageNet. For other sample runs with different backbones, larger images or ensembles, see
`sample_training.sh`.

Given a pretrained PatchCore model (or models for all MVTec AD subdatasets), these can be evaluated using

```shell
datapath=/path_to_mvtec_folder/mvtec
loadpath=/path_to_pretrained_patchcores_models
modelfolder=IM224_WR50_L2-3_P001_D1024-1024_PS-3_AN-1_S0
savefolder=evaluated_results'/'$modelfolder

datasets=('bottle'  'cable'  'capsule'  'carpet'  'grid'  'hazelnut' 'leather'  'metal_nut'  'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))
model_flags=($(for dataset in "${datasets[@]}"; do echo '-p '$loadpath'/'$modelfolder'/models/mvtec_'$dataset; done))

python bin/mvtec/load_and_evaluate.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
```

A set of pretrained PatchCores are hosted here: __add link__. To use them (and replicate training),
check out `sample_evaluation.sh` and `sample_training.sh`.

---

## In-Depth Description

### Requirements

Our results were computed using Python 3.8, with packages and respective version noted in
`requirements.txt`. In general, the majority of experiments should not exceed 11GB of GPU memory;
however using significantly large input images will incur higher memory cost.

### Setting up MVTec AD

To set up the main MVTec AD benchmark, download it from here: <https://www.mvtec.com/company/research/datasets/mvtec-ad>.
Place it in some location `datapath`. Make sure that it follows the following data tree:

```shell
mvtec
|-- bottle
|-----|----- ground_truth
|-----|----- test
|-----|--------|------ good
|-----|--------|------ broken_large
|-----|--------|------ ...
|-----|----- train
|-----|--------|------ good
|-- cable
|-- ...
```

containing in total 15 subdatasets: `bottle`, `cable`, `capsule`, `carpet`, `grid`, `hazelnut`,
`leather`, `metal_nut`, `pill`, `screw`, `tile`, `toothbrush`, `transistor`, `wood`, `zipper`.

### "Training" PatchCore

PatchCore extracts a (coreset-subsampled) memory of pretrained, locally aggregated training patch features:

![patchcore_architecture](images/architecture.png)

To do so, we have provided `bin/mvtec/run_patchcore.py`, which uses `click` to manage and aggregate input
arguments. This looks something like

```shell
python bin/mvtec/run_patchcore.py \
--gpu <gpu_id> --seed <seed> # Set GPU-id & reproducibility seed.
--save_patchcore_model # If set, saves the patchcore model(s).
--log_online # If set, logs results to a Weights & Biases account.
--log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 --log_project MVTecAD_Results results # Logging details: Name of the run & Name of the overall project folder.

patch_core  # We now pass all PatchCore-related parameters.
-b wideresnet50  # Which backbone to use.
-le layer2 -le layer3 # Which layers to extract features from.
--faiss_on_gpu # If similarity-searches should be performed on GPU.
--pretrain_embed_dimension 1024  --target_embed_dimension 1024 # Dimensionality of features extracted from backbone layer(s) and final aggregated PatchCore Dimensionality
--anomaly_scorer_num_nn 1 --patchsize 3 # Num. nearest neighbours to use for anomaly detection & neighbourhoodsize for local aggregation.

sampler # We now pass all the (Coreset-)subsampling parameters.
-p 0.1 approx_greedy_coreset # Subsampling percentage & exact subsampling method.

dataset # We now pass all the Dataset-relevant parameters.
--resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath # Initial resizing shape and final imagesize (centercropped) as well as the MVTec subdatasets to use.
```

Note that `sample_runs.sh` contains exemplary training runs to achieve strong AD performance. Due to
repository changes (& hardware differences), results may deviate slightly from those reported in the
paper, but should generally be very close or even better. As mentioned previously, for re-use and
replicability we have also provided several pretrained PatchCore models hosted at __add link__ -
download the folder, extract, and pass the model of your choice to
`bin/mvtec/load_and_evaluate.py` which showcases an exemplary evaluation process.

During (after) training, the following information will be stored:

```shell
|PatchCore model (if --save_patchcore_model is set)
|-- models
|-----|----- mvtec_bottle
|-----|-----------|------- nnscorer_search_index.faiss
|-----|-----------|------- patchcore_params.pkl
|-----|----- mvtec_cable
|-----|----- ...
|-- results.csv # Contains performance for each subdataset.

|Sample_segmentations (if --save_segmentation_images is set)
```

In addition to the main training process, we have also included Weights-&-Biases logging, which
allows you to log all training & test performances online to Weights-and-Biases servers
(<https://wandb.ai>). To use that, include the `--log_online` flag and provide your W&B key in
`run_patchcore.py > --log_wandb_key`.

Finally, due to the effectiveness and efficiency of PatchCore, we also incorporate the option to use
an ensemble of backbone networks and network featuremaps. For this, provide the list of backbones to
use (as listed in `/src/anomaly_detection/backbones.py`) with `-b <backbone` and, given their
ordering, denote the layers to extract with `-le idx.<layer_name>`. An example with three different
backbones would look something like

```shell
python bin/mvtec/run_patchcore.py --gpu <gpu_id> --seed <seed> --save_patchcore_model --log_group <log_name> --log_online --log_project <log_project> results \

patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu \

--pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath

```

When using `--save_patchcore_model`, in the case of ensembles, a respective ensemble of PatchCore parameters is stored.

### Evaluating a pretrained PatchCore model

To evaluate a/our pretrained PatchCore model(s), run

```shell
python bin/mvtec/load_and_evaluate.py --gpu <gpu_id> --seed <seed> $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
```

assuming your pretrained model locations to be contained in `model_flags`; one for each subdataset
in `dataset_flags`. Results will then be stored in `savefolder`. Example model & dataset flags:

```shell
model_flags=('-p', 'path_to_mvtec_bottle_patchcore_model', '-p', 'path_to_mvtec_cable_patchcore_model', ...)
dataset_flags=('-d', 'bottle', '-d', 'cable', ...)
```

### Expected performance of pretrained models

While there may be minor changes in performance due to software & hardware differences, the provided
pretrained models should achieve the performances provided in their respective `results.csv`-files.
The mean performance (particularly for the baseline WR50 as well as the larger Ensemble model)
should look something like:

| Model | Mean AUROC | Mean Seg. AUROC | Mean PRO
|---|---|---|---|
| WR50-baseline | 99.2% | 98.1% | 94.4%
| Ensemble | __99.6%__ | __98.2%__ | __94.9%__

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

## Organisation des scripts et des sorties

Deux axes, dans cet ordre : le dataset, puis la phase. Le fit est la moitié
coûteuse et hors-ligne de PatchCore (extraction des features + coreset) et
n'écrit que des banques mémoire ; l'inférence recharge une banque et n'écrit que
des figures et des mesures. Les séparer évite qu'un script de scoring déclenche
un fit par inadvertance, et rend visible ce qui est cher à reproduire.

```
bin/
  celeba/
    fit/memory_bank.py      # écrit models/celeba/<tag>/
    infer/heatmap.py        # -> results/celeba/heatmaps/
    infer/histogram.py      # -> results/celeba/histograms/
    infer/benchmark.py      # -> results/celeba/benchmarks/
  atr/
    fit/                    # à écrire (cf. src/patchcore/datasets/atr.py)
  mvtec/
    run_patchcore.py        # amont : fit + éval en un seul script
    load_and_evaluate.py
  live_camera.py            # agnostique : le dataset vient de --bank_dir

models/<dataset>/<tag>/     # banques mémoire (gitignoré pour celeba)
results/<dataset>/<sortie>/ # figures et mesures (gitignoré en entier)
results/_archive/           # sorties d'anciens scripts, conservées telles quelles
```

`live_camera.py` est le seul script agnostique du dataset : il vaut ce que vaut
la banque qu'on lui passe, et déduit de `--bank_dir` où écrire ses instantanés
(`models/celeba/…` -> `results/celeba/live/`).

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

Reproduction :

```shell
./sweep_histograms.sh                        # en local
./grid5000_run.sh sweep_histograms.sh        # sur Grid'5000
```

Les variables d'environnement disponibles (tailles, pourcentages, nombre de
voisins, emplacement des banques, reprise) sont documentées en tête de
`sweep_histograms.sh`.

## COCO + Open Images V7 — dataset fusionné « personne + couteau »

Même protocole one-class que CelebA, mais l'anomalie est un couteau tenu par une
personne : la banque est construite sur des images de personnes SANS couteau, et
l'on compare les scores personne-sans-couteau vs personne-avec-couteau sur un test
équilibré. Deux jeux fusionnés, aux schémas d'annotation différents :

- **COCO** : annoté exhaustivement par image → on lit les bounding boxes `person`
  et `knife` (anomalie = image contenant une personne ET une box couteau).
- **Open Images V7** : les boxes d'OID sont annotées par classe et
  non-exhaustivement (sur une image « Knife », la personne n'est souvent pas
  boxée). On utilise donc les **labels image-level positifs** : anomalie =
  `Person` ET `Knife` positifs, normal = `Person` positif sans `Knife`.

Composition (l'anomalie prend toutes les images disponibles ; le normal est
plafonné à ~50 % par source pour équilibrer la banque) :

| Source | Personne + couteau (anomalie) | Part de l'anomalie | Normal |
| --- | --- | --- | --- |
| COCO (train+val)   | 2 459 | ~99 % | plafonné à ~50 % de la banque |
| Open Images V7     | 28    | ~1 %  | plafonné à ~50 % de la banque |
| **Fusion**         | **2 487** | 100 % | ~50 / 50 COCO / OIV7 |

**Constat clé** : Open Images V7 n'apporte quasiment pas d'anomalies utiles —
seulement ~4 % de ses images « Knife » contiennent une personne, le reste étant
des couteaux de cuisine, dagues et armes (labels co-occurrents `Kitchen knife`,
`Dagger`, `Cutting board`, `Weapon`…). COCO, exhaustivement annoté, reste la seule
source substantielle de scènes « personne + couteau » ; la banque fusionnée est
donc ~50 / 50 côté normal mais son anomalie est à ~99 % du COCO.

## MLflow

Tous les runs vivent dans une base unique, y compris ceux rapatriés des serveurs
distants. Chaque run porte un tag `origin` (`local`, `g5k`, `metz`) et une
expérience par tâche (`celeba-histograms`, `celeba-heatmap`) :

```shell
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

Les runs distants sont fusionnés dans cette base au moment du rapatriement
(`./grid5000_run.sh --fetch`), via `tools/mlflow_import.py`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
