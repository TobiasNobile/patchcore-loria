# Towards Total Recall in Industrial Anomaly Detection

This repository contains the implementation for `PatchCore` as proposed in Roth et al. (2021), <https://arxiv.org/abs/2106.08265>.

It also provides various pretrained models that can achieve up to _99.6%_ image-level anomaly
detection AUROC, _98.4%_ pixel-level anomaly localization AUROC and _>95%_ PRO score (although the
later metric is not included for license reasons).

![defect_segmentation](images/patchcore_defect_segmentation.png)

_For questions & feedback, please reach out to karsten.rh1@gmail.com!_

---

## Quick Guide

### 1. Environment setup

```shell
# Install dependencies
uv sync --extra dev

# Configure local paths (never committed)
cp .env.example .env   # then edit if your paths differ
source .env
```

The dataset lives **outside the repository**, at a stable path on your machine or compute server:

| Machine        | `MVTEC_PATH`                                              |
|----------------|-----------------------------------------------------------|
| Local (dev)    | `$HOME/dev/telecom/stage_1a_data/mvtec_anomaly_detection` |
| Compute server | set in `.env` on that machine (e.g. `/scratch/<user>/...`) |

Download MVTec AD from <https://www.mvtec.com/company/research/datasets/mvtec-ad> and extract it so the layout is:

```
$MVTEC_PATH/
├── bottle/
├── cable/
└── ...   (15 categories total)
```

### 2. Reference training run

To train PatchCore on MVTec AD (as described below), run

```
datapath=/path_to_mvtec_folder/mvtec datasets=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut'
'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
dataset_flags=($(for dataset in "${datasets[@]}"; do echo '-d '$dataset; done))


python bin/run_patchcore.py --gpu 0 --seed 0 --save_patchcore_model \
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

python bin/load_and_evaluate_patchcore.py --gpu 0 --seed 0 $savefolder \
patch_core_loader "${model_flags[@]}" --faiss_on_gpu \
dataset --resize 366 --imagesize 320 "${dataset_flags[@]}" mvtec $datapath
```

A set of pretrained PatchCores are hosted here: __add link__. To use them (and replicate training),
check out `sample_evaluation.sh` and `sample_training.sh`.

---

## MLflow Tracking

Runs are tracked locally with MLflow. Start the UI with:

```shell
uv run mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5001
# then open http://localhost:5001
```

### Naming convention

| MLflow field  | Value                                   | Example                                          |
|---------------|-----------------------------------------|--------------------------------------------------|
| `experiment`  | Dataset name                            | `mvtec`                                          |
| `run_name`    | `{backbone}-{sampler}-p{pct}-im{size}` | `wideresnet50-approx_greedy_coreset-p10-im224`   |
| tag `category`| MVTec subdataset                        | `bottle`                                         |

**Backbone** — model name as passed to `-b` (e.g. `wideresnet50`). For ensembles, join with `+`:
`wideresnet50+resnext101`.

**Sampler** — coreset method (`approx_greedy_coreset`, `greedy_coreset`, `identity`).

**pct** — coreset percentage × 100, zero-padded to 2 digits: `-p 0.1` → `p10`, `-p 0.01` → `p01`.

**im** — final image size after centre-crop (`--imagesize`).

### Logged metrics (per run)

| Key                   | Description                                      |
|-----------------------|--------------------------------------------------|
| `instance_auroc`      | Image-level AUROC                                |
| `full_pixel_auroc`    | Pixel-level AUROC on all test images             |
| `anomaly_pixel_auroc` | Pixel-level AUROC restricted to anomalous images |

Use `patchcore.tracking.make_run_name` to build the run name programmatically:

```python
from patchcore.tracking import make_run_name, patchcore_run

run_name = make_run_name(["wideresnet50"], "approx_greedy_coreset", 0.1, 224)
# → "wideresnet50-approx_greedy_coreset-p10-im224"

with patchcore_run(experiment="mvtec", run_name=run_name, params=config) as run:
    run.log_metrics({"instance_auroc": auroc, ...})
```

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

To do so, we have provided `bin/run_patchcore.py`, which uses `click` to manage and aggregate input
arguments. This looks something like

```shell
python bin/run_patchcore.py \
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
`bin/load_and_evaluate_patchcore.py` which showcases an exemplary evaluation process.

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
python bin/run_patchcore.py --gpu <gpu_id> --seed <seed> --save_patchcore_model --log_group <log_name> --log_online --log_project <log_project> results \

patch_core -b wideresnet101 -b resnext101 -b densenet201 -le 0.layer2 -le 0.layer3 -le 1.layer2 -le 1.layer3 -le 2.features.denseblock2 -le 2.features.denseblock3 --faiss_on_gpu \

--pretrain_embed_dimension 1024  --target_embed_dimension 384 --anomaly_scorer_num_nn 1 --patchsize 3 sampler -p 0.01 approx_greedy_coreset dataset --resize 256 --imagesize 224 "${dataset_flags[@]}" mvtec $datapath

```

When using `--save_patchcore_model`, in the case of ensembles, a respective ensemble of PatchCore parameters is stored.

### Evaluating a pretrained PatchCore model

To evaluate a/our pretrained PatchCore model(s), run

```shell
python bin/load_and_evaluate_patchcore.py --gpu <gpu_id> --seed <seed> $savefolder \
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

## Jalon 1 — PatchCore on CelebA

### Dataset

Loaded via HuggingFace `datasets` (see `src/patchcore/celeba.py`):

```python
from datasets import load_dataset
ds = load_dataset("flwrlabs/celeba")  # train / valid / test splits, cached locally
```

Cached locally under `~/.cache/huggingface/hub/datasets--flwrlabs--celeba/.../img_align+identity+attr/*.parquet`.
Each row has an `image` plus 40 boolean CelebA attributes (`Wearing_Hat`, `Male`, `Smiling`, ...).

### Hat / No-hat class imbalance

Counted directly from the `Wearing_Hat` boolean column of the cached parquet shards (no image
decoding required), 2026-07-01:

| Split   | Hat   | No hat  | Total   | Hat %  |
|---------|-------|---------|---------|--------|
| train   | 8,039 | 154,731 | 162,770 | 4.94%  |
| valid   |   940 |  18,927 |  19,867 | 4.73%  |
| test    |   839 |  19,123 |  19,962 | 4.20%  |
| **all** | **9,818** | **192,781** | **202,599** | **4.85%** |

→ Roughly 1 "hat" image for every ~20 "no-hat" images (ratio ≈ 1:19.6 across the full dataset).
This is a strong, naturally-occurring class imbalance — relevant if `Wearing_Hat` is used to
define the normal/anomalous split for the PatchCore experiment on CelebA.

### One-class train / test split

Implemented in `src/patchcore/datasets/celeba.py` (`CelebADataset`, mirrors the `MVTecDataset`
interface). `Wearing_Hat` defines the anomaly label:

- **Train** (`DatasetSplit.TRAIN`): only "no-hat" images from the CelebA `train` split — the
  PatchCore memory bank only ever sees "normal" data. 154,731 images.
- **Test** (`DatasetSplit.TEST`): hat + no-hat images from the CelebA `test` split, **balanced**
  by randomly subsampling the majority (no-hat) class down to the minority (hat) count
  (`seed`-controlled, default `0`). 839 hat + 839 no-hat = 1,678 images.

```python
from patchcore.datasets.celeba import CelebADataset, DatasetSplit

train_dataset = CelebADataset(split=DatasetSplit.TRAIN)               # 154,731 no-hat images
test_dataset = CelebADataset(split=DatasetSplit.TEST, seed=0)         # 839 + 839 balanced images
```

### Running PatchCore on CelebA

Fitting and inference are two separate scripts, because they have very different costs. Building the
memory bank is dominated by the coreset — a sequential selection whose iteration count grows with
`TRAIN_SUBSET` × `PERCENTAGE` — and it only has to happen once. Scoring an image against a saved bank
is the fast half, and the one that has to run in real time. Both scripts hold their hyperparameters
in a `CONFIG` block at the top of the file; edit it and run, no flags to remember.

```shell
python bin/fit_memory_bank_celeba.py                    # once: writes models/celeba/<tag>/
python bin/infer_heatmap_celeba.py                      # heatmap overlay, default image
python bin/infer_heatmap_celeba.py --image_index 900    # any other test image
```

`<tag>` encodes the config (backbone, sampler, percentage, train subset, seed), so different configs
never overwrite each other, and each bank carries a `fit_config.json` recording how it was built.
`infer_heatmap_celeba.py` reads `resize` / `imagesize` back from that file rather than restating
them: a query embedded differently from the bank it is searched against would give meaningless
distances.

Pixel-level evaluation (real "hat" segmentation ground-truth, sourced from CelebAMask-HQ) is a
follow-up step, not yet wired in.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
