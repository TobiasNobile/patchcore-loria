#!/usr/bin/env python
"""oiv7_fetch.py — Sous-ensemble Open Images V7 « personne + couteau » via FiftyOne,
en se basant sur les LABELS IMAGE-LEVEL (pas les boxes).

Raison : OID annote les bounding boxes par classe, non-exhaustivement — sur une
image « Knife » la personne n'est souvent pas boxée. Les labels image-level
(positive_labels) sont, eux, multi-classes par image. On dérive donc :
  - anomalie : Person ET Knife en positive_labels ;
  - normal   : Person en positive_labels SANS Knife.
Pas de bbox couteau (masque = zéros) : sans impact sur l'histogramme image-level
ni sur les heatmaps (overlay PatchCore). Manifest fusionnable (comme coco_fetch).

    DEST=/tmp/$USER/merged/oiv7 CAP_NORMAL=30000 python tools/oiv7_fetch.py

NB : OIV7 n'a que ~4 % de ses images couteau avec une personne (le reste =
couteaux de cuisine/armes) -> l'apport en anomalies est faible (~qq dizaines).
"""
import json
import os
import sys

DEST = os.environ.get("DEST", os.path.expanduser("/tmp/{}/merged/oiv7".format(os.environ.get("USER", "u"))))
CAP_NORMAL = int(os.environ.get("CAP_NORMAL", "30000"))
CAP_ANOMALY = int(os.environ.get("CAP_ANOMALY", "0"))  # 0 = toutes
SEED = int(os.environ.get("SEED", "0"))
SPLITS = os.environ.get("OIV7_SPLITS", "train,validation").split(",")

os.makedirs(DEST, exist_ok=True)
os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", os.path.join(DEST, "zoo"))
os.environ.setdefault("FIFTYONE_DATABASE_DIR", os.path.join(DEST, ".fiftyone", "db"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")

import PIL.Image  # noqa: E402
import fiftyone as fo  # noqa: E402
import fiftyone.zoo as foz  # noqa: E402


def _positive_labels(sample):
    """Labels image-level POSITIFS de l'image (champ 'positive_labels' d'OID ;
    fallback : toute Classifications de confidence >= 0.5)."""
    labs = set()
    for name, f in sample.iter_fields():
        if isinstance(f, fo.Classifications) and getattr(f, "classifications", None):
            if name == "negative_labels":
                continue
            for cc in f.classifications:
                if cc.confidence is None or cc.confidence >= 0.5:
                    labs.add(cc.label)
    return labs


def _load(classes, max_samples):
    """Charge les images OIV7 (splits poolés) contenant `classes`, avec leurs
    labels image-level (classifications)."""
    parts = []
    for split in SPLITS:
        parts.append(foz.load_zoo_dataset(
            "open-images-v7", split=split, label_types=["classifications"],
            classes=classes, max_samples=max_samples, only_matching=False,
            shuffle=True, seed=SEED,
            dataset_name="oiv7il_{}_{}_{}".format("-".join(classes), split, max_samples or "all"),
        ))
    return parts


def _records(datasets, keep_anomaly):
    """keep_anomaly=True -> Person∧Knife (image-level) ; False -> Person sans Knife."""
    recs, seen = [], set()
    for ds in datasets:
        for sample in ds:
            labs = _positive_labels(sample)
            has_person = "Person" in labs
            has_knife = "Knife" in labs
            if keep_anomaly:
                if not (has_person and has_knife):
                    continue
            else:
                if not has_person or has_knife:
                    continue
            path = sample.filepath
            key = os.path.basename(path)
            if key in seen or not os.path.isfile(path):
                continue
            seen.add(key)
            try:
                w, h = PIL.Image.open(path).size
            except Exception:
                continue
            recs.append({"src_path": path, "key": key, "width": w, "height": h,
                         "is_anomaly": int(keep_anomaly)})
    return recs


def main():
    print("OIV7 fetch (image-level) -> {} (CAP_NORMAL={}, splits={})".format(DEST, CAP_NORMAL, SPLITS))
    anomaly = _records(_load(["Knife"], CAP_ANOMALY or None), keep_anomaly=True)
    normal = _records(_load(["Person"], CAP_NORMAL), keep_anomaly=False)
    print("OIV7 sélection : {} personne-sans-couteau + {} personne-couteau".format(
        len(normal), len(anomaly)))

    img_dir = os.path.join(DEST, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = []
    for rec in normal + anomaly:
        link = os.path.join(img_dir, "oiv7_" + rec["key"])
        if not os.path.exists(link):
            try:
                os.symlink(rec["src_path"], link)
            except OSError:
                continue
        manifest.append({
            "file": os.path.join("images", "oiv7_" + rec["key"]),
            "is_anomaly": rec["is_anomaly"],
            "knife_boxes": [],  # image-level : pas de bbox (masque = zéros)
            "width": rec["width"],
            "height": rec["height"],
        })
    n_anom = sum(m["is_anomaly"] for m in manifest)
    with open(os.path.join(DEST, "manifest.json"), "w") as fh:
        json.dump({"images": manifest}, fh)
    print("OIV7 manifest : {} images ({} normal / {} couteau) -> {}".format(
        len(manifest), len(manifest) - n_anom, n_anom, os.path.join(DEST, "manifest.json")))
    if n_anom == 0:
        print("ATTENTION : 0 image person+knife image-level récupérée.", file=sys.stderr)


if __name__ == "__main__":
    main()
