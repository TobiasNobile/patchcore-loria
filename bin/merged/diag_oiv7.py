#!/usr/bin/env python
"""Diagnostic v2 : sur les images couteau OIV7 du split TRAIN, combien ont
Person en IMAGE-LEVEL (labels), et FiftyOne me les rend-il ? (vs les boxes)."""
import os
from collections import Counter

DEST = "/tmp/{}/diag".format(os.environ.get("USER", "u"))
os.makedirs(DEST, exist_ok=True)
os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", os.path.join(DEST, "zoo"))
os.environ.setdefault("FIFTYONE_DATABASE_DIR", os.path.join(DEST, "db"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")

import fiftyone as fo
import fiftyone.zoo as foz


def pos_imagelevel(sample, cls_fields):
    """Labels image-level POSITIFS (confidence >= 0.5 ; OID: 1=positif, 0=négatif)."""
    labs = set()
    for fld in cls_fields:
        c = sample[fld]
        if c and getattr(c, "classifications", None):
            for cc in c.classifications:
                if cc.confidence is None or cc.confidence >= 0.5:
                    labs.add(cc.label)
    return labs


def box_labels(sample, det_fields):
    labs = set()
    for fld in det_fields:
        d = sample[fld]
        if d and getattr(d, "detections", None):
            for det in d.detections:
                labs.add(det.label)
    return labs


print("=== TRAIN, classes=['Knife'], label_types=[detections,classifications], max=500 ===")
ds = foz.load_zoo_dataset(
    "open-images-v7", split="train",
    label_types=["detections", "classifications"],
    classes=["Knife"], max_samples=500, only_matching=False,
    dataset_name="diag_knife_train",
)
s0 = ds.first()
cls_fields = [n for n, f in s0.iter_fields() if isinstance(f, fo.Classifications)]
det_fields = [n for n, f in s0.iter_fields() if isinstance(f, fo.Detections)]
print("  champs classifications :", cls_fields, "| détections :", det_fields)

total = p_label = p_box = pk_label = 0
co = Counter()
for s in ds:
    total += 1
    im = pos_imagelevel(s, cls_fields)
    bx = box_labels(s, det_fields)
    if "Person" in im:
        p_label += 1
    if "Person" in bx:
        p_box += 1
    if "Person" in im and "Knife" in im:
        pk_label += 1
    for l in im:
        co[l] += 1
print("  images couteau (train)          :", total)
print("  avec Person IMAGE-LEVEL         :", p_label)
print("  avec Person BOX                 :", p_box)
print("  avec Person+Knife IMAGE-LEVEL   :", pk_label)
print("  co-labels image-level (top15)   :", co.most_common(15))
