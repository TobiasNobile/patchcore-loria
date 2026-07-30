#!/usr/bin/env python
"""Diagnostic : les images couteau OIV7 portent-elles une box Person quand on
charge TOUS les labels ? Tranche entre (A) FiftyOne charge en class-scoped (fixable)
et (B) OID ne co-annote pas (il faudrait relâcher/lâcher)."""
import os
from collections import Counter

DEST = "/tmp/{}/diag".format(os.environ.get("USER", "u"))
os.makedirs(DEST, exist_ok=True)
os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", os.path.join(DEST, "zoo"))
os.environ.setdefault("FIFTYONE_DATABASE_DIR", os.path.join(DEST, "db"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")

import fiftyone as fo
import fiftyone.zoo as foz


def det_fields(sample):
    return [n for n, f in sample.iter_fields() if isinstance(f, fo.Detections)]


def labels_of(sample, fields):
    labs = set()
    for fld in fields:
        d = sample[fld]
        if d and getattr(d, "detections", None):
            for det in d.detections:
                labs.add(det.label)
    return labs


def check(desc, **load_kwargs):
    print("\n===", desc, "===")
    ds = foz.load_zoo_dataset(
        "open-images-v7", split="validation", label_types=["detections"],
        **load_kwargs,
    )
    fields = det_fields(ds.first())
    print("  champs Detections :", fields)
    withp = knives = 0
    co = Counter()
    for s in ds:
        labs = labels_of(s, fields)
        if "Knife" in labs:
            knives += 1
            for l in labs:
                co[l] += 1
            if "Person" in labs:
                withp += 1
    print("  images couteau : {} | dont avec box Person : {}".format(knives, withp))
    print("  labels co-occurrents sur images couteau :", co.most_common(12))


# A) tel qu'on faisait : classes=["Knife"]
check("classes=['Knife'], only_matching=False, max=59",
      classes=["Knife"], max_samples=59, only_matching=False)

# B) en demandant AUSSI Person dans classes (pour forcer le DL des annos Person)
check("classes=['Person','Knife'], only_matching=False, max=400",
      classes=["Person", "Knife"], max_samples=400, only_matching=False)
