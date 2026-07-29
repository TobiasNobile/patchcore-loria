#!/usr/bin/env python
"""oiv7_fetch.py — Récupère un sous-ensemble Open Images V7 « personne + couteau »
via FiftyOne, dans le MÊME format de manifest que coco_fetch.py (fusionnable).

FiftyOne télécharge à la demande, seulement les images des classes voulues :
  - anomalie : toutes les images avec une personne ET un couteau ;
  - normal   : images avec une personne SANS couteau (plafonné à CAP_NORMAL).
Écrit DEST/manifest.json ({file, is_anomaly, knife_boxes, width, height}) et des
symlinks DEST/images/ vers les images du cache FiftyOne (node-local, éphémère).

    DEST=/tmp/$USER/merged/oiv7 CAP_NORMAL=30000 python tools/oiv7_fetch.py

Prérequis : `pip install fiftyone` (gros paquet, MongoDB embarqué). Le wrapper
bin/merged/fit_and_score.sh l'installe si absent. Tout le cache va sur node-local
(FIFTYONE_* pointés sur DEST) pour ne pas toucher au quota /home.
"""
import json
import os
import sys

DEST = os.environ.get("DEST", os.path.expanduser("/tmp/{}/merged/oiv7".format(os.environ.get("USER", "u"))))
CAP_NORMAL = int(os.environ.get("CAP_NORMAL", "30000"))
CAP_ANOMALY = int(os.environ.get("CAP_ANOMALY", "0"))  # 0 = toutes
SEED = int(os.environ.get("SEED", "0"))
SPLITS = os.environ.get("OIV7_SPLITS", "train,validation").split(",")

# Cache FiftyOne + Mongo sur le disque local du nœud (hors quota home).
os.makedirs(DEST, exist_ok=True)
_fo_home = os.path.join(DEST, ".fiftyone")
os.environ.setdefault("FIFTYONE_DEFAULT_DATASET_DIR", os.path.join(DEST, "zoo"))
os.environ.setdefault("FIFTYONE_DATABASE_DIR", os.path.join(_fo_home, "db"))
os.environ.setdefault("FIFTYONE_DO_NOT_TRACK", "true")

import PIL.Image  # noqa: E402
import fiftyone as fo  # noqa: E402
import fiftyone.zoo as foz  # noqa: E402


def _detections_field(sample):
    """Nom du champ Detections (OIV7 le nomme en général 'detections')."""
    for name, field in sample.iter_fields():
        if isinstance(field, fo.Detections):
            return name
    return None


def _load(classes, max_samples):
    """Charge le sous-ensemble OIV7 (les splits poolés) filtré sur `classes`."""
    parts = []
    for split in SPLITS:
        ds = foz.load_zoo_dataset(
            "open-images-v7",
            split=split,
            label_types=["detections"],
            classes=classes,
            max_samples=max_samples,
            only_matching=True,
            shuffle=True,
            seed=SEED,
            dataset_name="oiv7_{}_{}_{}".format("-".join(classes), split, max_samples or "all"),
        )
        parts.append(ds)
    return parts


def _records_from(datasets, keep_anomaly):
    """Extrait les enregistrements manifest des datasets FiftyOne.
    keep_anomaly=True -> garde person+knife ; False -> garde person sans knife."""
    recs = []
    seen = set()
    for ds in datasets:
        field = None
        for sample in ds:
            if field is None:
                field = _detections_field(sample)
            dets = getattr(sample[field], "detections", []) if field else []
            labels = {d.label for d in dets}
            has_person = "Person" in labels
            has_knife = "Knife" in labels
            if not has_person:
                continue
            if keep_anomaly and not has_knife:
                continue
            if (not keep_anomaly) and has_knife:
                continue
            path = sample.filepath
            key = os.path.basename(path)
            if key in seen:
                continue
            seen.add(key)
            try:
                w, h = PIL.Image.open(path).size
            except Exception:
                continue
            knife_boxes = []
            for d in dets:
                if d.label != "Knife":
                    continue
                x, y, bw, bh = d.bounding_box  # relatif [0,1]
                knife_boxes.append([int(x * w), int(y * h), int((x + bw) * w), int((y + bh) * h)])
            recs.append({"src_path": path, "key": key, "width": w, "height": h,
                         "is_anomaly": int(keep_anomaly), "knife_boxes": knife_boxes})
    return recs


def main():
    print("OIV7 fetch -> {} (CAP_NORMAL={}, splits={})".format(DEST, CAP_NORMAL, SPLITS))
    # Anomalie : toutes les images couteau, filtrées personne.
    anomaly = _records_from(_load(["Knife"], CAP_ANOMALY or None), keep_anomaly=True)
    # Normal : personnes (plafonné), filtrées sans couteau.
    normal = _records_from(_load(["Person"], CAP_NORMAL), keep_anomaly=False)
    print("OIV7 sélection : {} personne-sans-couteau + {} personne-couteau".format(len(normal), len(anomaly)))

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
            "knife_boxes": rec["knife_boxes"],
            "width": rec["width"],
            "height": rec["height"],
        })
    n_anom = sum(m["is_anomaly"] for m in manifest)
    with open(os.path.join(DEST, "manifest.json"), "w") as fh:
        json.dump({"images": manifest}, fh)
    print("OIV7 manifest : {} images ({} normal / {} couteau) -> {}".format(
        len(manifest), len(manifest) - n_anom, n_anom, os.path.join(DEST, "manifest.json")))
    if n_anom == 0:
        print("ATTENTION : aucune image couteau OIV7 récupérée.", file=sys.stderr)


if __name__ == "__main__":
    main()
