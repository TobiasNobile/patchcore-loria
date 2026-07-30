#!/usr/bin/env python
"""o365_fetch.py — Sous-ensemble Objects365 « personne + couteau » à partir des
annotations COCO-format DÉJÀ présentes sur disque.

O365 est verrouillé côté Chine : c'est TOI qui poses le JSON d'annotations
(zhiyuan_objv2_{train,val}.json) + le dossier d'images sur g5k ; ce script ne
télécharge RIEN, il ne fait qu'indexer et écrire un manifest fusionnable
(même format que coco_fetch.py / CocoDataset).

    O365_ANN=/chemin/zhiyuan_objv2_val.json O365_IMG=/chemin/images \
    DEST=/tmp/$USER/merged/o365 CAP_NORMAL=30000 python tools/o365_fetch.py

Env : O365_ANN (JSON COCO-format), O365_IMG (racine des images), DEST,
      CAP_NORMAL, CAP_ANOMALY (0=toutes), SEED. Person/Knife détectés par NOM
      dans le bloc `categories` (robuste aux id v1/v2).
"""
import json
import os
import random
import sys

import PIL.Image

DEST = os.environ.get("DEST", os.path.expanduser("/tmp/{}/merged/o365".format(os.environ.get("USER", "u"))))
ANN = os.environ.get("O365_ANN")
IMG_ROOT = os.environ.get("O365_IMG")
CAP_NORMAL = int(os.environ.get("CAP_NORMAL", "30000"))
CAP_ANOMALY = int(os.environ.get("CAP_ANOMALY", "0"))  # 0 = toutes
SEED = int(os.environ.get("SEED", "0"))


def _resolve_image(file_name):
    """O365 stocke souvent file_name en chemin relatif (images/v1/patchX/....jpg)
    ou juste le basename. On tente les deux sous O365_IMG."""
    for cand in (
        os.path.join(IMG_ROOT, file_name),
        os.path.join(IMG_ROOT, os.path.basename(file_name)),
    ):
        if os.path.isfile(cand):
            return cand
    return None


def main():
    if not ANN or not IMG_ROOT:
        sys.exit("O365_ANN (JSON) et O365_IMG (racine images) sont requis.")
    print("O365 fetch : ann={} img={} -> {}".format(ANN, IMG_ROOT, DEST))
    with open(ANN) as fh:
        data = json.load(fh)

    name_to_id = {c["name"].strip().lower(): c["id"] for c in data["categories"]}
    person_id, knife_id = name_to_id.get("person"), name_to_id.get("knife")
    if person_id is None or knife_id is None:
        sys.exit("Catégories person/knife introuvables. Exemples: {}".format(
            [c["name"] for c in data["categories"][:30]]))

    imgs = {im["id"]: {"file_name": im["file_name"], "width": im.get("width", 0),
                       "height": im.get("height", 0), "has_person": False, "knife_boxes": []}
            for im in data["images"]}
    for ann in data["annotations"]:
        rec = imgs.get(ann["image_id"])
        if rec is None:
            continue
        if ann["category_id"] == person_id:
            rec["has_person"] = True
        elif ann["category_id"] == knife_id:
            x, y, w, h = ann["bbox"]
            rec["knife_boxes"].append([int(x), int(y), int(x + w), int(y + h)])

    samples = [
        {"file_name": r["file_name"], "width": r["width"], "height": r["height"],
         "is_anomaly": int(len(r["knife_boxes"]) > 0), "knife_boxes": r["knife_boxes"]}
        for r in imgs.values() if r["has_person"]  # scènes de personnes seulement
    ]
    normal = [s for s in samples if not s["is_anomaly"]]
    anomaly = [s for s in samples if s["is_anomaly"]]
    rng = random.Random(SEED)
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    if CAP_NORMAL > 0:
        normal = normal[:CAP_NORMAL]
    if CAP_ANOMALY > 0:
        anomaly = anomaly[:CAP_ANOMALY]
    print("O365 sélection : {} personne-sans-couteau + {} personne-couteau".format(
        len(normal), len(anomaly)))

    img_dir = os.path.join(DEST, "images")
    os.makedirs(img_dir, exist_ok=True)
    manifest = []
    missing = 0
    for rec in normal + anomaly:
        src = _resolve_image(rec["file_name"])
        if src is None:
            missing += 1
            continue
        w, h = rec["width"], rec["height"]
        if w <= 0 or h <= 0:
            try:
                w, h = PIL.Image.open(src).size
            except Exception:
                continue
        key = "o365_" + os.path.basename(rec["file_name"])
        link = os.path.join(img_dir, key)
        if not os.path.exists(link):
            try:
                os.symlink(src, link)
            except OSError:
                continue
        manifest.append({"file": os.path.join("images", key), "is_anomaly": rec["is_anomaly"],
                         "knife_boxes": rec["knife_boxes"], "width": w, "height": h})

    n_anom = sum(m["is_anomaly"] for m in manifest)
    with open(os.path.join(DEST, "manifest.json"), "w") as fh:
        json.dump({"images": manifest}, fh)
    print("O365 manifest : {} images ({} normal / {} couteau), {} images introuvables -> {}".format(
        len(manifest), len(manifest) - n_anom, n_anom, missing, os.path.join(DEST, "manifest.json")))
    if n_anom == 0:
        print("ATTENTION : 0 image person+knife (vérifie les catégories / le split).", file=sys.stderr)


if __name__ == "__main__":
    main()
