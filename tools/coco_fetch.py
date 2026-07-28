#!/usr/bin/env python
"""coco_fetch.py — Récupère un sous-ensemble COCO « personne + couteau » SANS
saturer le /home.

On ne télécharge QUE les images utiles au one-class :
  - normal : images avec une personne ET aucun couteau (plafonné à CAP_NORMAL) ;
  - anomalie : images avec une personne ET au moins un couteau (toutes).
Les images vont sur le disque local du nœud (DEST, éphémère). On écrit un
manifest.json compact (chemin, is_anomaly, bbox couteau, dims) que CocoDataset
relit — pas besoin de re-parser les 450 Mo d'annotations au fit.

Aucune dépendance lourde : annotations JSON parsées à la main, bbox couteau
rasterisés en masque plus tard (pas de pycocotools), images tirées des URLs
publiques COCO en parallèle.

    DEST=/tmp/$USER/coco CAP_NORMAL=40000 python tools/coco_fetch.py

Env : DEST, CAP_NORMAL, CAP_ANOMALY (0=toutes), SEED, COCO_SPLITS (train2017,val2017),
      G5K_PROXY (proxy sortant si le nœud l'exige), COCO_WORKERS.
"""
import concurrent.futures as cf
import io
import json
import os
import random
import sys
import urllib.request
import zipfile

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_URL = "http://images.cocodataset.org/{split}/{filename}"

DEST = os.environ.get("DEST", os.path.expanduser("/tmp/{}/coco".format(os.environ.get("USER", "u"))))
CAP_NORMAL = int(os.environ.get("CAP_NORMAL", "40000"))
CAP_ANOMALY = int(os.environ.get("CAP_ANOMALY", "0"))  # 0 = toutes
SEED = int(os.environ.get("SEED", "0"))
SPLITS = os.environ.get("COCO_SPLITS", "train2017,val2017").split(",")
WORKERS = int(os.environ.get("COCO_WORKERS", "16"))


def _setup_proxy():
    """Le nœud g5k hérite parfois d'un http_proxy qui ne résout pas en batch.
    Direct par défaut ; proxy explicite si G5K_PROXY est fourni (cf. ucf_fetch)."""
    proxy = os.environ.get("G5K_PROXY")
    if proxy:
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ[k] = proxy
    else:
        for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(k, None)
    print("Proxy:", proxy or "(direct)")


def _download_annotations(dest):
    """instances_{split}.json depuis le zip officiel (250 Mo), extraits une fois."""
    ann_dir = os.path.join(dest, "annotations")
    need = [os.path.join(ann_dir, "instances_{}.json".format(s)) for s in SPLITS]
    if all(os.path.isfile(p) for p in need):
        return
    os.makedirs(ann_dir, exist_ok=True)
    print("Téléchargement des annotations (~250 Mo)...")
    with urllib.request.urlopen(ANN_URL, timeout=120) as resp:
        buf = io.BytesIO(resp.read())
    with zipfile.ZipFile(buf) as zf:
        for s in SPLITS:
            member = "annotations/instances_{}.json".format(s)
            target = os.path.join(ann_dir, "instances_{}.json".format(s))
            if not os.path.isfile(target):
                with zf.open(member) as src, open(target, "wb") as out:
                    out.write(src.read())
    print("Annotations extraites.")


def _index_split(ann_path):
    """Parcourt une annotation COCO -> par image : présence personne, bbox couteau.
    Renvoie une liste de dicts (filename, split, width, height, is_anomaly,
    knife_boxes) pour les seules images CONTENANT une personne."""
    with open(ann_path) as fh:
        data = json.load(fh)
    name_to_id = {c["name"]: c["id"] for c in data["categories"]}
    person_id, knife_id = name_to_id.get("person"), name_to_id.get("knife")
    if person_id is None or knife_id is None:
        raise RuntimeError("Catégories person/knife introuvables dans {}".format(ann_path))

    imgs = {im["id"]: {"file_name": im["file_name"], "width": im["width"],
                       "height": im["height"], "has_person": False, "knife_boxes": []}
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

    split = os.path.splitext(os.path.basename(ann_path))[0].replace("instances_", "")
    out = []
    for rec in imgs.values():
        if not rec["has_person"]:
            continue  # on ne veut que des scènes de personnes
        out.append({
            "filename": rec["file_name"],
            "split": split,
            "width": rec["width"],
            "height": rec["height"],
            "is_anomaly": int(len(rec["knife_boxes"]) > 0),
            "knife_boxes": rec["knife_boxes"],
        })
    return out


def _fetch_one(rec, img_dir):
    dst = os.path.join(img_dir, rec["filename"])
    if os.path.isfile(dst) and os.path.getsize(dst) > 0:
        return True
    url = IMG_URL.format(split=rec["split"], filename=rec["filename"])
    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                blob = resp.read()
            tmp = dst + ".part"
            with open(tmp, "wb") as out:
                out.write(blob)
            os.replace(tmp, dst)
            return True
        except Exception:
            if attempt == 2:
                return False
    return False


def main():
    _setup_proxy()
    os.makedirs(DEST, exist_ok=True)
    _download_annotations(DEST)

    samples = []
    for s in SPLITS:
        samples += _index_split(os.path.join(DEST, "annotations", "instances_{}.json".format(s)))

    normal = [s for s in samples if not s["is_anomaly"]]
    anomaly = [s for s in samples if s["is_anomaly"]]
    rng = random.Random(SEED)
    rng.shuffle(normal)
    rng.shuffle(anomaly)
    if CAP_NORMAL > 0:
        normal = normal[:CAP_NORMAL]
    if CAP_ANOMALY > 0:
        anomaly = anomaly[:CAP_ANOMALY]
    chosen = normal + anomaly
    print("Sélection : {} personne-sans-couteau + {} personne-avec-couteau = {} images".format(
        len(normal), len(anomaly), len(chosen)))

    img_dir = os.path.join(DEST, "images")
    os.makedirs(img_dir, exist_ok=True)
    print("Téléchargement de {} images ({} threads)...".format(len(chosen), WORKERS))
    ok = 0
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futures = {ex.submit(_fetch_one, r, img_dir): r for r in chosen}
        for i, fut in enumerate(cf.as_completed(futures), 1):
            if fut.result():
                ok += 1
            if i % 2000 == 0:
                print("  {}/{} ({} ok)".format(i, len(chosen), ok))

    # Manifest : ne garde que les images effectivement présentes sur disque.
    manifest = []
    for r in chosen:
        p = os.path.join(img_dir, r["filename"])
        if os.path.isfile(p) and os.path.getsize(p) > 0:
            manifest.append({
                "file": os.path.join("images", r["filename"]),
                "is_anomaly": r["is_anomaly"],
                "knife_boxes": r["knife_boxes"],
                "width": r["width"],
                "height": r["height"],
            })
    n_norm = sum(1 for m in manifest if not m["is_anomaly"])
    n_anom = len(manifest) - n_norm
    with open(os.path.join(DEST, "manifest.json"), "w") as fh:
        json.dump({"images": manifest}, fh)
    print("Manifest écrit : {} images ({} normal / {} couteau) -> {}".format(
        len(manifest), n_norm, n_anom, os.path.join(DEST, "manifest.json")))
    if n_anom == 0:
        print("ATTENTION : aucune image couteau récupérée — le test sera vide.", file=sys.stderr)


if __name__ == "__main__":
    main()
