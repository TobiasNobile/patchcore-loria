#!/usr/bin/env python
"""merge_manifests.py — Fusionne plusieurs manifests source en un manifest unique
lisible par CocoDataset.

Chaque source est un dossier (produit par coco_fetch.py / oiv7_fetch.py) contenant
manifest.json + images/. On préfixe le champ `file` de chaque entrée par le nom de
la source, de sorte que le manifest fusionné, placé à la RACINE commune, résolve
`<racine>/<source>/images/...`.

    python tools/merge_manifests.py /tmp/$USER/merged \\
        /tmp/$USER/merged/coco:coco  /tmp/$USER/merged/oiv7:oiv7

-> écrit /tmp/$USER/merged/manifest.json (les sous-dossiers coco/ oiv7/ doivent
   donc être SOUS la racine).
"""
import json
import os
import sys


def main():
    out_dir = sys.argv[1]
    specs = sys.argv[2:]
    if not specs:
        sys.exit("usage: merge_manifests.py OUT_DIR SRC_DIR:name [SRC_DIR:name ...]")

    merged = []
    for spec in specs:
        src_dir, name = spec.rsplit(":", 1)
        path = os.path.join(src_dir, "manifest.json")
        if not os.path.isfile(path):
            print("  (source ignorée, pas de manifest) :", path, file=sys.stderr)
            continue
        images = json.load(open(path))["images"]
        for e in images:
            e2 = dict(e)
            e2["file"] = os.path.join(name, e["file"])  # ex: coco/images/x.jpg
            merged.append(e2)
        n_a = sum(e["is_anomaly"] for e in images)
        print("  {} : {} images ({} normal / {} couteau)".format(name, len(images), len(images) - n_a, n_a))

    os.makedirs(out_dir, exist_ok=True)
    n_anom = sum(e["is_anomaly"] for e in merged)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"images": merged}, fh)
    print("MERGE : {} images ({} normal / {} couteau) -> {}".format(
        len(merged), len(merged) - n_anom, n_anom, os.path.join(out_dir, "manifest.json")))


if __name__ == "__main__":
    main()
