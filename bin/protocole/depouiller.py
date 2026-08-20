"""Rejoue les deux références sur les scores mesurés et dit laquelle détecte.

    python bin/protocole/depouiller.py --nom scene1

Une seule passe d'inférence a produit results/protocole/<nom>.jsonl ; les deux
échelles s'y appliquent après coup. Ce qui est comparé est donc bien l'échelle,
sur des scores identiques.

Le seuil est le même des deux côtés, et sans dimension : le q999 des scores
nominaux mesuré au fit, exprimé en écarts. C'est tout l'intérêt de normaliser —
un seuil qui se transporte.
"""

import json
import logging
import os
import statistics as st
from collections import deque

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import patchcore.banks  # noqa: E402

LOGGER = logging.getLogger(__name__)

COULEURS = {"absent": "#7f8c8d", "loin": "#2E86AB", "proche": "#C0442E"}


def _echelle_banque(fit_config):
    patch = ((fit_config.get("nominal_scores") or {}).get("patch") or {})
    if not patch or patch.get("sigma", 0) <= 0:
        raise SystemExit("Banque sans échelle mesurée — refitter pour l'obtenir.")
    return patch["median"], patch["sigma"], patch["q999"]


def _fenetre(lignes, secondes, fps_scorees):
    """Échelle glissante, comme en direct : médiane des médianes et des MAD."""
    n = max(1, int(round(secondes * fps_scorees)))
    recentes, sortie = deque(maxlen=n), []
    for l in lignes:
        recentes.append((l["med"], l["sigma"]))
        sortie.append((st.median([m for m, _ in recentes]),
                       st.median([s for _, s in recentes])))
    return sortie, n


@click.command()
@click.option("--nom", required=True)
@click.option("--fenetre_s", default=20.0, show_default=True)
def main(nom, fenetre_s):
    lignes = [json.loads(l) for l in open(os.path.join("results", "protocole", nom + ".jsonl"))]
    labels = json.load(open(os.path.join("data", "protocole", nom + ".labels.json")))

    # La condition et le temps se relisent sur le NUMÉRO DE FRAME, jamais sur
    # l'horloge du fichier : une webcam qui annonce un fps qu'elle ne tient pas
    # fait encoder un conteneur dont la timeline ne veut rien dire.
    fps = labels["fps_mesure"] or labels["fps_nominal"]
    for l in lignes:
        l["t_reel"] = l["frame"] / fps
        for seg in labels["segments"]:
            if seg["frame0"] <= l["frame"] < seg["frame1"]:
                l["condition"] = seg["condition"]
                break
    racine = os.path.join("models", "protocole", nom)
    bank_dir = os.path.join(racine, sorted(os.listdir(racine))[0])
    fit_config = json.load(open(os.path.join(bank_dir, patchcore.banks.CONFIG_FILENAME)))

    med_b, sig_b, q999_b = _echelle_banque(fit_config)
    seuil = (q999_b - med_b) / sig_b     # sans dimension, valable pour les deux
    duree = lignes[-1]["t_reel"] - lignes[0]["t_reel"]
    fps_scorees = len(lignes) / max(duree, 1e-6)
    fen, n_fen = _fenetre(lignes, fenetre_s, fps_scorees)

    z_banque = [(l["max"] - med_b) / sig_b for l in lignes]
    z_fen = [(l["max"] - m) / max(s, 1e-6) for l, (m, s) in zip(lignes, fen)]
    temps = [l["t_reel"] for l in lignes]
    cond = [l["condition"] for l in lignes]

    print("\n  {} — {} frames scorées, {:.1f}/s, fenêtre = {} frames ({:.0f} s)".format(
        nom, len(lignes), fps_scorees, n_fen, fenetre_s))
    print("  banque : médiane {:.3f}  sigma {:.3f}  ->  seuil {:.1f} écarts\n".format(
        med_b, sig_b, seuil))

    print("  {:<9} {:>7} {:>21} {:>21}".format("", "frames", "réf. BANQUE", "réf. FENÊTRE"))
    print("  {:<9} {:>7} {:>10} {:>10} {:>10} {:>10}".format(
        "", "", "pic méd.", "au-dessus", "pic méd.", "au-dessus"))
    for c in ("absent", "loin", "proche"):
        idx = [i for i, x in enumerate(cond) if x == c]
        if not idx:
            continue
        zb = [z_banque[i] for i in idx]
        zf = [z_fen[i] for i in idx]
        print("  {:<9} {:>7} {:>9.1f}σ {:>9.0f}% {:>9.1f}σ {:>9.0f}%".format(
            c, len(idx), st.median(zb), 100 * sum(z > seuil for z in zb) / len(zb),
            st.median(zf), 100 * sum(z > seuil for z in zf) / len(zf)))

    print("\n  Le pic s'efface-t-il pendant que l'objet reste dans le champ ?")
    for seg in labels["segments"]:
        if seg["condition"] == "absent":
            continue
        print("   {} ({:.0f}-{:.0f} s)".format(seg["condition"], seg["t0"], seg["t1"]))
        pas = (seg["t1"] - seg["t0"]) / 4
        for k in range(4):
            a, b = seg["t0"] + k * pas, seg["t0"] + (k + 1) * pas
            idx = [i for i, t in enumerate(temps) if a <= t < b]
            if not idx:
                continue
            print("     +{:>3.0f} à +{:>3.0f} s   banque {:>5.1f}σ   fenêtre {:>5.1f}σ".format(
                a - seg["t0"], b - seg["t0"],
                st.median([z_banque[i] for i in idx]), st.median([z_fen[i] for i in idx])))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    for seg in labels["segments"]:
        ax.axvspan(seg["t0"], seg["t1"], color=COULEURS[seg["condition"]], alpha=0.10, lw=0)
        ax.text((seg["t0"] + seg["t1"]) / 2, 0.97, seg["condition"], ha="center",
                va="top", transform=ax.get_xaxis_transform(), fontsize=8,
                color=COULEURS[seg["condition"]])
    ax.plot(temps, z_banque, lw=1.1, color="#1f6b8c", label="référence : enrôlement")
    ax.plot(temps, z_fen, lw=1.1, color="#B23A22", label="référence : fenêtre {:.0f} s".format(fenetre_s))
    ax.axhline(seuil, color="#333", ls="--", lw=1,
               label="seuil = q999 nominal ({:.1f}σ)".format(seuil))
    ax.set_xlabel("temps dans le clip (s)")
    ax.set_ylabel("pic de la carte, en écarts robustes")
    ax.set_title("{} — le pic survit-il à la recalibration ?".format(nom))
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(alpha=0.2)
    chemin = os.path.join("results", "protocole", nom + ".png")
    fig.savefig(chemin, bbox_inches="tight", dpi=130)
    print("\n  figure : {}\n".format(chemin))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
