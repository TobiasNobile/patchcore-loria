"""Filme un clip labellisé, pour mesurer si la fenêtre glissante absorbe l'objet.

    python bin/protocole/capture.py --nom essai1
    python bin/protocole/capture.py --nom essai1 --rapide     # répétition à blanc

La question est simple et le protocole en découle : la fenêtre de calibration
couvre 20 secondes. Si elle absorbe ce qui reste dans le champ, un objet présent
pendant 60 secondes — trois fois la fenêtre — doit voir son score décroître
jusqu'à se confondre avec le fond. S'il tient, elle n'absorbe pas.

Deux tailles apparentes, parce que la médiane ne cède que lorsque l'objet occupe
une grande part de l'image : c'est la variable qui décide, pas la durée.

Écrit data/protocole/<nom>.mp4 et <nom>.labels.json. L'incrustation à l'écran
n'est PAS enregistrée : la vidéo ne contient que la scène, sinon la consigne
elle-même deviendrait une anomalie.
"""

import json
import logging
import os
import platform
import time

if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import click
import cv2

LOGGER = logging.getLogger(__name__)

# (durée en secondes, condition, consigne affichée)
SEGMENTS = (
    (30, "absent", "RIEN dans le champ"),
    (60, "loin", "OBJET dans le champ, LOIN"),
    (15, "absent", "SORS l'objet du champ"),
    (60, "proche", "OBJET dans le champ, PRES"),
    (15, "absent", "SORS l'objet du champ"),
)

# Repères d'affichage. Le texte porte déjà la consigne : la couleur ne fait que
# la doubler, pour être lisible du coin de l'œil pendant qu'on manipule l'objet.
COULEURS = {
    "absent": (90, 170, 90),
    "loin": (60, 170, 230),
    "proche": (70, 70, 220),
}

PREPARATION = 5.0   # secondes avant que l'enregistrement ne commence


def _bandeau(image, condition, consigne, restant, total_restant):
    """Incruste consigne et décompte. Renvoie une copie : l'original est écrit tel quel."""
    vue = image.copy()
    h, w = vue.shape[:2]
    couleur = COULEURS.get(condition, (200, 200, 200))

    cv2.rectangle(vue, (0, 0), (w, 96), (20, 20, 20), -1)
    cv2.rectangle(vue, (0, 0), (w, 8), couleur, -1)
    cv2.putText(vue, consigne, (18, 52), cv2.FONT_HERSHEY_SIMPLEX, 1.0, couleur, 2)
    cv2.putText(vue, "%2ds" % int(restant + 0.999), (w - 120, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (240, 240, 240), 2)
    cv2.putText(vue, "reste %d:%02d au total" % (total_restant // 60, total_restant % 60),
                (18, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1)

    # Barre de progression du segment courant.
    duree = next(d for d, c, t in SEGMENTS if t == consigne) if consigne else 1
    cv2.rectangle(vue, (0, h - 10), (int(w * (1 - restant / max(duree, 1e-6))), h), couleur, -1)
    return vue


def _cadence_reelle(capture, annonce, frames=60):
    """Cadence effective, mesurée sur les premières frames."""
    t0 = time.perf_counter()
    lues = 0
    for _ in range(frames):
        if not capture.read()[0]:
            break
        lues += 1
    ecoule = time.perf_counter() - t0
    if lues < 10 or ecoule <= 0:
        return annonce
    mesure = lues / ecoule
    if abs(mesure - annonce) > 0.15 * annonce:
        print("  cadence annoncee {:.0f} fps, mesuree {:.0f} : on garde la mesuree."
              .format(annonce, mesure))
    return round(mesure)


@click.command()
@click.option("--nom", required=True, help="Nom du clip, sans extension.")
@click.option("--source", default="0", show_default=True,
              help="Index de webcam, ou fichier vidéo pour répéter à blanc.")
@click.option("--out", default="data/protocole", show_default=True)
@click.option("--rapide", is_flag=True,
              help="Divise toutes les durées par 6 : pour vérifier le montage.")
@click.option("--enrolement-par-s", default=4, show_default=True,
              help="Images par seconde extraites du premier segment pour la banque. "
                   "Deux images consécutives n'apprennent rien de neuf.")
@click.option("--show/--no-show", default=True, show_default=True)
def main(nom, source, out, rapide, enrolement_par_s, show):
    facteur = 1 / 6 if rapide else 1.0
    segments = [(d * facteur, c, t) for d, c, t in SEGMENTS]
    total = sum(d for d, _, _ in segments)

    capture = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not capture.isOpened():
        raise SystemExit("Impossible d'ouvrir la source '{}'.".format(source))
    fichier = not source.isdigit()
    largeur = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    hauteur = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_nominal = capture.get(cv2.CAP_PROP_FPS) or 30.0

    os.makedirs(out, exist_ok=True)
    chemin = os.path.join(out, nom + ".mp4")
    # Le fps annoncé n'engage pas la caméra : la mienne annonce 15 et en délivre
    # 30, ce qui encode un fichier joué au ralenti et décale toute la timeline.
    # On le mesure sur quelques frames avant d'ouvrir l'encodeur.
    fps = fps_nominal if fichier else _cadence_reelle(capture, fps_nominal)
    writer = cv2.VideoWriter(chemin, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (largeur, hauteur))

    print("\n  {} — {:.0f} s, {}x{} à {:.0f} fps".format(chemin, total, largeur, hauteur, fps_nominal))
    print("  La camera ne doit PAS bouger de tout le clip : un mouvement de camera")
    print("  se confondrait avec l'apparition de l'objet.\n")
    depart = 0.0
    for duree, condition, consigne in segments:
        print("   {:>5.0f} s  {:<8s} {}".format(depart, condition, consigne))
        depart += duree
    print("   {:>5.0f} s  fin\n".format(depart))

    if show:
        print("  q pour abandonner. Preparation...\n")
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < PREPARATION * facteur:
            ok, frame = capture.read()
            if not ok:
                break
            restant = PREPARATION * facteur - (time.perf_counter() - t0)
            vue = frame.copy()
            cv2.putText(vue, "PRET DANS %d" % int(restant + 0.999), (18, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (240, 240, 240), 3)
            cv2.imshow("protocole", vue)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                capture.release(); cv2.destroyAllWindows()
                raise SystemExit("Abandon avant le depart.")

    dossier_enrolement = os.path.join(out, nom + "_enrolement", "normal")
    os.makedirs(dossier_enrolement, exist_ok=True)
    enroles = 0

    labels, frames, debut = [], 0, time.perf_counter()
    abandon = False
    for index, (duree, condition, consigne) in enumerate(segments):
        premier_absent = index == 0
        t_seg = time.perf_counter()
        frame0 = frames
        print("  -> {}".format(consigne))
        while time.perf_counter() - t_seg < duree:
            ok, frame = capture.read()
            if not ok:
                # Une webcam ne s'arrête pas ; un fichier, si. En répétition on
                # le reboucle pour que le déroulé dure ce qu'il doit durer.
                if fichier:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                abandon = True
                break
            if fichier:
                # Sans horloge, un fichier défile à la vitesse du disque et les
                # segments ne durent plus rien : on le cadence sur son propre fps.
                attendu = debut + frames / max(fps_nominal, 1e-6)
                avance = attendu - time.perf_counter()
                if avance > 0:
                    time.sleep(min(avance, 0.05))
            writer.write(frame)          # la scène seule, sans incrustation
            # Le premier segment sert aussi de jeu d'enrôlement : c'est la même
            # scène, filmée avant que l'objet n'entre, donc le « normal » exact.
            if premier_absent and enrolement_par_s > 0:
                pas = max(1, int(round(fps_nominal / enrolement_par_s)))
                if frames % pas == 0:
                    cv2.imwrite(os.path.join(dossier_enrolement,
                                             "{:05d}.jpg".format(frames)), frame)
                    enroles += 1
            frames += 1
            if show:
                restant = duree - (time.perf_counter() - t_seg)
                total_restant = int(total - (time.perf_counter() - debut))
                cv2.imshow("protocole",
                           _bandeau(frame, condition, consigne, restant, max(total_restant, 0)))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    abandon = True
                    break
        labels.append({
            "condition": condition,
            "consigne": consigne,
            "t0": round(t_seg - debut, 3),
            "t1": round(time.perf_counter() - debut, 3),
            "frame0": frame0,
            "frame1": frames,
        })
        if abandon:
            break

    ecoule = time.perf_counter() - debut
    capture.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()

    meta = {
        "nom": nom,
        "clip": chemin,
        "largeur": largeur, "hauteur": hauteur,
        "fps_nominal": round(fps_nominal, 3),
        "fps_conteneur": round(fps, 3),
        # Une webcam ne tient pas son fps annoncé : c'est le mesuré qui convertit
        # un numéro de frame en seconde, donc qui aligne les labels sur les scores.
        "fps_mesure": round(frames / ecoule, 3) if ecoule else 0.0,
        "frames": frames,
        "duree_s": round(ecoule, 3),
        "fenetre_calibration_s": 20,
        "complet": not abandon,
        "enrolement": os.path.dirname(dossier_enrolement),
        "images_enrolement": enroles,
        "segments": labels,
    }
    chemin_labels = os.path.join(out, nom + ".labels.json")
    with open(chemin_labels, "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)

    print("\n  {} frames en {:.1f} s ({:.1f} fps mesures)".format(
        frames, ecoule, meta["fps_mesure"]))
    print("  {} images d'enrolement dans {}".format(enroles, os.path.dirname(dossier_enrolement)))
    print("  {}\n  {}{}\n".format(chemin, chemin_labels,
                                  "" if not abandon else "   [INCOMPLET]"))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
