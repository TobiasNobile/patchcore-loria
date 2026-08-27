"""Ce que le serveur live calcule sans caméra : le rendu et ses bornes.

La boucle de scoring, elle, demande une source et une banque : elle est
vérifiée de bout en bout à la main, pas ici.
"""

import numpy as np

from live import server


# ─── Le seuil d'affichage ───────────────────────────────────────────────────

def test_sous_le_seuil_l_image_reste_nue():
    """Une découpe, pas un fondu : sous `seuil × vmax`, rien n'est peint.

    C'est ce que l'exposant d'opacité d'avant ne savait pas faire — il laissait
    partout un voile, d'autant plus visible que le fond nominal était large.
    """
    frame = np.full((4, 4, 3), 130, dtype=np.uint8)
    # Deux moitiés : 30 % de vmax en haut, 90 % en bas.
    heatmap = np.array([[3.0] * 4, [3.0] * 4, [9.0] * 4, [9.0] * 4], dtype=np.float32)

    sortie = server.overlay_heatmap(frame, heatmap, 0.0, 10.0, 0.7)

    assert (sortie[:2] == 130).all(), "sous le seuil, la vignette doit rester intacte"
    assert not (sortie[2:] == 130).any(), "au-dessus, la couleur doit couvrir"


def test_seuil_nul_peint_toute_la_carte():
    frame = np.full((2, 2, 3), 130, dtype=np.uint8)
    heatmap = np.zeros((2, 2), dtype=np.float32)

    assert not (server.overlay_heatmap(frame, heatmap, 0.0, 10.0, 0.0) == 130).any()
    # Le même fond, au moindre seuil non nul, disparaît.
    assert (server.overlay_heatmap(frame, heatmap, 0.0, 10.0, 0.02) == 130).all()


def test_le_seuil_est_borne():
    assert server.clamp_seuil(-1.0) == 0.0
    assert server.clamp_seuil(4.0) == 1.0
    assert server.clamp_seuil(0.7) == 0.7


# ─── La fiche de banque affichée par le bandeau ─────────────────────────────

def test_la_fiche_porte_l_echelle_mesuree():
    cfg = {
        "backbone_name": "resnet18", "layers_to_extract_from": ["layer2", "layer3"],
        "sampler_name": "approx_greedy_coreset", "coreset_pct": 0.05,
        "memory_bank_size": 1097, "n_train_images": 28,
        "vmax_holdout": {"vmax": 73.06, "n_images": 7},
    }
    fiche = server.bank_summary(cfg, "/tmp/x.pkg", "x", stored=False)

    assert fiche["vmax"] == 73.06 and fiche["vmax_images"] == 7
    assert fiche["layers"] == "l2-l3" and fiche["coreset"] == "p0.05"
    assert fiche["stored"] is False


def test_une_banque_d_avant_n_a_pas_d_echelle():
    """Le champ manque : la page doit pouvoir retomber sur sa table."""
    fiche = server.bank_summary({"backbone_name": "resnet18"}, "/tmp/y.pkg", "y")

    assert fiche["vmax"] is None and fiche["vmax_images"] is None
    assert fiche["stored"] is True
