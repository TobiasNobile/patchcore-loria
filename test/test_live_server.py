"""Ce que le serveur live calcule sans caméra : le rendu et ses bornes.

La boucle de scoring, elle, demande une source et une banque : elle est
vérifiée de bout en bout à la main, pas ici.
"""

import cv2
import numpy as np

from live import scoring, server


# ─── La normalisation et le canal alpha ─────────────────────────────────────

def test_la_normalisation_est_ce_qui_depasse_vmax():
    """max(heatmap / vmax - 1, 0) : nul sous vmax, l'écart relatif au-dessus."""
    heatmap = np.array([[5.0, 10.0], [15.0, 20.0]], dtype=np.float32)

    sortie = scoring.normalize_heatmap(heatmap, 10.0)

    assert sortie.tolist() == [[0.0, 0.0], [0.5, 1.0]]


def test_sous_vmax_l_image_reste_nue():
    frame = np.full((4, 4, 3), 130, dtype=np.uint8)
    # Deux moitiés : sous vmax en haut, au double de vmax en bas.
    heatmap = np.array([[9.0] * 4, [9.0] * 4, [20.0] * 4, [20.0] * 4], dtype=np.float32)

    sortie = scoring.overlay_heatmap(frame, heatmap, 10.0, 1.0)

    assert (sortie[:2] == 130).all(), "sous vmax, la vignette doit rester intacte"
    assert not (sortie[2:] == 130).any(), "au-dessus, la couleur doit couvrir"


def test_l_exposant_regle_l_opacite():
    """Même carte, deux exposants : le plus petit peint plus fort."""
    frame = np.full((2, 2, 3), 130, dtype=np.uint8)
    heatmap = np.full((2, 2), 12.0, dtype=np.float32)   # 0,2 une fois normalisé

    def ecart(img):
        return float(np.abs(img.astype(float) - 130).mean())

    doux = scoring.overlay_heatmap(frame, heatmap, 10.0, 1.0)
    fort = scoring.overlay_heatmap(frame, heatmap, 10.0, 0.2)

    assert ecart(fort) > ecart(doux)


def test_alpha_nul_peint_toute_la_carte():
    frame = np.full((2, 2, 3), 130, dtype=np.uint8)
    heatmap = np.zeros((2, 2), dtype=np.float32)

    assert not (scoring.overlay_heatmap(frame, heatmap, 10.0, 0.0) == 130).any()
    # Le même fond nul, au moindre exposant non nul, disparaît.
    assert (scoring.overlay_heatmap(frame, heatmap, 10.0, 0.02) == 130).all()


def test_le_canal_alpha_se_superpose_aux_trois_canaux():
    """a = normalisé ** alpha, mélangé à la vignette sur B, G et R."""
    frame = np.full((1, 1, 3), 130, dtype=np.uint8)
    heatmap = np.full((1, 1), 15.0, dtype=np.float32)   # normalisé = 0,5
    alpha = 0.5

    sortie = scoring.overlay_heatmap(frame, heatmap, 10.0, alpha)

    a = 0.5 ** alpha
    ramp = np.clip(np.float32([[0.5]]), scoring.COLORMAP_LOW, scoring.COLORMAP_HIGH)
    colored = cv2.applyColorMap((ramp * 255).astype(np.uint8), cv2.COLORMAP_JET)
    attendu = (colored[0, 0] * a + 130 * (1 - a)).astype(np.uint8)

    assert sortie[0, 0].tolist() == attendu.tolist()


def test_l_exposant_est_borne():
    assert server.clamp_alpha(-1.0) == 0.0
    assert server.clamp_alpha(4.0) == 1.0
    assert server.clamp_alpha(0.5) == 0.5


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
    """Le champ manque : la page le dit plutôt que d'inventer une échelle."""
    fiche = server.bank_summary({"backbone_name": "resnet18"}, "/tmp/y.pkg", "y")

    assert fiche["vmax"] is None and fiche["vmax_images"] is None
    assert fiche["stored"] is True
