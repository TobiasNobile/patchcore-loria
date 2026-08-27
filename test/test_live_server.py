"""La phase de test du mode auto-calibré, sans caméra.

La boucle de scoring est remplacée par une fonction qui rend la mesure qu'on lui
dicte : ce qui est vérifié ici, c'est ce qu'on en fait — la garder, l'appliquer,
ou ne rien garder du tout.
"""

import pytest

from live import server


@pytest.fixture
def runner():
    return server.Runner()


def test_le_quantile_du_test_devient_le_vmax(runner, monkeypatch):
    """C'est le p90 qui est appliqué, pas le pic : le pic n'est que montré."""
    monkeypatch.setattr(runner, "_run",
                        lambda params: {"vmax": 121.5, "max": 137.5, "n": 200})

    suite = runner._tester_puis_calibrer({"vmax": 20.0, "bank_vmax": 61.3})

    assert suite["vmax"] == 121.5
    # Posé pour toute la démo qui suit : la page le lit quand elle veut.
    assert runner.state()["calib_kept"] == 121.5


def test_le_coefficient_multiplie_la_mesure(runner, monkeypatch):
    """La marge s'applique à l'échelle mesurée, et à elle seule.

    `calib_kept` reste la mesure brute : c'est ce que la page multiplie de son
    côté pour afficher le calcul, et les deux doivent parler du même nombre.
    """
    monkeypatch.setattr(runner, "_run",
                        lambda params: {"vmax": 100.0, "max": 130.0, "n": 300})

    suite = runner._tester_puis_calibrer({"vmax": 20.0, "vmax_coef": 0.8})

    assert suite["vmax"] == pytest.approx(80.0)
    assert runner.state()["calib_kept"] == 100.0


def test_le_coefficient_est_borne():
    """Hors bornes, la marge n'ajuste plus une mesure, elle l'invente."""
    assert server.clamp_coef(0.0) == server.VMAX_COEF_MIN
    assert server.clamp_coef(9.0) == server.VMAX_COEF_MAX
    assert server.clamp_coef(0.8) == 0.8


def test_les_reglages_du_test_suivent_dans_la_demo(runner, monkeypatch):
    """Le zoom réglé en présentant l'anomalie vaut aussi pour ce qui suit.

    Sinon l'image changerait entre la calibration et ce qu'elle calibre.
    """
    def _run(params):
        # Réglé en cours de test, comme la page le fait : avant, `_preparer_run`
        # repose les réglages sur ceux du formulaire.
        runner.update_live({"zoom": 2.5, "stride": 4, "smoothing": "max"})
        return {"vmax": 90.0, "max": 99.0, "n": 42}

    monkeypatch.setattr(runner, "_run", _run)

    suite = runner._tester_puis_calibrer({"vmax": 20.0, "zoom": 1.0, "stride": 1})

    assert suite["zoom"] == 2.5
    assert suite["stride"] == 4
    assert suite["smoothing"] == "max"


def test_un_test_sans_score_laisse_le_vmax_du_fit(runner, monkeypatch):
    """Bouton pressé avant la première inférence : rien à garder."""
    monkeypatch.setattr(runner, "_run", lambda params: None)

    suite = runner._tester_puis_calibrer({"vmax": 20.0})

    assert suite["vmax"] == 20.0
    assert runner.state()["calib_kept"] is None


def test_un_test_abandonne_ne_garde_rien(runner, monkeypatch):
    """Arrêter n'est pas Terminer : la séquence tombe, l'échelle avec elle."""

    def _run(params):
        runner._stop.set()      # ce que fait /api/stop pendant la boucle
        return {"vmax": 121.5, "max": 137.5, "n": 200}

    monkeypatch.setattr(runner, "_run", _run)

    with pytest.raises(KeyboardInterrupt):
        runner._tester_puis_calibrer({"vmax": 20.0})
    assert runner.state()["calib_kept"] is None


def test_stop_sort_aussi_d_une_phase_de_test(runner):
    """Une phase de test attend son bouton : `stop` doit poser les deux drapeaux,
    sinon la boucle ne rendrait jamais la main."""
    runner.stop()

    assert runner._stop.is_set()
    assert runner._fin_test.is_set()


def test_end_test_ne_touche_pas_a_l_arret(runner):
    runner.end_test()

    assert runner._fin_test.is_set()
    assert not runner._stop.is_set()


# ─── Le seuil d'affichage ───────────────────────────────────────────────────

def test_sous_le_seuil_l_image_reste_nue():
    """Une découpe, pas un fondu : sous `seuil × vmax`, rien n'est peint.

    C'est ce que l'exposant d'opacité d'avant ne savait pas faire — il laissait
    partout un voile, d'autant plus visible que le fond nominal était large.
    """
    import numpy as np

    frame = np.full((4, 4, 3), 130, dtype=np.uint8)
    # Deux moitiés : 30 % de vmax en haut, 90 % en bas.
    heatmap = np.array([[3.0] * 4, [3.0] * 4, [9.0] * 4, [9.0] * 4], dtype=np.float32)

    sortie = server.overlay_heatmap(frame, heatmap, 0.0, 10.0, 0.7)

    assert (sortie[:2] == 130).all(), "sous le seuil, la vignette doit rester intacte"
    assert not (sortie[2:] == 130).any(), "au-dessus, la couleur doit couvrir"


def test_seuil_nul_peint_toute_la_carte():
    import numpy as np

    frame = np.full((2, 2, 3), 130, dtype=np.uint8)
    heatmap = np.zeros((2, 2), dtype=np.float32)

    assert not (server.overlay_heatmap(frame, heatmap, 0.0, 10.0, 0.0) == 130).any()
    # Le même fond, au moindre seuil non nul, disparaît.
    assert (server.overlay_heatmap(frame, heatmap, 0.0, 10.0, 0.02) == 130).all()


def test_le_seuil_est_borne():
    assert server.clamp_seuil(-1.0) == 0.0
    assert server.clamp_seuil(4.0) == 1.0
    assert server.clamp_seuil(0.7) == 0.7
