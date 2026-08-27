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
