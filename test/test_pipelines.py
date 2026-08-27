"""La calibration de l'échelle : ce qu'elle mesure, et ce qu'elle refuse.

Sans modèle ni banque — une instance factice rend les scores qu'on lui dicte.
Ce qui est testé ici n'est pas la valeur d'un score, c'est le choix des images
qui la produisent : le nominal hors banque, et lui seul.
"""

import numpy as np
import pytest
import torch

from experiments.pipelines import Spec, holdout_vmax

CFG = {"resize": 8, "imagesize": 8}


class _Dataset(torch.utils.data.Dataset):
    """Un split TEST : du holdout nominal, et d'éventuels contre-exemples."""

    def __init__(self, labels):
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            "image": torch.zeros(3, 8, 8),
            "mask": torch.zeros(1, 8, 8),
            "is_anomaly": self.labels[index],
            # L'indice voyage avec l'image : c'est ce qui permet à l'instance
            # factice de dire *quelles* images lui ont été soumises.
            "index": index,
        }


class _Instance:
    """Rend un score par image, pris dans une table indexée comme le dataset."""

    def __init__(self, scores):
        self._scores = scores
        self.vus = []

    def predict(self, loader):
        for batch in loader:
            self.vus.extend(int(i) for i in batch["index"])
        scores = [self._scores[i] for i in self.vus]
        # La carte vaut la moitié du score : le flou rabaisse le pic, et on
        # vérifie au passage que les deux grandeurs ne sont pas confondues.
        cartes = [np.full((8, 8), s / 2, dtype=np.float32) for s in scores]
        return scores, cartes, [], []


def _spec(dataset):
    return Spec(name="factice", build=lambda **kw: dataset,
                normal="Normal", anomaly="Anomalie")


def _calibrer(instance, dataset, num_workers=0):
    return holdout_vmax(instance, _spec(dataset), CFG, seed=0,
                        device=torch.device("cpu"), extra_config=None,
                        num_workers=num_workers)


def test_vmax_est_le_plus_grand_score_nominal():
    dataset = _Dataset([0, 0, 0, 0])
    stats = _calibrer(_Instance([1.0, 7.5, 3.0, 2.0]), dataset)

    assert stats["vmax"] == pytest.approx(7.5)
    assert stats["n_images"] == 4
    assert stats["score_median"] == pytest.approx(2.5)
    assert stats["heatmap_max"] == pytest.approx(3.75)


def test_les_anomalies_ne_font_pas_l_echelle():
    """Un anomaly/ dans le zip rejoint le holdout dans le split TEST.

    Le scorer sur du 900 ferait une échelle où plus rien ne ressort : c'est
    l'erreur que ce test interdit.
    """
    dataset = _Dataset([0, 1, 0, 1])
    instance = _Instance([2.0, 900.0, 5.0, 900.0])
    stats = _calibrer(instance, dataset)

    assert instance.vus == [0, 2]
    assert stats["vmax"] == pytest.approx(5.0)
    assert stats["n_images"] == 2


def test_sans_nominal_hors_banque_pas_d_echelle():
    """Le fit continue sans : le vmax reste ce qu'il était, un réglage manuel."""
    assert _calibrer(_Instance([900.0]), _Dataset([1])) is None
    assert _calibrer(_Instance([]), _Dataset([])) is None


def test_calibration_coupee_par_l_environnement(monkeypatch):
    monkeypatch.setenv("FIT_CALIB_IMAGES", "0")
    assert _calibrer(_Instance([1.0]), _Dataset([0])) is None


def test_plafond_du_nombre_d_images(monkeypatch):
    monkeypatch.setenv("FIT_CALIB_IMAGES", "3")
    instance = _Instance([float(i) for i in range(10)])
    stats = _calibrer(instance, _Dataset([0] * 10))

    assert stats["n_images"] == 3
    assert len(instance.vus) == 3
    # Tirage sous le seed, donc reproductible d'un fit à l'autre.
    assert instance.vus == sorted(instance.vus)


def test_un_dataset_illisible_ne_tue_pas_le_fit():
    """Le split TEST peut manquer (dossier de transit déjà nettoyé, par ex.)."""

    def build(**kw):
        raise FileNotFoundError("normal/ introuvable")

    spec = Spec(name="factice", build=build, normal="Normal", anomaly="Anomalie")
    assert holdout_vmax(_Instance([]), spec, CFG, seed=0,
                        device=torch.device("cpu"), extra_config=None,
                        num_workers=0) is None
