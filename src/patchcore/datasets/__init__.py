"""Le découpage commun aux datasets one-class : les images normales alimentent
la banque (TRAIN), l'évaluation mélange normal et anomalie (TEST).

Un seul Enum partagé, et non un par dataset : `split == DatasetSplit.TRAIN` doit
rester vrai quand le split traverse deux modules (FolderDataset le reçoit de
experiments/datasets.py). MVTec garde le sien, qui a un VAL en plus.
"""

from enum import Enum


class DatasetSplit(Enum):
    TRAIN = "train"
    TEST = "test"
