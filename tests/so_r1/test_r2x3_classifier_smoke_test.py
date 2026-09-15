from pathlib import Path

from src.skin_optics_so_r1.realface_classifier_smoke_test import _normalise, _select_batch


def test_smoke_split_is_fixed_and_balanced():
    import pandas as pd
    root = Path(__file__).resolve().parents[2]
    frame = pd.read_csv(root / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv", usecols=["ID", "fold", "binary_label", "SEX"])
    train, validation, batch = _select_batch(frame)
    assert len(train) == 320 and len(validation) == 80 and len(batch) == 16
    assert set(batch.binary_label) == {0, 1}


def test_smoke_normalization_contract():
    import torch
    x = torch.ones(2, 5, 3, 3)
    y = _normalise(x, "RGB_B1MH")
    assert y.shape == x.shape and torch.isfinite(y).all()

