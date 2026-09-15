from pathlib import Path

import pandas as pd
import torch

from src.skin_optics_so_r1 import realface_mh_only_ablation as mh


def test_four_locked_mh_only_conditions_and_no_rgb_source():
    assert mh.CONDITIONS == {
        "B1_MH_ONLY": ("rgb_b1mh", (3, 4)),
        "B2_MH_ONLY": ("rgb_b2mh", (3, 4)),
        "B2_M_ONLY": ("rgb_b2mh", (3,)),
        "B2_H_ONLY": ("rgb_b2mh", (4,)),
    }
    source = Path(mh.__file__).read_text(encoding="utf-8")
    assert '"rgb"' not in source
    assert "mmap_mode=\"r\"" in source and "source[list(self.indices)]" in source
    assert "_mh_channel_sha" in source


def test_nested_direct_split_is_320_80_and_leak_free():
    root = Path(__file__).resolve().parents[2]
    table = pd.read_csv(root / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv", usecols=mh.FIELDS)
    train, validation = mh._inner_split(table[table.fold != 0], 0)
    assert len(train) == 320 and len(validation) == 80
    assert not set(train.ID) & set(validation.ID)


def test_one_and_two_channel_conv1_use_imagenet_channel_mean():
    two = mh._model(2)
    one = mh._model(1)
    assert two.conv1.weight.shape[1] == 2 and one.conv1.weight.shape[1] == 1
    assert torch.allclose(two.conv1.weight[:, 0], two.conv1.weight[:, 1])
    assert two.fc.out_features == one.fc.out_features == 2


def test_fixed_comparisons_are_descriptive_only():
    pooled = pd.DataFrame({"condition": list(mh.CONDITIONS), **{metric: [0.1, 0.2, 0.3, 0.4] for metric in mh.METRICS}})
    fold = pd.concat([pd.DataFrame({"condition": [condition] * 5, "fold": range(5), **{metric: range(5) for metric in mh.METRICS}}) for condition in mh.CONDITIONS], ignore_index=True)
    result = mh._comparison(pooled, fold)
    assert set(result.comparison) == {"B2_MH_ONLY_minus_B1_MH_ONLY", "B2_MH_ONLY_minus_B2_M_ONLY", "B2_MH_ONLY_minus_B2_H_ONLY"}
    assert len(result) == 15


def test_mh_channel_hash_ignores_rgb_but_detects_mh_changes(tmp_path):
    import numpy as np
    path = tmp_path / "five_channel.npy"
    value = np.zeros((5, 320, 256), dtype=np.float32)
    value[3] = 0.25; value[4] = 0.75
    np.save(path, value)
    first = mh._mh_channel_sha(path)
    value[0] = 99.0
    np.save(path, value)
    assert mh._mh_channel_sha(path) == first
    value[3, 0, 0] = 0.5
    np.save(path, value)
    assert mh._mh_channel_sha(path) != first


def test_completed_outputs_have_full_coverage_and_zero_leakage():
    import json
    root = Path(__file__).resolve().parents[2]
    report = root / "reports/so_r2x3_mh_only_ablation"
    acceptance = json.loads((root / "data/processed/SO_R2X3_MHOnlyAblation_v1/MH_ABLATION_ACCEPTANCE.json").read_text(encoding="utf-8"))
    leakage = json.loads((report / "split_leakage_audit.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(report / "oof_predictions.csv", dtype={"ID": str})
    assert acceptance["oof_coverage"] == {condition: 500 for condition in mh.CONDITIONS}
    assert acceptance["outer_test_model_selection_access"] == 0
    assert oof.groupby("condition").size().to_dict() == {condition: 500 for condition in mh.CONDITIONS}
    assert not oof.duplicated(["condition", "ID"]).any()
    assert leakage["all_pass"] is True
    assert all(item["outer_test_model_selection_access"] == 0 for item in leakage["folds"])
