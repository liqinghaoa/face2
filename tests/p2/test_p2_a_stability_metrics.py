from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from p2_counterfactual.assets import PRESET_NAMES
from p2_counterfactual.stability_metrics import compute_stability_outputs


def test_stability_metrics_known_values(tmp_path: Path) -> None:
    original = pd.DataFrame(
        [
            {"case_id": "a", "patient_group_id": "g1", "fold": 0, "label": 0, "prob_control": 0.8, "prob_patient": 0.2, "predicted_label": 0},
            {"case_id": "b", "patient_group_id": "g2", "fold": 1, "label": 1, "prob_control": 0.2, "prob_patient": 0.8, "predicted_label": 1},
        ]
    )
    rows = []
    for cid, label, group in [("a", 0, "g1"), ("b", 1, "g2")]:
        for idx, preset in enumerate(PRESET_NAMES):
            prob = 0.1 + idx * 0.01 if cid == "a" else 0.9 - idx * 0.01
            rows.append({"case_id": cid, "patient_group_id": group, "fold": label, "label": label, "preset_name": preset, "prob_control": 1 - prob, "prob_patient": prob, "predicted_label": int(prob >= 0.5)})
    relighted = pd.DataFrame(rows)
    np.savez_compressed(tmp_path / "fo.npz", case_ids=np.asarray(["a", "b"]), features=np.ones((2, 512), np.float32))
    np.savez_compressed(tmp_path / "fr.npz", case_ids=np.asarray(["a", "b"]), preset_names=np.asarray(PRESET_NAMES), features=np.ones((2, 6, 512), np.float32))
    out = compute_stability_outputs(
        original_predictions=original,
        relighted_predictions=relighted,
        original_features_npz=tmp_path / "fo.npz",
        relighted_features_npz=tmp_path / "fr.npz",
        output_dir=tmp_path / "summary",
    )
    assert out["metrics"]["case_level_label_flip_rate"] == 0.0
    assert out["metrics"]["worst_light_auc"] == 1.0
    assert abs(out["metrics"]["mean_feature_cosine"] - 1.0) < 1e-6
    assert (tmp_path / "summary/case_level_stability.csv").is_file()
