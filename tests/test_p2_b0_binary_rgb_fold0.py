from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from datasets.nyha_3class_face_dataset import build_transforms
from losses.classification_losses import compute_class_weights
from models.nyha_backbone_factory import build_nyha_classification_model
from scripts.evaluate.summarize_p2_b0_binary_rgb_fold0 import validate_fold0_predictions
from utils.experiment_utils import load_yaml
from utils.p2_b0_protocol import (
    PROJECT_ROOT,
    check_p1_ready_case_alignment,
    validate_p2_b0_config,
    verify_fold0_split_protocol,
)


CONFIG_PATH = PROJECT_ROOT / "config" / "p2" / "p2_b0_binary_rgb_fold0_v1.yaml"


def test_p2_b0_config_is_fixed_e0b_protocol():
    config = load_yaml(CONFIG_PATH)
    validate_p2_b0_config(config)
    assert config["data"]["image_root"] == "data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images"
    assert config["task"]["type"] == "binary_control_vs_patient"
    assert config["task"]["num_classes"] == 2
    assert config["p2"]["fixed_fold"] == 0
    assert all(config["p2"][key] is False for key in ("use_p1_features", "use_relighting", "use_auxiliary_features", "use_reliability_gate"))


@pytest.mark.parametrize("section,key,value", [
    ("task", "num_classes", 3), ("model", "num_classes", 3), ("p2", "fixed_fold", 1),
    ("p2", "use_p1_features", True), ("p2", "use_relighting", True),
    ("p2", "use_auxiliary_features", True), ("p2", "use_reliability_gate", True),
])
def test_p2_b0_rejects_protocol_mutations(section, key, value):
    config = copy.deepcopy(load_yaml(CONFIG_PATH))
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_p2_b0_config(config)


def test_fold0_counts_groups_weights_and_p1_exact_alignment(tmp_path):
    config = load_yaml(CONFIG_PATH)
    counts = verify_fold0_split_protocol(config)
    assert counts["train_control"] == 92 and counts["train_patient"] == 308
    assert counts["val_control"] == 23 and counts["val_patient"] == 77
    split = PROJECT_ROOT / config["data"]["split_dir"]
    train = pd.read_csv(split / "fold_0_train.csv", dtype={"patient_group_id": "string"})
    val = pd.read_csv(split / "fold_0_val.csv", dtype={"patient_group_id": "string"})
    assert not set(train["patient_group_id"]).intersection(val["patient_group_id"])
    labels = [0 if x == 0 else 1 for x in train["label_3class"].astype(int)]
    assert np.allclose(compute_class_weights(labels, 2).numpy(), [400 / 184, 400 / 616])
    alignment = check_p1_ready_case_alignment(config, tmp_path)
    assert alignment["status"] == "passed"
    assert alignment["p1_rows"] == alignment["fixed_oof_unique_ids"] == 500
    assert alignment["p1_artifacts_read"] == []


def test_transforms_and_model_are_binary_without_validation_augmentation():
    train_tf = build_transforms("train", horizontal_flip=True)
    val_tf = build_transforms("val", horizontal_flip=False)
    assert any(transform.__class__.__name__ == "RandomHorizontalFlip" for transform in train_tf.transforms)
    assert not any(transform.__class__.__name__ == "RandomHorizontalFlip" for transform in val_tf.transforms)
    model = build_nyha_classification_model("resnet18", num_classes=2, pretrained=False, freeze_backbone=False)
    with torch.no_grad():
        assert tuple(model(torch.zeros(2, 3, 224, 224)).shape) == (2, 2)


def _expected_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"], "expected_patient_group_id": ["ga", "gb", "gc", "gd"],
        "NYHA": [0, 1, 0, 3], "label_3class": [0, 1, 0, 2], "expected_binary_label": [0, 1, 0, 1],
    })


def _prediction_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "sample_id": ["a", "b", "c", "d"], "patient_group_id": ["ga", "gb", "gc", "gd"], "fold": [0] * 4,
        "original_label": [0, 1, 0, 3], "original_three_class_label": [0, 1, 0, 2], "binary_label": [0, 1, 0, 1],
        "logit_normal": [1.0, 0.0, 1.0, 0.0], "logit_patient": [0.0, 1.0, 0.0, 1.0],
        "prob_normal": [.9, .2, .8, .1], "prob_patient": [.1, .8, .2, .9], "pred_class": [0, 1, 0, 1],
        "image_path": ["a.png", "b.png", "c.png", "d.png"], "selected_epoch": [1] * 4, "checkpoint_path": ["x.pth"] * 4,
    })


def test_single_fold_summary_rejects_duplicate_ids_and_invalid_probabilities():
    expected, predictions = _expected_frame(), _prediction_frame()
    with pytest.raises(ValueError, match="100 unique"):
        validate_fold0_predictions(predictions, expected)
    # The production validator is intentionally fixed to a 100-case fold.  Build
    # a 100-case valid fixture to exercise AUC-from-prob_patient and rejection paths.
    expected = pd.concat([expected.assign(sample_id=lambda x, n=n: x["sample_id"] + str(n), expected_patient_group_id=lambda x, n=n: x["expected_patient_group_id"] + str(n)) for n in range(25)], ignore_index=True)
    predictions = pd.concat([predictions.assign(sample_id=lambda x, n=n: x["sample_id"] + str(n), patient_group_id=lambda x, n=n: x["patient_group_id"] + str(n)) for n in range(25)], ignore_index=True)
    assert validate_fold0_predictions(predictions, expected)["metrics"]["macro_auc"] == 1.0
    duplicate = predictions.copy(); duplicate.loc[1, "sample_id"] = duplicate.loc[0, "sample_id"]
    with pytest.raises(ValueError):
        validate_fold0_predictions(duplicate, expected)
    invalid = predictions.copy(); invalid.loc[0, "prob_normal"] = .8; invalid.loc[0, "prob_patient"] = .8
    with pytest.raises(ValueError, match="sum"):
        validate_fold0_predictions(invalid, expected)
