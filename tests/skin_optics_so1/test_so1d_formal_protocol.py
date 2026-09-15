from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch
import yaml

from skin_optics_so1.decomposition.formal_protocol import (
    FORBIDDEN_SPLITS,
    FORMAL_PROTOCOL_VERSION,
    build_formal_identity,
    file_sha256,
    formal_resume_signature,
    split_access_payload,
    validate_formal_config,
    verify_history_continuity,
)
from skin_optics_so1.decomposition.metrics import PredictionMonitoringAggregator


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config/train/skin_optics_so1/so1d_formal_train_v1.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def test_so1d_formal_config_is_exactly_frozen() -> None:
    config = load_config()
    validate_formal_config(config)
    assert config["protocol_version"] == FORMAL_PROTOCOL_VERSION
    assert config["scheduler"]["T_max"] == config["training"]["max_epochs"] == 30
    assert config["training"] == {"batch_size": 8, "amp": True, "max_epochs": 30}
    assert config["early_stopping"]["patience"] == 5
    assert config["checkpoint"]["selection_metric"] == "val_M_MAE_plus_val_H_MAE"
    assert config["data"]["forbidden_splits"] == FORBIDDEN_SPLITS
    assert len(file_sha256(CONFIG_PATH)) == 64


def test_formal_config_rejects_protocol_changes() -> None:
    config = load_config()
    config["training"]["batch_size"] = 4
    with pytest.raises(ValueError, match="training"):
        validate_formal_config(config)


def test_formal_resume_signature_covers_training_protocol() -> None:
    config = load_config()
    original = formal_resume_signature(config)
    config["loss"]["beta"] = 0.2
    assert formal_resume_signature(config) != original


def test_formal_identity_contains_full_resume_provenance() -> None:
    identity = build_formal_identity(contract_hash="contract", config=load_config())
    assert identity["dataset_contract_hash"] == "contract"
    assert identity["SO0_version"] == "SO0_Forward_Model_v1.1"
    assert identity["SO0_config_hash"]
    assert identity["normalization"]["P_norm"] == "P / 0.10"
    assert len(identity["formal_resume_signature"]) == 64


def test_split_access_audit_never_marks_forbidden_splits() -> None:
    audit = split_access_payload(train_accessed=True, validation_accessed=True)
    assert audit["status"] == "PASS"
    assert all(not audit[split]["accessed"] for split in FORBIDDEN_SPLITS)
    assert audit["forbidden_splits_accessed"] == []


def test_history_resume_requires_continuous_epochs(tmp_path: Path) -> None:
    path = tmp_path / "training_history.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerows([{"epoch": 1}, {"epoch": 2}])
    assert verify_history_continuity(path, checkpoint_epoch=2) == [1, 2]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch"])
        writer.writeheader()
        writer.writerows([{"epoch": 1}, {"epoch": 3}])
    with pytest.raises(RuntimeError, match="REFUSE_RESUME"):
        verify_history_continuity(path, checkpoint_epoch=3)


def test_prediction_monitoring_uses_only_valid_pixels() -> None:
    prediction = torch.tensor(
        [[[[0.0, 1.0]]], [[[0.2, 0.8]]], [[[0.3, 0.7]]], [[[0.0, 0.5]]]],
        dtype=torch.float32,
    ).permute(1, 0, 2, 3)
    target = prediction.clone()
    mask = torch.tensor([[[[1.0, 0.0]]]])
    metrics = PredictionMonitoringAggregator()
    metrics.update(prediction, target, mask)
    result = metrics.compute()
    assert result["M_pred_mean"] == pytest.approx(0.0)
    assert result["H_pred_mean"] == pytest.approx(0.2)
    assert result["M_pred_le_0.01_fraction"] == pytest.approx(1.0)
    assert result["P_active_recall"] != result["P_active_recall"]
