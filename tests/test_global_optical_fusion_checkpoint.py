from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from scripts.train import train_global_optical_fusion_5fold as train_script
from trainers.global_optical_fusion_trainer import (
    capture_rng_state,
    load_torch_checkpoint,
    make_data_generator,
    restore_rng_state,
    validate_checkpoint_metadata,
)
from utils.experiment_utils import load_yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config/train/global_optical_fusion/global_resnet18_optical_fusion.yaml"


def test_training_does_not_require_git(monkeypatch):
    def missing_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(train_script.subprocess, "run", missing_git)
    assert train_script._optional_git_commit() is None


def _metadata():
    return {
        "variant": "global_raw", "fold": 0,
        "feature_names": ["f1", "f2", "forehead_available"],
        "feature_scaler_sha256": "scaler-hash", "split_sha256": "split-hash",
        "feature_source_sha256": {"train": "train-hash", "val": "val-hash"},
        "feature_schema_sha256": "schema-hash",
        "upstream_manifest_sha256": "manifest-hash",
        "train_id_sha256": "train-id-hash", "val_id_sha256": "val-id-hash",
        "config_sha256": "config-hash", "implementation_signature": "code-hash",
        "auxiliary_input_dim": 7, "fused_input_dim": 519,
    }


def test_checkpoint_model_optimizer_and_predictions_restore(tmp_path):
    torch.manual_seed(10)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(5, 4)
    loss = model(inputs).sum()
    loss.backward()
    optimizer.step()
    expected = model(inputs).detach().clone()
    payload = {
        **_metadata(), "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(), "epoch": 4,
        "best_epoch": 3, "best_macro_auc": 0.7, "patience_counter": 1,
    }
    path = tmp_path / "checkpoint.pth"
    torch.save(payload, path)
    restored_model = nn.Linear(4, 3)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    checkpoint = load_torch_checkpoint(path)
    validate_checkpoint_metadata(checkpoint, _metadata())
    restored_model.load_state_dict(checkpoint["model_state_dict"])
    restored_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    assert torch.equal(restored_model(inputs), expected)
    assert checkpoint["epoch"] == 4
    assert checkpoint["patience_counter"] == 1
    assert restored_optimizer.state_dict()["state"]


@pytest.mark.parametrize(
    "key", [
        "variant", "feature_names", "feature_scaler_sha256", "split_sha256",
        "feature_source_sha256", "config_sha256", "implementation_signature",
        "train_id_sha256", "val_id_sha256",
    ]
)
def test_checkpoint_identity_mismatch_is_rejected(key):
    checkpoint = _metadata()
    expected = _metadata()
    expected[key] = "different"
    with pytest.raises(ValueError, match="mismatch"):
        validate_checkpoint_metadata(checkpoint, expected)


def test_rng_and_loader_generator_restore():
    loader = DataLoader(
        TensorDataset(torch.arange(20)), batch_size=4, shuffle=True,
        generator=make_data_generator(123),
    )
    torch.manual_seed(7)
    np.random.seed(7)
    state = capture_rng_state(loader)
    expected_torch = torch.rand(4)
    expected_numpy = np.random.rand(4)
    expected_order = [batch[0].tolist() for batch in loader]
    restore_rng_state(state, loader)
    assert torch.equal(torch.rand(4), expected_torch)
    assert np.array_equal(np.random.rand(4), expected_numpy)
    assert [batch[0].tolist() for batch in loader] == expected_order


def test_resume_loads_persisted_scaler_without_refit(tmp_path, monkeypatch):
    config = load_yaml(CONFIG_PATH)
    train_split = PROJECT_ROOT / "data/processed/splits_500/fold_0_train.csv"
    val_split = PROJECT_ROOT / "data/processed/splits_500/fold_0_val.csv"
    train_ids = pd.read_csv(
        train_split, dtype={"ID": "string"}
    )["ID"].astype(str).tolist()
    val_ids = pd.read_csv(
        val_split, dtype={"ID": "string"}
    )["ID"].astype(str).tolist()
    _, _, fitted, _ = train_script._prepare_features(
        config, "global_raw", 0, train_ids, val_ids, train_split, CONFIG_PATH
    )
    assert fitted is not None
    scaler_path = tmp_path / "feature_scaler.json"
    fitted.save_json(scaler_path)
    original_hash = fitted.payload_sha256

    def fail_if_refit(*args, **kwargs):
        raise AssertionError("resume attempted to refit the scaler")

    monkeypatch.setattr(train_script, "fit_feature_scaler", fail_if_refit)
    _, _, restored, _ = train_script._prepare_features(
        config, "global_raw", 0, train_ids, val_ids, train_split, CONFIG_PATH,
        existing_scaler_path=scaler_path,
    )
    assert restored is not None
    assert restored.payload_sha256 == original_hash
