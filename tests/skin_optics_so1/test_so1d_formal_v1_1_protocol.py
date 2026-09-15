from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from skin_optics_so1.decomposition.checkpoint import save_checkpoint
from skin_optics_so1.decomposition.formal_protocol_v1_1 import (
    NUMERICAL_POLICY,
    PROTOCOL_VERSION,
    OverflowFrequencyGate,
    audit_resume,
    build_identity,
    validate_config,
)
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer, count_parameters


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config/train/skin_optics_so1/so1d_formal_train_v1_1.yaml"
V10_LAST = ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train/checkpoints/last.pt"
V11_DIR = ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train_v1_1"


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_v1_1_config_is_frozen_and_localized() -> None:
    config = load_config()
    validate_config(config)
    assert config["protocol_version"] == PROTOCOL_VERSION
    assert config["numerical_precision"] == NUMERICAL_POLICY
    assert config["training"]["batch_size"] == 8
    assert config["training"]["max_epochs"] == 30


def test_v1_1_model_topology_and_precision_policy() -> None:
    original = SO1UNetDecomposer()
    model = SO1UNetDecomposer(numerical_precision=NUMERICAL_POLICY)
    assert count_parameters(model) == {
        "parameter_count": 31_037_828,
        "trainable_parameter_count": 31_037_828,
    }
    assert list(original.state_dict()) == list(model.state_dict())
    assert model.up1.conv_fp32 is True
    assert all(not block.conv_fp32 for block in (model.up2, model.up3, model.up4))


def test_overflow_frequency_gate_uses_frozen_thresholds() -> None:
    gate = OverflowFrequencyGate(max_consecutive=5, max_per_100=10)
    assert [gate.update(True) for _ in range(4)] == [False] * 4
    assert gate.update(True)
    gate = OverflowFrequencyGate(max_consecutive=5, max_per_100=10)
    results = [gate.update(index % 10 == 0) for index in range(100)]
    assert results[-1]
    assert gate.maximum_per_100 == 10


def test_overflow_frequency_gate_state_can_be_restored() -> None:
    gate = OverflowFrequencyGate(max_consecutive=5, max_per_100=10)
    for value in (False, True, True):
        gate.update(value)
    restored = OverflowFrequencyGate(max_consecutive=5, max_per_100=10)
    restored.consecutive = gate.consecutive
    restored.window.extend(gate.window)
    restored.maximum_consecutive = gate.maximum_consecutive
    restored.maximum_per_100 = gate.maximum_per_100
    assert restored.consecutive == 2
    assert list(restored.window) == [False, True, True]
    assert not restored.update(True)


def test_v1_1_output_is_isolated_and_v10_hashes_are_historical() -> None:
    assert V11_DIR.resolve() != V10_LAST.parent.parent.resolve()
    assert V10_LAST.is_file()
    import hashlib
    digest = hashlib.sha256(V10_LAST.read_bytes()).hexdigest()
    assert digest == "85e559ac4cb9a3da5e889b57db7fc45974cdc14f8f8d1312479c1a980e755349"


def test_resume_rejects_v10_checkpoint_before_loading(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="only formal_train_v1_1"):
        audit_resume(
            V10_LAST,
            expected_path=V11_DIR / "checkpoints/last.pt",
            config_sha256="unused",
            identity={},
            history_path=tmp_path / "missing.csv",
        )


def test_checkpoint_extra_payload_is_additive_and_collision_safe(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    identity = {
        "target_order": [], "dataset_contract_hash": "d", "generator_config_hash": "g",
        "SO0_version": "s", "SO0_config_hash": "h", "model_architecture": "m",
        "normalization": {}, "formal_resume_signature": "r", "protocol_version": PROTOCOL_VERSION,
    }
    path = tmp_path / "checkpoint.pt"
    kwargs = dict(
        model=model, optimizer=optimizer, scheduler=scheduler, grad_scaler=scaler,
        epoch=1, global_step=1, best_selection_metric=1.0, early_stopping_counter=0,
        identity=identity, resolved_config={}, dataset_contract={}, dataset_fingerprint={}, seed=1,
    )
    save_checkpoint(path, **kwargs, extra_payload={"optimizer_update_count": 1})
    assert torch.load(path, weights_only=False)["optimizer_update_count"] == 1
    with pytest.raises(ValueError, match="overlaps"):
        save_checkpoint(path, **kwargs, extra_payload={"epoch": 2})
