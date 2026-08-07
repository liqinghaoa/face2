from __future__ import annotations

from pathlib import Path

import pytest
import torch

from skin_optics_so1.decomposition.checkpoint import load_checkpoint, save_checkpoint
from skin_optics_so1.decomposition.unet_decomposer import SO1UNetDecomposer


def _identity() -> dict:
    return {
        "target_order": ["M_norm", "H_norm", "S_norm", "P_norm"],
        "dataset_contract_hash": "contract-hash",
        "generator_config_hash": "78256a4f18072257d407c46f311c5903a66e0242ee945ff62e1d1a18afcc5cdf",
        "SO0_version": "SO0_Forward_Model_v1.1",
        "SO0_config_hash": "so0-hash",
        "model_architecture": SO1UNetDecomposer.architecture_id,
    }


def test_checkpoint_roundtrip_and_identity_rejection(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = SO1UNetDecomposer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=2)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    path = tmp_path / "last.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        grad_scaler=scaler,
        epoch=1,
        global_step=3,
        best_selection_metric=0.5,
        early_stopping_counter=0,
        identity=_identity(),
        resolved_config={"ok": True},
        dataset_contract={"schema": "test"},
        dataset_fingerprint={"hash": "x"},
        seed=20260801,
    )
    loaded_model = SO1UNetDecomposer()
    loaded_optimizer = torch.optim.AdamW(loaded_model.parameters(), lr=1.0e-3)
    loaded_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(loaded_optimizer, T_max=2)
    loaded_scaler = torch.amp.GradScaler("cuda", enabled=False)
    checkpoint = load_checkpoint(
        path,
        model=loaded_model,
        optimizer=loaded_optimizer,
        scheduler=loaded_scheduler,
        grad_scaler=loaded_scaler,
        expected_identity=_identity(),
        map_location="cpu",
        restore_rng=True,
    )
    assert checkpoint["epoch"] == 1
    for left, right in zip(model.parameters(), loaded_model.parameters()):
        assert torch.equal(left, right)
    bad_identity = _identity()
    bad_identity["target_order"] = ["H_norm", "M_norm", "S_norm", "P_norm"]
    with pytest.raises(ValueError, match="target_order"):
        load_checkpoint(
            path,
            model=loaded_model,
            optimizer=None,
            scheduler=None,
            grad_scaler=None,
            expected_identity=bad_identity,
            map_location="cpu",
            restore_rng=False,
        )
