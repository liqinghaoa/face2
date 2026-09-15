from __future__ import annotations

from pathlib import Path

import pytest
import torch

from skin_optics_so1.decomposition.amp_recovery import (
    RECOVERY_HISTORY_FIELDS,
    adjudicate_recovery_window,
    adam_state_sha256,
    diagnostic_path_isolated,
    grad_scaler_found_inf,
    optimizer_group_sha256,
    parameter_state_sha256,
    scaler_step_and_update,
)


class FakeScaler:
    def __init__(self, *, overflow: bool) -> None:
        self.scale = 16.0
        self.overflow = overflow
        self.step_called = False
        self.update_called = False

    def get_scale(self) -> float:
        return self.scale

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_called = True
        if not self.overflow:
            optimizer.step()

    def update(self) -> None:
        self.update_called = True
        if self.overflow:
            self.scale *= 0.5


def initialized_runtime() -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_nonfinite_detection_path_uses_scaler_skip_not_optimizer_step() -> None:
    model, optimizer = initialized_runtime()
    before = parameter_state_sha256(model)
    result = scaler_step_and_update(FakeScaler(overflow=True), optimizer)
    assert result["scaler_step_called"] and result["scaler_update_called"]
    assert result["optimizer_step_skipped"] and not result["optimizer_step_executed"]
    assert result["scale_after"] < result["scale_before"]
    assert parameter_state_sha256(model) == before


def test_overflow_batch_preserves_parameter_adam_and_group_state() -> None:
    model, optimizer = initialized_runtime()
    before = (
        parameter_state_sha256(model), adam_state_sha256(optimizer, model),
        optimizer_group_sha256(optimizer),
    )
    scaler_step_and_update(FakeScaler(overflow=True), optimizer)
    after = (
        parameter_state_sha256(model), adam_state_sha256(optimizer, model),
        optimizer_group_sha256(optimizer),
    )
    assert after == before


def test_finite_chain_executes_optimizer_step() -> None:
    model, optimizer = initialized_runtime()
    model(torch.ones(1, 2)).sum().backward()
    before = parameter_state_sha256(model)
    result = scaler_step_and_update(FakeScaler(overflow=False), optimizer)
    assert result["optimizer_step_executed"] and not result["optimizer_step_skipped"]
    assert parameter_state_sha256(model) != before


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GradScaler contract")
def test_grad_scaler_found_inf_reuses_unscale_result() -> None:
    model = torch.nn.Linear(2, 1).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.amp.GradScaler("cuda")
    scaler.scale(model(torch.ones(1, 2, device="cuda")).sum()).backward()
    scaler.unscale_(optimizer)
    assert not grad_scaler_found_inf(scaler, optimizer)
    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

    scaler.scale(model(torch.ones(1, 2, device="cuda")).sum()).backward()
    next(model.parameters()).grad.fill_(float("inf"))
    scaler.unscale_(optimizer)
    assert grad_scaler_found_inf(scaler, optimizer)


def test_recovery_history_contract_and_stability_rule() -> None:
    assert RECOVERY_HISTORY_FIELDS == [
        "batch_index", "global_step", "scale_before", "scale_after",
        "forward_finite", "loss_finite", "gradient_finite", "overflow_detected",
        "optimizer_step_skipped", "optimizer_step_executed", "parameters_finite",
        "BN_finite", "Adam_finite",
    ]
    rows = [{"overflow_detected": index in (2, 20)} for index in range(100)]
    assert adjudicate_recovery_window(rows)["AMP_RECOVERY_STABLE"]
    rows[75]["overflow_detected"] = True
    result = adjudicate_recovery_window(rows)
    assert not result["AMP_RECOVERY_STABLE"]
    assert not result["last_50_overflow_free"]


def test_r3_diagnostic_artifacts_are_isolated_from_formal_checkpoints() -> None:
    root = Path("experiments/skin_optics_so1/SO1_Decomposition_UNet_v1")
    diagnostic = root / "diagnostics/so1d_r3_gradscaler_recoverability"
    formal = root / "formal_train/checkpoints"
    assert diagnostic_path_isolated(diagnostic, formal)
