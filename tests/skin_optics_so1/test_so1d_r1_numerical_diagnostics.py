from __future__ import annotations

from pathlib import Path

import torch

from skin_optics_so1.decomposition.numerical_diagnostics import (
    all_rows_finite,
    atomic_torch_save,
    batchnorm_health,
    epoch7_order_from_generator_state,
    gradient_health,
    model_parameter_health,
    optimizer_state_health,
    sample_order_sha256,
    tensor_health,
)


def test_tensor_health_detects_first_nonfinite_evidence() -> None:
    result = tensor_health(torch.tensor([1.0, float("nan"), float("inf")]))
    assert not result["finite"]
    assert result["num_nan"] == 1
    assert result["num_inf"] == 1
    assert result["min"] == 1.0


def test_checkpoint_health_helpers_cover_model_bn_and_adam() -> None:
    model = torch.nn.Sequential(torch.nn.Conv2d(1, 2, 1), torch.nn.BatchNorm2d(2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.ones(2, 1, 2, 2)).sum()
    loss.backward()
    optimizer.step()
    assert all_rows_finite(model_parameter_health(model))
    bn = batchnorm_health(model)
    assert all_rows_finite(bn)
    assert all(row["nonnegative"] is not False for row in bn)
    optimizer_rows = optimizer_state_health(optimizer, model)
    assert optimizer_rows and all_rows_finite(optimizer_rows)


def test_gradient_health_localizes_nonfinite_parameter() -> None:
    model = torch.nn.Linear(2, 1)
    model.weight.grad = torch.tensor([[float("nan"), 1.0]])
    model.bias.grad = torch.ones_like(model.bias)
    _, summary = gradient_health(model)
    assert not summary["finite"]
    assert summary["first_nonfinite_parameter"] == "weight"


def test_epoch7_order_is_state_exact_and_stable() -> None:
    generator = torch.Generator().manual_seed(20260801)
    state = generator.get_state()
    expected = torch.randperm(20, generator=generator).tolist()
    assert epoch7_order_from_generator_state(
        sample_count=20, batch_size=8, generator_state=state
    ) == expected


def test_sample_order_sha256_changes_with_order() -> None:
    rows = [{
        "batch_index": 0, "position_in_batch": index, "split_index": index,
        "sample_id": f"train_{index:06d}", "base_latent_id": index,
        "acquisition_variant_id": 0, "camera": "C", "light": "L",
        "camera_light_pair": "C / L",
    } for index in range(2)]
    first = sample_order_sha256(rows)
    rows.reverse()
    assert sample_order_sha256(rows) != first


def test_safe_anchor_atomic_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "anchors" / "latest_safe_anchor.pt"
    atomic_torch_save({"batch_index": 0}, path)
    atomic_torch_save({"batch_index": 100}, path)
    assert torch.load(path, weights_only=False)["batch_index"] == 100
    assert not path.with_suffix(".pt.tmp").exists()


def test_diagnostic_output_path_is_isolated_from_formal_checkpoints() -> None:
    diagnostic = Path("experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/diagnostics/so1d_r1_epoch7_numerical_failure")
    formal = Path("experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train/checkpoints")
    assert formal not in diagnostic.parents
    assert diagnostic not in formal.parents
