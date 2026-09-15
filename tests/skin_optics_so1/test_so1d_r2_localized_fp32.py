from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from skin_optics_so1.decomposition.unet_decomposer import (
    SO1UNetDecomposer,
    count_parameters,
    parse_numerical_precision_policy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
R2_CONFIG = PROJECT_ROOT / "config/train/skin_optics_so1/so1d_localized_fp32_repair_v1.yaml"
FORMAL_LAST = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/formal_train/checkpoints/last.pt"
R2_OUTPUT = PROJECT_ROOT / "experiments/skin_optics_so1/SO1_Decomposition_UNet_v1/diagnostics/so1d_r2_localized_fp32_repair"


def policy() -> dict:
    return yaml.safe_load(R2_CONFIG.read_text(encoding="utf-8"))["numerical_precision"]


def test_localized_fp32_config_parsing_is_restricted() -> None:
    assert parse_numerical_precision_policy(policy()) == {"up1.conv"}
    assert parse_numerical_precision_policy(None) == set()
    with pytest.raises(ValueError, match="restricted"):
        parse_numerical_precision_policy({"mode": "localized_fp32", "fp32_blocks": ["up2.conv"]})


def test_state_dict_and_parameter_topology_are_unchanged() -> None:
    original = SO1UNetDecomposer()
    repaired = SO1UNetDecomposer(numerical_precision=policy())
    assert list(original.state_dict()) == list(repaired.state_dict())
    assert count_parameters(original) == count_parameters(repaired)


def test_repair_disabled_preserves_v1_behavior() -> None:
    torch.manual_seed(7)
    original = SO1UNetDecomposer().eval()
    disabled = SO1UNetDecomposer(numerical_precision={"mode": "disabled"}).eval()
    disabled.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(original(x), disabled(x))


def test_up1_conv_is_fp32_and_other_blocks_remain_autocast_managed() -> None:
    model = SO1UNetDecomposer(numerical_precision=policy()).eval()
    observed: dict[str, torch.dtype] = {}
    handles = [
        model.up1.conv.register_forward_pre_hook(
            lambda _m, inputs: observed.__setitem__("up1_input", inputs[0].dtype)
        ),
        model.up1.conv.net[0].register_forward_hook(
            lambda _m, _i, output: observed.__setitem__("up1_conv0", output.dtype)
        ),
        model.up1.conv.net[1].register_forward_hook(
            lambda _m, _i, output: observed.__setitem__("up1_bn0", output.dtype)
        ),
        model.up1.conv.register_forward_hook(
            lambda _m, _i, output: observed.__setitem__("up1_output", output.dtype)
        ),
        model.up2.up.register_forward_hook(
            lambda _m, _i, output: observed.__setitem__("up2_up", output.dtype)
        ),
    ]
    try:
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(torch.randn(1, 3, 32, 32))
    finally:
        for handle in handles:
            handle.remove()
    assert observed["up1_input"] == torch.float32
    assert observed["up1_conv0"] == torch.float32
    assert observed["up1_bn0"] == torch.float32
    assert observed["up1_output"] == torch.float32
    assert observed["up2_up"] == torch.bfloat16


def test_localized_guard_is_valid_when_global_amp_is_disabled() -> None:
    original = SO1UNetDecomposer().eval()
    repaired = SO1UNetDecomposer(numerical_precision=policy()).eval()
    repaired.load_state_dict(original.state_dict(), strict=True)
    x = torch.randn(1, 3, 32, 32)
    assert torch.equal(original(x), repaired(x))


def test_epoch6_checkpoint_strictly_loads_and_paths_are_isolated() -> None:
    checkpoint = torch.load(FORMAL_LAST, map_location="cpu", weights_only=False)
    model = SO1UNetDecomposer(numerical_precision=policy())
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    formal_checkpoints = FORMAL_LAST.parent.resolve()
    diagnostic = R2_OUTPUT.resolve()
    assert formal_checkpoints not in diagnostic.parents
    assert diagnostic not in formal_checkpoints.parents


def test_diagnostic_checkpoint_contract_is_explicit() -> None:
    config = yaml.safe_load(R2_CONFIG.read_text(encoding="utf-8"))
    assert config["diagnostic"]["diagnostic_only"] is True
    assert config["protocol_version"] == "SO1D_Localized_FP32_Repair_v1"
    assert config["base_protocol"] == "SO1D_Formal_Train_v1.0"


def test_state_gate_ignores_empty_first_failure_metadata() -> None:
    import importlib.util

    path = PROJECT_ROOT / "scripts/analysis/run_skin_optics_so1d_r2_localized_fp32_repair.py"
    spec = importlib.util.spec_from_file_location("so1d_r2_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.finite_state_passes({
        "parameters_finite": True,
        "optimizer_state_finite": True,
        "batchnorm_buffers_finite": True,
        "first_nonfinite_parameter": None,
        "first_nonfinite_optimizer_state": None,
        "first_nonfinite_batchnorm": None,
    })


def test_rng_restore_accepts_device_mapped_byte_tensors() -> None:
    import importlib.util
    import random
    import numpy as np

    path = PROJECT_ROOT / "scripts/analysis/run_skin_optics_so1d_r2_localized_fp32_repair.py"
    spec = importlib.util.spec_from_file_location("so1d_r2_rng_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = {
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    module.restore_anchor_rng(state)


def test_r2_b_failure_classification_distinguishes_overflow_from_backward() -> None:
    import importlib.util

    path = PROJECT_ROOT / "scripts/analysis/run_skin_optics_so1d_r2_localized_fp32_repair.py"
    spec = importlib.util.spec_from_file_location("so1d_r2_classification_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.classify_r2_b_failure({
        "stage": "FORWARD", "health": {"dtype": "torch.float16", "finite": False}
    }) == ("PARTIAL", True)
    assert module.classify_r2_b_failure({
        "stage": "BACKWARD", "module": "inc.net.0.weight"
    }) == ("FAIL", False)
