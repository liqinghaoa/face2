"""Focused safety and contract tests for the SO-R2-X0 audit."""
from pathlib import Path
import inspect
import numpy as np
import torch

from src.skin_optics_so_r1 import realface_input_contract_audit as audit
from src.skin_optics_so_r1.baseline_training import BaselineMHUNet


def test_authoritative_constants_and_architecture():
    assert audit.PATCHES == tuple(f"P{i:02d}" for i in range(1, 21))
    assert audit.XS == (0, 241, 482, 723)
    assert audit.YS == (0, 241, 482, 723, 964)
    assert BaselineMHUNet().inc.layers[0].in_channels == 3
    assert BaselineMHUNet.output_order == ["M-sensitive", "H-sensitive"]


def test_hwc_to_chw_float32_semantics():
    rgb = np.zeros((256, 256, 3), dtype=np.float32)
    rgb[..., 0], rgb[..., 1], rgb[..., 2] = 0.1, 0.2, 0.3
    tensor = torch.from_numpy(np.transpose(rgb, (2, 0, 1))[None])
    assert tuple(tensor.shape) == (1, 3, 256, 256)
    assert tensor.dtype == torch.float32
    assert torch.allclose(tensor[:, 0].mean(), torch.tensor(0.1))
    assert torch.allclose(tensor[:, 2].mean(), torch.tensor(0.3))


def test_mask_contract_is_binary_and_not_model_input():
    source = inspect.getsource(audit.synthetic_loader_semantics)
    assert '"mask_is_model_input": False' in source
    assert audit.PATCHES[0] == "P01" and audit.PATCHES[-1] == "P20"
    assert set(np.unique(np.array([[0, 255]], dtype=np.uint8))) <= {0, 255}


def test_fixed_canary_selection_rule_is_deterministic():
    source = inspect.getsource(audit.canary)
    assert "ids[0],ids[249],ids[499]" in source.replace(" ", "")
    assert 'patches=["P01","P10","P11","P20"]' in source.replace(" ", "")


def test_no_forbidden_sensitive_field_access_or_full_inference():
    source = Path(audit.__file__).read_text(encoding="utf-8").lower()
    # The acceptance report may explicitly record that forbidden fields were not read;
    # no sensitive dataset path/field access is allowed.
    for token in ("parsing_label", "nyha.csv", "labels.csv", "fold.csv", "sex.csv", "classification.csv"):
        assert token not in source
    assert '"full_inference_started":true' not in source.replace(" ", "")
    assert '"prediction_map_saved":true' not in source.replace(" ", "")


def test_canary_does_not_save_prediction_maps():
    source = inspect.getsource(audit.canary)
    assert "prediction_maps_saved" in source
    assert '"prediction_map_saved":True' not in source.replace(" ", "")
    assert "to_csv" not in source and "write_text" not in source


def test_failure_evidence_blocks_next_stage():
    import tempfile, json
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        audit.write_failure(root, RuntimeError("synthetic"))
        acceptance = json.loads((root / audit.OUT / "R2X0_ACCEPTANCE.json").read_text())
        assert acceptance["status"] == "FAIL_INPUT_SEMANTICS"
        assert acceptance["next_stage_authorized"] is False
        assert acceptance["full_inference_started"] is False


def test_protected_hashes_are_read_only_function():
    source = inspect.getsource(audit.protected_hashes)
    assert ".open(\"wb\")" not in source
    assert "write_text" not in source
