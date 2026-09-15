from pathlib import Path
import inspect
import numpy as np
from src.skin_optics_so_r1 import realface_frozen_inference as x1


def test_fusion_formula_and_contract_constants():
    assert x1.XS == (0, 241, 482, 723) and x1.YS == (0, 241, 482, 723, 964)
    src = inspect.getsource(x1._fuse)
    assert "sums[:, y0:y0+256, x0:x0+256] += pred * mask[None]" in src
    assert "weights[y0:y0+256, x0:x0+256] += mask" in src
    assert "sums[:, ~valid]" not in src


def test_outputs_are_frozen_fp32_and_no_checkpoint_write():
    src = inspect.getsource(x1)
    assert "torch.no_grad()" in src and "astype(np.float32)" in src
    assert "torch.save" not in src and "checkpoint" in src
    assert "np.save(" not in src and "np.savez_compressed" in src


def test_no_sensitive_fields_or_classification():
    src = Path(x1.__file__).read_text(encoding="utf-8").lower()
    for token in ("nyha", "sex", "fold", "clinical"):
        assert token not in src


def test_manifest_and_output_contracts_are_declared():
    assert x1.OUT.parts[-1] == "SO_R2X1_RealFaceFrozenInference_v1"
    assert x1.REPORT.parts[-1] == "so_r2x1_realface_frozen_inference"
    assert np.zeros((2, 1220, 979), dtype=np.float32).dtype == np.float32
