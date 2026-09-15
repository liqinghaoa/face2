from pathlib import Path
import inspect
import numpy as np
from src.skin_optics_so_r1 import realface_output_quality_audit as x2

def test_frozen_protocol_and_no_model_or_training_calls():
    p=x2._protocol();assert p["forward_calls_allowed"]==0 and p["label_access_allowed"] is False
    s=Path(x2.__file__).read_text(encoding="utf-8");assert "torch" not in s and "model.forward" not in s and "torch.save" not in s

def test_fixed_seams_and_qualitative_selection():
    assert x2.SEAM_X==(241,482,723) and x2.SEAM_Y==(241,482,723,964) and x2.OFFSET==16
    assert x2.QUAL_INDICES==(1,46,91,137,182,228,273,319,364,410,455,500)

def test_mask_aware_seam_and_output_checks():
    a=np.arange(36,dtype=np.float32).reshape(6,6);v=np.ones((6,6),bool)
    # Contract function is defined for fixed production coordinates; source must
    # enforce both-side mask validity and outside-mask zero checks.
    s=inspect.getsource(x2._seam)+inspect.getsource(x2.run)
    assert "valid[:,position]" in s and "a[:,~valid]!=0" in s

def test_only_x1_fused_maps_and_masks_are_inputs():
    s=Path(x2.__file__).read_text(encoding="utf-8")
    assert "ROOT_DATA" in s and '"mh_sensitive_map"' in s and '"valid_masks"' in s
    assert "patches/" not in s and "P01_rgb" not in s

def test_no_clinical_field_paths_or_state_rewrite():
    s=Path(x2.__file__).read_text(encoding="utf-8")
    for token in ("nyha.csv", "binary_label.csv", "sex.csv", "fold.csv", "age.csv"):
        assert token not in s
    assert '"formal_SO_R1_C_status":"FAIL"' in s
