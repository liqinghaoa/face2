from pathlib import Path
import inspect
import numpy as np

from src.skin_optics_so_r1 import realface_classifier_input_preparation as x3


def test_geometry_and_interpolation_contract():
    assert x3.SOURCE_SHAPE == (1220, 979)
    assert x3.RESIZED_SHAPE == (319, 256)
    assert x3.TARGET_SHAPE == (320, 256)
    assert "align_corners=False" in inspect.getsource(x3._resize_chw)
    a = np.ones((3, 319, 256), dtype=np.float32)
    out = x3._pad_bottom(a)
    assert out.shape == (3, 320, 256) and np.all(out[:, 319] == 0)


def test_canonical_order_uses_x1_manifest_and_source_full_ids():
    source = Path(x3.__file__).read_text(encoding="utf-8")
    assert "inference_manifest.csv" in source and "full_ids.txt" in source


def test_channel_contract_is_declared_and_rgb_reused():
    source = Path(x3.__file__).read_text(encoding="utf-8")
    assert "RGB_B1MH" in source and "RGB_B2MH" in source
    assert "np.concatenate((rgb_small, b1_small)" in source
    assert "np.concatenate((rgb_small, b2_small)" in source
    assert "np.array_equal(rgb, b1[:3])" in source


def test_x3_is_label_blind_and_never_runs_forward_or_training():
    source = Path(x3.__file__).read_text(encoding="utf-8").lower()
    assert "model.forward" not in source and "torch.save" not in source
    assert '"model_forward_calls": 0' in source
    assert '"training_calls": 0' in source


def test_mask_resize_is_nearest_and_output_mask_is_binary():
    source = inspect.getsource(x3._prepare_case)
    assert '"nearest"' in source and '"bilinear"' in source
    assert "* mask_small[None]" in source


def test_output_audit_has_hard_checks_for_dtype_mask_and_inventory():
    source = inspect.getsource(x3._audit_outputs)
    assert "dtype_ok" in source
    assert "mask_binary" in source
    assert "exact_file_inventory" in source
    assert "path.is_file()" in source
