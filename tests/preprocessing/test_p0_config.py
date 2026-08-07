from pathlib import Path
import pytest
from preprocessing.p0_physics_assets.config import load_config

def test_config_uses_global_hybrid_constants():
    config=load_config(Path('config/preprocess/p0_physics_audit_assets_v1.yaml'),Path.cwd())
    assert config.runtime.image_size == 224 and config.global_mask.feather_kernel == 11
    assert (config.global_mask.forehead_expand_ratio,config.global_mask.side_expand_ratio,config.global_mask.chin_expand_ratio)==(.18,.05,.03)

def test_config_rejects_roi_mask_constants(tmp_path):
    text=Path('config/preprocess/p0_physics_audit_assets_v1.yaml').read_text(encoding='utf-8').replace('forehead_expand_ratio: 0.18','forehead_expand_ratio: 0.15')
    path=tmp_path/'bad.yaml'; path.write_text(text,encoding='utf-8')
    with pytest.raises(ValueError,match='Global mask'): load_config(path,Path.cwd())
