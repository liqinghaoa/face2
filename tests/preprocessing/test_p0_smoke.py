from pathlib import Path
from preprocessing.p0_physics_assets.builder import preflight
from preprocessing.p0_physics_assets.config import load_config

def test_real_metadata_preflight_without_models():
    config=load_config(Path('config/preprocess/p0_physics_audit_assets_v1.yaml'),Path.cwd())
    split,exif,assets,records=preflight(config)
    assert len(split)==len(exif)==len(records)==500
    assert all(len(value)==500 for value in assets.values())
