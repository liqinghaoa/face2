from pathlib import Path
import yaml
from src.skin_optics_hsi.km_bio_v2r1_stage_c1 import _resolve
ROOT=Path(__file__).resolve().parents[2]
def test_c1_config_train_only_and_scope():
    c=yaml.safe_load((ROOT/'configs/skin_optics_hsi/km_bio_v2r1_stage_c1.yaml').read_text(encoding='utf-8'))
    assert c['selection']['candidates']==['V2R-PS','V2R-PSG']
    assert all(c['execution'][k] is False for k in ['raw_hsi_content_allowed','rgb_content_allowed','validation_content_allowed','test_content_allowed','clinical_500_content_allowed'])
def test_c1_inputs_exist():
    c=yaml.safe_load((ROOT/'configs/skin_optics_hsi/km_bio_v2r1_stage_c1.yaml').read_text(encoding='utf-8'))
    assert _resolve(ROOT,c['inputs']['r_c0r_decision']).is_file()
    assert _resolve(ROOT,c['inputs']['observation_manifest']).is_file()
def test_c1_sensitivity_grid():
    c=yaml.safe_load((ROOT/'configs/skin_optics_hsi/km_bio_v2r1_stage_c1.yaml').read_text(encoding='utf-8'))
    assert c['sensitivity']['diameter_um']==[0.0,7.5,15.0,30.0]
    assert c['sensitivity']['bandwidth']==['420_680','430_670','440_660']
