from pathlib import Path
import torch
from datasets.p2_r1_relighting_dataset import P2R1PairedTransform,P2R1RelightingDataset,identity_pil,audit_p1_assets
from datasets.control_patient_binary_dataset import ControlPatientFaceDataset,map_three_class_to_binary
from losses.p2_r1_consistency_loss import js_divergence,consistency_weight
from models.nyha_backbone_factory import build_nyha_classification_model
from utils.experiment_utils import load_yaml
ROOT=Path(__file__).resolve().parents[1]; C=load_yaml(ROOT/'config/p2/p2_r1_binary_deca_relighting_fold0_v1.yaml')
def ds(train=True):
 d=C['data'];base=ControlPatientFaceDataset(ROOT/d['train_csv' if train else 'val_csv'],identity_pil,ROOT/d['image_root'],d['image_filename_template']);return P2R1RelightingDataset(base,ROOT/C['p1']['root'],C['p1'],P2R1PairedTransform(train,C['normalize']['mean'],C['normalize']['std']),train)
def test_fixed_protocol_and_mapping():
 assert C['task']['type']=='binary_control_vs_patient' and C['data']['fold']==0 and C['relighting']['num_presets']==6
 assert [map_three_class_to_binary(i) for i in (0,1,2)]==[0,1,1]
def test_real_p1_pair_shapes_and_rotation():
 x=ds(True);x.set_epoch(0);a=x[0];x.set_epoch(1);b=x[0]
 assert a['preset_index']==0 and b['preset_index']==1 and a['preset_name']=='neutral_front'
 assert a['raw_image'].shape==a['relight_image'].shape==(3,224,224) and torch.isfinite(a['relight_image']).all()
def test_p1_ready_equals_fixed_oof_and_has_six_assets(tmp_path):
 r=audit_p1_assets(ROOT,C,tmp_path);assert r['status']=='passed' and r['ready_equals_fixed_oof'] and r['ready_unique_case_ids']==r['fixed_oof_unique_ids']==500 and not r['missing_from_p1'] and not r['extra_in_p1']
def test_js_warmup_and_shared_binary_model():
 z=torch.randn(3,2);assert js_divergence(z,z).abs()<1e-6 and js_divergence(z,z+1).isfinite();assert [consistency_weight(i) for i in range(4)]==[0.,0.,0.,.2]
 m=build_nyha_classification_model('resnet18',2,False,False);out=m(torch.zeros(4,3,224,224));assert out.shape==(4,2)
def test_no_deca_runtime_import_or_latent_input():
 text=(ROOT/'datasets/p2_r1_relighting_dataset.py').read_text(encoding='utf-8').lower();assert 'deca_runtime' not in text and 'latents.npz' not in text and 'torch.load' not in text
def test_stability_evaluator_uses_relight_tensor_not_raw_tensor():
 text=(ROOT/'scripts/evaluate/summarize_p2_r1_vs_b0.py').read_text(encoding='utf-8');assert '_,rt=ds.transform' in text
def test_smoke_and_formal_guardrails_exist():
 text=(ROOT/'scripts/run/run_p2_r1_binary_deca_relighting_fold0.py').read_text(encoding='utf-8');assert "out.name.endswith('_smoke')" in text and 'skip-final-comparison is allowed only for an isolated smoke run' in text
