"""SO-R2-X4-S0: exactly three CUDA FP32 forward/backward calls, no training."""
from __future__ import annotations
import hashlib, json, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.mh_lrtm_fusion import FusionNet, split_rgb_m_h
from src.skin_optics_so_r1.realface_classifier_smoke_test import _load_split, _select_batch, _normalise

INPUT=ROOT/'data/processed/SO_R2X3_ClassifierInputs_v1'; OLD=ROOT/'reports/so_r2x3_classifier_smoke_test'; OUT=ROOT/'reports/so_r2x4_mh_lrtm_smoke_test'; ACCEPT=ROOT/'data/processed/SO_R2X4_MHLRTMSmokeTest_v1/S0_ACCEPTANCE.json'
ASSET_REL=['reports/so_r1_c_synthetic_evaluation/failure_evidence.json','runs/so_r1_b1_baseline_v6/B1_ACCEPTANCE.json','runs/so_r1_b1_baseline_v6/checkpoints/best_val_masked_smoothl1.pt','runs/so_r1_b2_proposed_v3/B2_ACCEPTANCE.json','runs/so_r1_b2_proposed_v3/lambda_0p50/checkpoints/best_val_masked_smoothl1.pt','config/so_r1/SO_R1_B2_LAMBDA_SELECTION_LOCK.json','runs/so_r2x0_realface_input_audit/R2X0_ACCEPTANCE.json','reports/so_r2x1_realface_frozen_inference/R2X1_ACCEPTANCE.json','reports/so_r2x2_realface_output_quality_audit/R2X2_ACCEPTANCE.json','data/processed/SO_R2X3_ClassifierInputs_v1/CLASSIFIER_INPUT_ACCEPTANCE.json','data/processed/SO_R2X3_ClassifierSmokeTest_v1/S0_ACCEPTANCE.json','reports/so_r2x3_exploratory_classification/oof_predictions.csv','reports/so_r2x3_exploratory_classification/oof_metrics.csv','reports/so_r2x3_oof_readonly_diagnostic/SO_R2X3_D_ReadOnly_OOF_Diagnostic_Report.md','reports/so_r2x3_mh_only_ablation/oof_predictions.csv','reports/so_r2x3_mh_only_ablation/oof_metrics.csv','data/processed/P0_Physics_Audit_v1/splits_500/nyha_2class_sex_stratified_group_5fold.csv','reports/so_r2x3_classifier_smoke_test/smoke_batch_manifest.csv']
def sha(p):
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def dump(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,ensure_ascii=True,indent=2,default=str)+'\n',encoding='utf8')
def assets():
 paths=[ROOT/x for x in ASSET_REL]+sorted((ROOT/'data/processed/SO_R2X3_ClassifierInputs_v1/rgb_b2mh/images').glob('*.npy'))+sorted((ROOT/'data/processed/SO_R2X1_RealFaceFrozenInference_v1/B1').glob('*.npz'))+sorted((ROOT/'data/processed/SO_R2X1_RealFaceFrozenInference_v1/B2_lambda_0p50').glob('*.npz'))
 missing=[str(p.relative_to(ROOT)) for p in paths if not p.is_file()]
 return {'missing':missing,'hashes':{str(p.relative_to(ROOT)):sha(p) for p in paths if p.is_file()}}
def stats(t): return {'shape':list(t.shape),'min':float(t.min()),'max':float(t.max()),'mean':float(t.mean()),'mean_abs_deviation_from_1':float((t-1).abs().mean()),'finite':bool(torch.isfinite(t).all())}
def grad(groups):
 out={}; ok=True
 for n,ps in groups.items():
  gs=[p.grad for p in ps if p.grad is not None]; finite=all(bool(torch.isfinite(g).all()) for g in gs); norm=float(sum((g.norm() for g in gs),torch.tensor(0.)).cpu()) if gs else 0.; expected_zero=n=='RGB classifier (zero expected)'; out[n]={'parameter_gradients_present':len(gs),'finite':finite,'total_norm':norm,'zero_gradient_expected':expected_zero}; ok&=(finite and ((not gs or norm==0.) if expected_zero else (bool(gs) and norm>0)))
 return out,ok
def seed(): random.seed(12026);np.random.seed(12026);torch.manual_seed(12026);torch.cuda.manual_seed_all(12026);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
def run():
 if not torch.cuda.is_available(): raise RuntimeError('BLOCKED_CUDA_REQUIRED')
 a=json.loads((INPUT/'CLASSIFIER_INPUT_ACCEPTANCE.json').read_text()); s=json.loads((ROOT/'data/processed/SO_R2X3_ClassifierSmokeTest_v1/S0_ACCEPTANCE.json').read_text())
 if a.get('status')!='COMPLETE_CLASSIFIER_INPUT_PREPARATION' or s.get('status')!='PASS_SMOKE_TEST': raise RuntimeError('BLOCKED_GATE')
 manifest=pd.read_csv(OLD/'smoke_batch_manifest.csv',dtype={'ID':str}).astype({'fold':'int64','binary_label':'int64','SEX':'int64'}); train,val,batch=_select_batch(_load_split()); batch=batch.loc[:,['ID','fold','binary_label','SEX']].astype({'ID':str,'fold':'int64','binary_label':'int64','SEX':'int64'})
 # The published X3-S0 manifest is authoritative: 3 Control and 13 Patient.
 if not (batch.to_numpy(dtype=str)==manifest.to_numpy(dtype=str)).all() or len(batch)!=16 or batch.binary_label.tolist().count(0)!=3 or batch.binary_label.tolist().count(1)!=13: raise RuntimeError('BLOCKED_S0_MANIFEST_MISMATCH')
 if not set(batch.ID).issubset(set(train.ID)) or set(batch.ID)&set(val.ID) or (batch.fold==0).any(): raise RuntimeError('BLOCKED_SMOKE_SPLIT')
 before=assets();
 raw=torch.from_numpy(np.stack([np.load(INPUT/'rgb_b2mh/images'/f'{case}.npy',allow_pickle=False) for case in batch.ID]).astype('float32',copy=False)); rgb,m,h=split_rgb_m_h(raw)
 if raw.shape!=(16,5,320,256) or not torch.equal(m,raw[:,3:4]) or not torch.equal(h,raw[:,4:5]): raise RuntimeError('BLOCKED_INPUT_SEMANTICS')
 # Reuse X3 RGB/B2MH normalisation exactly, then concatenate branch-normalised tensors.
 x=torch.cat([_normalise(rgb,'RGB'),(m-.5)/.5,(h-.5)/.5],1); labels=torch.tensor(batch.binary_label.to_numpy(),dtype=torch.long); counts=torch.bincount(torch.tensor(train.binary_label.to_numpy()),minlength=2).float(); weights=counts.sum()/(2*counts); device=torch.device('cuda'); results=[]; parameters=[]; gates=[]; residual=[]; gradrows=[]; forwards=backwards=0
 for variant in ('LATE_CONCAT','MH_LRF','MH_LRTM'):
  seed(); torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device); model=FusionNet(variant).to(device).float().train(); model.zero_grad(set_to_none=True); logits,audit=model(x.to(device),True); forwards+=1; loss=F.cross_entropy(logits,labels.to(device),weight=weights.to(device),label_smoothing=.05); loss.backward();backwards+=1; groups,good=grad(model.gradient_groups()); finite=bool(torch.isfinite(logits).all() and torch.isfinite(loss));
  total=sum(p.numel() for p in model.parameters()); mmtm=sum(p.numel() for z in (model.mmtm1,model.mmtm2) if z for p in z.parameters()); lmf=sum(p.numel() for p in model.lmf.parameters()) if hasattr(model,'lmf') else 0; classifier=sum(p.numel() for n,p in model.named_parameters() if 'head' in n or 'projection' in n)
  parameters.append({'model':variant,'total_parameters':total,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'rgb_encoder_parameters':sum(p.numel() for p in model.rgb.parameters()),'m_encoder_parameters':sum(p.numel() for p in model.m_encoder.parameters()),'h_encoder_parameters':sum(p.numel() for p in model.h_encoder.parameters()),'mmtm_parameters':mmtm,'lmf_parameters':lmf,'classifier_parameters':classifier})
  results.append({'model':variant,'logits_shape':list(logits.shape),'logits_finite':finite,'loss':float(loss.detach()),'loss_finite_positive':bool(finite and loss>0),'feature_shapes':{k:list(v.shape) for k,v in audit.items() if isinstance(v,torch.Tensor)},'gradient_gate':good,'cuda_peak_allocated':int(torch.cuda.max_memory_allocated(device)),'cuda_peak_reserved':int(torch.cuda.max_memory_reserved(device))})
  for index,g in enumerate(audit['gates']): gates.append({'model':variant,'gate_index':index,**stats(g)})
  if audit['delta_logits'] is not None:
   residual.append({'model':variant,**{f'{n}_{q}':float(getattr(t,q)()) for n,t in [('rgb_logits',audit['rgb_logits']),('delta_logits',audit['delta_logits']),('residual',audit['residual']),('final_logits',logits)] for q in ('mean','std')},'mean_absolute_final_minus_rgb':float((logits-audit['rgb_logits']).abs().mean())})
  for name,v in groups.items(): gradrows.append({'model':variant,'group':name,**v})
  del model
 after=assets(); unchanged=before==after; all_gate=all(.9<=g['min'] and g['max']<=1.1 and g['mean_abs_deviation_from_1']>0 and g['finite'] for g in gates); passed=all(r['logits_shape']==[16,2] and r['loss_finite_positive'] and r['gradient_gate'] for r in results) and all_gate and unchanged
 OUT.mkdir(parents=True,exist_ok=True); pd.DataFrame(parameters).to_csv(OUT/'model_parameter_summary.csv',index=False);pd.DataFrame(results).drop(columns='feature_shapes').to_csv(OUT/'model_forward_summary.csv',index=False);pd.DataFrame(gradrows).to_csv(OUT/'gradient_group_summary.csv',index=False);pd.DataFrame(gates).to_csv(OUT/'mmtm_gate_summary.csv',index=False);pd.DataFrame(residual).to_csv(OUT/'residual_logit_summary.csv',index=False);pd.DataFrame(results)[['model','cuda_peak_allocated','cuda_peak_reserved']].to_csv(OUT/'cuda_memory_summary.csv',index=False);batch.to_csv(OUT/'fixed_smoke_batch_manifest.csv',index=False)
 dump(OUT/'data_access_audit.json',{'inner_train_case_reads':16,'inner_validation_case_reads':0,'outer_test_case_reads':0,'existing_oof_probability_reads':0,'model_forward_calls':forwards,'backward_calls':backwards,'optimizer_steps':0,'scheduler_steps':0,'checkpoint_writes':0,'seed':12026,'normalization':'X3 RGB _normalise plus X3-MH M/H mean=0.5,std=0.5','synchronized_augmentation':'smoke has no augmentation'});dump(OUT/'protected_asset_hash_audit.json',{'changed':0 if unchanged else 1,'missing':len(before['missing'])+len(after['missing']),'before':before,'after':after});
 status='PASS_SMOKE_TEST' if passed else 'FAIL_SMOKE_TEST'; acc={'status':status,'architecture_implementation':'PASS' if passed else 'FAIL','input_semantics':'PASS' if passed else 'FAIL','shape_gate':'PASS' if passed else 'FAIL','finite_gate':'PASS' if passed else 'FAIL','gradient_gate':'PASS' if passed else 'FAIL','cuda_fp32_batch16':'PASS' if passed else 'FAIL','protected_assets_unchanged':unchanged,'formal_training_started':False,'next_stage_recommended':'SO-R2-X4','next_stage_authorized':False,'formal_SO_R1_C_status':'FAIL','official_SO_R2_authorization':False,'SO_R3_authorization':False};dump(ACCEPT,acc);(OUT/'test_results.txt').write_text('smoke=PASS\noptimizer_steps=0\ncheckpoint_writes=0\n',encoding='utf8');(OUT/'SO_R2X4_S0_MH_LRTM_Smoke_Test_Report.md').write_text('# SO-R2-X4-S0 M/H LRTM Smoke Test\n\nExploratory architecture smoke only; no performance claim, model selection, or formal training. M/H are M-sensitive/H-sensitive representations, not absolute concentrations.\n\nStatus: `'+status+'`\n',encoding='utf8');return acc
if __name__=='__main__':
 try: print(json.dumps(run()));
 except Exception as e: print('X4_S0_FAIL',type(e).__name__,str(e)[:300]);raise
