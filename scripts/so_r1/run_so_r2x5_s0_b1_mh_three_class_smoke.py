"""SO-R2-X5-S0: CUDA FP32 forward/backward smoke, not a training run."""
from __future__ import annotations
import hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.realface_b1_mh_three_class_exploratory import build_model, inner_split, load_table

def sha_mh(p:Path)->dict[str,str]:
    a=np.load(p,mmap_mode='r',allow_pickle=False); out={}
    for i,n in ((3,'M'),(4,'H')): out[n]=hashlib.sha256(np.ascontiguousarray(a[i]).tobytes()).hexdigest()
    return out
def dump(p:Path,x:object)->None:
    p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=True,indent=2,default=str)+'\n',encoding='utf-8')
def run(root:Path=ROOT)->dict:
    root=root.resolve()
    if not torch.cuda.is_available():raise RuntimeError('BLOCKED_CUDA_REQUIRED')
    table=load_table(root); train,val=inner_split(table[table.fold!=0].copy(),0)
    # Canonical ID ordering removes random batch selection; require every class.
    batch=train.sort_values('ID').groupby('class3',group_keys=False).head(1)
    remain=train[~train.ID.isin(batch.ID)].sort_values('ID'); batch=pd.concat([batch,remain]).head(16).sort_values('ID')
    if len(batch)!=16 or set(batch.class3)!={0,1,2} or set(batch.ID)&set(val.ID) or (batch.fold==0).any():raise RuntimeError('BLOCKED_X5_SMOKE_BATCH')
    directory=root/'data/processed/SO_R2X3_ClassifierInputs_v1/rgb_b1mh/images';paths=[directory/f'{i}.npy' for i in batch.ID];before={str(p.relative_to(root)):sha_mh(p) for p in paths}
    raw=np.stack([np.array(np.load(p,mmap_mode='r',allow_pickle=False)[[3,4]],dtype=np.float32,copy=True) for p in paths]);
    if raw.shape!=(16,2,320,256) or not np.isfinite(raw).all():raise RuntimeError('BLOCKED_X5_SMOKE_INPUT')
    x=torch.from_numpy((raw-.5)/.5);y=torch.tensor(batch.class3.to_numpy(),dtype=torch.long);counts=np.bincount(train.class3,minlength=3);w=torch.tensor(len(train)/(3*counts),dtype=torch.float32);torch.manual_seed(2026);torch.cuda.manual_seed_all(2026);model=build_model().cuda().float().train();logits=model(x.cuda());loss=F.cross_entropy(logits,y.cuda(),weight=w.cuda(),label_smoothing=.05);loss.backward()
    grads={'conv1':model.conv1.weight.grad,'fc':model.fc.weight.grad};audit={k:{'finite':bool(torch.isfinite(g).all()),'nonzero':bool(g.norm()>0),'norm':float(g.norm())} for k,g in grads.items()};allfinite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.requires_grad and p.grad is not None);after={str(p.relative_to(root)):sha_mh(p) for p in paths};unchanged=before==after;ok=logits.shape==(16,3) and bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)) and float(loss)>0 and all(x['finite'] and x['nonzero'] for x in audit.values()) and allfinite and unchanged
    report=root/'reports/so_r2x5_b1_mh_three_class_smoke_test';report.mkdir(parents=True,exist_ok=True);batch[['ID','patient_group_id','SEX','NYHA','fold','class3']].to_csv(report/'smoke_batch_manifest.csv',index=False);dump(report/'model_interface_audit.json',{'input_shape':[16,2,320,256],'dtype':'float32','logits_shape':list(logits.shape),'class_order':['Normal','Mild','Severe'],'imagenet_conv1_mean_initialization':True});dump(report/'gradient_audit.json',{'groups':audit,'all_trainable_gradients_finite':allfinite});dump(report/'protected_asset_hash_audit.json',{'unchanged':unchanged,'before':before,'after':after});dump(report/'data_access_audit.json',{'inner_train_reads':16,'inner_validation_reads':0,'outer_test_reads':0,'model_forward_calls':1,'backward_calls':1,'optimizer_step_count':0,'checkpoint_write_count':0});(report/'SO_R2X5_S0_B1_MH_Three_Class_Smoke_Report.md').write_text('# SO-R2-X5-S0 B1 M/H three-class smoke\n\nForward/backward interface validation only; no training or outer-test inference.\n',encoding='utf-8');acc={'status':'PASS_SMOKE_TEST' if ok else 'FAIL_SMOKE_TEST','input_contract_pass':bool(raw.shape==(16,2,320,256)),'logits_contract_pass':bool(logits.shape==(16,3)),'gradient_gate_pass':bool(all(x['finite'] and x['nonzero'] for x in audit.values()) and allfinite),'protected_assets_unchanged':unchanged,'optimizer_step_count':0,'checkpoint_write_count':0,'outer_test_forward_count':0,'formal_SO_R1_C_status':'FAIL','official_SO_R2_authorization':False,'SO_R3_authorization':False,'next_stage_authorized':False};dump(root/'data/processed/SO_R2X5_B1MHThreeClassSmokeTest_v1/S0_ACCEPTANCE.json',acc);return acc
if __name__=='__main__':print(json.dumps(run(),ensure_ascii=True,sort_keys=True))
