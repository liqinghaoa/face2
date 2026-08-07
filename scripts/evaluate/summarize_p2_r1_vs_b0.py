"""Raw performance and six-view frozen-relighting stability comparison for B0/R1."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
from datasets.control_patient_binary_dataset import ControlPatientFaceDataset
from datasets.p2_r1_relighting_dataset import P2R1PairedTransform,P2R1RelightingDataset,identity_pil
from metrics.binary_classification_metrics import compute_binary_metrics,flatten_metrics
from models.nyha_backbone_factory import build_nyha_classification_model
from utils.experiment_utils import load_yaml
def pp(x):p=Path(x);return p if p.is_absolute() else ROOT/p
def parse():p=argparse.ArgumentParser();p.add_argument('--experiment-dir',type=Path,required=True);return p.parse_args()
def model_for(cfg,ck,dev):
 m=build_nyha_classification_model('resnet18',2,False,False).to(dev);m.load_state_dict(torch.load(ck,map_location=dev,weights_only=False)['model_state_dict']);m.eval();return m
@torch.no_grad()
def collect(model,ds,dev):
 rows=[]
 for i in range(len(ds)):
  x=ds[i];raw=x['raw_image'].unsqueeze(0).to(dev);rawp=torch.softmax(model(raw),1)[0].cpu().numpy();views=[]
  for k in range(6):
   _,r=ds._arrays(i,k);_,rt=ds.transform(ds.base_dataset[i]['image'],r);views.append(float(torch.softmax(model(rt.unsqueeze(0).to(dev)),1)[0,1]))
  rv=np.asarray(views);rows.append({'sample_id':x['sample_id'],'patient_group_id':x['patient_group_id'],'true_label':int(x['label']),'original_nyha':int(x['original_nyha']),'raw_prob_patient':float(rawp[1]),'raw_pred':int(rawp.argmax()),'relight_probs':rv})
 return rows
def stability(rows):
 sh=np.array([np.abs(x['relight_probs']-x['raw_prob_patient']) for x in rows]);fl=np.array([((x['relight_probs']>=.5).astype(int)!=x['raw_pred']) for x in rows]);return {'MeanAbsShift':float(sh.mean()),'MeanMaxShift':float(sh.max(1).mean()),'ViewFlipRate':float(fl.mean()),'AnyFlipCaseRate':float(fl.any(1).mean()),'MeanRelightStd':float(np.array([x['relight_probs'].std() for x in rows]).mean())}
def main():
 out=pp(parse().experiment_dir);c=load_yaml(out/'config_snapshot.yaml');b0=pp(c['evaluation']['reference_b0_dir']);bc=load_yaml(b0/'config_snapshot.yaml');d=c['data'];base=ControlPatientFaceDataset(pp(d['val_csv']),identity_pil,pp(d['image_root']),d['image_filename_template']);ds=P2R1RelightingDataset(base,pp(c['p1']['root']),c['p1'],P2R1PairedTransform(False,c['normalize']['mean'],c['normalize']['std']),False);reference=pd.read_csv(b0/'p2_b0_predictions.csv',dtype={'sample_id':'string'});expected=set(base.frame['ID'].astype(str));
 if len(reference)!=100 or set(reference['sample_id'].astype(str))!=expected: raise ValueError('B0/R1 must use the exact same 100 fold-0 validation IDs')
 dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');bm=model_for(bc,b0/'fold_0/checkpoints/best_macro_auc.pth',dev);rm=model_for(c,out/'fold_0/checkpoints/best_raw_macro_auc.pth',dev);br,rr=collect(bm,ds,dev),collect(rm,ds,dev)
 def met(x):return compute_binary_metrics(np.array([z['true_label'] for z in x]),np.array([[1-z['raw_prob_patient'],z['raw_prob_patient']] for z in x]))
 mb,mr=met(br),met(rr);sb,sr=stability(br),stability(rr);cases=[]
 for a,b in zip(br,rr):
  ac=a['raw_pred']==a['true_label'];bcorr=b['raw_pred']==b['true_label'];trans='both_correct' if ac and bcorr else 'b0_wrong_r1_correct' if not ac and bcorr else 'b0_correct_r1_wrong' if ac else 'both_wrong';cases.append({'sample_id':a['sample_id'],'patient_group_id':a['patient_group_id'],'true_label':a['true_label'],'original_nyha':a['original_nyha'],'b0_raw_prob_patient':a['raw_prob_patient'],'r1_raw_prob_patient':b['raw_prob_patient'],'b0_raw_pred':a['raw_pred'],'r1_raw_pred':b['raw_pred'],'b0_correct':ac,'r1_correct':bcorr,'b0_mean_abs_relight_shift':float(np.abs(a['relight_probs']-a['raw_prob_patient']).mean()),'r1_mean_abs_relight_shift':float(np.abs(b['relight_probs']-b['raw_prob_patient']).mean()),'b0_max_relight_shift':float(np.abs(a['relight_probs']-a['raw_prob_patient']).max()),'r1_max_relight_shift':float(np.abs(b['relight_probs']-b['raw_prob_patient']).max()),'b0_any_flip':bool(((a['relight_probs']>=.5).astype(int)!=a['raw_pred']).any()),'r1_any_flip':bool(((b['relight_probs']>=.5).astype(int)!=b['raw_pred']).any()),'correctness_transition':trans})
 frame=pd.DataFrame(cases);frame.to_csv(out/'p2_r1_vs_b0_case_comparison.csv',index=False,encoding='utf-8-sig');pd.DataFrame([{'model':'B0',**flatten_metrics(mb),**sb},{'model':'R1',**flatten_metrics(mr),**sr}]).to_csv(out/'p2_r1_vs_b0_metrics.csv',index=False);json.dump(sb,open(out/'b0_relighting_stability.json','w'),indent=2);json.dump(sr,open(out/'r1_relighting_stability.json','w'),indent=2)
 fn_b0=int(((frame.true_label==1)&(frame.b0_raw_pred==0)).sum());fp_b0=int(((frame.true_label==0)&(frame.b0_raw_pred==1)).sum());detail={'b0_wrong_r1_correct':int((frame.correctness_transition=='b0_wrong_r1_correct').sum()),'b0_correct_r1_wrong':int((frame.correctness_transition=='b0_correct_r1_wrong').sum()),'b0_patient_fn_corrected':int(((frame.true_label==1)&(frame.b0_raw_pred==0)&(frame.r1_raw_pred==1)).sum()),'b0_control_fp_corrected':int(((frame.true_label==0)&(frame.b0_raw_pred==1)&(frame.r1_raw_pred==0)).sum()),'r1_patient_fn':int(((frame.true_label==1)&(frame.r1_raw_pred==0)).sum()),'r1_control_fp':int(((frame.true_label==0)&(frame.r1_raw_pred==1)).sum())};stable=mr['macro_auc']>=mb['macro_auc']-.01 and all(sr[k]<sb[k] for k in sb);improve=mr['macro_auc']>mb['macro_auc'] or (mr['balanced_accuracy']>mb['balanced_accuracy'] or mr['macro_f1']>mb['macro_f1']);decision={'classification_signal':bool(improve),'stability_signal':bool(stable),'recommend_p3':bool(improve or stable),'details':detail,'b0_patient_fn':fn_b0,'b0_control_fp':fp_b0};(out/'p2_r1_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2),encoding='utf-8');(out/'p2_r1_summary.md').write_text(f'# P2-R1 vs B0\n\n决策：`{decision["recommend_p3"]}`。B0/R1 原始 AUC：{mb["macro_auc"]:.4f}/{mr["macro_auc"]:.4f}。\n\nB0稳定性：{sb}\n\nR1稳定性：{sr}\n\n配对变化：{detail}\n',encoding='utf-8');print(f'P2_R1_SUMMARY_DIR={out}')
if __name__=='__main__':main()
