"""Train P1-RGB using only P0-aligned RGB and the immutable P0 five-fold split."""
from __future__ import annotations
import argparse, json, os, platform, random, subprocess, sys, time, hashlib
from datetime import datetime
from pathlib import Path
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights, resnet18

ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from datasets.nyha_3class_face_dataset import build_transforms
from datasets.p1_rgb_binary_dataset import P1RGBBinaryDataset
from losses.classification_losses import compute_class_weights
from metrics.binary_classification_metrics import compute_binary_metrics
from utils.experiment_utils import load_yaml, save_yaml, set_random_seed
from utils.p1_rgb_audit import aggregate_groups, image_equivalence, load_p1_frame, split_equivalence, write_source_metadata

def parse():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--fold',type=int,action='append');p.add_argument('--epochs',type=int);p.add_argument('--smoke',action='store_true');return p.parse_args()
def path(v): return Path(v) if Path(v).is_absolute() else ROOT/Path(v)
def device():
 if not torch.cuda.is_available():
  raise RuntimeError('P1-RGB formal and smoke training require CUDA. Activate the face2 CUDA environment before running.')
 return torch.device('cuda')
def make_model():
 m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1);m.fc=nn.Linear(m.fc.in_features,2)
 if not all(p.requires_grad for p in m.parameters()): raise RuntimeError('P1-RGB requires full finetune')
 return m
def metrics(y,p):
 base=compute_binary_metrics(y,p); pred=p.argmax(1); cm=np.asarray(base['confusion_matrix']);
 base.update({'pr_auc':float(average_precision_score(y,p[:,1])),'patient_sensitivity':float(recall_score(y,pred,pos_label=1,zero_division=0)),'control_specificity':float(recall_score(y,pred,pos_label=0,zero_division=0)),'ppv':float(precision_score(y,pred,pos_label=1,zero_division=0)),'npv':float(precision_score(y,pred,pos_label=0,zero_division=0) )}); return base
def metric_flat(m): return {k:float(v) for k,v in m.items() if np.isscalar(v)}
def loader(ds,batch,shuffle,seed):
 g=torch.Generator();g.manual_seed(seed);return DataLoader(ds,batch_size=batch,shuffle=shuffle,num_workers=0,pin_memory=False,generator=g)
@torch.no_grad()
def infer(model,dl,dev,fold,epoch,checkpoint):
 model.eval();out=[]; probs=[]; labels=[]
 for b in dl:
  logits=model(b['image'].to(dev)); pr=torch.softmax(logits,1).cpu().numpy(); lab=b['label_binary'].numpy(); probs.append(pr);labels.extend(lab.tolist())
  for i in range(len(lab)): out.append({'case_id':str(b['case_id'][i]),'patient_group_id':str(b['patient_group_id'][i]),'fold':int(b['fold'][i]),'label_original':int(b['label_original'][i]),'label_3class':int(b['label_3class'][i]),'label_binary':int(lab[i]),'prob_control':float(pr[i,0]),'prob_patient':float(pr[i,1]),'pred_binary':int(pr[i].argmax()),'best_epoch':epoch,'checkpoint_path':str(checkpoint),'image_path':str(b['image_path'][i])})
 frame=pd.DataFrame(out);return frame,metrics(np.asarray(labels),np.concatenate(probs))
def prepare(out,cfg):
 out.mkdir(parents=True,exist_ok=True);(out/'metadata').mkdir(exist_ok=True)
 manifest=path(cfg['data']['master_manifest']);split=path(cfg['data']['fixed_split']); frame=load_p1_frame(manifest,split)
 split_status=split_equivalence(frame,ROOT/'data/processed/splits_500',out/'metadata');image_status=image_equivalence(frame,ROOT/'data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images',out/'metadata');identity=write_source_metadata(frame,manifest,split,out/'metadata',split_status,image_status)
 report=f"# P1-RGB data audit\n\n- P1 manifest: `{manifest}`\n- Immutable split: `{split}`\n- Cases: {len(frame)}, patient groups: {frame.patient_group_id.nunique()}\n- Split equivalence: `{split_status}`\n- RGB equivalence: `{image_status}`\n- Protocol identity: `{identity}`\n- All 16 P0 QC cases remain in the 500-case manifest; no EXIF-based removal was performed.\n"
 (out/'data_audit_report.md').write_text(report,encoding='utf-8');return frame,identity,split_status,image_status
def environment(out,cfg,config_path):
 import torchvision
 payload={'started_at':datetime.now().isoformat(timespec='seconds'),'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'cuda_available':torch.cuda.is_available(),'cuda_version':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,'seed':cfg['training']['seed'],'config_sha256':hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),'git_commit':subprocess.run(['git','rev-parse','HEAD'],cwd=ROOT,capture_output=True,text=True).stdout.strip()}
 (out/'environment.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
def run_fold(frame,cfg,out,fold,epochs,dev):
 seed=int(cfg['training']['seed']);set_random_seed(seed);train=frame[frame.fold!=fold].copy();val=frame[frame.fold==fold].copy()
 tf=build_transforms('train',224,cfg['normalize']['mean'],cfg['normalize']['std'],True);vf=build_transforms('val',224,cfg['normalize']['mean'],cfg['normalize']['std'],False)
 tr=P1RGBBinaryDataset(train,tf);va=P1RGBBinaryDataset(val,vf); tl=loader(tr,16,True,seed+fold);vl=loader(va,16,False,seed+1000+fold)
 model=make_model().to(dev);weights=compute_class_weights(tr.labels,2).to(dev);expected=np.array([400/(2*92),400/(2*308)],dtype=np.float32)
 if not np.allclose(weights.cpu().numpy(),expected,rtol=1e-6): raise ValueError('fold-specific class weights incorrect')
 opt=torch.optim.AdamW(model.parameters(),lr=float(cfg['optimizer']['learning_rate']),weight_decay=float(cfg['optimizer']['weight_decay']));criterion=nn.CrossEntropyLoss(weight=weights)
 d=out/f'fold_{fold}';ck=d/'checkpoints';ck.mkdir(parents=True,exist_ok=True);history=[];best=-np.inf;best_epoch=0;patience=0;started=time.perf_counter()
 for epoch in range(1,epochs+1):
  model.train(); total=0.;seen=0
  for b in tl:
   x,y=b['image'].to(dev),b['label_binary'].to(dev);opt.zero_grad(set_to_none=True);loss=criterion(model(x),y)
   if not torch.isfinite(loss): raise RuntimeError('nonfinite loss')
   loss.backward();opt.step();total+=float(loss.detach())*len(y);seen+=len(y)
  _,vm=infer(model,vl,dev,fold,epoch,ck/'best_macro_auc.pth');improved=float(vm['macro_auc'])>best;history.append({'epoch':epoch,'train_loss':total/seen,'val_macro_auc':vm['macro_auc'],'val_accuracy':vm['accuracy'],'val_macro_precision':vm['macro_precision'],'val_macro_recall':vm['macro_recall'],'val_macro_f1':vm['macro_f1'],'val_balanced_accuracy':vm['balanced_accuracy'],'learning_rate':opt.param_groups[0]['lr'],'is_best':improved})
  torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'best_macro_auc':best if np.isfinite(best) else None},ck/'last.pth')
  if improved: best=float(vm['macro_auc']);best_epoch=epoch;patience=0;torch.save({'model_state_dict':model.state_dict(),'epoch':epoch,'best_macro_auc':best},ck/'best_macro_auc.pth')
  else: patience+=1
  if patience>=int(cfg['training']['early_stopping_patience']): break
 pd.DataFrame(history).to_csv(d/'training_history.csv',index=False,encoding='utf-8-sig')
 state=torch.load(ck/'best_macro_auc.pth',map_location=dev,weights_only=False);model.load_state_dict(state['model_state_dict']);case,cm= infer(model,vl,dev,fold,int(state['epoch']),ck/'best_macro_auc.pth');group=aggregate_groups(case)
 case.to_csv(d/'val_predictions_case.csv',index=False,encoding='utf-8-sig');group.to_csv(d/'val_predictions_group.csv',index=False,encoding='utf-8-sig');gm=metrics(group.label_binary.to_numpy(),group[['prob_control','prob_patient']].to_numpy())
 (d/'metrics_case.json').write_text(json.dumps({**metric_flat(cm),'confusion_matrix':np.asarray(cm['confusion_matrix']).tolist()},indent=2));(d/'metrics_group.json').write_text(json.dumps({**metric_flat(gm),'confusion_matrix':np.asarray(gm['confusion_matrix']).tolist()},indent=2))
 pd.DataFrame(cm['confusion_matrix'],index=['control','patient'],columns=['control','patient']).to_csv(d/'confusion_matrix_case.csv');pd.DataFrame(gm['confusion_matrix'],index=['control','patient'],columns=['control','patient']).to_csv(d/'confusion_matrix_group.csv')
 summary={'fold':fold,'best_epoch':int(state['epoch']),'training_seconds':time.perf_counter()-started,'train_n':len(train),'val_n':len(val),'train_control':int((train.label_binary==0).sum()),'train_patient':int((train.label_binary==1).sum()),'class_weight_control':float(weights[0]),'class_weight_patient':float(weights[1]),**metric_flat(cm),'group_count':len(group)}
 (d/'fold_summary.json').write_text(json.dumps(summary,indent=2));return summary
def main():
 a=parse();cfg=load_yaml(a.config);out=path(a.output_dir);save_yaml(cfg,out/'config_snapshot.yaml');environment(out,cfg,a.config);frame,_,_,_=prepare(out,cfg);folds=a.fold if a.fold else list(range(5));epochs=int(a.epochs or cfg['training']['max_epochs']);dev=device()
 for f in folds: run_fold(frame,cfg,out,f,epochs,dev)
 (out/'run_manifest.json').write_text(json.dumps({'status':'smoke_complete' if a.smoke else 'training_complete','folds':folds,'device':str(dev),'epochs':epochs,'finished_at':datetime.now().isoformat(timespec='seconds')},indent=2));print(f'P1_RGB_TRAINING_COMPLETE={out}')
if __name__=='__main__': main()
