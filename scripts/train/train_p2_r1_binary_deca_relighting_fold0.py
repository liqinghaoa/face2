"""Fixed fold-0 P2-R1 trainer: shared ResNet18 over raw and frozen relighting views."""
from __future__ import annotations
import argparse,json,logging,time,sys
from datetime import datetime
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch.utils.data import DataLoader
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT)) if str(ROOT) not in sys.path else None
from datasets.control_patient_binary_dataset import ControlPatientFaceDataset
from datasets.p2_r1_relighting_dataset import P2R1PairedTransform,P2R1RelightingDataset,identity_pil
from losses.classification_losses import compute_class_weights
from losses.p2_r1_consistency_loss import js_divergence,consistency_weight
from metrics.binary_classification_metrics import compute_binary_metrics,flatten_metrics
from models.nyha_backbone_factory import build_nyha_classification_model,count_parameters
from scripts.train.train_e0b_global_resnet18_control_patient_binary_5fold import loader,save_environment
from utils.e0b_binary_audit import audit_fixed_splits
from utils.experiment_utils import load_yaml,save_yaml,set_random_seed,configure_logging
LOG=logging.getLogger('p2_r1')
def pp(v): p=Path(v);return p if p.is_absolute() else ROOT/p
def args():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--epochs',type=int);return p.parse_args()
def validate(c,smoke=False):
 if c['task']['type']!='binary_control_vs_patient' or c['task']['num_classes']!=2 or c['data']['fold']!=0:raise ValueError('P2-R1 fixed binary fold0 protocol')
 if c['model']['num_classes']!=2 or c['model']['backbone']!='resnet18' or c['model']['freeze_backbone']:raise ValueError('P2-R1 requires full-finetuned ResNet18/2')
 if c['training']['monitor']!='raw_val_macro_auc' or c['training']['amp'] or c['evaluation']['threshold_search']:raise ValueError('P2-R1 monitor/AMP/threshold contract')
 if c['relighting']['num_presets']!=6 or len(c['p1']['preset_names'])!=6:raise ValueError('P2-R1 requires six fixed presets')
 if c.get('p2',{}).get('use_deca_latents') is not False or c['p2'].get('use_reliability_gate') is not False or c['p2'].get('use_auxiliary_features') is not False:raise ValueError('P2-R1 forbids latent, gate, and auxiliary inputs')
 t=c['training'];
 if t['optimizer'].lower()!='adamw' or t['learning_rate']!=1e-4 or t['weight_decay']!=1e-4 or t['seed']!=2026 or t['batch_size'] not in (8,16):raise ValueError('P2-R1 fixed optimization protocol')
def build(c,train):
 d=c['data'];base=ControlPatientFaceDataset(pp(d['train_csv'] if train else d['val_csv']),identity_pil,pp(d['image_root']),d['image_filename_template']);tf=P2R1PairedTransform(train,c['normalize']['mean'],c['normalize']['std']);return P2R1RelightingDataset(base,pp(c['p1']['root']),c['p1'],tf,train)
@torch.no_grad()
def evaluate(model,ds,device,epoch,ckpt):
 dl=DataLoader(ds,batch_size=16,shuffle=False,num_workers=0);rows=[];ys=[];ps=[];model.eval()
 for b in dl:
  logits=model(b['raw_image'].to(device));prob=torch.softmax(logits,1).cpu().numpy();log=logits.cpu().numpy();lab=b['label'].numpy();ys.extend(lab);ps.append(prob)
  for i in range(len(lab)):rows.append({'sample_id':b['sample_id'][i],'patient_group_id':b['patient_group_id'][i],'fold':0,'original_label':int(b['original_nyha'][i]),'original_three_class_label':int(b['original_three_class_label'][i]),'binary_label':int(lab[i]),'logit_normal':float(log[i,0]),'logit_patient':float(log[i,1]),'prob_normal':float(prob[i,0]),'prob_patient':float(prob[i,1]),'pred_class':int(prob[i].argmax()),'is_correct':int(prob[i].argmax()==lab[i]),'image_path':str(ds.base_dataset.image_root/f"{b['sample_id'][i]}.png"),'selected_epoch':epoch,'checkpoint_path':str(ckpt)})
 return pd.DataFrame(rows),compute_binary_metrics(np.asarray(ys),np.concatenate(ps))
def main():
 a=args();c=load_yaml(a.config);smoke=a.epochs is not None;validate(c,smoke);c['training']['max_epochs']=a.epochs or c['training']['max_epochs'];out=pp(a.output_dir);out.mkdir(parents=True,exist_ok=True);configure_logging(out/'experiment.log');save_yaml(c,out/'config_snapshot.yaml');save_environment(out);audit_fixed_splits(c,out);set_random_seed(c['training']['seed']);device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 tr,va=build(c,True),build(c,False);tl=loader(tr,c['training']['batch_size'],True,c['training']['num_workers'],2026,c['training']['pin_memory']);model=build_nyha_classification_model('resnet18',2,'imagenet',False).to(device);weights=compute_class_weights(tr.base_dataset.labels,2).to(device);criterion=torch.nn.CrossEntropyLoss(weight=weights);opt=torch.optim.AdamW(model.parameters(),lr=c['training']['learning_rate'],weight_decay=c['training']['weight_decay']);fd=out/'fold_0';cd=fd/'checkpoints';cd.mkdir(parents=True,exist_ok=True);(fd/'training_parameters.json').write_text(json.dumps({'fold':0,'train_normal_count':92,'train_patient_count':308,'weight_normal':float(weights[0]),'weight_patient':float(weights[1]),**count_parameters(model),'patient_batch_size':c['training']['batch_size']},indent=2),encoding='utf-8')
 best=-np.inf;pat=0;hist=[];start=time.perf_counter()
 for e in range(int(c['training']['max_epochs'])):
  tr.set_epoch(e);model.train();tot=rawv=relv=jsv=0.;n=0;lam=consistency_weight(e,c['loss']['consistency_start_epoch'],c['loss']['consistency_weight'])
  for b in tl:
   raw,rel,y=b['raw_image'].to(device),b['relight_image'].to(device),b['label'].to(device);z=model(torch.cat([raw,rel]));rl,vl=z[:len(y)],z[len(y):];rce=criterion(rl,y);lce=criterion(vl,y);js=js_divergence(rl,vl);loss=rce+.5*lce+lam*js;opt.zero_grad(set_to_none=True);loss.backward();opt.step();q=len(y);n+=q;tot+=loss.item()*q;rawv+=rce.item()*q;relv+=lce.item()*q;jsv+=js.item()*q
  _,m=evaluate(model,va,device,e+1,cd/'best_raw_macro_auc.pth');row={'epoch':e+1,'learning_rate':c['training']['learning_rate'],'lambda_cons':lam,'train_total_loss':tot/n,'train_raw_ce':rawv/n,'train_relight_ce':relv/n,'train_js':jsv/n,**{'raw_val_'+k:v for k,v in flatten_metrics(m).items()}};hist.append(row);auc=m['macro_auc']
  if auc>best:best=auc;pat=0;torch.save({'model_state_dict':model.state_dict(),'epoch':e+1,'best_raw_macro_auc':auc,'config':c},cd/'best_raw_macro_auc.pth')
  else:pat+=1
  torch.save({'model_state_dict':model.state_dict(),'epoch':e+1,'best_raw_macro_auc':best,'config':c},cd/'last.pth');LOG.info('epoch=%d total=%.5f raw_auc=%.4f best=%.4f patience=%d/%d',e+1,row['train_total_loss'],auc,best,pat,c['training']['early_stopping_patience'])
  if pat>=c['training']['early_stopping_patience']:break
 pd.DataFrame(hist).to_csv(fd/'training_history.csv',index=False,encoding='utf-8-sig');ck=torch.load(cd/'best_raw_macro_auc.pth',map_location=device,weights_only=False);model.load_state_dict(ck['model_state_dict']);frame,m=evaluate(model,va,device,ck['epoch'],cd/'best_raw_macro_auc.pth');frame.to_csv(fd/'raw_val_predictions.csv',index=False,encoding='utf-8-sig');(fd/'raw_metrics.json').write_text(json.dumps(flatten_metrics(m),indent=2),encoding='utf-8');pd.DataFrame(m['confusion_matrix'],index=['control','patient'],columns=['control','patient']).to_csv(fd/'confusion_matrix.csv',index_label='true\\pred');(fd/'metrics.json').write_text(json.dumps({'best_epoch':ck['epoch'],'training_seconds':time.perf_counter()-start,**flatten_metrics(m)},indent=2),encoding='utf-8');print(f'P2_R1_EXPERIMENT_DIR={out}')
if __name__=='__main__':main()
