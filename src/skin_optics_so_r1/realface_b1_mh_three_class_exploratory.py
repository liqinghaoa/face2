"""Independent SO-R2-X5 B1 M/H-only three-class exploratory pipeline.

No existing X3/X4 implementation or frozen result is modified. RGB tensor bytes
are never materialised: only B1 channels 3 and 4 are loaded through mmap.
"""
from __future__ import annotations
import hashlib, json, random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

FIELDS = ["ID", "patient_group_id", "SEX", "NYHA", "fold"]
CLASSES = ("Normal", "Mild", "Severe")

def _seed(v:int)->None:
    random.seed(v); np.random.seed(v); torch.manual_seed(v); torch.cuda.manual_seed_all(v)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False

def _dump(p:Path,x:Any)->None:
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,ensure_ascii=True,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')

def three_class_metrics(y:Any, prob:Any)->dict[str,Any]:
    y=np.asarray(y,dtype=int); p=np.asarray(prob,dtype=float); pred=p.argmax(1)
    if p.shape!=(len(y),3) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError('invalid three-class probability')
    cm=confusion_matrix(y,pred,labels=[0,1,2]); recall=np.divide(np.diag(cm),cm.sum(1),out=np.zeros(3),where=cm.sum(1)!=0)
    specificity=[]
    for k in range(3):
        rest=cm.sum()-cm[k,:].sum()-cm[:,k].sum()+cm[k,k]; neg=cm.sum()-cm[k,:].sum()
        specificity.append(float(rest/neg) if neg else 0.)
    return {'macro_ovr_auc':float(roc_auc_score(y,p,multi_class='ovr',average='macro')),'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,average='macro',zero_division=0)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'class_recall':recall.tolist(),'class_specificity':specificity,'confusion_matrix':cm}

def load_table(root:Path)->pd.DataFrame:
    p=root/'data/processed/global_face_R3DPR/nyha_2class_sex_stratified_group_5fold.csv'
    d=pd.read_csv(p,usecols=FIELDS,dtype={'ID':str,'patient_group_id':str})
    if len(d)!=500 or d.ID.duplicated().any() or set(d.fold)!={0,1,2,3,4} or set(d.NYHA)!={0,1,2,3,4}: raise RuntimeError('BLOCKED_X5_SPLIT_CONTRACT')
    d['class3']=d.NYHA.map(lambda x:0 if x==0 else 1 if x in (1,2) else 2).astype(int)
    if d.class3.value_counts().sort_index().tolist()!=[115,238,147]: raise RuntimeError('BLOCKED_X5_LABEL_MAPPING')
    return d

def inner_split(development:pd.DataFrame, outer_fold:int)->tuple[pd.DataFrame,pd.DataFrame]:
    strata=development.class3.astype(str)+'__'+development.SEX.astype(str)
    tr,va=next(StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=2026+outer_fold).split(development,strata,development.patient_group_id))
    train,valid=development.iloc[tr].copy(),development.iloc[va].copy()
    if set(train.ID)&set(valid.ID) or set(train.patient_group_id)&set(valid.patient_group_id): raise RuntimeError('BLOCKED_X5_GROUP_LEAKAGE')
    if len(train)+len(valid)!=400 or set(valid.class3)!={0,1,2}: raise RuntimeError('BLOCKED_X5_INNER_SPLIT')
    return train,valid

class B1MHDataset(Dataset):
    def __init__(self,frame:pd.DataFrame,image_dir:Path,train:bool): self.frame,self.image_dir,self.train=frame.reset_index(drop=True),image_dir,train
    def __len__(self): return len(self.frame)
    def __getitem__(self,i:int)->dict[str,Any]:
        r=self.frame.iloc[i]; a=np.load(self.image_dir/f'{r.ID}.npy',mmap_mode='r',allow_pickle=False)
        if a.shape!=(5,320,256) or a.dtype!=np.float32: raise RuntimeError('BLOCKED_B1_MH_CONTRACT')
        x=np.array(a[[3,4]],dtype=np.float32,copy=True)
        if not np.isfinite(x).all(): raise RuntimeError('BLOCKED_B1_MH_NONFINITE')
        t=torch.from_numpy(x)
        if self.train and bool(torch.rand(())<.5): t=torch.flip(t,dims=(2,))
        return {'image':(t-.5)/.5,'label':torch.tensor(int(r.class3)), 'ID':str(r.ID), 'group':str(r.patient_group_id), 'fold':int(r.fold)}

def build_model()->nn.Module:
    m=resnet18(weights=ResNet18_Weights.IMAGENET1K_V1); old=m.conv1; conv=nn.Conv2d(2,old.out_channels,old.kernel_size,old.stride,old.padding,bias=False)
    with torch.no_grad(): conv.weight[:]=old.weight.mean(1,keepdim=True).repeat(1,2,1,1)
    m.conv1=conv; m.fc=nn.Linear(m.fc.in_features,3); return m

def _loader(f:pd.DataFrame,d:Path,train:bool,seed:int)->DataLoader:
    g=torch.Generator();g.manual_seed(seed);return DataLoader(B1MHDataset(f,d,train),batch_size=16,shuffle=train,num_workers=0,pin_memory=True,generator=g)

def _eval(m:nn.Module,l:DataLoader,w:torch.Tensor,dev:torch.device)->tuple[dict[str,Any],pd.DataFrame]:
    m.eval(); ys=[];ps=[];ids=[];folds=[]
    with torch.no_grad():
        for b in l:
            p=torch.softmax(m(b['image'].to(dev,non_blocking=True)),1).cpu().numpy();ys+=b['label'].tolist();ps.append(p);ids+=b['ID'];folds+=b['fold'].tolist()
    p=np.concatenate(ps); return three_class_metrics(ys,p),pd.DataFrame({'case_id':ids,'outer_fold':folds,'true_label':ys,'prob_normal':p[:,0],'prob_mild':p[:,1],'prob_severe':p[:,2],'predicted_label':p.argmax(1)})

def run(root:Path)->dict[str,Any]:
    root=root.resolve()
    if not torch.cuda.is_available(): raise RuntimeError('BLOCKED_CUDA_REQUIRED')
    gate=json.loads((root/'data/processed/SO_R2X3_ClassifierInputs_v1/CLASSIFIER_INPUT_ACCEPTANCE.json').read_text())
    if gate.get('status')!='COMPLETE_CLASSIFIER_INPUT_PREPARATION': raise RuntimeError('BLOCKED_X5_INPUT_GATE')
    table=load_table(root); image_dir=root/'data/processed/SO_R2X3_ClassifierInputs_v1/rgb_b1mh/images'; dev=torch.device('cuda'); rows=[];preds=[];audits=[]
    for fold in range(5):
        development,outer=table[table.fold!=fold].copy(),table[table.fold==fold].copy();train,val=inner_split(development,fold); seed=2026+fold;_seed(seed)
        out=root/'runs/so_r2x5_b1_mh_three_class_exploratory'/f'fold_{fold}';(out/'checkpoints').mkdir(parents=True,exist_ok=True)
        for n,f in [('inner_train',train),('inner_validation',val),('outer_test',outer)]:f.to_csv(out/f'{n}.csv',index=False,encoding='utf-8-sig')
        counts=np.bincount(train.class3,minlength=3); w=torch.tensor(len(train)/(3*counts),dtype=torch.float32,device=dev);m=build_model().to(dev).float();opt=torch.optim.AdamW(m.parameters(),lr=1e-4,weight_decay=1e-4);best=-np.inf;best_epoch=0;stale=0;hist=[];ckpt=out/'checkpoints/inner_best_macro_ovr_auc.pt'
        for epoch in range(1,51):
            m.train();loss_sum=0.
            for b in _loader(train,image_dir,True,seed):
                opt.zero_grad(set_to_none=True);loss=F.cross_entropy(m(b['image'].to(dev,non_blocking=True)),b['label'].to(dev),weight=w,label_smoothing=.05);loss.backward();opt.step();loss_sum+=float(loss.detach())*len(b['label'])
            metric,_=_eval(m,_loader(val,image_dir,False,seed),w,dev);hist.append({'epoch':epoch,'train_loss':loss_sum/len(train),'inner_val_macro_ovr_auc':metric['macro_ovr_auc']})
            if metric['macro_ovr_auc']>best:best,best_epoch,stale=float(metric['macro_ovr_auc']),epoch,0;torch.save({'state_dict':m.state_dict(),'selected_epoch':epoch,'seed':seed},ckpt)
            else:stale+=1
            if stale>=10:break
        pd.DataFrame(hist).to_csv(out/'inner_selection_history.csv',index=False,encoding='utf-8-sig');m.load_state_dict(torch.load(ckpt,map_location=dev,weights_only=True)['state_dict']);metric,pred=_eval(m,_loader(outer,image_dir,False,seed),w,dev);pred['selected_epoch']=best_epoch;pred['model_name']='B1_MH_ONLY_3CLASS';pred.to_csv(out/'outer_test_predictions.csv',index=False,encoding='utf-8-sig');rows.append({'fold':fold,'selected_epoch':best_epoch,'inner_best_macro_ovr_auc':best,**{k:metric[k] for k in ('macro_ovr_auc','accuracy','macro_f1','balanced_accuracy')}});preds.append(pred);audits.append({'fold':fold,'train':len(train),'validation':len(val),'outer':len(outer),'group_overlap_train_validation':0,'id_overlap_train_outer':0,'id_overlap_validation_outer':0});del m;torch.cuda.empty_cache()
    oof=pd.concat(preds,ignore_index=True);metric=three_class_metrics(oof.true_label,oof[['prob_normal','prob_mild','prob_severe']]);report=root/'reports/so_r2x5_b1_mh_three_class_exploratory';report.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(report/'fold_metrics.csv',index=False);oof.to_csv(report/'oof_predictions.csv',index=False);pd.DataFrame([{'scope':'pooled_oof',**{k:v for k,v in metric.items() if k not in ('confusion_matrix','class_recall','class_specificity')}}]).to_csv(report/'oof_metrics.csv',index=False);pd.DataFrame(metric['confusion_matrix'],index=CLASSES,columns=CLASSES).to_csv(report/'oof_confusion_matrix.csv');_dump(report/'split_leakage_audit.json',{'folds':audits,'all_pass':True});_dump(report/'input_provenance_audit.json',{'source':'rgb_b1mh','channels':[3,4],'rgb_payload_materialised':False});_dump(report/'field_access_audit.json',{'fields':FIELDS,'NYHA_used_only_for_frozen_three_class_mapping':True,'no_other_clinical_fields':True});(report/'SO_R2X5_B1_MH_Three_Class_Report.md').write_text('# SO-R2-X5 B1 M/H-only three-class exploratory classification\n\nInternal OOF exploration only; no winner or authorization change.\n',encoding='utf-8');acc={'status':'COMPLETE_EXPLORATORY_B1_MH_THREE_CLASS','formal_SO_R1_C_status':'FAIL','official_SO_R2_authorization':False,'SO_R3_authorization':False,'next_stage_authorized':False,'classes':list(CLASSES),'oof_cases':int(len(oof))};_dump(root/'data/processed/SO_R2X5_B1MHThreeClassExploratory_v1/X5_ACCEPTANCE.json',acc);return acc
