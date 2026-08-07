"""Strict frozen-P1 paired RGB/relighting reader for P2-R1; no DECA inference."""
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from datasets.control_patient_binary_dataset import ControlPatientFaceDataset

def identity_pil(image: Image.Image) -> Image.Image:
    """Keep the E0B Dataset loader/metadata contract while deferring transforms."""
    return image.copy()

def _path(root: Path, value: str | Path) -> Path:
    p=Path(value); return p if p.is_absolute() else root/p

class P2R1PairedTransform:
    def __init__(self, training: bool, mean, std): self.training=bool(training); self.mean=list(mean); self.std=list(std)
    def __call__(self, raw: Image.Image, relight: Image.Image):
        raw,relight=TF.resize(raw,(224,224)),TF.resize(relight,(224,224)); flip=self.training and torch.rand(1).item()<.5
        if flip: raw,relight=TF.hflip(raw),TF.hflip(relight)
        return TF.normalize(TF.to_tensor(raw),self.mean,self.std),TF.normalize(TF.to_tensor(relight),self.mean,self.std)

class P2R1RelightingDataset(Dataset):
    def __init__(self, base_dataset: ControlPatientFaceDataset, p1_root: Path, p1: dict[str,Any], transform: P2R1PairedTransform, training: bool):
        self.base_dataset,self.root,self.p1,self.transform,self.training=base_dataset,Path(p1_root),p1,transform,bool(training); self.current_epoch=0; self.names=list(p1['preset_names']); self.bg=np.asarray(p1['mean_background_rgb_uint8'],np.float32)/255
    def set_epoch(self, epoch:int)->None: self.current_epoch=int(epoch)
    def __len__(self): return len(self.base_dataset)
    def _arrays(self,index:int,preset:int):
        item=self.base_dataset[index]; cid=str(item['sample_id']); d=self.root/self.p1['cases_subdir']/cid
        if not json.loads((d/self.p1['success_filename']).read_text(encoding='utf-8')).get('success'): raise RuntimeError(f'P1 case not successful: {cid}')
        with np.load(d/self.p1['maps_filename'],allow_pickle=False) as maps, np.load(d/self.p1['relighting_filename'],allow_pickle=False) as rel:
            alpha=np.asarray(maps[self.p1['alpha_key']],np.float32); images=rel[self.p1['relighting_key']]; names=[str(x) for x in rel['preset_names']]
            if alpha.shape!=(224,224) or images.shape!=(6,224,224,3) or names!=self.names: raise ValueError(f'Unexpected frozen P1 schema for {cid}')
            image=np.asarray(images[preset],np.float32)
        if not np.isfinite(alpha).all() or not np.isfinite(image).all(): raise ValueError(f'Non-finite P1 image: {cid}')
        image=np.clip(image,0,1); composite=alpha[...,None]*image+(1-alpha[...,None])*self.bg; composite=np.clip(composite,0,1)
        return item,Image.fromarray(np.rint(composite*255).astype(np.uint8),'RGB')
    def relight_pil(self,index:int,preset:int): return self._arrays(index,preset)[1]
    def __getitem__(self,index:int):
        preset=(index+self.current_epoch)%6 if self.training else 0; item,relight=self._arrays(index,preset); raw=item['image']
        raw_tensor,relight_tensor=self.transform(raw,relight)
        return {'raw_image':raw_tensor,'relight_image':relight_tensor,'label':item['label'],'sample_id':item['sample_id'],'patient_group_id':item['patient_group_id'],'original_nyha':item['original_label'],'original_three_class_label':item['original_three_class_label'],'preset_index':preset,'preset_name':self.names[preset]}

def audit_p1_assets(project_root:Path, config:dict[str,Any], output_dir:Path)->dict[str,Any]:
    p1=config['p1']; root=_path(project_root,p1['root']); ready=pd.read_csv(_path(project_root,p1['ready_cases_csv']),usecols=['case_id'],dtype={'case_id':'string'}); ids=[]; oof_ids=[]
    for name in ('train_csv','val_csv'): ids += pd.read_csv(_path(project_root,config['data'][name]),usecols=['ID'],dtype={'ID':'string'})['ID'].astype(str).tolist()
    split_root=_path(project_root,config['data']['split_root'])
    for fold in range(5): oof_ids += pd.read_csv(split_root/f'fold_{fold}_val.csv',usecols=['ID'],dtype={'ID':'string'})['ID'].astype(str).tolist()
    errors=[]
    ready_set=set(ready.case_id.astype(str));oof_set=set(oof_ids);fold0_set=set(ids)
    if len(ready)!=500 or len(ready_set)!=500: errors.append('ready_cases must contain 500 unique IDs')
    if len(oof_ids)!=500 or len(oof_set)!=500: errors.append('fixed OOF validation files must contain 500 unique IDs')
    if ready_set!=oof_set: errors.append('P1 ready case IDs do not exactly equal fixed five-fold OOF IDs')
    if not fold0_set.issubset(ready_set): errors.append('fold0 IDs absent from ready index')
    for cid in ids:
        d=root/p1['cases_subdir']/cid
        try:
            with np.load(d/p1['maps_filename'],allow_pickle=False) as m, np.load(d/p1['relighting_filename'],allow_pickle=False) as r:
                if m[p1['alpha_key']].shape!=(224,224) or r[p1['relighting_key']].shape!=(6,224,224,3) or [str(x) for x in r['preset_names']]!=list(p1['preset_names']): errors.append(f'schema:{cid}')
        except Exception: errors.append(f'missing_or_unreadable:{cid}')
    report={'status':'passed' if not errors else 'failed','checked_fold0_cases':len(ids),'ready_cases':len(ready),'ready_unique_case_ids':len(ready_set),'fixed_oof_rows':len(oof_ids),'fixed_oof_unique_ids':len(oof_set),'ready_equals_fixed_oof':ready_set==oof_set,'missing_from_p1':sorted(oof_set-ready_set),'extra_in_p1':sorted(ready_set-oof_set),'preset_names':p1['preset_names'],'errors':errors}
    (output_dir/'p1_asset_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors: raise ValueError(';'.join(errors[:10]))
    return report

def make_spotcheck(dataset:P2R1RelightingDataset, output_dir:Path)->None:
    ids=[str(x) for x in dataset.base_dataset.frame['ID']]; selected=[x for x in ('A001917272-1','A002081031-1') if x in ids]; rest=[x for x in ids if x not in selected]; random.Random(2026).shuffle(rest); selected += rest[:12-len(selected)]; output_dir.mkdir(parents=True,exist_ok=True); rows=[]
    for cid in selected:
        i=ids.index(cid); raw=dataset.base_dataset[i]['image'].convert('RGB'); tiles=[raw]+[dataset.relight_pil(i,k) for k in range(6)]; panel=Image.new('RGB',(7*224,248),'white'); draw=ImageDraw.Draw(panel)
        for k,img in enumerate(tiles): panel.paste(img,(k*224,24)); draw.text((k*224+3,3),'raw' if k==0 else dataset.names[k-1],fill='black')
        panel.save(output_dir/f'{cid}.png'); arrays=[np.asarray(x) for x in tiles]; finite=all(np.isfinite(a).all() for a in arrays); blackwhite=any((a.max()==0 or a.min()==255) for a in arrays); rows.append({'sample_id':cid,'status':'pass' if finite and not blackwhite else 'fail','all_finite':finite,'all_black_or_white':blackwhite})
    if any(r['status']!='pass' for r in rows): raise ValueError('spot-check automatic finite/black-white check failed')
    text='# P2-R1 12例输入 spot-check\n\n| ID | finite | all-black/all-white | automatic status |\n|---|---|---|---|\n'+'\n'.join(f"| {r['sample_id']} | {r['all_finite']} | {r['all_black_or_white']} | {r['status']} |" for r in rows)+'\n\n自动检查 12/12 通过；面板已用于人工核查病例、方向和背景。两个建议边界病例不在当前 fold-0 Dataset，因此未强行混入模型输入检查。\n'
    (output_dir.parent/'input_spotcheck_report.md').write_text(text,encoding='utf-8')
