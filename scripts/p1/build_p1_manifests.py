from __future__ import annotations
import hashlib,json
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2]
P0=ROOT/'data/processed/P0_Physics_Audit_v1';OUT=ROOT/'data/manifests/p1'
def h(p):
 d=hashlib.sha256();d.update(p.read_bytes());return d.hexdigest()
def main():
 df=pd.read_csv(P0/'metadata/master_index.csv',dtype={'ID':str,'patient_group_id':str})
 if len(df)!=500 or df.ID.nunique()!=500:raise RuntimeError('P0 master must be frozen 500 unique cases')
 rows=[];acq=[];image_metrics={}
 for _,r in df.sort_values('ID').iterrows():
  path=(P0/str(r.aligned_scene_relpath)).resolve()
  if not path.is_file():raise FileNotFoundError(path)
  from PIL import Image
  with Image.open(path) as im:w,hgt=im.size
  image=cv2.imread(str(path),cv2.IMREAD_COLOR)
  if image is None:raise FileNotFoundError(path)
  scene=image.astype(np.float32)/255.0
  cid=str(r.ID);group=str(r.patient_group_id)
  mask=cv2.imread(str(P0/f'masks/physics_core_skin_224/{cid}.png'),cv2.IMREAD_GRAYSCALE)
  image_metrics[cid]={'input_brightness':float(scene.mean()),'input_saturation':float((scene>=1).mean()),'skin_fraction':float((mask>0).mean()) if mask is not None else float('nan')}
  rows.append({'case_id':cid,'patient_group_id':group,'fold_id':int(r.fold),'input_path_wsl':str(path),'input_sha256':h(path),'input_width':w,'input_height':hgt,'fixed_input_mode':'direct_p0_aligned','source_asset_version':'P0_Physics_Audit_v1','expected_output_dir':str(ROOT/'data/processed/P1_DECA_Full500_v1/cases'/cid),'canary_flag':False,'canary_reason':''})
  def val(k):return r[k] if k in r and pd.notna(r[k]) else ''
  acq.append({'case_id':cid,'patient_group_id':group,'camera_make':val('camera_make'),'camera_model':val('camera_model'),'exposure_time_seconds':val('exposure_time_s'),'f_number':val('f_number'),'iso':val('iso'),'brightness_value':val('brightness_value'),'focal_length':'','capture_datetime':val('datetime_original'),'exif_available':bool(val('camera_make') or val('iso')),'exposure_missing':not bool(val('exposure_time_s')),'f_number_missing':not bool(val('f_number')),'iso_missing':not bool(val('iso')),'brightness_missing':not bool(val('brightness_value')),'acquisition_group':f"{val('camera_make')} {val('camera_model')}".strip()})
 gen=pd.DataFrame(rows);ac=pd.DataFrame(acq);banned={'label','nyha','class','binary_label','three_class_label'}
 if any(any(x in c.lower() for x in banned) for c in gen.columns):raise RuntimeError('generation manifest label-field leak')
 OUT.mkdir(parents=True,exist_ok=True);gen.to_csv(OUT/'p1_generation_manifest_v1.csv',index=False);ac.to_csv(OUT/'p1_acquisition_metadata_v1.csv',index=False)
 # Deterministic coverage set: no clinical fields are involved.
 chosen={x for x in ['A001917272-1','A002081031-1'] if x in set(gen.case_id)};reasons={x:['prespecified_boundary_case'] for x in chosen}
 mp=pd.read_csv(ROOT/'data/processed/P0B_DECA_Pilot12_v1/pilot_manifest/p0b_pilot12_id_mapping.csv',dtype=str);cand=['P0B-008','P0B-012','P0B-005','P0B-010','P0B-002','P0B-007','P0B-006','P0B-011']
 for aid in cand:
  v=mp.loc[mp.audit_id.eq(aid),'sample_id']
  if len(v):chosen.add(v.iloc[0]);reasons.setdefault(v.iloc[0],[]).append('p0b2_near_duplicate_candidate')
 for col in ['camera_make','camera_model']:
  for _,g in ac.groupby(col):
   x=sorted(g.case_id)[0];chosen.add(x);reasons.setdefault(x,[]).append(f'{col}_representative')
 for col in ['exposure_time_seconds','iso','brightness_value']:
  x=pd.to_numeric(ac[col],errors='coerce');
  for i in list(x.nsmallest(2).index)+list(x.nlargest(2).index):
   cid=ac.loc[i,'case_id'];chosen.add(cid);reasons.setdefault(cid,[]).append(f'{col}_extreme')
 # Missing EXIF and image-domain extremes are explicit canary strata, rather
 # than being accidentally represented only through numeric EXIF extremes.
 missing=ac.loc[~ac.exif_available.astype(bool),'case_id'].sort_values().tolist()
 if missing:
  cid=missing[0];chosen.add(cid);reasons.setdefault(cid,[]).append('exif_missing')
 metric=pd.DataFrame.from_dict(image_metrics,orient='index').rename_axis('case_id').reset_index()
 for col in ['input_brightness','input_saturation','skin_fraction']:
  vals=pd.to_numeric(metric[col],errors='coerce').dropna()
  for i in [vals.idxmin(),vals.idxmax()]:
   cid=metric.loc[i,'case_id'];chosen.add(cid);reasons.setdefault(cid,[]).append(f'{col}_extreme')
 for _,g in ac.groupby('patient_group_id'):
  if len(g)>=2:
   cid=sorted(g.case_id)[0];chosen.add(cid);reasons.setdefault(cid,[]).append('multi_image_patient_group')
 # Preserve prescribed cases, then greedily maximize required-stratum coverage
 # before deterministic case-id filling.  This avoids lexical truncation.
 prescribed=[x for x in ['A001917272-1','A002081031-1'] if x in chosen]
 p0b2_cases=sorted(cid for cid,tags in reasons.items() if 'p0b2_near_duplicate_candidate' in tags)
 priority=prescribed+p0b2_cases
 required={'p0b2_near_duplicate_candidate','camera_make_representative','camera_model_representative','exif_missing','exposure_time_seconds_extreme','iso_extreme','brightness_value_extreme','input_brightness_extreme','input_saturation_extreme','skin_fraction_extreme','multi_image_patient_group'}
 ordered=list(priority);covered={tag for cid in ordered for tag in reasons[cid]}
 while len(ordered)<24:
  remaining=[cid for cid in sorted(chosen) if cid not in ordered]
  if not remaining:break
  cid=max(remaining,key=lambda x:(len((set(reasons[x])&required)-covered),len(set(reasons[x])&required),-len(x),x))
  ordered.append(cid);covered.update(reasons[cid])
 if len(ordered)<24:
  for cid in sorted(gen.case_id):
   if cid not in chosen:ordered.append(cid);reasons[cid]=['case_id_fill'];
   if len(ordered)==24:break
 can=gen[gen.case_id.isin(ordered)].copy();can['canary_flag']=True;can['canary_reason']=can.case_id.map(lambda x:';'.join(reasons[x]));can=can.set_index('case_id').loc[ordered].reset_index();can.to_csv(OUT/'p1_canary_manifest_v1.csv',index=False)
 print({'generation_cases':len(gen),'canary_cases':len(can),'label_columns':False})
if __name__=='__main__':main()
