"""P0-B2 engineering audit for representation non-collapse; no labels or training."""
from __future__ import annotations
import csv, hashlib, json, math, random
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image

PRIMARY = ("tex_code", "albedo_like", "shape_code", "detail_code", "normal_coarse")
APPEARANCE = {"tex_code", "albedo_like"}; STRUCTURE = {"shape_code", "detail_code", "normal_coarse"}
MODES=("direct_p0_aligned","mask_bbox_crop"); SEED=20260726

def _hash(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def _rows(p:Path)->list[dict[str,str]]:
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def _write(p:Path,rows:list[dict[str,Any]])->None:
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=sorted({k for r in rows for k in r}));w.writeheader();w.writerows(rows)
def _dist(a:np.ndarray,b:np.ndarray,kind:str)->float:
 a=a.reshape(-1).astype(float);b=b.reshape(-1).astype(float)
 if kind=='cosine':
  d=np.linalg.norm(a)*np.linalg.norm(b);return float(1-(a@b/d if d>1e-12 else 1.0))
 return float(np.sqrt(np.mean((a-b)**2)))
def _matrix(x:list[np.ndarray],kind:str)->np.ndarray:
 return np.array([[_dist(a,b,kind) for b in x] for a in x],dtype=np.float64)
def _effective(x:np.ndarray)->dict[str,Any]:
 x=x-x.mean(0,keepdims=True);s=np.linalg.svd(x,compute_uv=False);e=s*s
 p=e/e.sum() if e.sum()>0 else np.zeros_like(e);rank=float(np.exp(-(p[p>0]*np.log(p[p>0])).sum())) if p.any() else 0.
 c=np.cumsum(p);return {'effective_rank':rank,'effective_rank_fraction':rank/11,'nonzero_variance_dimensions':int((x.std(0)>1e-10).sum()),'pc90':int(np.searchsorted(c,.9)+1) if c.size else 0,'pc95':int(np.searchsorted(c,.95)+1) if c.size else 0,'singular_values':s,'cumulative_variance':c}
def _heat(a:np.ndarray,p:Path)->None:
 x=np.asarray(a,float); lo,hi=np.nanmin(x),np.nanmax(x);u=np.zeros_like(x,dtype=np.uint8) if hi<=lo else np.uint8(np.clip((x-lo)/(hi-lo)*255,0,255));Image.fromarray(u).resize((480,480),Image.Resampling.NEAREST).save(p)
def _pca_plot(x:np.ndarray,p:Path)->None:
 x=x-x.mean(0);u,s,_=np.linalg.svd(x,full_matrices=False);z=u[:,:2]*s[:2];im=Image.new('RGB',(480,480),'white');pix=im.load();
 for q in z:
  xx=int(240+(q[0]/(np.max(np.abs(z[:,0]))+1e-9))*190);yy=int(240-(q[1]/(np.max(np.abs(z[:,1]))+1e-9))*190)
  for i in range(-3,4):
   for j in range(-3,4):
    if 0<=xx+i<480 and 0<=yy+j<480:pix[xx+i,yy+j]=(20,70,160)
 im.save(p)

def _validate(root:Path)->tuple[list[str],dict[str,dict[str,Path]],list[dict[str,str]]]:
 base=root/'data/processed/P0B_DECA_Pilot12_v1'; metrics=_rows(base/'input_mode_comparison/p0b_b1_input_mode_case_metrics.csv'); mapping=_rows(base/'pilot_manifest/p0b_pilot12_id_mapping.csv')
 cases=[r['audit_id'] for r in mapping]; pairs={(r['audit_id'],r['input_mode']) for r in metrics}
 if len(cases)!=12 or len(set(cases))!=12 or pairs!={(c,m) for c in cases for m in MODES}:raise RuntimeError('expected exactly frozen 12x2 source runs')
 files={};
 for c in cases:
  files[c]={}
  for m in MODES:
   d=base/'input_mode_comparison'/m/c
   for n in ('physical_maps_float.npz','input.png','reconstruction.png','albedo_like.png','shading_like.png','normal_coarse.png','alpha_mask.png','residual_abs.png'):
    if not (d/n).is_file():raise FileNotFoundError(d/n)
   if not (base/'noncollapse_audit_v1/extracted_latent_codes'/m/f'{c}_latent_codes.npz').is_file():raise FileNotFoundError('missing validated latent code')
   files[c][m]=d
 return cases,files,mapping

def _inventory(out:Path,cases:list[str],files:dict[str,dict[str,Path]],base:Path)->None:
 rows=[]
 for c in cases:
  for m,d in files[c].items():
   for p in sorted(d.iterdir()):
    if p.is_file() and p.suffix in ('.png','.npz'):
     if p.suffix=='.npz':
      with np.load(p) as z:
       for k in z.files:
        a=z[k];rows.append({'case_id':c,'input_mode':m,'representation_name':k,'file_path':str(p),'shape':str(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()),'minimum':float(np.nanmin(a)),'maximum':float(np.nanmax(a)),'mean':float(np.nanmean(a)),'standard_deviation':float(np.nanstd(a)),'sha256':_hash(p)})
     else:
      a=cv2.imread(str(p),cv2.IMREAD_UNCHANGED);rows.append({'case_id':c,'input_mode':m,'representation_name':p.stem,'file_path':str(p),'shape':str(a.shape),'dtype':str(a.dtype),'finite':True,'minimum':int(a.min()),'maximum':int(a.max()),'mean':float(a.mean()),'standard_deviation':float(a.std()),'sha256':_hash(p)})
   p=base/'noncollapse_audit_v1/extracted_latent_codes'/m/f'{c}_latent_codes.npz'
   with np.load(p) as z:
    for k in z.files:
     a=z[k];rows.append({'case_id':c,'input_mode':m,'representation_name':f'{k}_code','file_path':str(p),'shape':str(a.shape),'dtype':str(a.dtype),'finite':bool(np.isfinite(a).all()),'minimum':float(a.min()),'maximum':float(a.max()),'mean':float(a.mean()),'standard_deviation':float(a.std()),'sha256':_hash(p)})
 _write(out/'representation_inventory.csv',rows)

def _image(a:np.ndarray,mask:np.ndarray)->np.ndarray:
 x=cv2.resize(a,(64,64),interpolation=cv2.INTER_AREA);m=cv2.resize(mask.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)>=.5
 return x,m
def _features(cases,files,base):
 data={r:{} for r in PRIMARY}; masks={}
 for c in cases:
  for m,d in files[c].items():
   with np.load(d/'physical_maps_float.npz') as z:
    masks[c,m]=z['alpha']>.5
    for r,k in [('albedo_like','albedo_like'),('normal_coarse','normal_coarse')]:data[r][c,m]=z[k]
   with np.load(base/'noncollapse_audit_v1/extracted_latent_codes'/m/f'{c}_latent_codes.npz') as z:
    data['tex_code'][c,m]=z['tex'].reshape(-1);data['shape_code'][c,m]=z['shape'].reshape(-1);data['detail_code'][c,m]=z['detail'].reshape(-1)
 # fixed direct mask frequency >=10/12
 fixed=sum(masks[c,'direct_p0_aligned'] for c in cases)>=10
 Image.fromarray((fixed*255).astype(np.uint8)).save(base/'noncollapse_audit_v1/analysis_mask.png')
 (base/'noncollapse_audit_v1/analysis_mask.json').write_text(json.dumps({'rule':'direct alpha >0.5 in at least 10 of 12','area_pixels':int(fixed.sum()),'resolution':[224,224]},indent=2))
 return data,masks,fixed
def _imgvec(a,mask,channel_mean,channel_std):
 x,m=_image(a,mask); y=(x-channel_mean)/channel_std;return y[m].reshape(-1)

def run(config:Any)->dict[str,Any]:
 root=config.project_root;base=config.output_root;out=base/'noncollapse_audit_v1';out.mkdir(exist_ok=True);(out/'plots').mkdir(exist_ok=True);(out/'logs').mkdir(exist_ok=True)
 cases,files,mapping=_validate(root); before={str(p):_hash(p) for c in cases for m in MODES for p in files[c][m].rglob('*') if p.is_file()}
 _inventory(out,cases,files,base);data,masks,fixed=_features(cases,files,base)
 group={r['audit_id']:r['acquisition_group'] for r in mapping}; all_keys=[(c,m) for c in cases for m in MODES]
 retrieval=[];summ=[];direct_summ=[];rank_rows=[];pca_rows=[];acq_rows=[];near=[];matrices={};representation_pass={};image_stats={}
 rng=np.random.default_rng(SEED)
 for rep in PRIMARY:
  is_img=rep in {'albedo_like','normal_coarse'}
  if is_img:
   values=np.concatenate([data[rep][k][masks[k]] for k in all_keys],axis=0);mean=values.mean(0);std=values.std(0);std[std<1e-8]=1;image_stats[rep]=(mean,std)
   direct_raw=[];direct_std=[]
   for c in cases:
    x,m=_image(data[rep][c,'direct_p0_aligned'],fixed);direct_raw.append(x[m].reshape(-1));direct_std.append(((x-mean)/std)[m].reshape(-1))
  else:
   allx=np.stack([data[rep][k] for k in all_keys]);mean=allx.mean(0);std=allx.std(0);std[std<1e-8]=1;direct_raw=[data[rep][c,'direct_p0_aligned'] for c in cases];direct_std=[(data[rep][c,'direct_p0_aligned']-mean)/std for c in cases]
  er=_effective(np.stack(direct_std));rank_rows.append({'representation':rep,**{k:v for k,v in er.items() if k not in ('singular_values','cumulative_variance') }});pca_rows.extend({'representation':rep,'component':i+1,'singular_value':float(s),'cumulative_variance':float(er['cumulative_variance'][i])} for i,s in enumerate(er['singular_values']))
  _pca_plot(np.stack(direct_std),out/'plots'/f'{rep}_pca.png');_heat(_matrix(direct_std,'normalized_rms_l2'),out/'plots'/f'{rep}_direct_distance.png')
  direct_matrix=_matrix(direct_std,'normalized_rms_l2');vals=direct_matrix[np.triu_indices(12,1)];direct_summ.append({'representation':rep,'distance_median':float(np.median(vals)),'distance_iqr':float(np.subtract(*np.percentile(vals,[75,25]))),'distance_cv':float(vals.std()/(vals.mean()+1e-12)),'minimum_nonzero_distance':float(vals[vals>0].min()) if np.any(vals>0) else 0})
  # near duplicate candidate criterion uses all distinct direct pairs, standardized metrics.
  cos=_matrix(direct_std,'cosine');rms=direct_matrix;tc=np.quantile(cos[np.triu_indices(12,1)],.005);tr=np.quantile(rms[np.triu_indices(12,1)],.005)
  for i in range(12):
   for j in range(i+1,12):
    if cos[i,j]<tc and rms[i,j]<tr:near.append({'representation':rep,'case_a':cases[i],'case_b':cases[j],'cosine_distance':float(cos[i,j]),'rms_distance':float(rms[i,j]),'near_duplicate_candidate':True})
  passes=[]
  for kind in ('cosine','normalized_rms_l2'):
   # standardized retrieval; image pairs use per-pair alpha intersection.
   dm=np.zeros((12,12))
   for i,c in enumerate(cases):
    for j,b in enumerate(cases):
     if is_img:
      mask=masks[c,'direct_p0_aligned']&masks[b,'mask_bbox_crop'];q=_imgvec(data[rep][c,'direct_p0_aligned'],mask,mean,std);g=_imgvec(data[rep][b,'mask_bbox_crop'],mask,mean,std)
     else:q=(data[rep][c,'direct_p0_aligned']-mean)/std;g=(data[rep][b,'mask_bbox_crop']-mean)/std
     dm[i,j]=_dist(q,g,kind)
   matrices[f'{rep}_{kind}']=dm; ranks=np.argsort(np.argsort(dm,axis=1),axis=1).diagonal()+1;match=np.diag(dm);unmatch=dm[~np.eye(12,dtype=bool)]
   perm_rank=[];perm_dist=[]
   for _ in range(10000):
    p=rng.permutation(12);perm_rank.append(float(np.mean(np.argsort(np.argsort(dm[:,p],axis=1),axis=1).diagonal()+1)));perm_dist.append(float(np.mean(dm[np.arange(12),p])))
   pr=(1+sum(x<=float(np.mean(ranks)) for x in perm_rank))/10001;pd=(1+sum(x<=float(np.mean(match)) for x in perm_dist))/10001
   for i,c in enumerate(cases):retrieval.append({'representation':rep,'distance':kind,'case_id':c,'mate_distance':float(match[i]),'mate_rank':int(ranks[i]),'top1':bool(ranks[i]<=1),'top3':bool(ranks[i]<=3),'mate_percentile':float(ranks[i]/12)})
   summ.append({'representation':rep,'distance':kind,'top1_accuracy':float(np.mean(ranks==1)),'top3_accuracy':float(np.mean(ranks<=3)),'median_mate_rank':float(np.median(ranks)),'mean_mate_percentile':float(np.mean(ranks/12)),'matched_distance_median':float(np.median(match)),'unmatched_distance_median':float(np.median(unmatch)),'matched_unmatched_ratio':float(np.median(match)/np.median(unmatch)),'mean_rank_permutation_p':pr,'matched_distance_permutation_p':pd})
   passes.append(np.mean(ranks==1)>=.5 and np.median(ranks)<=3 and pd<.05)
  representation_pass[rep]=bool(all(passes) and er['effective_rank']>=3)
  # acquisition warning calculation on direct standardized L2
  dm=_matrix(direct_std,'normalized_rms_l2');within=[];between=[];pred=[]
  for i,c in enumerate(cases):
   near_i=np.argsort(np.where(np.arange(12)==i,np.inf,dm[i]))[0];pred.append(group[cases[near_i]]==group[c]);
   for j in range(i+1,12):(within if group[c]==group[cases[j]] else between).append(dm[i,j])
  obs=float(np.mean(pred));labels=np.array([group[c] for c in cases]);perm=[]
  for _ in range(10000):
   lab=rng.permutation(labels);perm.append(float(np.mean([lab[np.argsort(np.where(np.arange(12)==i,np.inf,dm[i]))[0]]==lab[i] for i in range(12)])))
  p=(1+sum(x>=obs for x in perm))/10001;acq_rows.append({'representation':rep,'within_distance_mean':float(np.mean(within)),'between_distance_mean':float(np.mean(between)),'within_between_ratio':float(np.mean(within)/np.mean(between)),'loo_1nn_accuracy':obs,'permutation_p':p,'acquisition_group_dominance_warning':bool(p<.05 and obs>1/3)})
 # Image settings sensitivity: raw and globally standardized values, same paired masks.
 sensitivity=[];sensitivity_conflict=False
 for rep,(mean,std) in image_stats.items():
  setting_pass=[]
  for setting in ('raw','global_standardized'):
   metric_pass=[]
   for kind in ('cosine','normalized_rms_l2'):
    dm=np.zeros((12,12))
    for i,c in enumerate(cases):
     for j,b in enumerate(cases):
      mask=masks[c,'direct_p0_aligned']&masks[b,'mask_bbox_crop'];q,_=_image(data[rep][c,'direct_p0_aligned'],mask);g,_=_image(data[rep][b,'mask_bbox_crop'],mask);mm=cv2.resize(mask.astype(np.float32),(64,64),interpolation=cv2.INTER_AREA)>=.5
      if setting=='global_standardized':q=(q-mean)/std;g=(g-mean)/std
      dm[i,j]=_dist(q[mm].reshape(-1),g[mm].reshape(-1),kind)
    ranks=np.argsort(np.argsort(dm,axis=1),axis=1).diagonal()+1;matched=np.diag(dm);metric_pass.append(np.mean(ranks==1)>=.5 and np.median(ranks)<=3)
    sensitivity.append({'representation':rep,'setting':setting,'distance':kind,'top1_accuracy':float(np.mean(ranks==1)),'median_mate_rank':float(np.median(ranks)),'matched_distance_median':float(np.median(matched)),'retrieval_condition_passed':metric_pass[-1]})
   setting_pass.append(all(metric_pass))
  if setting_pass[0]!=setting_pass[1]:sensitivity_conflict=True
 _write(out/'sensitivity_analysis.csv',sensitivity)
 for item in near:
  a=files[item['case_a']]['direct_p0_aligned']/ 'input.png';b=files[item['case_b']]['direct_p0_aligned']/ 'input.png';ia=Image.open(a).convert('RGB');ib=Image.open(b).convert('RGB');panel=Image.new('RGB',(448,224));panel.paste(ia,(0,0));panel.paste(ib,(224,0));panel.save(out/'plots'/f"near_duplicate_{item['representation']}_{item['case_a']}_{item['case_b']}.png")
 np.savez_compressed(out/'cross_mode_distance_matrices.npz',**matrices);_write(out/'cross_mode_retrieval_per_case.csv',retrieval);_write(out/'cross_mode_retrieval_summary.csv',summ);_write(out/'direct_pairwise_distance_summary.csv',direct_summ);_write(out/'effective_rank_summary.csv',rank_rows);_write(out/'pca_summary.csv',pca_rows);_write(out/'acquisition_group_sensitivity.csv',acq_rows);_write(out/'near_duplicate_candidates.csv',near or [{'representation':'none','near_duplicate_candidate':False}])
 # reliability is deliberately QC only.
 rel=[];names=('neutral_front','left','right','top','dim_front','bright_front')
 for c in cases:
  d=files[c]['direct_p0_aligned'];imgs=[cv2.cvtColor(cv2.imread(str(d/'relighting'/f'{n}.png')),cv2.COLOR_BGR2RGB).astype(float)/255 for n in names];stack=np.stack(imgs);finite=bool(np.isfinite(stack).all());rel.append({'case_id':c,'relighting_finite':finite,'relighting_variation_rms':float(stack.std(0).mean()),'identical_relighting':len({_hash(d/'relighting'/f'{n}.png') for n in names})==1,'saturation_fraction':float((stack>=1).mean()),'black_fraction':float((stack<=0).mean()),'alpha_coverage':float(masks[c,'direct_p0_aligned'].mean()),'reconstruction_mae':float(np.abs(np.load(d/'physical_maps_float.npz')['residual_abs']).mean()),'residual_energy':float((np.load(d/'physical_maps_float.npz')['residual_abs']**2).mean())})
 _write(out/'case_reliability_metrics.csv',rel)
 appearance=[r for r in PRIMARY if r in APPEARANCE and representation_pass[r]];structure=[r for r in PRIMARY if r in STRUCTURE and representation_pass[r]];warn_acq=any(r['acquisition_group_dominance_warning'] for r in acq_rows);relight_ok=all(r['relighting_finite'] and not r['identical_relighting'] for r in rel)
 if sensitivity_conflict:decision='REVIEW_REQUIRED';stage='Resolve raw-versus-standardized image sensitivity conflict before any 500-case run.'
 elif appearance and structure and relight_ok:decision='GO_FULL_AUX';stage='Run controlled 500-case auxiliary DECA generation with camera/acquisition control.'
 elif structure and relight_ok:decision='GO_GEOMETRY_ONLY';stage='Run 500-case DECA only for structure/detail, light, relighting consistency and reliability.'
 elif not appearance and not structure:decision='STOP_DECA_REPRESENTATION';stage='Do not use DECA representations in the current mainline.'
 else:decision='REVIEW_REQUIRED';stage='Resolve sensitivity conflicts before any 500-case run.'
 result={'status':'completed','decision':decision,'fixed_input_mode':config.fixed_input_mode,'appearance_gate':bool(appearance),'structure_gate':bool(structure),'relighting_gate':relight_ok,'acquisition_group_dominance_warning':warn_acq,'failed_representations':[r for r in PRIMARY if not representation_pass[r]],'passed_representations':[r for r in PRIMARY if representation_pass[r]],'warnings':(['acquisition_group_dominance_warning'] if warn_acq else [])+(['sensitivity_conflict'] if sensitivity_conflict else []),'reasons':['engineering non-collapse audit; no clinical labels or classification used'],'recommended_next_stage':stage,'allowed_future_inputs':['original RGB','normal','shape/detail','light','relighting consistency','reconstruction reliability'] if decision!='STOP_DECA_REPRESENTATION' else ['original RGB only'],'prohibited_future_claims':['clinical validity','classification performance','true physiological reflectance','paper-ready efficacy'],'source_hashes_unchanged':before=={str(p):_hash(p) for c in cases for m in MODES for p in files[c][m].rglob('*') if p.is_file()},'sensitivity_conflict':sensitivity_conflict}
 (out/'noncollapse_decision.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));(out/'noncollapse_audit_summary.json').write_text(json.dumps({'cases':12,'modes':list(MODES),'primary_representations':list(PRIMARY),'source_hashes_unchanged':result['source_hashes_unchanged'],'decision':decision},ensure_ascii=False,indent=2))
 return result
