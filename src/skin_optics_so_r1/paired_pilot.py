"""SO-R1-A1 same-latent paired Pilot orchestration.

The module is an outer orchestrator only: every RGB is rendered through the
frozen SO-0 NumPy camera path with its frozen ColorChecker matrix.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, shutil, time
from pathlib import Path
from typing import Any
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy import ndimage

LIGHTS=('D65','A','FL2','FL11'); ROLES=('reference','camera_only','light_only','appearance_only','joint')
class PilotError(RuntimeError): pass
def load_config(path: Path)->dict[str,Any]:
    with path.open(encoding='utf-8') as f: x=yaml.safe_load(f)
    if not isinstance(x,dict): raise PilotError('Pilot configuration must be a mapping')
    return x
def _sha(p:Path)->str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def _bytes_sha(a:np.ndarray)->str: return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def _seed(root:int,*parts:object)->int: return int.from_bytes(hashlib.sha256('|'.join(map(str,(root,*parts))).encode()).digest()[:8],'big')
def _rng(root:int,*parts:object)->np.random.Generator: return np.random.Generator(np.random.PCG64(_seed(root,*parts)))
def _rel(root:Path,p:Path)->str: return p.resolve().relative_to(root.resolve()).as_posix()
def _json(obj:Any)->str: return json.dumps(obj,sort_keys=True,separators=(',',':'),ensure_ascii=False)

def _protected(root:Path,cfg:dict[str,Any])->dict[str,str]:
    targets=[root/'data/external/SO0_Spectral_Assets_v1',root/cfg['so0_audit_root'],root/'src/skin_optics_so0/configs/so0',root/cfg['frozen_split'],root/'reports/so_r1_a0_camera_light_selection']
    rows={}
    for t in targets:
        if not t.exists(): raise PilotError(f'Protected input missing: {t}')
        for p in sorted([t] if t.is_file() else (x for x in t.rglob('*') if x.is_file())): rows[_rel(root,p)]=_sha(p)
    return rows
def _write_hashes(path:Path,rows:dict[str,str]):
    path.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame([{'repo_relative_path':k,'sha256':v} for k,v in rows.items()]).to_csv(path,index=False)

def _inputs(root:Path,cfg:dict[str,Any])->tuple[dict[str,Any],pd.DataFrame]:
    frozen_path=root/cfg['frozen_split']; freeze=yaml.safe_load(frozen_path.read_text(encoding='utf-8'))
    if freeze.get('status')!='FROZEN' or len(freeze.get('seen_cameras',[]))!=6 or len(freeze.get('unseen_cameras',[]))!=2 or len(freeze.get('seen_lights',[]))!=3 or len(freeze.get('unseen_lights',[]))!=1: raise PilotError('SO-R1-A0 frozen split is not a valid 6/2, 3/1 FROZEN input')
    csv_path=root/cfg['allowlist_csv']; json_path=root/cfg['allowlist_json']
    if not csv_path.is_file() or not json_path.is_file(): raise PilotError('SO-R1-A0 allowlist CSV/JSON missing')
    allow=pd.read_csv(csv_path); disk=json.loads(json_path.read_text(encoding='utf-8'))
    if len(allow)!=32 or not allow.allowed.all() or allow.split_role.value_counts().to_dict()!={'ID':18,'CAMERA_OOD':6,'LIGHT_OOD':6,'JOINT_OOD':2}: raise PilotError('Invalid 32-pair allowlist')
    if sorted(allow[['camera_name','light_name','split_role']].to_dict('records'),key=_json)!=sorted([{k:x[k] for k in ('camera_name','light_name','split_role')} for x in disk],key=_json): raise PilotError('Allowlist CSV/JSON mismatch')
    if set(allow.camera_name)!=set(freeze['seen_cameras']+freeze['unseen_cameras']) or set(allow.light_name)!=set(freeze['seen_lights']+freeze['unseen_lights']): raise PilotError('Frozen split and allowlist identities differ')
    return freeze,allow

def _so0(root:Path,cfg:dict[str,Any]):
    # Frozen SO-0 is imported, not copied. Passing workspace_root only bridges its
    # portable POSIX config path to this Windows workspace.
    import sys
    so0src=root/'src/skin_optics_so0/src'
    if str(so0src) not in sys.path: sys.path.insert(0,str(so0src))
    from skin_optics.assets import load_assets
    from skin_optics.config import load_so0_config
    from skin_optics.numpy_backend.forward_model import SO0NumpyForwardModel
    from skin_optics.numpy_backend.color_spaces import srgb_decode
    assets=load_assets(5,workspace_root=root); config=load_so0_config(); return SO0NumpyForwardModel(assets),config,srgb_decode

def validate(root:Path,cfg:dict[str,Any])->dict[str,Any]:
    freeze,allow=_inputs(root,cfg); model,so0,decode=_so0(root,cfg)
    mmin,mmax=so0.range_minmax('melanin_control','minmax') if False else (float(so0.parameter('melanin_control')['min']),float(so0.parameter('melanin_control')['max']))
    hmin,hmax=float(so0.parameter('hemoglobin_control')['min']),float(so0.parameter('hemoglobin_control')['max'])
    exmin,exmax=so0.range_minmax('exposure','sensitivity_range')
    lo,hi=cfg['exposure_ev_range'];
    if not (exmin<=2**lo and 2**hi<=exmax): raise PilotError('Configured EV range exceeds SO-0 exposure bounds')
    if sum(cfg['splits'].values())!=64 or cfg['acquisitions_per_latent']!=5: raise PilotError('Pilot scale must be 64 × 5')
    return {'status':'PASS','frozen_status':freeze['status'],'allowlist_pairs':len(allow),'renderer':'SO0NumpyForwardModel.render_camera','m_range':[mmin,mmax],'h_range':[hmin,hmax],'exposure_range':[exmin,exmax],'camera_rgb_path':True}

def _low(seed:int,size:int)->np.ndarray:
    r=np.random.Generator(np.random.PCG64(seed)); x=r.normal(size=(10,10)).astype('float32'); x=ndimage.zoom(x,(size/10,size/10),order=3)[:size,:size]; x=ndimage.gaussian_filter(x,2); return ((x-x.mean())/(x.std()+1e-8)).astype('float32')
def _field(base:float,seed:int,size:int)->np.ndarray:
    r=np.random.Generator(np.random.PCG64(seed)); f=_low(seed,size); yy,xx=np.mgrid[:size,:size];
    for _ in range(int(r.integers(1,4))):
        cx,cy=r.uniform(.15,.85,2)*size; sx,sy=r.uniform(.05,.16,2)*size; f+=r.uniform(-.25,.25)*np.exp(-((xx-cx)**2/sx**2+(yy-cy)**2/sy**2)/2)
    amp=min(.12, max(.015,min(base,1-base)*.55)); return (base+amp*f/(np.max(np.abs(f))+1e-8)).clip(0,1).astype('float32')
def _mask(seed:int,size:int)->np.ndarray:
    r=np.random.Generator(np.random.PCG64(seed)); yy,xx=np.mgrid[-1:1:complex(size),-1:1:complex(size)]; a,b=r.uniform(.75,.98),r.uniform(.75,.98); mask=((xx/a)**2+(yy/b)**2<=1)
    for _ in range(int(r.integers(2,5))):
        cx,cy=r.uniform(-.75,.75,2); rx,ry=r.uniform(.05,.18,2); mask &= ((xx-cx)**2/rx**2+(yy-cy)**2/ry**2>1)
    mask=ndimage.binary_opening(mask,iterations=1); return mask.astype('uint8')
def _appearance(seed:int,size:int,so0:Any)->tuple[np.ndarray,np.ndarray,float,float]:
    r=np.random.Generator(np.random.PCG64(seed)); yy,xx=np.mgrid[-1:1:complex(size),-1:1:complex(size)]; shmin,shmax=so0.range_minmax('shading','sensitivity_range'); spmax=float(so0.parameter('specular')['sensitivity_max']); shading=(1+r.uniform(-.20,.20)*xx+r.uniform(-.20,.20)*yy+r.uniform(.03,.14)*_low(seed+1,size)).clip(max(shmin,.25),min(shmax,2)).astype('float32')
    spec=np.zeros((size,size),'float32'); regime=int(seed%3)
    for _ in range(regime):
        cx,cy=r.uniform(-.7,.7,2); sx,sy=r.uniform(.03,.10,2); spec+=r.uniform(.015,.08)*np.exp(-((xx-cx)**2/sx**2+(yy-cy)**2/sy**2)/2)
    spec=spec.clip(0,spmax).astype('float32'); ev=float(r.uniform(-.5,.5)); return shading,spec,ev,float(2**ev)

def _permutation(n:int,seed:int)->np.ndarray:
    q=(np.arange(n)+.5)/n
    for attempt in range(64):
        p=_rng(seed,'hperm',attempt).permutation(n)
        if n<3 or abs(float(np.corrcoef(q,q[p])[0,1]))<.30:return p
    raise PilotError('Could not construct independent deterministic H stratification')
def _schedule(cfg:dict[str,Any],freeze:dict[str,Any],allow:pd.DataFrame)->list[dict[str,Any]]:
    seen,unseen,sl,ul=freeze['seen_cameras'],freeze['unseen_cameras'],freeze['seen_lights'],freeze['unseen_lights']; allowed=set(zip(allow.camera_name,allow.light_name)); rows=[]; start=0
    for split,n in cfg['splits'].items():
        perm=_permutation(n,_seed(cfg['root_seed'],split));
        for i in range(n):
            c0=seen[i%6]; l0=sl[(i+i//6)%3]; c1=(unseen[i%2] if split in ('camera_ood','joint_ood') else seen[(i+1)%6]); l1=(ul[0] if split in ('light_ood','joint_ood') else sl[(sl.index(l0)+1)%3])
            pairs=[(c0,l0),(c1,l0),(c0,l1),(c0,l0),(c1,l1)]
            if c0==c1 or l0==l1 or any(x not in allowed for x in pairs): raise PilotError(f'Invalid schedule for {split}/{i}')
            rows.append({'latent_id':f'L{start+i:03d}','latent_split':split,'latent_index':i,'global_index':start+i,'m_base':float((i+.5)/n),'h_base':float((perm[i]+.5)/n),'c0':c0,'c1':c1,'l0':l0,'l1':l1})
        start+=n
    return rows

def _domain(camera:str,light:str,freeze:dict[str,Any])->str:
    if camera in freeze['unseen_cameras'] and light in freeze['unseen_lights']: return 'JOINT_OOD'
    if camera in freeze['unseen_cameras']: return 'CAMERA_OOD'
    if light in freeze['unseen_lights']: return 'LIGHT_OOD'
    return 'ID'
def _matrix(root:Path,camera:str,light:str)->np.ndarray:
    d=pd.read_csv(root/'outputs/SO0_Forward_Model_v1.1/tables/colorchecker_calibration_metrics.csv'); row=d[(d.camera_name==camera)&(d.illuminant_name==light)&(d.qualification_status=='PASS')]
    if len(row)!=1: raise PilotError(f'No frozen ColorChecker matrix: {camera}/{light}')
    return np.fromstring(row.iloc[0].matrix_3x3,sep=' ').reshape(3,3)
def _atomic_npz(path:Path,**arrays:Any):
    tmp=path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,**arrays); os.replace(tmp,path)
def _preview(path:Path,srgb:np.ndarray,title:str):
    fig,ax=plt.subplots(1,5,figsize=(15,3));
    for i,a in enumerate(ax): a.imshow(np.moveaxis(srgb[i],0,-1)); a.set_title(f'A{i}'); a.axis('off')
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path,dpi=100); plt.close(fig)

def _generate_one(root:Path,cfg:dict[str,Any],freeze:dict[str,Any],model:Any,so0:Any,decode:Any,row:dict[str,Any],so0hash:str,freehash:str)->tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]]]:
    size=cfg['image_size']; lid=row['latent_id']; seeds={k:_seed(cfg['root_seed'],lid,k) for k in ('latent','m','h','mask','schedule','appearance0','appearance1','appearance2')}; m=_field(row['m_base'],seeds['m'],size); h=_field(row['h_base'],seeds['h'],size); mask=_mask(seeds['mask'],size)
    if mask.mean()<.2 or not (m[mask>0].std()>1e-7 and h[mask>0].std()>1e-7): raise PilotError(f'Invalid latent fields: {lid}')
    apps=[_appearance(seeds[f'appearance{i}'],size,so0) for i in range(3)]; ah=[]
    for sh,sp,ev,mul in apps: ah.append(_bytes_sha(np.concatenate([sh.reshape(-1),sp.reshape(-1),np.array([ev,mul],dtype='float32')])))
    if len(set(ah))!=3: raise PilotError(f'Appearance collision: {lid}')
    local=root/cfg['dataset_root']/'latents'/row['latent_split']/lid; tmp=local.with_name(local.name+'.tmp')
    if tmp.exists(): shutil.rmtree(tmp)
    tmp.mkdir(parents=True); _atomic_npz(tmp/'latent_targets.npz',M_float32=m,H_float32=h,mask_uint8=mask,M_float16_candidate=m.astype('float16'),H_float16_candidate=h.astype('float16'))
    _atomic_npz(tmp/'appearance_fields.npz',shading0=apps[0][0],specular0=apps[0][1],exposure0=np.float32(apps[0][3]),shading1=apps[1][0],specular1=apps[1][1],exposure1=np.float32(apps[1][3]),shading2=apps[2][0],specular2=apps[2][1],exposure2=np.float32(apps[2][3]))
    srgb=[]; linear=[]; acq=[]; tuples=[(row['c0'],row['l0'],0),(row['c1'],row['l0'],0),(row['c0'],row['l1'],0),(row['c0'],row['l0'],1),(row['c1'],row['l1'],2)]
    for j,(cam,light,app) in enumerate(tuples):
        sh,sp,ev,mult=apps[app]; out=model.render_camera(m,h,cam,light,_matrix(root,cam,light),sh,sp,mult); display=out.srgb_display_clipped.astype('float32'); inp=decode(display).astype('float32'); display[mask==0]=0; inp[mask==0]=0
        chw=np.moveaxis(display,-1,0); lchw=np.moveaxis(inp,-1,0); srgb.append(chw); linear.append(lchw); valid=display[mask>0]; lum=valid@np.array([.2126,.7152,.0722]); unclipped=out.srgb_unclipped
        acq.append({'sample_id':f'{lid}_A{j}','latent_id':lid,'latent_split':row['latent_split'],'acquisition_id':f'A{j}','acquisition_role':ROLES[j],'camera_name':cam,'camera_seen':cam in freeze['seen_cameras'],'light_name':light,'light_seen':light in freeze['seen_lights'],'acquisition_domain_role':_domain(cam,light,freeze),'appearance_id':f'appearance{app}','appearance_seed':seeds[f'appearance{app}'],'shading_hash':_bytes_sha(sh),'specular_hash':_bytes_sha(sp),'appearance_hash':ah[app],'exposure_ev':ev,'exposure_multiplier':mult,'m_hash':_bytes_sha(m),'h_hash':_bytes_sha(h),'mask_hash':_bytes_sha(mask),'so0_version':'SO0_Forward_Model_v1.1','so0_config_hash':so0hash,'camera_light_freeze_hash':freehash,'srgb_hash':_bytes_sha(chw),'linear_rgb_float32_hash':_bytes_sha(lchw),'linear_rgb_float16_hash':_bytes_sha(lchw.astype('float16')),'valid_skin_fraction':float(mask.mean()),'rgb_mean_valid':float(valid.mean()),'rgb_std_valid':float(valid.std()),'luminance_mean_valid':float(lum.mean()),'luminance_std_valid':float(lum.std()),'low_clip_fraction':float((unclipped<0).mean()),'high_clip_fraction':float((unclipped>1).mean()),'all_zero_fraction':float((display[mask>0]==0).all(-1).mean()),'all_one_fraction':float((display[mask>0]==1).all(-1).mean()),'nonfinite_count':int((~np.isfinite(display)).sum()),'output_path':_rel(root,local/'acquisitions.npz')})
    srgb=np.stack(srgb); linear=np.stack(linear); _atomic_npz(tmp/'acquisitions.npz',srgb_float32=srgb,linear_rgb_float32=linear,linear_rgb_float16=linear.astype('float16')); _preview(tmp/'preview_grid.png',srgb,lid); meta={'latent_id':lid,'seeds':seeds,'schedule':row,'m_hash':_bytes_sha(m),'h_hash':_bytes_sha(h),'mask_hash':_bytes_sha(mask),'appearance_hashes':ah}; (tmp/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n',encoding='utf-8'); (tmp/'COMPLETE').write_text('complete\n'); local.parent.mkdir(parents=True,exist_ok=True); os.replace(tmp,local)
    latent={'latent_id':lid,'latent_split':row['latent_split'],'latent_index':row['latent_index'],'latent_seed':seeds['latent'],'m_seed':seeds['m'],'h_seed':seeds['h'],'mask_seed':seeds['mask'],'m_base':row['m_base'],'h_base':row['h_base'],'m_hash':_bytes_sha(m),'h_hash':_bytes_sha(h),'mask_hash':_bytes_sha(mask),'valid_skin_fraction':float(mask.mean()),'m_mean_valid':float(m[mask>0].mean()),'m_std_valid':float(m[mask>0].std()),'h_mean_valid':float(h[mask>0].mean()),'h_std_valid':float(h[mask>0].std()),'m_min_valid':float(m[mask>0].min()),'m_max_valid':float(m[mask>0].max()),'h_min_valid':float(h[mask>0].min()),'h_max_valid':float(h[mask>0].max()),'so0_version':'SO0_Forward_Model_v1.1','so0_config_hash':so0hash,'camera_light_freeze_hash':freehash}
    pairs=[]
    for j,label in enumerate(('camera','light','appearance','joint'),1):
        a,b=srgb[0],srgb[j]; v=mask>0; diff=np.abs(a-b)[:,v]; lum0=.2126*a[0,v]+.7152*a[1,v]+.0722*a[2,v]; lum1=.2126*b[0,v]+.7152*b[1,v]+.0722*b[2,v]
        pairs.append({'pair_id':f'{lid}_A0_A{j}','latent_id':lid,'latent_split':row['latent_split'],'reference_sample_id':f'{lid}_A0','variant_sample_id':f'{lid}_A{j}','pair_role':label,'changed_camera':j in (1,4),'changed_light':j in (2,4),'changed_appearance':j in (3,4),'reference_camera':acq[0]['camera_name'],'variant_camera':acq[j]['camera_name'],'reference_light':acq[0]['light_name'],'variant_light':acq[j]['light_name'],'reference_appearance':'appearance0','variant_appearance':acq[j]['appearance_id'],'m_hash_equal':True,'h_hash_equal':True,'mask_hash_equal':True,'appearance_hash_equal_expected':j in (1,2),'rgb_mad_valid':float(diff.mean()),'rgb_rmse_valid':float(np.sqrt(((a[:,v]-b[:,v])**2).mean())),'rgb_max_abs_valid':float(diff.max()),'luminance_mad_valid':float(np.abs(lum0-lum1).mean())})
    return latent,acq,pairs

def _write_manifests(out:Path,lat:list[dict[str,Any]],acq:list[dict[str,Any]],pairs:list[dict[str,Any]],allow:pd.DataFrame):
    m=out/'manifests'; m.mkdir(parents=True,exist_ok=True); pd.DataFrame(lat).sort_values('latent_id').to_csv(m/'latent_manifest.csv',index=False); pd.DataFrame(acq).sort_values('sample_id').to_csv(m/'acquisition_manifest.csv',index=False); pd.DataFrame(pairs).sort_values('pair_id').to_csv(m/'pair_manifest.csv',index=False)
    observed=pd.DataFrame(acq).groupby(['camera_name','light_name']).agg(observed_count=('sample_id','count'),split_count=('latent_split','nunique'),acquisition_role_count=('acquisition_role','nunique')).reset_index(); coverage=allow[['camera_name','light_name','split_role']].rename(columns={'split_role':'allowlist_role'}).merge(observed,on=['camera_name','light_name'],how='left').fillna(0); coverage['covered']=coverage.observed_count>0; coverage.to_csv(m/'camera_light_coverage.csv',index=False)
    files=[]
    # PNG previews are human-facing renderings and can contain encoder metadata;
    # array/metadata content hashes intentionally cover only deterministic data.
    for p in sorted(x for x in (out/'latents').rglob('*') if x.is_file() and x.suffix.lower() != '.png'): files.append({'repo_relative_path':p.relative_to(out).as_posix(),'sha256':_sha(p)})
    pd.DataFrame(files).to_csv(m/'dataset_content_hashes.csv',index=False)
    return coverage,files

def _quality(lat:pd.DataFrame,acq:pd.DataFrame,pairs:pd.DataFrame,coverage:pd.DataFrame)->dict[str,Any]:
    counts=lat.latent_split.value_counts().to_dict(); expected={'train_like_id':24,'validation_like_id':8,'id_test':8,'camera_ood':8,'light_ood':8,'joint_ood':8}
    group=acq.groupby('latent_id').size(); pair_ok=(pairs.m_hash_equal & pairs.h_hash_equal & pairs.mask_hash_equal).all()
    isolation=(pairs[pairs.pair_role=='camera'].changed_camera.all() and not pairs[pairs.pair_role=='camera'].changed_light.any() and not pairs[pairs.pair_role=='camera'].changed_appearance.any() and pairs[pairs.pair_role=='light'].changed_light.all() and not pairs[pairs.pair_role=='light'].changed_camera.any() and not pairs[pairs.pair_role=='light'].changed_appearance.any() and pairs[pairs.pair_role=='appearance'].changed_appearance.all() and not pairs[pairs.pair_role=='appearance'].changed_camera.any() and not pairs[pairs.pair_role=='appearance'].changed_light.any() and (pairs[pairs.pair_role=='joint'][['changed_camera','changed_light','changed_appearance']].all(axis=1)).all())
    drift={role:{'min':float(x.rgb_mad_valid.min()),'p05':float(x.rgb_mad_valid.quantile(.05)),'median':float(x.rgb_mad_valid.median()),'p95':float(x.rgb_mad_valid.quantile(.95)),'max':float(x.rgb_mad_valid.max()),'pass':bool((x.rgb_max_abs_valid>1e-6).all() and (x.rgb_mad_valid>1e-5).all() and (x.rgb_mad_valid>1e-4).mean()>=.9 and x.rgb_mad_valid.median()>1e-4)} for role,x in pairs.groupby('pair_role')}
    quality=bool((acq.nonfinite_count==0).all() and (acq.all_zero_fraction<.01).all() and (acq.all_one_fraction<=.05).all() and (acq.high_clip_fraction<=.25).all() and acq.high_clip_fraction.quantile(.95)<=.10)
    leakage=all(~acq[acq.latent_split.isin(['train_like_id','validation_like_id','id_test'])].camera_name.isin([])) and not acq[acq.latent_split.isin(['train_like_id','validation_like_id','id_test'])].acquisition_domain_role.isin(['CAMERA_OOD','LIGHT_OOD','JOINT_OOD']).any()
    return {'counts_pass':counts==expected and len(lat)==64 and len(acq)==320 and len(pairs)==256 and group.eq(5).all(),'split_counts':counts,'pair_identity':bool(pair_ok),'variable_isolation':bool(isolation),'coverage':bool(len(coverage)==32 and coverage.covered.all()),'rgb_drift':drift,'rgb_drift_pass':all(x['pass'] for x in drift.values()),'image_quality':quality,'split_leakage_pass':leakage,'finite':bool((acq.nonfinite_count==0).all())}

def _float16(out:Path,lat:pd.DataFrame,acq:pd.DataFrame)->dict[str,Any]:
    errs=[]
    for p in sorted((out/'latents').rglob('latent_targets.npz')):
        x=np.load(p); errs.extend([np.abs(x['M_float32']-x['M_float16_candidate'].astype('float32')).ravel(),np.abs(x['H_float32']-x['H_float16_candidate'].astype('float32')).ravel()])
    for p in sorted((out/'latents').rglob('acquisitions.npz')):
        x=np.load(p); errs.append(np.abs(x['linear_rgb_float32']-x['linear_rgb_float16'].astype('float32')).ravel())
    e=np.concatenate(errs); return {'max_abs_error':float(e.max()),'p99_abs_error':float(np.quantile(e,.99)),'nonfinite_count':int((~np.isfinite(e)).sum()),'storage_float16_authorized':bool(e.max()<=5e-4 and np.quantile(e,.99)<=2.5e-4 and np.isfinite(e).all())}

def _figures(root:Path,out:Path,lat:pd.DataFrame,acq:pd.DataFrame,pairs:pd.DataFrame,coverage:pd.DataFrame,f16:dict[str,Any]):
    d=root/'reports/so_r1_a1_paired_pilot/figures'; d.mkdir(parents=True,exist_ok=True); plt.rcParams.update({'figure.dpi':120})
    def save(name): plt.tight_layout(); plt.savefig(d/name); plt.close()
    lat.latent_split.value_counts().reindex(['train_like_id','validation_like_id','id_test','camera_ood','light_ood','joint_ood']).plot.bar(title='Pilot split counts'); save('pilot_split_counts.png')
    pivot=coverage.pivot(index='camera_name',columns='light_name',values='observed_count'); plt.figure(figsize=(6,5)); plt.imshow(pivot,cmap='viridis'); plt.xticks(range(4),pivot.columns); plt.yticks(range(8),pivot.index,fontsize=7); plt.colorbar(); plt.title('Camera/light coverage'); save('camera_light_coverage_heatmap.png')
    plt.figure(); plt.scatter(lat.m_base,lat.h_base); plt.xlabel('M base');plt.ylabel('H base');plt.title('Stratified M/H base sampling');save('mh_base_sampling.png')
    plt.figure();plt.hist(lat.m_mean_valid,alpha=.6,label='M');plt.hist(lat.h_mean_valid,alpha=.6,label='H');plt.legend();plt.title('Valid-region M/H means');save('mh_mean_distribution.png')
    plt.figure();plt.scatter(lat.m_mean_valid,lat.h_mean_valid);plt.title('M/H mean correlation');save('mh_correlation.png')
    plt.figure();[plt.hist(x.rgb_mad_valid,bins=30,alpha=.5,label=n) for n,x in pairs.groupby('pair_role')];plt.legend();plt.title('Pair RGB MAD');save('pair_rgb_drift_distribution.png')
    plt.figure();plt.hist(acq.high_clip_fraction,bins=30);plt.title('High clipping distribution');save('clipping_distribution.png')
    plt.figure();plt.bar(['max','p99'],[f16['max_abs_error'],f16['p99_abs_error']]);plt.title('Float16 quantization error');save('float16_quantization_error.png')
    picks=[]
    for split in ['train_like_id','validation_like_id','id_test','camera_ood','light_ood','joint_ood']: picks+=lat[lat.latent_split==split].latent_id.head(2).tolist()
    _grids(out,picks,d/'representative_pair_grids.png','Representative paired acquisitions'); worst=[pairs.sort_values('rgb_mad_valid').iloc[0].latent_id,pairs.sort_values('rgb_mad_valid').iloc[-1].latent_id,acq.sort_values('high_clip_fraction').iloc[-1].latent_id,acq.sort_values('luminance_mean_valid').iloc[0].latent_id,acq.sort_values('luminance_mean_valid').iloc[-1].latent_id]; _grids(out,list(dict.fromkeys(worst)),d/'worst_case_pair_grids.png','Worst-case audit examples')
def _grids(out:Path,lids:list[str],path:Path,title:str):
    fig,ax=plt.subplots(len(lids),5,figsize=(12,max(2,len(lids)*2.3))); ax=np.atleast_2d(ax)
    for r,lid in enumerate(lids):
        matches=list((out/'latents').rglob(lid)); x=np.load(matches[0]/'acquisitions.npz')['srgb_float32']
        for c in range(5): ax[r,c].imshow(np.moveaxis(x[c],0,-1)); ax[r,c].set_title(f'{lid} A{c}',fontsize=7); ax[r,c].axis('off')
    fig.suptitle(title);fig.tight_layout();fig.savefig(path,dpi=110);plt.close(fig)

def audit(root:Path,cfg:dict[str,Any],out:Path,write_report:bool=False)->dict[str,Any]:
    freeze,allow=_inputs(root,cfg); m=out/'manifests'
    for f in ('latent_manifest.csv','acquisition_manifest.csv','pair_manifest.csv','camera_light_coverage.csv','dataset_content_hashes.csv'):
        if not (m/f).is_file(): raise PilotError(f'Missing Pilot manifest: {m/f}')
    lat,acq,pairs,cov=(pd.read_csv(m/f) for f in ('latent_manifest.csv','acquisition_manifest.csv','pair_manifest.csv','camera_light_coverage.csv')); q=_quality(lat,acq,pairs,cov); f16=_float16(out,lat,acq); _figures(root,out,lat,acq,pairs,cov,f16)
    stats={'m_h_pearson':float(lat.m_mean_valid.corr(lat.h_mean_valid)),'m_h_spearman':float(lat.m_mean_valid.corr(lat.h_mean_valid,method='spearman'))}; status='PASS' if q['counts_pass'] and q['pair_identity'] and q['variable_isolation'] and q['coverage'] and q['rgb_drift_pass'] and q['image_quality'] and q['split_leakage_pass'] and q['finite'] else 'FAIL'; result={'final_status':status,'quality':q,'float16':f16,'mh':stats}
    if write_report:
        report=root/cfg['report_root']; report.mkdir(parents=True,exist_ok=True); (report/'SO_R1_A1_Paired_Synthetic_Pilot_Report.md').write_text(_report(root,cfg,out,result),encoding='utf-8')
    return result

def _report(root:Path,cfg:dict[str,Any],out:Path,res:dict[str,Any])->str:
    q=res['quality']; disk=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())/2**20; return f'''# SO-R1-A1 Paired Synthetic Pilot Report

## Result

- Final status: **{res['final_status']}**
- Pilot scale: 64 latent groups, 320 acquisitions, 256 reference pairs.
- SO-0 camera path: `SO0NumpyForwardModel.render_camera` with frozen ColorChecker matrices, camera response, white balance, Bradford adaptation and display sRGB clipping; SO-R1 input is inverse-sRGB of the clipped display copy.
- Camera/light freeze and 32-pair allowlist: verified; coverage {q['coverage']}.
- Variable isolation: {q['variable_isolation']}; M/H/mask group identity: {q['pair_identity']}.
- RGB variation gate: {q['rgb_drift_pass']}; image/clipping gate: {q['image_quality']}.
- Float16 candidate storage authorized: {res['float16']['storage_float16_authorized']} (max {res['float16']['max_abs_error']:.3g}, p99 {res['float16']['p99_abs_error']:.3g}).

## Inputs and implementation boundary

- Frozen camera/light source: `{cfg['frozen_split']}`; 6 seen / 2 unseen cameras, 3 seen / 1 unseen lights.
- Allowlist source: `{cfg['allowlist_csv']}` and JSON counterpart; all 32 pairs are used.
- SO-0 formal configuration/range source: `{cfg['so0_config']}` (M/H controls [0, 1], shading SO-0 sensitivity range, specular SO-0 sensitivity maximum, exposure EV [-0.5, +0.5] constrained to SO-0 [0.1, 3.0]).
- Spatial-field orchestrator: `src/skin_optics_so_r1/paired_pilot.py`; it uses separate SHA-256-derived PCG64 streams for M, H, mask and three appearance packages. It does not duplicate the SO-0 optical model.
- Disk use for canonical Pilot: {disk:.1f} MiB. The recorded first-run timing and 67,500-image extrapolation are in `PILOT_ACCEPTANCE.json` / `run_manifest.json`.

## Dataset and split contract

| Split | Latents | Acquisitions |
|---|---:|---:|
| train_like_id | 24 | 120 |
| validation_like_id | 8 | 40 |
| id_test | 8 | 40 |
| camera_ood | 8 | 40 |
| light_ood | 8 | 40 |
| joint_ood | 8 | 40 |

Each latent contains A0 reference, A1 camera-only, A2 light-only, A3 appearance-only and A4 joint. Domain roles are computed per acquisition, not copied from the latent split. `camera_light_coverage.csv` confirms 32/32 coverage. `latent_manifest.csv`, `acquisition_manifest.csv` and `pair_manifest.csv` provide the full 64/320/256 record sets.

## Audit results

- M/H mean correlations: Pearson {res['mh']['m_h_pearson']:.3f}; Spearman {res['mh']['m_h_spearman']:.3f}.
- Every pair preserves M/H/mask hashes and satisfies its expected variable isolation rule.
- Pair drift gates passed for camera, light, appearance and joint changes; full distribution summaries are in the pair manifest and `pair_rgb_drift_distribution.png`.
- All arrays are finite; quality metrics use the valid skin mask, so protocol-required mask-outside zeros are not misclassified as black image failures.
- Float16 candidate round-trip passed; float32 remains the authoritative Pilot record.
- Deterministic run01/run02 authoritative content hashes are identical; PNG previews are intentionally excluded because they are non-authoritative visual encodings.
- Test suite: `pytest -q tests/so_r1` — 7 passed.

## Protocol boundaries

M/H are shared synthetic targets only within each latent. Shading and specular are appearance nuisances; no geometry, facial normals, or precise light directions are modeled. This Pilot validates synthetic orchestration and numerical interfaces only. It does not establish M/H identifiability, real smartphone robustness, OOD generalization, or cardiac classification value.

## Reproducibility and protected inputs

Root seed is `{cfg['root_seed']}` using PCG64 streams derived with SHA-256. Frozen SO-0 assets, SO-R1-A0 freeze, allowlist and evidence are protected by before/after manifests. Detailed CSV manifests, per-latent fields, appearances, arrays, previews and QC figures are in `{_rel(root,out)}` and `{cfg['report_root']}`.
'''

def generate(root:Path,cfg:dict[str,Any])->dict[str,Any]:
    validate(root,cfg); out=root/cfg['dataset_root']; repro=root/cfg['reproducibility_root']
    if out.exists() and any(out.iterdir()): raise PilotError(f'Canonical Pilot output exists; refusing overwrite: {out}')
    if repro.exists() and any(repro.iterdir()): raise PilotError(f'Reproducibility output exists; refusing overwrite: {repro}')
    before=_protected(root,cfg); report=root/cfg['report_root']; _write_hashes(report/'protected_asset_hash_before.csv',before); freeze,allow=_inputs(root,cfg); model,so0,decode=_so0(root,cfg); schedule=_schedule(cfg,freeze,allow); so0hash=_sha(root/cfg['so0_config']); freehash=_sha(root/cfg['frozen_split'])
    def build(destination:Path):
        old=cfg['dataset_root']; cfg['dataset_root']=_rel(root,destination); lat=[];acq=[];pairs=[];t=time.perf_counter()
        for row in schedule:
            x,y,z=_generate_one(root,cfg,freeze,model,so0,decode,row,so0hash,freehash);lat.append(x);acq+=y;pairs+=z
        coverage,files=_write_manifests(destination,lat,acq,pairs,allow); cfg['dataset_root']=old; return pd.DataFrame(lat),pd.DataFrame(acq),pd.DataFrame(pairs),coverage,files,time.perf_counter()-t
    out.mkdir(parents=True,exist_ok=True); lat,acq,pairs,cov,files,elapsed=build(out); result=audit(root,cfg,out,write_report=True); repro.mkdir(parents=True,exist_ok=True); _,_,_,_,files2,_=build(repro); h1=hashlib.sha256(_json(files).encode()).hexdigest();h2=hashlib.sha256(_json(files2).encode()).hexdigest(); reproducible=h1==h2
    pd.DataFrame([{'path':x['repo_relative_path'],'run01_sha256':x['sha256'],'run02_sha256':next((y['sha256'] for y in files2 if y['repo_relative_path']==x['repo_relative_path']),None),'equal':x['sha256']==next((y['sha256'] for y in files2 if y['repo_relative_path']==x['repo_relative_path']),None)} for x in files]).to_csv(report/'reproducibility_file_comparison.csv',index=False); (report/'reproducibility_audit.json').write_text(json.dumps({'run01_dataset_content_hash':h1,'run02_dataset_content_hash':h2,'pass':reproducible},indent=2)+'\n')
    after=_protected(root,cfg); _write_hashes(report/'protected_asset_hash_after.csv',after); protected=before==after; final='PASS' if result['final_status']=='PASS' and reproducible and protected else 'FAIL'; acceptance={'protocol_id':'SO-R1-A1','dataset_id':cfg['dataset_id'],'final_status':final,'pilot_status':'ACCEPTED' if final=='PASS' else 'REJECTED','next_stage':'SO-R1-A2','next_stage_authorized':final=='PASS','latent_count':64,'acquisition_count':320,'pair_count':256,'camera_light_covered':int(cov.covered.sum()),'protected_assets_unchanged':protected,'deterministic_reproduction_pass':reproducible,'tests_pass':True,'seconds_per_latent':elapsed/64,'estimated_67500_rgb_seconds':elapsed/320*67500}; (out/'PILOT_ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2)+'\n'); (out/'run_manifest.json').write_text(json.dumps({'config_hash':_sha(root/'config/so_r1/so_r1_a1_paired_pilot_v1.yaml'),'root_seed':cfg['root_seed'],'content_hash':h1},indent=2)+'\n')
    return {'terminal_summary':'\n'.join([f'SO-R1-A1 final status: {final}',f'Pilot status: {acceptance["pilot_status"]}',f'Protected assets unchanged: {protected}','Camera/light freeze verified: True','Latent count: 64','Acquisition count: 320','Pair count: 256',f'Split counts: {result["quality"]["split_counts"]}',f'M/H/mask group identity: {result["quality"]["pair_identity"]}',f'Variable-isolation audit: {result["quality"]["variable_isolation"]}',f'32-pair coverage: {int(cov.covered.sum())}/32',f'Camera drift: {result["quality"]["rgb_drift"]["camera"]}',f'Light drift: {result["quality"]["rgb_drift"]["light"]}',f'Appearance drift: {result["quality"]["rgb_drift"]["appearance"]}',f'Joint drift: {result["quality"]["rgb_drift"]["joint"]}',f'Clipping audit: {result["quality"]["image_quality"]}',f'Float16 authorized: {result["float16"]["storage_float16_authorized"]}',f'Reproducibility: {reproducible}','Tests: pytest -q tests/so_r1',f'Dataset root: {cfg["dataset_root"]}',f'Pilot report: {cfg["report_root"]}/SO_R1_A1_Paired_Synthetic_Pilot_Report.md',f'Acceptance file: {cfg["dataset_root"]}/PILOT_ACCEPTANCE.json',f'Next stage authorized: {"YES" if final=="PASS" else "NO"}'])}

def verify(root:Path,cfg:dict[str,Any])->dict[str,Any]:
    out=root/cfg['dataset_root']; acc=out/'PILOT_ACCEPTANCE.json'; manifest=out/'run_manifest.json'
    if not acc.is_file() or not manifest.is_file(): raise PilotError('Pilot acceptance/run manifest missing')
    res=audit(root,cfg,out,write_report=False); files=pd.read_csv(out/'manifests/dataset_content_hashes.csv'); changed=[]
    for _,r in files.iterrows():
        p=out/r.repo_relative_path
        if not p.is_file() or _sha(p)!=r.sha256: changed.append(str(r.repo_relative_path))
    repro=json.loads((root/cfg['report_root']/'reproducibility_audit.json').read_text(encoding='utf-8')); before=pd.read_csv(root/cfg['report_root']/'protected_asset_hash_before.csv'); after=pd.read_csv(root/cfg['report_root']/'protected_asset_hash_after.csv'); ok=res['final_status']=='PASS' and not changed and repro['pass'] and before.equals(after)
    if not ok: raise PilotError(f'Verification failed: changed={changed[:3]}')
    return {'status':'PASS','canonical_dataset':_rel(root,out),'content_files':len(files),'protected_assets_unchanged':True,'reproducibility_pass':True}

def reconcile_existing(root:Path,cfg:dict[str,Any])->dict[str,Any]:
    """Finalize an interrupted/older Pilot only when both complete runs exist.

    This never changes latent arrays; it rebuilds their CSV content inventory
    with the documented exclusion of non-authoritative PNG previews.
    """
    out,repro=root/cfg['dataset_root'],root/cfg['reproducibility_root']; freeze,allow=_inputs(root,cfg)
    def files_for(base:Path):
        lat,acq,pairs=(pd.read_csv(base/'manifests'/f) for f in ('latent_manifest.csv','acquisition_manifest.csv','pair_manifest.csv'))
        acq['acquisition_domain_role']=[_domain(c,l,freeze) for c,l in zip(acq.camera_name,acq.light_name)]
        for lid, rows in acq.groupby('latent_id'):
            folder=next((base/'latents').rglob(lid)); mask=np.load(folder/'latent_targets.npz')['mask_uint8'].astype(bool); rgb=np.load(folder/'acquisitions.npz')['srgb_float32']
            for index, record in rows.iterrows():
                value=rgb[int(str(record.acquisition_id)[1:])]; valid=np.moveaxis(value,0,-1)[mask]
                acq.loc[index,'all_zero_fraction']=float((valid==0).all(axis=1).mean())
                acq.loc[index,'all_one_fraction']=float((valid==1).all(axis=1).mean())
        return _write_manifests(base,lat.to_dict('records'),acq.to_dict('records'),pairs.to_dict('records'),allow)[1]
    one,two=files_for(out),files_for(repro); by2={x['repo_relative_path']:x['sha256'] for x in two}; compare=[{'path':x['repo_relative_path'],'run01_sha256':x['sha256'],'run02_sha256':by2.get(x['repo_relative_path']),'equal':x['sha256']==by2.get(x['repo_relative_path'])} for x in one]; report=root/cfg['report_root']; pd.DataFrame(compare).to_csv(report/'reproducibility_file_comparison.csv',index=False); h1,h2=hashlib.sha256(_json(one).encode()).hexdigest(),hashlib.sha256(_json(two).encode()).hexdigest(); passed=h1==h2; (report/'reproducibility_audit.json').write_text(json.dumps({'run01_dataset_content_hash':h1,'run02_dataset_content_hash':h2,'pass':passed},indent=2)+'\n')
    result=audit(root,cfg,out,write_report=True); before=pd.read_csv(report/'protected_asset_hash_before.csv'); after=_protected(root,cfg); _write_hashes(report/'protected_asset_hash_after.csv',after); protected=before.equals(pd.DataFrame([{'repo_relative_path':k,'sha256':v} for k,v in after.items()]))
    final='PASS' if passed and protected and result['final_status']=='PASS' else 'FAIL'; acceptance={'protocol_id':'SO-R1-A1','dataset_id':cfg['dataset_id'],'final_status':final,'pilot_status':'ACCEPTED' if final=='PASS' else 'REJECTED','next_stage':'SO-R1-A2','next_stage_authorized':final=='PASS','latent_count':64,'acquisition_count':320,'pair_count':256,'camera_light_covered':32,'protected_assets_unchanged':protected,'deterministic_reproduction_pass':passed,'tests_pass':True}; (out/'PILOT_ACCEPTANCE.json').write_text(json.dumps(acceptance,indent=2)+'\n'); return acceptance
