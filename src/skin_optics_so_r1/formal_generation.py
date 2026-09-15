"""G1 deterministic CPU generation.  Workers own only per-latent files."""
from __future__ import annotations
import hashlib, json, os, platform, shutil, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import psutil
from scipy.stats import pearsonr, spearmanr
from .paired_pilot import _field, _appearance, _matrix, _so0, ROLES
from .mask_benchmark_closure import _derive_mask

PROTOCOL='SO-R1-A2-D0'; VERSION='1.0'; ROOT_SEED=20260822
FLAGS={'AUDIT_ONLY':False,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':False}
CAL_FLAGS={'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':True}
SPLIT_DIR={'Train':'train','Validation':'val','ID Test':'id_test','Camera-OOD':'camera_ood','Light-OOD':'light_ood','Joint-OOD':'joint_ood'}

def _stable(*x):return int.from_bytes(hashlib.blake2b('|'.join(map(str,x)).encode(),digest_size=8).digest(),'big')
def _canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False)
def _hash_arr(a):
    h=hashlib.sha256();h.update(str(a.dtype).encode());h.update(_canon(list(a.shape)).encode());h.update(np.ascontiguousarray(a).tobytes());return h.hexdigest()
def _hash_content(m,h,mask,rgb,meta):
    z=hashlib.sha256()
    for x in (m,h,mask,rgb): z.update(str(x.dtype).encode());z.update(_canon(list(x.shape)).encode());z.update(np.ascontiguousarray(x).tobytes())
    z.update(_canon(meta).encode());return z.hexdigest()
def _write(p,x):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True,default=str)+'\n',encoding='utf-8')
def _sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def _metrics(un,post,mask):
    x=un[mask>0]; d=post[mask>0]; hi=x>=1-1e-6;lo=x<=1e-6
    q={'high_clip_element_fraction':float(hi.mean()),'low_clip_element_fraction':float(lo.mean()),'nonfinite_count':int((~np.isfinite(d)).sum()),'all_zero':bool((d==0).all()),'all_one':bool((d==1).all())}
    for i,c in enumerate('RGB'):q[f'{c}_zero_fraction']=float((d[:,i]==0).mean());q[f'{c}_one_fraction']=float((d[:,i]>=1-1e-6).mean())
    return q
def _env_one_thread():
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'

def _checkpoint_path(outroot:Path,lid:str)->Path:return outroot/'manifests'/'checkpoints'/f'{lid}.json'
def _valid_checkpoint(outroot:Path,row:dict):
    """Validate both the retained NPZ canonical bytes and its checkpoint."""
    lid=row['latent_id']; cp=_checkpoint_path(outroot,lid); dst=outroot/SPLIT_DIR[row['split']]/f'{lid}.npz'
    if not(cp.is_file() and dst.is_file()):return None
    try:
        rec=json.loads(cp.read_text(encoding='utf-8'))
        with np.load(dst,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata']))
            content=_hash_content(z['m'],z['h'],z['mask'],z['rgb'],meta)
            ok=(z['rgb'].shape==(5,3,256,256) and z['rgb'].dtype==np.float16 and z['m'].shape==(256,256) and z['h'].shape==(256,256) and z['mask'].dtype==np.uint8 and np.isfinite(z['rgb']).all() and meta['latent_id']==lid and meta['split']==row['split'] and content==rec['latent']['canonical_content_hash'])
        return rec if ok else None
    except Exception:return None

def _generate_task(task:dict):
    """Spawn-safe immutable worker entry point; no final CSV writes."""
    _env_one_thread(); root=Path(task['root']); row=task['row']; outroot=Path(task['outroot']); calibration=task['calibration']
    if not calibration:
        prior=_valid_checkpoint(outroot,row)
        if prior is not None:return prior
    model,so0,decode=_so0(root,{})
    lid=row['latent_id']; split=row['split']; m=_field(float(row['m_base']),int(row['m_seed']),256);h=_field(float(row['h_base']),int(row['h_seed']),256)
    cat=row['mask_category']; sampler_cat=cat.lower(); tr=None if cat=='Full' else ((.72,.93) if cat=='Mild' else (.32,.68)); mask,coverage,mask_try,mask_hash,lcc=_derive_mask(ROOT_SEED,lid,sampler_cat,0,256,tr,128,.90)
    combos=[(row['c0'],row['l0'],0),(row['c1'],row['l0'],0),(row['c0'],row['l1'],0),(row['c0'],row['l0'],1),(row['c1'],row['l1'],2)]
    apps=[]
    for a in range(3):
        seed=int(row[f'appearance{a}_seed']);sh,sp,_,_=_appearance(seed,256,so0);apps.append((seed,sh,sp))
    rgb=[]; aq=[];qcs=[]
    for j,(cam,light,a) in enumerate(combos):
        attempt=0; reason=''; final=None
        while attempt<128:
            # Retry derives all nuisance components only from frozen protocol fields.
            seed=_stable(PROTOCOL,VERSION,ROOT_SEED,split,int(row['split_index']),f'appearance{a}',attempt)
            sh,sp,_,_= _appearance(seed,256,so0)
            ev=float(np.random.Generator(np.random.PCG64(_stable(PROTOCOL,VERSION,ROOT_SEED,split,int(row['split_index']),f'exposure{a}',attempt))).uniform(-.5,.31))
            z=model.render_camera(m,h,cam,light,_matrix(root,cam,light),sh,sp,float(2**ev)); un=z.srgb_unclipped.astype('float32');post=decode(z.srgb_display_clipped).astype('float32');post[mask==0]=0;q=_metrics(un,post,mask)
            collapse=max(q[k] for k in q if k.endswith('_zero_fraction') or k.endswith('_one_fraction'))>=.5
            if q['high_clip_element_fraction']<=.1 and q['low_clip_element_fraction']<=.1 and q['nonfinite_count']==0 and not q['all_zero'] and not q['all_one'] and not collapse:
                final=(sh,sp,ev,post,q,seed);break
            attempt+=1;reason='qc_high_low_clip_nonfinite_or_channel_collapse'
        if final is None:raise RuntimeError(f'FAIL_QC_RETRY_EXHAUSTED:{lid}:A{j}')
        sh,sp,ev,post,q,seed=final;rgb.append(np.moveaxis(post,-1,0).astype('float16')); apphash=hashlib.sha256(np.ascontiguousarray(np.concatenate([sh.ravel(),sp.ravel(),np.asarray([ev],dtype=np.float32)])).tobytes()).hexdigest()
        aq.append({'acquisition_id':f'{lid}_A{j}','latent_id':lid,'split':split,'acquisition_index':j,'role':ROLES[j],'camera':cam,'light':light,'camera_role':'seen' if cam in ('Canon 5DMarkII','Nikon D80','Olympus E-PL2','Canon 300D') else 'unseen','light_role':'seen' if light in ('D65','A','FL2') else 'unseen','appearance_id':f'appearance{a}','appearance_seed':seed,'appearance_hash':apphash,'exposure_ev':ev,'attempt_count':attempt+1,'rejection_reason':reason,'rgb_dtype':'float16','rgb_shape':'[3,256,256]',**q})
        qcs.append({'acquisition_id':f'{lid}_A{j}',**q,'attempt_count':attempt+1})
    rgb=np.stack(rgb); meta={'protocol_id':PROTOCOL,'protocol_version':VERSION,'latent_id':lid,'split':split,'split_index':int(row['split_index']),'mask_category':cat,'coverage':coverage,'mask_attempt':mask_try,'seeds':{k:int(row[k]) for k in row if k.endswith('_seed')},'camera_light':[(c,l) for c,l,_ in combos],'flags':CAL_FLAGS if calibration else FLAGS}
    content=_hash_content(m.astype('float16'),h.astype('float16'),mask,rgb,meta); mhash=_hash_arr(m.astype('float16'));hhash=_hash_arr(h.astype('float16'));rhash=_hash_arr(rgb);metahash=hashlib.sha256(_canon(meta).encode()).hexdigest()
    dst=outroot/SPLIT_DIR[split]/f'{lid}.npz';dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_name(dst.name+'.tmp.npz');np.savez_compressed(tmp,rgb=rgb,m=m.astype('float16'),h=h.astype('float16'),mask=mask,metadata=_canon(meta));
    with np.load(tmp,allow_pickle=False) as chk:
        if chk['rgb'].shape!=(5,3,256,256) or chk['rgb'].dtype!=np.float16 or chk['m'].dtype!=np.float16 or chk['mask'].dtype!=np.uint8 or not np.isfinite(chk['rgb']).all():raise RuntimeError(f'NPZ_READBACK_FAILURE:{lid}')
    os.replace(tmp,dst)
    latent={'latent_id':lid,'split':split,'latent_index':int(row['split_index']),'mask_category':cat,'valid_skin_fraction':coverage,'mask_lcc_fraction':lcc,'M_mean':float(m[mask>0].mean()),'H_mean':float(h[mask>0].mean()),'M_std':float(m[mask>0].std()),'H_std':float(h[mask>0].std()),'m_hash':mhash,'h_hash':hhash,'mask_hash':mask_hash,'rgb_stack_hash':rhash,'metadata_hash':metahash,'canonical_content_hash':content,'file_path':str(dst),'file_sha256':_sha(dst)}
    pairs=[]
    for j,kind in enumerate(('camera_only','light_only','appearance_only','joint'),1):pairs.append({'pair_id':f'{lid}_A0_A{j}','latent_id':lid,'split':split,'pair_index':j-1,'pair_type':kind,'reference_acquisition':f'{lid}_A0','target_acquisition':f'{lid}_A{j}','camera_changed':j in (1,4),'light_changed':j in (2,4),'appearance_changed':j in (3,4),'m_hash_match':True,'h_hash_match':True,'mask_hash_match':True,'rgb_mad':float(np.abs(rgb[0].astype('float32')-rgb[j].astype('float32')).mean())})
    result={'latent':latent,'acquisition':aq,'pairs':pairs,'qc':qcs}
    if not calibration:
        _write(_checkpoint_path(outroot,lid),result)
    return result

def _run(tasks,workers):
    t=time.perf_counter();r=[]
    if workers==1:
        for x in tasks:r.append(_generate_task(x))
    else:
        with ProcessPoolExecutor(max_workers=workers,initializer=_env_one_thread) as ex:
            futures=[ex.submit(_generate_task,x) for x in tasks]
            for f in as_completed(futures):r.append(f.result())
    return r,time.perf_counter()-t
def _collect(results):
    l=sorted((x['latent'] for x in results),key=lambda z:(z['split'],z['latent_index']));a=sorted((q for x in results for q in x['acquisition']),key=lambda z:(z['split'],z['latent_id'],z['acquisition_index']));p=sorted((q for x in results for q in x['pairs']),key=lambda z:(z['split'],z['latent_id'],z['pair_index']));q=sorted((z for x in results for z in x['qc']),key=lambda z:z['acquisition_id']);return l,a,p,q
def _rows(path):return pd.read_csv(path).to_dict('records')
def _copy_manifest(path,rows):pd.DataFrame(rows).to_csv(path,index=False)
def _selection(rows):
    need={'Train':48,'Validation':16,'ID Test':16,'Camera-OOD':16,'Light-OOD':16,'Joint-OOD':16};return [x for s,n in need.items() for x in sorted((z for z in rows if z['split']==s),key=lambda z:z['split_index'])[:n]]

def main(root:Path):
    out=root/'data/processed/SO_R1_A2_FormalPairedDataset_v1';cal=root/'data/processed/SO_R1_A2_G1_ParallelCalibration_v1';rep=root/'reports/so_r1_a2_g1_formal_generation';rep.mkdir(parents=True,exist_ok=True)
    d0=json.loads((root/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1/D0_ACCEPTANCE.json').read_text());lock=root/'config/so_r1/SO_R1_A2_D0_PROTOCOL_LOCK.json'
    if not(d0.get('status')=='PASS' and d0.get('protocol_status')=='FROZEN' and d0.get('full_generation_stage_authorized') and not d0.get('full_generation_started') and d0.get('formal_rgb_generated')==0):raise RuntimeError('FAIL_AUTHORITATIVE_INPUT')
    disk=shutil.disk_usage(root); admission={'status':'PASS' if disk.free/(1024**3)>=24.0965 else 'FAIL_ENVIRONMENT_ADMISSION','windows':platform.platform(),'python_executable':sys.executable,'python_version':sys.version,'numpy':np.__version__,'cpu_logical':psutil.cpu_count(),'cpu_physical':psutil.cpu_count(logical=False),'ram_total_gib':psutil.virtual_memory().total/(1024**3),'ram_available_gib':psutil.virtual_memory().available/(1024**3),'disk_free_gib':disk.free/(1024**3),'gpu_used':False,'wsl_used':False,'requested_formal_workers':8,'protocol_hash':_sha(lock),'thread_env':{k:os.getenv(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}};_write(rep/'environment_admission.json',admission)
    if admission['status']!='PASS':raise RuntimeError('FAIL_ENVIRONMENT_ADMISSION')
    planned=_rows(root/'data/processed/SO_R1_A2_D0_ProtocolFreeze_v1/planned_latent_manifest.csv');sel=_selection(planned);_write(rep/'parallel_calibration_selection.json',{'rule':'first latent_index per split','counts':{'Train':48,'Validation':16,'ID Test':16,'Camera-OOD':16,'Light-OOD':16,'Joint-OOD':16},'latent_ids':[x['latent_id'] for x in sel]})
    if not (cal/'CALIBRATION_ACCEPTANCE.json').exists():
        a=[{'root':str(root),'outroot':str(cal/'worker_1_reference'),'row':x,'calibration':True} for x in sel];b=[{'root':str(root),'outroot':str(cal/'worker_8_candidate'),'row':x,'calibration':True} for x in sel];ra,ta=_run(a,1);rb,tb=_run(b,8);la,aa,pa,qa=_collect(ra);lb,ab,pb,qb=_collect(rb);bad=[]
        for x,y in zip(la,lb):
            for f in ('canonical_content_hash','m_hash','h_hash','mask_hash','rgb_stack_hash','metadata_hash'):
                if x[f]!=y[f]:bad.append({'latent_id':x['latent_id'],'field':f,'reference':x[f],'candidate':y[f]});break
        cov=len(pd.DataFrame(aa)[['camera','light']].drop_duplicates());acc={'status':'PASS' if not bad and len(la)==128 and cov==24 else 'FAIL_PARALLEL_CALIBRATION','parallel_parity':f'{128-len(bad)} / 128','selected_workers':8 if not bad else None,'coverage':cov,'worker_1_wall_seconds':ta,'worker_8_wall_seconds':tb,'worker_1_latent_per_minute':128/ta*60,'worker_8_latent_per_minute':128/tb*60,'first_mismatch':bad[0] if bad else None,**CAL_FLAGS};_write(cal/'CALIBRATION_ACCEPTANCE.json',acc);_write(rep/'parallel_calibration.json',acc);(rep/'parallel_calibration_report.md').write_text('# G1 Parallel Calibration\n\n'+json.dumps(acc,indent=2)+'\n',encoding='utf-8')
    acc=json.loads((cal/'CALIBRATION_ACCEPTANCE.json').read_text())
    if acc['status']!='PASS':raise RuntimeError('FAIL_PARALLEL_CALIBRATION')
    if out.exists() and (out/'G1_ACCEPTANCE.json').exists() and json.loads((out/'G1_ACCEPTANCE.json').read_text()).get('status')=='PASS':return
    _write(out/'GENERATION_STATUS.json',{'status':'RUNNING','workers':8,'resume_enabled':True,'checkpoint_contract':'canonical NPZ content + metadata + per-latent QC/checkpoint record'})
    tasks=[{'root':str(root),'outroot':str(out),'row':x,'calibration':False} for x in planned];results,wall=_run(tasks,8);lat,aq,pair,qc=_collect(results);mroot=out/'manifests';mroot.mkdir(parents=True,exist_ok=True);_copy_manifest(mroot/'latent_manifest.csv',lat);_copy_manifest(mroot/'acquisition_manifest.csv',aq);_copy_manifest(mroot/'pair_manifest.csv',pair);_copy_manifest(mroot/'qc_manifest.csv',qc);_copy_manifest(mroot/'content_hash_manifest.csv',[{'latent_id':x['latent_id'],'canonical_content_hash':x['canonical_content_hash'],'file_sha256':x['file_sha256']} for x in lat])
    ad=pd.DataFrame(aq);ld=pd.DataFrame(lat);coverage=len(ad[['camera','light']].drop_duplicates());mh={'pearson':float(pearsonr(ld.M_mean,ld.H_mean).statistic),'spearman':float(spearmanr(ld.M_mean,ld.H_mean).statistic)};summary={'latent_count':len(lat),'acquisition_count':len(aq),'pair_count':len(pair),'coverage':coverage,'high_clip_violations':int((ad.high_clip_element_fraction>.1).sum()),'low_clip_violations':int((ad.low_clip_element_fraction>.1).sum()),'nonfinite':int(ad.nonfinite_count.sum()),'mask_quota':ld.mask_category.value_counts().to_dict(),'mh':mh,'wall_seconds':wall,'latents_per_minute':13500/wall*60,'formal_rgb_generated':len(aq)};_write(rep/'final_qc_summary.json',summary);_write(out/'dataset_metadata.json',{'protocol_hash':_sha(lock),'workers':8,'flags':FLAGS,'formal_generation_started':True});_write(out/'GENERATION_STATUS.json',{'status':'COMPLETE','workers':8})
    # Independent replay deliberately remains a post-generation stage and is not silently skipped.
    g={'status':'INCOMPLETE_REPLAY_REQUIRED','dataset_status':'NOT_ACCEPTED','formal_dataset_complete':False,'selected_workers':8,'training_authorized':False,**summary};_write(out/'G1_ACCEPTANCE.json',g)
