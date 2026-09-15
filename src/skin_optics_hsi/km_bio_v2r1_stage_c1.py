"""KM-BIO-v2R.1 R-C1 train-only identifiability and robustness audit."""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
import yaml
from .km_bio_inverse import FitSettings, fit_bounded_spectrum, spectral_metrics
from .km_bio_v2r import forward_preloaded_numpy, load_optical_numpy
from .km_bio_observation import sha256_file

FIT_LO=np.array([0.0,0.0]); FIT_HI=np.array([0.43,0.10])
WAVELENGTH=np.arange(400.,701.,10.)

def _resolve(root:Path, value:str)->Path:
    p=Path(value); return (root/p).resolve() if not p.is_absolute() else p.resolve()
def _jsonable(v:Any)->Any:
    if isinstance(v,(np.bool_,bool)): return bool(v)
    if isinstance(v,(np.integer,)): return int(v)
    if isinstance(v,(np.floating,)): return float(v)
    if isinstance(v,np.ndarray): return v.tolist()
    if isinstance(v,dict): return {str(k):_jsonable(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)): return [_jsonable(x) for x in v]
    return v
def _csv(df,path): df.to_csv(path,index=False,encoding='utf-8-sig')
def _fitsettings(c):
    s=c['solver']; return FitSettings(epsilon=float(c['analysis']['epsilon']),sobol_starts=int(s['sobol_starts']),seed=int(s['random_seed']),ftol=float(s['ftol']),xtol=float(s['xtol']),gtol=float(s['gtol']),max_nfev=int(s['max_nfev']))
def _cols(): return [f'observed_reflectance_{int(w)}nm' for w in WAVELENGTH]
def _fit_one(obs,opt,wfit,kwargs,s0,seed,c,hb_g_l=150.0):
    scaled_opt={k:np.array(v,copy=True) for k,v in opt.items()}
    hb_scale=float(hb_g_l)/150.0
    scaled_opt['mua_hbo2']*=hb_scale; scaled_opt['mua_hb']*=hb_scale
    of={k:v[(WAVELENGTH>=wfit[0])&(WAVELENGTH<=wfit[1])] for k,v in scaled_opt.items()}; mask=(WAVELENGTH>=wfit[0])&(WAVELENGTH<=wfit[1])
    def forward(theta): return forward_preloaded_numpy(theta,of,wavelength_nm=WAVELENGTH[mask],s0=s0,**kwargs)
    fit=fit_bounded_spectrum(obs[mask],forward,FIT_LO,FIT_HI,FitSettings(epsilon=float(c['analysis']['epsilon']),sobol_starts=int(c['solver']['sobol_starts']),seed=seed,ftol=float(c['solver']['ftol']),xtol=float(c['solver']['xtol']),gtol=float(c['solver']['gtol']),max_nfev=int(c['solver']['max_nfev'])))
    if not fit['success']: raise RuntimeError('non-converged sensitivity fit')
    pred=forward_preloaded_numpy(fit['theta'],scaled_opt,wavelength_nm=WAVELENGTH,s0=s0,**kwargs)
    m=spectral_metrics(pred[mask],obs[mask],float(c['analysis']['epsilon']))
    return fit,pred,m
def run_stage_c1(config_path: str|Path, project_root: str|Path)->dict[str,Any]:
    root=Path(project_root).resolve(); c=yaml.safe_load(Path(config_path).resolve().read_text(encoding='utf-8')); out=_resolve(root,c['output_directory'])
    if out.exists() and any(out.iterdir()): raise RuntimeError(f'output exists: {out}')
    out.mkdir(parents=True,exist_ok=True)
    for key in ('raw_hsi_content_allowed','rgb_content_allowed','validation_content_allowed','test_content_allowed','clinical_500_content_allowed'):
        if c['execution'].get(key): raise RuntimeError('forbidden data access enabled')
    rdir=_resolve(root,c['inputs']['r_c0r_output']); decision=json.loads(_resolve(root,c['inputs']['r_c0r_decision']).read_text(encoding='utf-8')); audit=json.loads(_resolve(root,c['inputs']['r_c0r_audit']).read_text(encoding='utf-8'))
    if decision.get('status')!='R_C0R_COMPLETE' or decision.get('selected_candidate')!='V2R-PS': raise RuntimeError('R-C0R selection/status mismatch')
    if audit.get('data_access',{}).get('validation_reads')!=0: raise RuntimeError('upstream validation access detected')
    manifest=pd.read_parquet(_resolve(root,c['inputs']['observation_manifest'])); cols=_cols()
    if len(manifest)!=44 or set(manifest['split'])!={'train'} or set(manifest['input_quality_status'])!={'PASS'}: raise RuntimeError('manifest integrity mismatch')
    obs=manifest[cols].to_numpy(float); opt=load_optical_numpy(WAVELENGTH,_resolve(root,c['inputs']['optical_asset_10nm']))
    fold=pd.read_csv(rdir/'fold_global_parameters.csv'); fold=fold[fold['candidate'].isin(c['selection']['candidates'])].copy(); _csv(fold,out/'fold_global_parameters_reused.csv')
    rows=[]; prof=[]; sens=[]; settings=_fitsettings(c)
    for cand in c['selection']['candidates']:
      for _,item in manifest.iterrows():
        sid=str(item.subject_id); frow=fold[(fold.candidate==cand)&(fold.outer_fold==int((pd.read_csv(rdir/'fold_manifest.csv').query('subject_id==@sid').iloc[0].outer_fold)))].iloc[0]; kwargs={'diameter_um':15.0,'scattering_amplitude':float(frow.A_s),'delta_bs':float(frow.delta_bs),'g0':float(frow.g0)}; ix=manifest.index.get_loc(item.name); fit,pred,m=_fit_one(obs[ix],opt,(420,680),kwargs,0.7,settings.seed+ix+(0 if cand=='V2R-PS' else 10000),c); u=(fit['theta']-FIT_LO)/(FIT_HI-FIT_LO)
        rows.append({'candidate':cand,'subject_id':sid,'outer_fold':int(frow.outer_fold),'f_mel':fit['theta'][0],'f_blood':fit['theta'][1],'logrmse':m['logrmse'],'rmse':m['rmse'],'sam_deg':m['sam_deg'],'f_mel_boundary':bool((u[0]<=.01)|(u[0]>=.99)),'f_blood_boundary':bool((u[1]<=.01)|(u[1]>=.99)),'converged':True})
        for parameter,values in [('diameter_um',c['sensitivity']['diameter_um']),('s0',c['sensitivity']['s0']),('epidermis_thickness_mm',c['sensitivity']['epidermis_thickness_mm']),('whole_blood_hb_g_l',c['sensitivity']['whole_blood_hb_g_l']),('scattering_amplitude_multiplier',c['sensitivity']['scattering_amplitude_multiplier']),('delta_bs_shift',c['sensitivity']['delta_bs_shift'])]:
          for value in values:
            kw=dict(kwargs); label=float(value)
            if parameter=='diameter_um': kw['diameter_um']=label
            elif parameter=='s0': pass
            elif parameter=='epidermis_thickness_mm': kw['epidermis_thickness_mm']=label
            elif parameter=='whole_blood_hb_g_l': kw['diameter_um']=kwargs['diameter_um']; label=float(value)
            elif parameter=='scattering_amplitude_multiplier': kw['scattering_amplitude']=kwargs['scattering_amplitude']*label
            elif parameter=='delta_bs_shift': kw['delta_bs']=kwargs['delta_bs']+label
            ss0=label if parameter=='s0' else 0.7
            if parameter=='epidermis_thickness_mm': pass
            fit2,p2,m2=_fit_one(obs[ix],opt,(420,680),kw,ss0,settings.seed+ix+20000,c,hb_g_l=label if parameter=='whole_blood_hb_g_l' else 150.0); uu=(fit2['theta']-FIT_LO)/(FIT_HI-FIT_LO)
            sens.append({'candidate':cand,'subject_id':sid,'outer_fold':int(frow.outer_fold),'setting':parameter,'value':float(value),'f_mel':fit2['theta'][0],'f_blood':fit2['theta'][1],'logrmse':m2['logrmse'],'rmse':m2['rmse'],'sam_deg':m2['sam_deg'],'delta_f_mel':fit2['theta'][0]-fit['theta'][0],'delta_f_blood':fit2['theta'][1]-fit['theta'][1],'boundary_any':bool(np.any((uu<=.01)|(uu>=.99))),'converged':True})
        for label,(lo,hi) in [('bandwidth',(430,670)),('bandwidth',(440,660))]:
          fit2,p2,m2=_fit_one(obs[ix],opt,(lo,hi),kwargs,0.7,settings.seed+ix+30000,c); uu=(fit2['theta']-FIT_LO)/(FIT_HI-FIT_LO)
          sens.append({'candidate':cand,'subject_id':sid,'outer_fold':int(frow.outer_fold),'setting':f'{label}_{lo}_{hi}','value':float(hi-lo),'f_mel':fit2['theta'][0],'f_blood':fit2['theta'][1],'logrmse':m2['logrmse'],'rmse':m2['rmse'],'sam_deg':m2['sam_deg'],'delta_f_mel':fit2['theta'][0]-fit['theta'][0],'delta_f_blood':fit2['theta'][1]-fit['theta'][1],'boundary_any':bool(np.any((uu<=.01)|(uu>=.99))),'converged':True})
    base=pd.DataFrame(rows); sdf=pd.DataFrame(sens); _csv(base,out/'baseline_subject_metrics.csv'); _csv(sdf,out/'sensitivity_subject_metrics.csv')
    for cand,g in base.groupby('candidate'):
      for p,gg in sdf[sdf.candidate==cand].groupby('setting'):
        prof.append({'candidate':cand,'setting':p,'setting_count':int(gg.value.nunique()),'median_logrmse':float(gg.logrmse.median()),'median_abs_delta_f_mel':float(gg.delta_f_mel.abs().median()),'median_abs_delta_f_blood':float(gg.delta_f_blood.abs().median()),'max_abs_delta_f_mel':float(gg.delta_f_mel.abs().max()),'max_abs_delta_f_blood':float(gg.delta_f_blood.abs().max()),'boundary_fraction':float(gg.boundary_any.mean()),'all_converged':bool(gg.converged.all())})
    ps=pd.DataFrame(prof); _csv(ps,out/'sensitivity_summary.csv')
    pp=[]
    for cand,g in base.groupby('candidate'):
      for param in ['f_mel','f_blood']:
        vals=g[param]; pp.append({'candidate':cand,'parameter':param,'median':float(vals.median()),'q05':float(vals.quantile(.05)),'q95':float(vals.quantile(.95)),'boundary_fraction':float(g[f'{param}_boundary'].mean()),'profile_identifiable':bool((vals.quantile(.95)-vals.quantile(.05))<=(.2*(.43 if param=='f_mel' else .10)))})
    pp=pd.DataFrame(pp); _csv(pp,out/'parameter_profile_summary.csv'); _csv(pd.DataFrame([{'candidate':cand,'subject_id':r.subject_id,'parameter':p,'value':r[p]} for _,r in base.iterrows() for p in ['f_mel','f_blood']]),out/'subject_parameter_profiles.csv')
    select=pd.DataFrame([{'candidate':x,'median_logrmse':float(base[base.candidate==x].logrmse.median()),'selected_by_r_c0r':x=='V2R-PS'} for x in c['selection']['candidates']]); _csv(select,out/'candidate_selection_recheck.csv')
    audit_out={'schema_version':1,'task_id':c['task_id'],'protocol_version':c['protocol_version'],'status':'R_C1_COMPLETE','data_access':{'manifest_content_reads':1,'raw_hsi_reads':0,'rgb_reads':0,'validation_reads':0,'test_reads':0,'clinical_500_reads':0},'subject_count':44,'candidates':c['selection']['candidates'],'all_sensitivity_converged':bool(sdf.converged.all()),'reflectance_clipping_applied':False,'fit_band_count':27,'edge_band_count':4}
    (out/'r_c1_integrity_audit.json').write_text(json.dumps(_jsonable(audit_out),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    dec={'schema_version':1,'task_id':c['task_id'],'protocol_version':c['protocol_version'],'status':'R_C1_COMPLETE','candidate_scope':c['selection']['candidates'],'selected_development_candidate':'V2R-PS','best_oof_candidate':'V2R-PSG','validation_executed':False,'test_executed':False,'clinical_500_executed':False,'rgb_encoder_training_executed':False,'interpretation':'Train-only identifiability and robustness audit; no physiological truth claim.'}
    (out/'r_c1_decision.json').write_text(json.dumps(dec,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    report=['# KM-BIO-v2R.1 R-C1 parameter identifiability and robustness audit','', '- Status: `R_C1_COMPLETE`','- Candidates audited: `V2R-PS`, `V2R-PSG`','- Scope: Train-only, fixed R-C0R global values, no Validation/Test/500/RGB access.','', 'See `parameter_profile_summary.csv` and `sensitivity_summary.csv` for deterministic profile and sensitivity results. Boundary and profile flags are diagnostic and do not establish physiological validity.']
    (out/'R_C1_REPORT.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    files=[('config', 'r_c1_config',Path(config_path).resolve()),('implementation','source',Path(__file__).resolve())]
    for p in [out/'r_c1_integrity_audit.json',out/'r_c1_decision.json',out/'R_C1_REPORT.md']+[out/x for x in ['candidate_selection_recheck.csv','subject_parameter_profiles.csv','parameter_profile_summary.csv','sensitivity_subject_metrics.csv','sensitivity_summary.csv','fold_global_parameters_reused.csv','baseline_subject_metrics.csv']]: files.append(('output',p.name,p))
    _csv(pd.DataFrame([{'role':r,'name':n,'path':str(p),'sha256':sha256_file(p)} for r,n,p in files]),out/'artifact_hash_manifest.csv')
    return {'status':'R_C1_COMPLETE','output_directory':str(out),'subject_count':44}
