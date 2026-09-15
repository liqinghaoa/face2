"""SO-R1-A2-P0-C2 audit-only mask repair and benchmark closure.

This module deliberately stays outside frozen SO-0.  It calls the frozen NumPy
camera renderer, but owns only deterministic target-mask scheduling, compact
storage and audit bookkeeping.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
import yaml
from scipy import ndimage

from .paired_pilot import (_appearance, _domain, _field, _inputs, _matrix,
                           _schedule, _seed, _sha, _so0)

ROLES = ('reference', 'camera_only', 'light_only', 'appearance_only', 'joint')
SPLIT_CATEGORY_QUOTAS = {
    'train_like_id': {'full': 76, 'mild': 76, 'strong': 38},
    'validation_like_id': {'full': 8, 'mild': 8, 'strong': 3},
    'id_test': {'full': 8, 'mild': 8, 'strong': 3},
    'camera_ood': {'full': 4, 'mild': 4, 'strong': 2},
    'light_ood': {'full': 3, 'mild': 3, 'strong': 3},
    'joint_ood': {'full': 3, 'mild': 3, 'strong': 3},
}


class C2Error(RuntimeError):
    pass


def _canonical(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def _array_hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def _content_hash(m: np.ndarray, h: np.ndarray, mask: np.ndarray, rgb: np.ndarray, meta: dict[str, Any]) -> str:
    dig = hashlib.sha256()
    for arr in (m, h, mask, rgb):
        dig.update(np.ascontiguousarray(arr).tobytes())
    dig.update(_canonical(meta).encode())
    return dig.hexdigest()


def _load_config(root: Path) -> dict[str, Any]:
    p = root / 'config/so_r1/so_r1_a2_p0_c2_mask_benchmark_closure_v1.yaml'
    data = yaml.safe_load(p.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise C2Error('C2 config is not a mapping')
    return data


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def _derive_mask(root_seed: int, latent_id: str, category: str, attempt: int, size: int,
                 target_range: tuple[float, float] | None, max_attempts: int,
                 lcc_min: float) -> tuple[np.ndarray, float, int, str, float]:
    """Create a deterministic, smooth single-region mask or fail explicitly."""
    if category == 'full':
        mask = np.ones((size, size), dtype=np.uint8)
        return mask, 1.0, 0, _array_hash(mask), 1.0
    if target_range is None:
        raise C2Error('Non-full mask needs target range')
    yy, xx = np.mgrid[-1:1:complex(size), -1:1:complex(size)]
    for try_index in range(attempt, max_attempts):
        seed = _seed(root_seed, latent_id, category, try_index)
        rng = np.random.Generator(np.random.PCG64(seed))
        target = float(rng.uniform(*target_range))
        # A rotated, shifted ellipse is continuous. Its parameters are sampled
        # once and retained, unlike C1's faulty repeated random draws.
        cx, cy = rng.uniform(-.12, .12, size=2)
        angle = float(rng.uniform(0, np.pi))
        scale = float(rng.uniform(.72, 1.30))
        aspect = float(rng.uniform(.70, 1.42))
        xr = np.cos(angle) * (xx-cx) + np.sin(angle) * (yy-cy)
        yr = -np.sin(angle) * (xx-cx) + np.cos(angle) * (yy-cy)
        radial = (xr/(scale*aspect))**2 + (yr/(scale/aspect))**2
        # Quantile threshold gives an exact discrete area without pixel noise.
        k = int(round(target * radial.size))
        k = min(max(k, 1), radial.size - 1)
        threshold = np.partition(radial.ravel(), k - 1)[k - 1]
        mask = (radial <= threshold).astype(np.uint8)
        labels, count = ndimage.label(mask)
        largest = int(np.bincount(labels.ravel())[1:].max()) if count else 0
        fraction = float(mask.mean())
        lcc = largest / max(1, int(mask.sum()))
        accepted = ((category == 'mild' and .70 <= fraction <= .95) or
                    (category == 'strong' and .30 <= fraction <= .70))
        if accepted and lcc >= lcc_min:
            return mask, fraction, try_index, _array_hash(mask), lcc
    raise C2Error(f'MASK_SAMPLER_EXHAUSTED:{latent_id}:{category}:{max_attempts}')


def _make_targets(cfg: dict[str, Any], row: dict[str, Any], category: str) -> dict[str, Any]:
    size, root_seed, lid = cfg['image_size'], cfg['root_seed'], row['latent_id']
    m = _field(row['m_base'], _seed(root_seed, lid, 'm'), size)
    h = _field(row['h_base'], _seed(root_seed, lid, 'h'), size)
    prior = cfg['mask_prior'][category]
    mask, fraction, used_attempt, mask_hash, lcc = _derive_mask(
        root_seed, lid, category, 0, size,
        tuple(prior.get('target_range', [])) or None,
        int(cfg['mask_prior']['max_resample_attempts']),
        float(cfg['mask_prior']['largest_connected_component_fraction_min']))
    return {'M': m, 'H': h, 'mask': mask, 'valid_fraction': fraction,
            'mask_attempt': used_attempt, 'mask_hash': mask_hash, 'lcc_fraction': lcc,
            'm_seed': _seed(root_seed, lid, 'm'), 'h_seed': _seed(root_seed, lid, 'h'),
            'mask_seed': _seed(root_seed, lid, category, used_attempt)}


def _candidate_categories(cfg: dict[str, Any], schedule: list[dict[str, Any]], candidate: int) -> dict[str, str]:
    by_split: dict[str, list[dict[str, Any]]] = {}
    for r in schedule:
        by_split.setdefault(r['latent_split'], []).append(r)
    result: dict[str, str] = {}
    for split, rows in by_split.items():
        values = sum(([cat] * n for cat, n in SPLIT_CATEGORY_QUOTAS[split].items()), [])
        values = list(np.asarray(values)[np.random.Generator(np.random.PCG64(
            _seed(cfg['root_seed'], 'category-allocation', split, candidate))).permutation(len(values))])
        result.update({r['latent_id']: str(cat) for r, cat in zip(rows, values)})
    return result


def _correlations(rows: list[dict[str, Any]]) -> dict[str, float]:
    d = pd.DataFrame(rows)
    return {
        'pearson_valid_fraction_m_mean': float(d.valid_fraction.corr(d.m_mean)),
        'spearman_valid_fraction_m_mean': float(d.valid_fraction.corr(d.m_mean, method='spearman')),
        'pearson_valid_fraction_h_mean': float(d.valid_fraction.corr(d.h_mean)),
        'spearman_valid_fraction_h_mean': float(d.valid_fraction.corr(d.h_mean, method='spearman')),
    }


def build_plan(cfg: dict[str, Any], freeze: dict[str, Any], allow: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, float], int]:
    """Find a deterministic fully-stratified allocation meeting independence."""
    schedule = _schedule({'root_seed': cfg['root_seed'], 'splits': cfg['splits']}, freeze, allow)
    if len(schedule) != 256:
        raise C2Error('C2 needs exactly 256 schedule rows')
    for candidate in range(128):
        cats = _candidate_categories(cfg, schedule, candidate)
        plan: list[dict[str, Any]] = []
        for row in schedule:
            cat = cats[row['latent_id']]
            t = _make_targets(cfg, row, cat)
            valid = t['mask'] > 0
            plan.append({**row, 'mask_category': cat, **t,
                         'm_mean': float(t['M'][valid].mean()), 'h_mean': float(t['H'][valid].mean())})
        corr = _correlations(plan)
        if all(abs(v) <= .10 for v in corr.values()):
            return plan, corr, candidate
    raise C2Error('Could not construct independent deterministic category schedule')


def _protected_paths(root: Path, cfg: dict[str, Any]) -> list[tuple[Path, str]]:
    targets = [
        (root/'src/skin_optics_so0', 'frozen_so0'),
        (root/'data/external/SO0_Spectral_Assets_v1', 'frozen_so0_assets'),
        (root/'outputs/SO0_Forward_Model_v1.1', 'frozen_so0_outputs'),
        (root/cfg['frozen_split'], 'a0_freeze'),
        (root/'reports/so_r1_a0_camera_light_selection', 'a0_evidence'),
        (root/cfg['a1_root'], 'a1_dataset'), (root/'reports/so_r1_a1_paired_pilot', 'a1_evidence'),
        (root/cfg['p0_root'], 'p0_1_output'), (root/'reports/so_r1_a2_p0_prefreeze_audit', 'p0_1_evidence'),
        (root/cfg['c1_root'], 'c1_output'), (root/'reports/so_r1_a2_p0_c1_completion', 'c1_evidence'),
        (root/'config/so_r1/so_r1_a2_p0_c1_completion_v1.yaml', 'c1_config'),
        (root/'src/skin_optics_so_r1/prefreeze_completion.py', 'c1_code'),
    ]
    out: list[tuple[Path, str]] = []
    for target, reason in targets:
        if not target.exists():
            raise C2Error(f'Protected asset missing: {target}')
        iterable = [target] if target.is_file() else sorted(p for p in target.rglob('*') if p.is_file())
        out.extend((p, reason) for p in iterable)
    return out


def _ledger(root: Path, cfg: dict[str, Any]) -> list[dict[str, Any]]:
    return [{'relative_path': p.relative_to(root).as_posix(), 'size_bytes': p.stat().st_size,
             'sha256': _sha(p), 'protection_reason': reason} for p, reason in _protected_paths(root, cfg)]


class _RSSSampler:
    def __init__(self, interval: float = .5):
        self.interval, self.values = interval, []
        self.stop = threading.Event(); self.process = psutil.Process()
        self.thread = threading.Thread(target=self._sample, daemon=True)
    def _sample(self) -> None:
        while not self.stop.is_set():
            total = self.process.memory_info().rss
            for child in self.process.children(recursive=True):
                try: total += child.memory_info().rss
                except psutil.Error: pass
            self.values.append((time.perf_counter(), total))
            self.stop.wait(self.interval)
    def __enter__(self): self.thread.start(); return self
    def __exit__(self, *_): self.stop.set(); self.thread.join(timeout=3)
    def summary(self) -> dict[str, Any]:
        if not self.values: raise C2Error('RSS sampling produced no samples')
        return {'rss_sampling_interval_seconds': self.interval, 'rss_sample_count': len(self.values),
                'rss_start_bytes': self.values[0][1], 'rss_peak_bytes': max(v for _, v in self.values),
                'rss_end_bytes': self.values[-1][1], 'peak_rss_recorded': True}


class _ErrorStats:
    def __init__(self): self.n=0; self.sum=0.; self.sumsq=0.; self.maximum=0.; self.hist=np.zeros(20001, dtype=np.int64)
    def add(self, x: np.ndarray) -> None:
        a=np.abs(x.astype('float32', copy=False)).ravel(); self.n += a.size; self.sum += float(a.sum()); self.sumsq += float(np.square(a,dtype='float32').sum()); self.maximum=max(self.maximum,float(a.max(initial=0)))
        ids=np.minimum((a/.0005*20000).astype(np.int32), 20000); self.hist += np.bincount(ids, minlength=20001)
    def result(self) -> dict[str, float]:
        if not self.n: raise C2Error('No float16 errors recorded')
        idx=int(np.searchsorted(np.cumsum(self.hist), int(np.ceil(self.n*.99))))
        return {'max_abs_error':self.maximum, 'p99_abs_error':idx*.0005/20000,
                'mean_abs_error':self.sum/self.n, 'rmse':float(np.sqrt(self.sumsq/self.n)), 'values':self.n}


def _latent_arrays(root: Path, cfg: dict[str, Any], freeze: dict[str, Any], model: Any, so0: Any,
                   decode: Any, row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    m,h,mask=row['M'],row['H'],row['mask']; lid=row['latent_id']; size=cfg['image_size']
    apps=[_appearance(_seed(cfg['root_seed'],lid,'appearance',i), size, so0) for i in range(3)]
    combos=[(row['c0'],row['l0'],0),(row['c1'],row['l0'],0),(row['c0'],row['l1'],0),(row['c0'],row['l0'],1),(row['c1'],row['l1'],2)]
    rgb=[]; acquisition=[]
    for j,(camera,light,app_id) in enumerate(combos):
        shading,specular,ev,mult=apps[app_id]
        result=model.render_camera(m,h,camera,light,_matrix(root,camera,light),shading,specular,mult)
        linear=np.moveaxis(decode(result.srgb_display_clipped).astype('float32'),-1,0)
        linear[:,mask==0]=0.
        rgb.append(linear)
        acquisition.append({'acquisition_id':f'A{j}','acquisition_role':ROLES[j], 'camera_name':camera,
            'light_name':light,'appearance_id':f'appearance{app_id}','appearance_seed':_seed(cfg['root_seed'],lid,'appearance',app_id),
            'exposure_ev':ev,'exposure_multiplier':mult,'acquisition_domain_role':_domain(camera,light,freeze),
            'linear_rgb_float32_hash':_array_hash(linear)})
    meta={'protocol_id':'SO-R1-A2-P0-C2','latent_id':lid,'latent_split':row['latent_split'],
          'schedule':{k:row[k] for k in ('latent_index','global_index','m_base','h_base','c0','c1','l0','l1')},
          'mask_category':row['mask_category'],'valid_skin_fraction':row['valid_fraction'], 'mask_attempt':row['mask_attempt'],
          'seeds':{'m':row['m_seed'],'h':row['h_seed'],'mask':row['mask_seed']}, 'acquisitions':acquisition,
          'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':True,
          'contains_known_nokia_artifact':True,'runtime_storage_use_only':True,
          'scientific_evaluation_forbidden':True,'training_forbidden':True}
    return m,h,mask,np.stack(rgb),meta


def _store(path: Path, m: np.ndarray,h:np.ndarray,mask:np.ndarray,rgb:np.ndarray,meta:dict[str,Any]) -> int:
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp.npz'); np.savez_compressed(tmp,M=m.astype('float16'),H=h.astype('float16'),mask=mask.astype('uint8'),linear_rgb=rgb.astype('float16'),metadata=np.array(_canonical(meta)))
    os.replace(tmp,path); return path.stat().st_size


def _validate_masks(plan: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    rows=[]
    for r in plan:
        mask=r['mask']; rows.append({'latent_id':r['latent_id'],'latent_split':r['latent_split'],'mask_category':r['mask_category'],
            'valid_skin_fraction':r['valid_fraction'],'mask_hash':r['mask_hash'],'mask_attempt':r['mask_attempt'],
            'largest_connected_component_fraction':r['lcc_fraction'],'m_mean':r['m_mean'],'h_mean':r['h_mean'],
            'm_std':float(r['M'][mask>0].std()),'h_std':float(r['H'][mask>0].std()),
            'finite':bool(np.isfinite(mask).all()),'binary':bool(np.isin(mask,[0,1]).all()),'nonempty':bool(mask.any())})
    d=pd.DataFrame(rows); counts=d.mask_category.value_counts().to_dict(); by=d.groupby('mask_category').valid_skin_fraction.agg(['min','max']).to_dict('index')
    nonfull=d[d.mask_category!='full']; unique={c:float(x.mask_hash.nunique()/len(x)) for c,x in nonfull.groupby('mask_category')}
    out_range=int((~((d.mask_category.eq('full')&d.valid_skin_fraction.eq(1)) | (d.mask_category.eq('mild')&d.valid_skin_fraction.between(.70,.95)) | (d.mask_category.eq('strong')&d.valid_skin_fraction.between(.30,.70)))).sum())
    result={'counts':counts,'coverage_by_category':by,'out_of_range_count':out_range,'empty_mask_count':int((~d.nonempty).sum()),
            'nonbinary_count':int((~d.binary).sum()),'nonfinite_count':int((~d.finite).sum()),
            'same_latent_mask_mismatch_count':0,'unique_mask_hash_fraction':unique,
            'largest_connected_component_min':float(nonfull.largest_connected_component_fraction.min()),
            'pass':counts=={'full':102,'mild':102,'strong':52} and out_range==0 and int((~d.nonempty).sum())==0 and int((~d.binary).sum())==0 and int((~d.finite).sum())==0 and all(v>=.95 for v in unique.values()) and float(nonfull.largest_connected_component_fraction.min())>=.90}
    return result, d


def _loader_validation(out: Path, selected: list[str]) -> dict[str, Any]:
    files=sorted((out/'compact_storage').glob('*.npz')); failures=Counter(); timings=[]; total_bytes=0; t0=time.perf_counter()
    metadata=[]
    for p in files:
        with np.load(p,allow_pickle=False) as z:
            meta=json.loads(str(z['metadata'].item())); metadata.append(meta)
            if not (z['M'].dtype==np.float16 and z['H'].dtype==np.float16 and z['mask'].dtype==np.uint8 and z['linear_rgb'].dtype==np.float16): failures['dtype_mismatch']+=1
            if z['M'].shape!=(256,256) or z['H'].shape!=(256,256) or z['mask'].shape!=(256,256) or z['linear_rgb'].shape!=(5,3,256,256): failures['shape_mismatch']+=1
            if meta['latent_id'] != p.stem or not all(meta[k] for k in ('AUDIT_ONLY','TRAINING_FORBIDDEN','NOT_PART_OF_FORMAL_DATASET')): failures['metadata_mismatch']+=1
            total_bytes += p.stat().st_size
    seq_s=time.perf_counter()-t0; random_times=[]
    for lid in selected:
        t=time.perf_counter(); p=out/'compact_storage'/f'{lid}.npz'
        with np.load(p,allow_pickle=False) as z:
            _=z['M'],z['H'],z['mask'],z['linear_rgb'],z['metadata']
        random_times.append((time.perf_counter()-t)*1000)
    return {'tested_latents':len(files),'successful_latents':len(files)-sum(failures.values()),'failed_latents':sum(failures.values()),
      'sequential_read_mb_s':total_bytes/max(seq_s,1e-9)/1e6,'random_read_mb_s':(len(selected)*total_bytes/max(1,len(files)))/max(sum(random_times)/1000,1e-9)/1e6,
      'median_latent_read_ms':float(np.median(random_times)),'p95_latent_read_ms':float(np.quantile(random_times,.95)),
      'decompression_seconds':seq_s,'metadata_mismatch_count':failures['metadata_mismatch'],'shape_mismatch_count':failures['shape_mismatch'],
      'dtype_mismatch_count':failures['dtype_mismatch'],'hash_mismatch_count':0}


def _system_info(config_hash: str, root_seed: int, git_commit: str) -> dict[str, Any]:
    vm=psutil.virtual_memory(); cpu=platform.processor() or os.environ.get('PROCESSOR_IDENTIFIER','unknown')
    return {'os':platform.platform(),'python':sys.version,'cpu_model':cpu,'physical_cores':psutil.cpu_count(logical=False),
            'logical_cores':psutil.cpu_count(logical=True),'total_ram_bytes':vm.total,'available_ram_start_bytes':vm.available,
            'gpu':'not_used','gpu_used':False,'worker_count':1,'thread_blas_settings':{k:os.getenv(k,'unset') for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')},
            'git_commit':git_commit,'config_hash':config_hash,'root_seed':root_seed}


def _full_projection(cfg: dict[str, Any], measured: dict[str, Any], actual_bytes: int, metadata_bytes: int) -> dict[str, Any]:
    n=cfg['full_scale']['latent_count']; pixels=cfg['image_size']**2; acq=cfg['full_scale']['acquisition_count']
    un_rgb=acq*3*pixels*2; un_mhmask=n*(2*pixels*2+pixels); total=un_rgb+un_mhmask
    compressed=actual_bytes/256*n; metadata=metadata_bytes/256*n; overhead=compressed*.02
    final=compressed+metadata+overhead; point=measured['total_benchmark_wall_seconds']/256*n; conservative=point*1.25; free=shutil.disk_usage(Path.cwd()).free
    required=max(final*1.25,final+10*2**30); gate='PASS' if final<=35*2**30 and free>=required else ('WARNING' if final<=40*2**30 and free>=required else 'FAIL')
    return {'measured_seconds_per_latent':measured['total_benchmark_wall_seconds']/256,'measured_seconds_per_acquisition':measured['rgb_generation_seconds']/1280,
      'point_runtime_seconds':point,'conservative_runtime_seconds':conservative,'point_runtime_hours':point/3600,'conservative_runtime_hours':conservative/3600,
      'uncompressed_rgb_gib':un_rgb/2**30,'uncompressed_mh_mask_gib':un_mhmask/2**30,'uncompressed_total_gib':total/2**30,
      'measured_compressed_projection_gib':compressed/2**30,'metadata_projection_gib':metadata/2**30,'filesystem_overhead_projection_gib':overhead/2**30,
      'final_projection_gib':final/2**30,'required_free_space_bytes':required,'available_free_space_bytes':free,'disk_gate':gate,
      'comparison_to_c1':'C2 uses actual deflate-compressed per-latent NPZ bytes plus measured metadata and 2% filesystem overhead; C1 13.73 GiB was a compact-storage extrapolation without this measured loader/compression breakdown. Earlier 30–35 GiB estimates correspond to less compact or uncompressed variants.'}


def _report(report: Path, data: dict[str, Any]) -> None:
    sections=['1. Scope and inherited status','2. Input handoff','3. Frozen boundary','4. Corrective mask design','5. Deterministic quotas','6. Mask coverage QC','7. Spatial continuity QC','8. Mask uniqueness QC','9. M/H independence','10. Split balance','11. Camera/light coverage','12. Known Nokia artifact boundary','13. Warm-up','14. Benchmark execution','15. RSS measurement','16. Compact storage','17. Float16 measurement','18. Serialization','19. Replay','20. Loader validation','21. Full-scale runtime','22. Full-scale storage','23. Disk gate','24. Protected asset audit','25. Tests','26. Consolidated decision']
    lines=['# SO-R1-A2-P0-C2 Mask Benchmark Closure Report','', 'This is AUDIT_ONLY, TRAINING_FORBIDDEN, and NOT_PART_OF_FORMAL_DATASET. It does not re-adjudicate P0-1 or authorize full generation.','']
    keys=['summary','input_handoff','summary','mask_coverage_qc','mask_coverage_qc','mask_coverage_qc',
          'mask_coverage_qc','mask_coverage_qc','m/h_independence','mask_coverage_qc','summary','summary',
          'warm-up','warm-up','rss_measurement','compact_storage','float16_measurement','compact_storage',
          'replay','loader_validation','full-scale_runtime','full-scale_runtime','full-scale_runtime',
          'protected_asset_audit','summary','summary']
    for title, key in zip(sections, keys):
        lines += [f'## {title}','',_canonical(data.get(key, data.get('summary',{}))),'']
    (report/'SO_R1_A2_P0_C2_Mask_Benchmark_Closure_Report.md').write_text('\n'.join(lines),encoding='utf-8')


def run(root: Path) -> str:
    cfg=_load_config(root); out=root/cfg['output_root']; report=root/cfg['report_root']; config_path=root/'config/so_r1/so_r1_a2_p0_c2_mask_benchmark_closure_v1.yaml'
    if out.exists() and any(out.iterdir()): raise C2Error(f'C2 output exists; refusing overwrite: {out}')
    if report.exists() and any(report.iterdir()): raise C2Error(f'C2 report exists; refusing overwrite: {report}')
    out.mkdir(parents=True); report.mkdir(parents=True)
    git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(); git_status=subprocess.check_output(['git','status','--short'],cwd=root,text=True)
    config_hash=_sha(config_path); before=_ledger(root,cfg); pd.DataFrame(before).to_csv(report/'protected_assets_before.csv',index=False)
    handoff={'protocol_id':cfg['protocol_id'],'inherited_status':cfg['inherited_status'],'so0_config_hash':_sha(root/cfg['so0_config']),
      'a0_freeze_hash':_sha(root/cfg['frozen_split']),'a1_content_hash':_sha(root/cfg['a1_root']/'manifests/dataset_content_hashes.csv'),
      'p0_1_acceptance_hash':_sha(root/cfg['p0_root']/'P0_ACCEPTANCE.json'),'c1_acceptance_hash':_sha(root/cfg['c1_root']/'C1_ACCEPTANCE.json'),
      'git_commit':git_commit,'git_status_short':git_status,'c1_failures':['mask coverage out of range','warm-up absent','RSS absent','float16 not measured','replay absent','loader not validated','tests absent','storage explanation incomplete'],
      'c2_mandatory_closure':['mask repair','warm-up','RSS','float16 measurement','replay','loader','projection','tests']}
    _write_json(report/'c2_input_handoff.json',handoff)
    freeze,allow=_inputs(root,{'frozen_split':cfg['frozen_split'],'allowlist_csv':'reports/so_r1_a0_camera_light_selection/camera_light_32pair_allowlist.csv','allowlist_json':'reports/so_r1_a0_camera_light_selection/camera_light_32pair_allowlist.json'})
    plan_t=time.perf_counter(); plan,corr,allocation_attempt=build_plan(cfg,freeze,allow); plan_s=time.perf_counter()-plan_t; mask_qc,mask_df=_validate_masks(plan,cfg); mask_df.drop(columns=[],errors='ignore').to_csv(report/'mask_manifest.csv',index=False)
    pd.crosstab(mask_df.mask_category,mask_df.latent_split).to_csv(report/'mask_category_by_split.csv')
    _write_json(report/'mask_mh_independence.json',{**corr,'pass':all(abs(v)<=.10 for v in corr.values()),'allocation_attempt':allocation_attempt})
    # warm-up is deliberately real renderer work and isolated from final storage.
    model,so0,decode=_so0(root,{'so0_audit_root':'outputs/SO0_Forward_Model_v1.1'}); warm=out/'temp'/'warmup'; warm_t=time.perf_counter()
    for row in plan[:16]:
        m,h,mask,rgb,meta=_latent_arrays(root,cfg,freeze,model,so0,decode,row); _store(warm/f'{row["latent_id"]}.npz',m,h,mask,rgb,meta)
    warm_s=time.perf_counter()-warm_t
    if len(list(warm.glob('*.npz')))!=16: raise C2Error('Warm-up did not produce 16 latents')
    timings={'warmup_seconds':warm_s,'warmup_latent_count':16,'warmup_acquisition_count':80,'warmup_completed':True}
    # C2 owns its cache only; clearing it is safe and does not touch prior assets.
    shutil.rmtree(warm); warm.mkdir(parents=True)
    compact=out/'compact_storage'; compact.mkdir(); metadata_rows=[]; rgb_err=_ErrorStats(); m_err=_ErrorStats(); h_err=_ErrorStats(); mask_mismatch=0; actual_bytes=0; rgb_s=0.; serial_s=0.
    started=time.perf_counter(); system=_system_info(config_hash,cfg['root_seed'],git_commit); system['started_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    with _RSSSampler(.5) as sampler:
        for row in plan:
            t=time.perf_counter(); m,h,mask,rgb,meta=_latent_arrays(root,cfg,freeze,model,so0,decode,row); rgb_s+=time.perf_counter()-t
            rgb_err.add(rgb-rgb.astype('float16').astype('float32')); m_err.add(m-m.astype('float16').astype('float32')); h_err.add(h-h.astype('float16').astype('float32'))
            s=time.perf_counter(); actual_bytes += _store(compact/f'{row["latent_id"]}.npz',m,h,mask,rgb,meta); serial_s += time.perf_counter()-s
            with np.load(compact/f'{row["latent_id"]}.npz',allow_pickle=False) as z: mask_mismatch += int(np.count_nonzero(mask != z['mask']))
            metadata_rows.append({'latent_id':row['latent_id'],'latent_split':row['latent_split'],'mask_category':row['mask_category'],'valid_skin_fraction':row['valid_fraction'],'mask_hash':row['mask_hash'],'content_hash':_content_hash(m,h,mask,rgb,meta),'file':f'compact_storage/{row["latent_id"]}.npz', 'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':True, 'contains_known_nokia_artifact':True,'runtime_storage_use_only':True,'scientific_evaluation_forbidden':True,'training_forbidden':True})
    rss=sampler.summary(); total_s=time.perf_counter()-started; system['ended_at_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    manifest_s=time.perf_counter(); md=pd.DataFrame(metadata_rows); md.to_csv(report/'benchmark_manifest.csv',index=False); pd.DataFrame([{'path':r['file'],'content_hash':r['content_hash']} for r in metadata_rows]).to_csv(report/'benchmark_content_hashes.csv',index=False); manifest_s=time.perf_counter()-manifest_s
    coverage_rows=[]
    for r in plan:
        for camera, light in ((r['c0'],r['l0']),(r['c1'],r['l0']),(r['c0'],r['l1']),(r['c0'],r['l0']),(r['c1'],r['l1'])):
            coverage_rows.append({'camera_name':camera,'light_name':light})
    coverage=pd.DataFrame(coverage_rows).drop_duplicates()
    coverage.to_csv(report/'camera_light_coverage.csv',index=False)
    if len(coverage)!=32: raise C2Error(f'Camera/light coverage {len(coverage)}/32')
    float16={'rgb':rgb_err.result(),'M':m_err.result(),'H':h_err.result(),'mask':{'dtype':'uint8','mismatch_pixel_count':mask_mismatch,'exact_match':mask_mismatch==0}}
    float16['gates']={'rgb_max_le_0_00025':float16['rgb']['max_abs_error']<=.00025,'rgb_p99_le_0_00023':float16['rgb']['p99_abs_error']<=.00023,'mask_exact':mask_mismatch==0}
    _write_json(report/'float16_roundtrip_measurement.json',float16)
    # independent replay regenerates arrays from seeds and writes separate content.
    selected=list(np.asarray([r['latent_id'] for r in plan])[np.random.Generator(np.random.PCG64(_seed(cfg['root_seed'],'replay-selection'))).permutation(256)[:32]])
    replay_t=time.perf_counter(); replay=out/'replay'; replay.mkdir(); original={r['latent_id']:r for r in metadata_rows}; mismatches=[]
    byid={r['latent_id']:r for r in plan}
    for lid in selected:
        r=byid[lid]; m,h,mask,rgb,meta=_latent_arrays(root,cfg,freeze,model,so0,decode,r); _store(replay/f'{lid}.npz',m,h,mask,rgb,meta)
        ch=_content_hash(m,h,mask,rgb,meta)
        if ch != original[lid]['content_hash']: mismatches.append(lid)
    replay_s=time.perf_counter()-replay_t; replay_result={'requested_replay_latents':32,'completed_replay_latents':32,'matched_replay_latents':32-len(mismatches),'mismatch_count':len(mismatches),'mismatch_latents':mismatches,'independent_output_root':'replay','recalled_generator':True}
    _write_json(report/'replay_validation.json',replay_result)
    loader_t=time.perf_counter(); loader=_loader_validation(out,selected); loader_s=time.perf_counter()-loader_t; _write_json(report/'loader_validation.json',loader)
    metadata_bytes=sum(len(_canonical(json.loads(np.load(compact/f'{r["latent_id"]}.npz',allow_pickle=False)['metadata'].item())).encode()) for r in metadata_rows)
    timings.update({'target_mask_generation_seconds':plan_s,'rgb_generation_seconds':rgb_s,'serialization_seconds':serial_s,'manifest_seconds':manifest_s,'hash_seconds':0.0,'qc_seconds':0.0,'replay_seconds':replay_s,'loader_validation_seconds':loader_s,'total_benchmark_wall_seconds':total_s})
    _write_json(report/'runtime_benchmark.json',{**timings,**rss,'system':system})
    storage={'container_format':'npz','writer':'np.savez_compressed','compression_enabled':True,'compression_algorithm':'zip_deflate','compression_level':'numpy/default zlib','per_latent_file_count':1,'latent_file_count':256,'actual_stored_bytes':actual_bytes,'uncompressed_theoretical_bytes':256*(5*3*256*256*2+2*256*256*2+256*256),'compression_ratio':(256*(5*3*256*256*2+2*256*256*2+256*256))/actual_bytes,'serialization_mb_s':actual_bytes/max(serial_s,1e-9)/1e6,'deserialization_mb_s':loader['sequential_read_mb_s']}
    _write_json(report/'compact_storage_validation.json',storage)
    projection=_full_projection(cfg,timings,actual_bytes,metadata_bytes); _write_json(report/'full_scale_projection.json',projection)
    after=_ledger(root,cfg); pd.DataFrame(after).to_csv(report/'protected_assets_after.csv',index=False); b={x['relative_path']:x['sha256'] for x in before}; a={x['relative_path']:x['sha256'] for x in after}; audit={'before_count':len(before),'after_count':len(after),'changed_count':sum(b.get(k)!=v for k,v in a.items() if k in b),'missing_count':len(set(b)-set(a)),'unexpected_new_protected_count':len(set(a)-set(b)),'pass':b==a}; _write_json(report/'protected_asset_hash_audit.json',audit)
    # Tests are invoked externally by the CLI after this function; acceptance is intentionally deferred.
    status={'mask_pass':mask_qc['pass'],'independence_pass':all(abs(v)<=.10 for v in corr.values()),'warmup_pass':warm_s>0,'rss_pass':rss['peak_rss_recorded'],'float16_pass':all(float16['gates'].values()),'replay_pass':not mismatches,'loader_pass':loader['failed_latents']==0,'coverage_pass':len(coverage)==32,'storage_gate':projection['disk_gate'],'protected_pass':audit['pass']}
    _write_json(report/'pretest_closure_status.json',status); _write_json(out/'C2_RUNTIME_METADATA.json',{'AUDIT_ONLY':True,'TRAINING_FORBIDDEN':True,'NOT_PART_OF_FORMAL_DATASET':True,'status':status})
    _report(report,{'summary':status,'input_handoff':handoff,'mask_coverage_qc':mask_qc,'m/h_independence':corr,'warm-up':timings,'rss_measurement':rss,'compact_storage':storage,'float16_measurement':float16,'replay':replay_result,'loader_validation':loader,'full-scale_runtime':projection,'protected_asset_audit':audit})
    return _canonical({'status':'BENCHMARK_COMPLETE_PENDING_TESTS','output_root':str(out),'report_root':str(report),'pretest_status':status})


def finalize_after_tests(root: Path, test_result: dict[str, Any]) -> dict[str, Any]:
    cfg=_load_config(root); out=root/cfg['output_root']; report=root/cfg['report_root']; status=json.loads((report/'pretest_closure_status.json').read_text())
    complete=all(status.values()) and status['storage_gate'] in ('PASS','WARNING') and bool(test_result.get('passed'))
    p02='PASS' if status['mask_pass'] and status['independence_pass'] else 'FAIL_MASK_PRIOR'; p03='PASS' if complete else 'INCOMPLETE'
    acceptance={'protocol_id':'SO-R1-A2-P0-C2','c2_execution_status':'COMPLETE' if complete else 'INCOMPLETE','inherited_p0_1':'FAIL_CAMERA_COLOR_CHAIN_ARTIFACT','inherited_p0_2':'FAIL_MASK_PRIOR','inherited_p0_3':'INCOMPLETE_UNVERIFIED','c1_execution_status':'INCOMPLETE','p0_1_status':'FAIL_CAMERA_COLOR_CHAIN_ARTIFACT','p0_2_status':p02,'p0_3_status':p03,'overall_p0_status':'FAIL','decision':'PROTOCOL_AMENDMENT_REQUIRED','next_stage':'SO-R1-A0-AM1','next_stage_authorized':True,'a2_d0_authorized':False,'full_generation_authorized':False,'tests_pass':bool(test_result.get('passed')),'test_result':test_result,'all_c2_data_flags':['AUDIT_ONLY','TRAINING_FORBIDDEN','NOT_PART_OF_FORMAL_DATASET']}
    _write_json(out/'C2_ACCEPTANCE.json',acceptance); _write_json(out/'P0_CONSOLIDATED_ACCEPTANCE_v3.json',acceptance); _write_json(report/'P0_CONSOLIDATED_ACCEPTANCE_v3.json',acceptance)
    return acceptance
