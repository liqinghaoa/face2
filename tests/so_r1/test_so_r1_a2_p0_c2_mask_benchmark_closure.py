from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.skin_optics_so_r1.mask_benchmark_closure import _derive_mask, _validate_masks, SPLIT_CATEGORY_QUOTAS


ROOT=Path(__file__).resolve().parents[2]

def _artifact_paths():
    return (ROOT/'data/processed/SO_R1_A2_P0_C2_MaskBenchmarkClosure_v1',
            ROOT/'reports/so_r1_a2_p0_c2_mask_benchmark_closure')

def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def test_fixed_split_category_quotas_are_256_and_global_102_102_52():
    counts={k:sum(v.get(k,0) for v in SPLIT_CATEGORY_QUOTAS.values()) for k in ('full','mild','strong')}
    assert counts=={'full':102,'mild':102,'strong':52}
    assert sum(sum(x.values()) for x in SPLIT_CATEGORY_QUOTAS.values())==256

def test_mask_sampler_is_deterministic_continuous_and_in_range():
    for cat, rng in [('mild',(.72,.93)),('strong',(.32,.68))]:
        a=_derive_mask(20260822,'L123',cat,0,256,rng,128,.90)
        b=_derive_mask(20260822,'L123',cat,0,256,rng,128,.90)
        assert np.array_equal(a[0],b[0]) and a[3]==b[3]
        assert set(np.unique(a[0])) <= {0,1} and a[0].any() and a[4]>=.9
        assert (.70<=a[1]<=.95) if cat=='mild' else (.30<=a[1]<=.70)

def test_c2_artifacts_if_benchmark_has_run_are_complete_and_not_training_data():
    out,report=_artifact_paths()
    if not out.exists():
        return
    manifest=pd.read_csv(report/'benchmark_manifest.csv')
    assert len(manifest)==256 and manifest.TRAINING_FORBIDDEN.all() and manifest.NOT_PART_OF_FORMAL_DATASET.all()
    assert len(list((out/'compact_storage').glob('*.npz')))==256
    assert len(list((out/'replay').glob('*.npz')))==32
    assert (report/'runtime_benchmark.json').is_file() and (report/'loader_validation.json').is_file()

def test_c2_mask_qc_and_independence_closure_if_present():
    out,report=_artifact_paths()
    if not out.exists(): return
    mask=pd.read_csv(report/'mask_manifest.csv')
    assert mask.mask_category.value_counts().to_dict()=={'full':102,'mild':102,'strong':52}
    assert (mask[mask.mask_category=='full'].valid_skin_fraction==1).all()
    assert mask[mask.mask_category=='mild'].valid_skin_fraction.between(.70,.95).all()
    assert mask[mask.mask_category=='strong'].valid_skin_fraction.between(.30,.70).all()
    assert mask.binary.all() and mask.finite.all() and mask.nonempty.all()
    assert (mask[mask.mask_category!='full'].largest_connected_component_fraction>=.90).all()
    for cat in ('mild','strong'):
        assert mask[mask.mask_category==cat].mask_hash.nunique()/len(mask[mask.mask_category==cat])>=.95
    corr=_read(report/'mask_mh_independence.json')
    assert corr['pass'] and all(abs(corr[k])<=.10 for k in corr if k.startswith(('pearson','spearman')))

def test_c2_warmup_rss_float16_and_compression_measurements_if_present():
    out,report=_artifact_paths()
    if not out.exists(): return
    runtime=_read(report/'runtime_benchmark.json'); f16=_read(report/'float16_roundtrip_measurement.json'); storage=_read(report/'compact_storage_validation.json')
    assert runtime['warmup_completed'] and runtime['warmup_latent_count']==16 and runtime['warmup_acquisition_count']==80 and runtime['warmup_seconds']>0
    assert runtime['peak_rss_recorded'] and runtime['rss_sample_count']>0 and runtime['rss_peak_bytes']>=runtime['rss_start_bytes']>0
    assert f16['gates']['rgb_max_le_0_00025'] and f16['gates']['rgb_p99_le_0_00023'] and f16['mask']['mismatch_pixel_count']==0
    assert storage['writer']=='np.savez_compressed' and storage['compression_algorithm']=='zip_deflate' and storage['compression_ratio']>1

def test_c2_replay_loader_coverage_and_protection_if_present():
    out,report=_artifact_paths()
    if not out.exists(): return
    replay=_read(report/'replay_validation.json'); loader=_read(report/'loader_validation.json'); protected=_read(report/'protected_asset_hash_audit.json')
    assert (replay['requested_replay_latents'],replay['completed_replay_latents'],replay['matched_replay_latents'],replay['mismatch_count'])==(32,32,32,0)
    assert loader['tested_latents']==loader['successful_latents']==256 and loader['failed_latents']==0
    assert all(loader[k]==0 for k in ('metadata_mismatch_count','shape_mismatch_count','dtype_mismatch_count','hash_mismatch_count'))
    assert len(pd.read_csv(report/'camera_light_coverage.csv'))==32
    assert protected['pass'] and protected['changed_count']==protected['missing_count']==protected['unexpected_new_protected_count']==0

def test_c2_consolidated_result_keeps_p0_failed_and_report_has_required_sections_if_present():
    out,report=_artifact_paths()
    if not out.exists(): return
    acc=_read(out/'P0_CONSOLIDATED_ACCEPTANCE_v3.json'); projection=_read(report/'full_scale_projection.json')
    assert acc['c2_execution_status']=='COMPLETE' and acc['p0_1_status']=='FAIL_CAMERA_COLOR_CHAIN_ARTIFACT'
    assert acc['p0_2_status']=='PASS' and acc['p0_3_status']=='PASS' and acc['overall_p0_status']=='FAIL'
    assert acc['next_stage']=='SO-R1-A0-AM1' and acc['next_stage_authorized'] and not acc['a2_d0_authorized'] and not acc['full_generation_authorized']
    assert projection['disk_gate']=='PASS' and projection['final_projection_gib']<=35
    text=(report/'SO_R1_A2_P0_C2_Mask_Benchmark_Closure_Report.md').read_text(encoding='utf-8')
    assert sum(line.startswith('## ') for line in text.splitlines())>=26
