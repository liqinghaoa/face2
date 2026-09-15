import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1 import formal_generation as g
g.PROTOCOL='SO-R1-A2-D0-AM1';g.VERSION='1.1';g.SEEN=['Canon 5DMarkII','Hasselblad H2','Nikon D80','Point Grey Grasshopper2 14S5C'];g.UNSEEN=['Canon 1DMarkIII','Nikon D5100']
def main():
 rows=g._rows(ROOT/'data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/manifests/latent_manifest.csv');sel=g._selection(rows)
 cal=ROOT/'data/processed/SO_R1_A2_G1_AM1_Calibration_v2';rep=ROOT/'reports/so_r1_a2_g1_am1_full_generation';rep.mkdir(parents=True,exist_ok=True)
 tasks=lambda d:[{'root':str(ROOT),'outroot':str(cal/d),'row':r,'calibration':True} for r in sel]
 a,ta=g._run(tasks('worker_1'),1);b,tb=g._run(tasks('worker_8'),8);la,aa,pa,qa=g._collect(a);lb,ab,pb,qb=g._collect(b)
 fields=['canonical_content_hash','m_hash','h_hash','mask_hash','rgb_stack_hash','metadata_hash'];bad=[x['latent_id'] for x,y in zip(la,lb) if any(x[k]!=y[k] for k in fields)]
 acc={'status':'PASS' if not bad else 'FAIL_CALIBRATION_PARITY_OR_QC','calibration_status':'PASS' if not bad else 'FAIL','workers_1_vs_8_parity':f'{128-len(bad)}/128 exact','formal_generation_authorized':not bool(bad),'training_authorized':False,'worker_1_seconds':ta,'worker_8_seconds':tb,'mismatch_latents':bad,'supersedes_failed_calibration':'SO_R1_A2_G1_AM1_Calibration_v1'}
 g._write(cal/'CALIBRATION_ACCEPTANCE.json',acc);g._write(rep/'calibration_workers_1_vs_8_parity.json',acc);pd=__import__('pandas');pd.DataFrame(sel).to_csv(rep/'calibration_manifest.csv',index=False);print(json.dumps(acc,indent=2))

if __name__=='__main__':
 main()
