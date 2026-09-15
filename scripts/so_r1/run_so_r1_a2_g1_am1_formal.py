import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import sys,json,time
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1 import formal_generation as g
g.PROTOCOL='SO-R1-A2-D0-AM1';g.VERSION='1.1';g.SEEN=['Canon 5DMarkII','Hasselblad H2','Nikon D80','Point Grey Grasshopper2 14S5C'];g.UNSEEN=['Canon 1DMarkIII','Nikon D5100']
def main():
 cal=json.loads((ROOT/'data/processed/SO_R1_A2_G1_AM1_Calibration_v2/CALIBRATION_ACCEPTANCE.json').read_text())
 if cal['status']!='PASS':raise RuntimeError('FAIL_CALIBRATION_GATE')
 out=ROOT/'data/processed/SO_R1_A2_G1_AM1_FullGeneration_v1';rep=ROOT/'reports/so_r1_a2_g1_am1_full_generation';out.mkdir(parents=True,exist_ok=True);rep.mkdir(parents=True,exist_ok=True)
 rows=g._rows(ROOT/'data/processed/SO_R1_A2_D0_AM1_ProtocolFreeze_v1/manifests/latent_manifest.csv');g._write(out/'GENERATION_STATUS.json',{'status':'RUNNING','workers':8,'formal_generation_started':True,'training_authorized':False,'renderer_backend':'CPU_NUMPY_FROZEN_PATH','gpu_used':False})
 try:
  result,seconds=g._run([{'root':str(ROOT),'outroot':str(out),'row':r,'calibration':False} for r in rows],8)
  lat,acq,pairs,qc=g._collect(result);m=out/'manifests';m.mkdir(parents=True,exist_ok=True)
  for n,x in [('latent_manifest.csv',lat),('acquisition_manifest.csv',acq),('pair_manifest.csv',pairs),('qc_manifest.csv',qc)]:__import__('pandas').DataFrame(x).to_csv(m/n,index=False)
  acc={'status':'INCOMPLETE_REPLAY_REQUIRED','dataset_status':'NOT_ACCEPTED','full_generation_status':'PASS','formal_generation_completed':True,'latent_count':len(lat),'acquisition_count':len(acq),'pair_count':len(pairs),'wall_seconds':seconds,'training_authorized':False};g._write(out/'G1_AM1_ACCEPTANCE.json',acc);g._write(out/'GENERATION_STATUS.json',{'status':'COMPLETE_PENDING_REPLAY','workers':8})
 except Exception as e:
  g._write(out/'GENERATION_STATUS.json',{'status':'FAIL_QC_OR_RUNTIME','dataset_status':'NOT_ACCEPTED','training_authorized':False,'error':repr(e)});raise
if __name__=='__main__':main()
