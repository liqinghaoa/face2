from pathlib import Path
import argparse, sys
ROOT=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.baseline_training import diagnose_gradient_stability, run, verify_workers2_one_epoch, write_failure, write_workers2_preflight_failure
if __name__ == '__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--project-root',type=Path,default=ROOT); parser.add_argument('--mode',choices=('train','diagnose','verify-workers2'),default='train'); parser.add_argument('--steps',type=int,default=250); args=parser.parse_args()
    try: print(diagnose_gradient_stability(args.project_root,args.steps) if args.mode == 'diagnose' else verify_workers2_one_epoch(args.project_root) if args.mode == 'verify-workers2' else run(args.project_root))
    except Exception as error:
        (write_workers2_preflight_failure if args.mode == 'verify-workers2' else write_failure)(args.project_root,error)
        raise
