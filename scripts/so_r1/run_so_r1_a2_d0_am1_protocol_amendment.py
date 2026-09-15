from pathlib import Path
import argparse,json,sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from src.skin_optics_so_r1.full_generation_protocol_amendment import build
p=argparse.ArgumentParser();p.add_argument('--verify-existing',action='store_true');p.add_argument('--project-root',default=str(ROOT));a=p.parse_args()
print(json.dumps(build(Path(a.project_root),a.verify_existing),indent=2))
