"""Run the independent SO-R1 A2 P0 C2 audit benchmark."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.skin_optics_so_r1.mask_benchmark_closure import finalize_after_tests, run

if __name__ == '__main__':
    print(run(ROOT))
    report = ROOT / 'reports/so_r1_a2_p0_c2_mask_benchmark_closure'
    commands = [
        [sys.executable, '-m', 'pytest', '-q', 'tests/so_r1/test_so_r1_a2_p0_c2_mask_benchmark_closure.py'],
        [sys.executable, '-m', 'pytest', '-q', 'tests/so_r1'],
    ]
    results=[]
    for command in commands:
        proc=subprocess.run(command,cwd=ROOT,text=True,capture_output=True)
        results.append({'command':' '.join(command),'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr,'passed':proc.returncode==0})
    passed=all(x['passed'] for x in results)
    (report/'c2_test_results.txt').write_text('\n\n'.join(x['command']+'\n'+x['stdout']+x['stderr'] for x in results),encoding='utf-8')
    (report/'c2_test_results.json').write_text(json.dumps({'passed':passed,'results':results},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(finalize_after_tests(ROOT, {'passed':passed,'results':[{k:v for k,v in x.items() if k not in ('stdout','stderr')} for x in results]}),ensure_ascii=False,indent=2))
