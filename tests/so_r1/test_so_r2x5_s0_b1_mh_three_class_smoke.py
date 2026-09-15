from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[2]
SCRIPT=ROOT/'scripts/so_r1/run_so_r2x5_s0_b1_mh_three_class_smoke.py'
def test_x5_smoke_is_single_batch_and_cannot_train_or_touch_outer_test():
    tree=ast.parse(SCRIPT.read_text(encoding='utf-8'))
    calls=[ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n,ast.Call)]
    assert 'optimizer.step' not in calls and 'torch.save' not in calls
    source=SCRIPT.read_text(encoding='utf-8')
    assert "outer_test_reads':0" in source and "outer_test_forward_count':0" in source
