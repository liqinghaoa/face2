from src.skin_optics_so_r1.formal_generation import _stable

def test_seed_is_stable_and_worker_independent():
    assert _stable('SO-R1-A2-D0','1.0',20260822,'Train',1,'m',0)==_stable('SO-R1-A2-D0','1.0',20260822,'Train',1,'m',0)
    assert _stable('SO-R1-A2-D0','1.0',20260822,'Train',1,'m',0)!=_stable('SO-R1-A2-D0','1.0',20260822,'Train',1,'m',1)
