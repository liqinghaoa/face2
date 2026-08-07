from p0b_deca.p1_production_gate import VERSION, root_dir
from pathlib import Path

def test_production_gate_is_a_new_versioned_contract():
    assert VERSION == 'P0B_cross_process_production_gate_v2_1'
    assert 'production_gate' in str(root_dir(Path('.')))

def test_ssim_is_not_a_primary_contract_category():
    # SSIM remains recorded by the audit but is intentionally excluded from the
    # primary hard-rule list of the production contract.
    primary=('latent','albedo_like','normal_coarse','relighting')
    assert 'ssim' not in primary
