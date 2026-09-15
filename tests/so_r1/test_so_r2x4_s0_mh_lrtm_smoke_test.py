import torch
from src.skin_optics_so_r1.mh_lrtm_fusion import FusionNet, LowRankMH, split_rgb_m_h


def test_strict_five_channel_split_and_mh_order():
    x=torch.zeros(2,5,320,256); x[:,3]=3; x[:,4]=4
    rgb,m,h=split_rgb_m_h(x)
    assert rgb.shape==(2,3,320,256) and torch.all(m==3) and torch.all(h==4)
    assert not torch.equal(m,h)


def test_lmf_rank_formula_shapes_and_no_full_outer_product():
    unit=LowRankMH(); m=torch.randn(2,128); h=torch.randn(2,128)
    assert unit.rank==4 and unit.U.shape==(4,129,128) and unit.V.shape==(4,129,128)
    assert unit(m,h).shape==(2,128)


def test_variants_have_locked_structure_and_independent_mh_encoders():
    late=FusionNet('LATE_CONCAT'); lrf=FusionNet('MH_LRF'); lrtm=FusionNet('MH_LRTM')
    assert late.mmtm1 is None and not hasattr(late,'lmf')
    assert lrf.mmtm1 is None and lrf.lmf.rank==4 and lrf.residual_scale==.1
    assert lrtm.mmtm1 is not None and lrtm.mmtm2 is not None
    assert all(a.data_ptr()!=b.data_ptr() for a,b in zip(lrf.m_encoder.parameters(),lrtm.h_encoder.parameters()))
    assert torch.count_nonzero(lrf.delta_head[-1].weight)>0 and torch.all(lrf.delta_head[-1].bias==0)
    assert not hasattr(late,'rgb_head') and not any('rgb_head' in k for k in late.state_dict())
    assert 'RGB classifier' not in late.gradient_groups()
    assert sum(p.numel() for p in late.parameters()) == 13_127_788 - 1_026


def test_shape_and_gradient_paths_on_synthetic_batch():
    model=FusionNet('MH_LRTM').train(); x=torch.randn(1,5,320,256); logits,audit=model(x,True); loss=logits.square().mean(); loss.backward()
    assert logits.shape==(1,2) and audit['rgb_stage4'].shape[1:]==(512,10,8)
    assert audit['m_stage4'].shape[1:]==(128,10,8) and audit['h_stage4'].shape[1:]==(128,10,8)
    assert len(audit['gates'])==6 and all(.9<=g.min() and g.max()<=1.1 and not torch.all(g==1) for g in audit['gates'])
    assert model.lmf.U.grad is not None and model.m_encoder.stem[0].weight.grad is not None and model.h_encoder.stem[0].weight.grad is not None
