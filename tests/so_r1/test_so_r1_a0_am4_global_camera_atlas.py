from src.skin_optics_so_r1.global_camera_qualification import panel_a,panel_b

def test_am4_panel_a_contract():
 p=panel_a()
 assert len(p)==216
 assert {x['m_base'] for x in p}=={.90,.95,.98}
 assert {x['h_base'] for x in p}=={.02,.50,.95}
 assert {x['mask_category'] for x in p}=={'Full','Mild','Strong'}

def test_am4_panel_b_contract_and_determinism():
 p=panel_b()
 assert p==panel_b() and len(p)==256
 assert sum(x['m_base']<.5 for x in p)==64
 assert sum(.5<=x['m_base']<.85 for x in p)==64
 assert sum(x['m_base']>=.85 for x in p)==128
 assert [sum((i/8)<=x['h_base']<((i+1)/8) for x in p) for i in range(8)]==[32]*8
 assert {x['mask_category']:sum(y['mask_category']==x['mask_category'] for y in p) for x in p}=={'Full':102,'Mild':102,'Strong':52}
