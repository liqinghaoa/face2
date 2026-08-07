from p0b_deca.grading import suggest_grade
def test_grading_rules():
 assert suggest_grade({'success_count':11,'geometry_usable':10,'albedo_usable':9,'relight_success_fraction':.9,'any_group_all_failed':False,'sh_direction_passed':True})=='A'; assert suggest_grade({})=='C'
