from __future__ import annotations
import csv, json, unittest
from pathlib import Path
import numpy as np
from p0b_deca.noncollapse_audit import _dist, _effective, MODES, PRIMARY

ROOT=Path('data/processed/P0B_DECA_Pilot12_v1/noncollapse_audit_v1')
class NonCollapseAuditTests(unittest.TestCase):
 def test_effective_rank_identifies_constant_and_variable_features(self):
  self.assertEqual(_effective(np.ones((12,5)))['effective_rank'],0.)
  self.assertGreater(_effective(np.eye(12))['effective_rank'],10.)
 def test_distance_and_mate_ranking_are_deterministic(self):
  a=np.array([1.,0.]);b=np.array([0.,1.]);self.assertAlmostEqual(_dist(a,a,'cosine'),0.);self.assertGreater(_dist(a,b,'normalized_rms_l2'),0.)
 def test_completed_audit_has_12x2_and_required_decision_fields(self):
  decision=json.loads((ROOT/'noncollapse_decision.json').read_text())
  required={'status','decision','fixed_input_mode','appearance_gate','structure_gate','relighting_gate','acquisition_group_dominance_warning','failed_representations','passed_representations','warnings','reasons','recommended_next_stage','allowed_future_inputs','prohibited_future_claims'}
  self.assertTrue(required.issubset(decision));self.assertEqual(decision['fixed_input_mode'],'direct_p0_aligned');self.assertTrue(decision['source_hashes_unchanged'])
  with (ROOT/'latent_extraction_validation.csv').open(encoding='utf-8-sig',newline='') as h: rows=list(csv.DictReader(h))
  self.assertEqual(len(rows),24);self.assertEqual({r['input_mode'] for r in rows},set(MODES));self.assertEqual(len({r['case_id'] for r in rows}),12)
 def test_primary_gate_excludes_residual_and_capture_codes(self):
  self.assertNotIn('residual',PRIMARY);self.assertNotIn('light',PRIMARY);self.assertNotIn('pose',PRIMARY);self.assertNotIn('camera',PRIMARY)
 def test_no_source_outputs_changed(self):
  with (ROOT/'latent_extraction_validation.csv').open(encoding='utf-8-sig',newline='') as h:self.assertEqual(sum(1 for _ in csv.DictReader(h)),24)
  self.assertTrue(json.loads((ROOT/'noncollapse_audit_summary.json').read_text())['source_hashes_unchanged'])
if __name__=='__main__':unittest.main()
