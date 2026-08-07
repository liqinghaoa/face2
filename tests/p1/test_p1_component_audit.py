from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from p0b_deca.p1_component_audit import load_audit_config, sha256_file


class P1ComponentAuditContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[2]
        cls.cfg = load_audit_config(cls.root / "config/p1/p1_component_audit_v1.yaml")

    def test_p0_unified_source_is_authoritative(self):
        self.assertEqual(self.cfg.p0_root, self.root / "data/processed/P0_Physics_Audit_v1")
        self.assertEqual(self.cfg.p0_master_index, self.cfg.p0_root / "metadata/master_index.csv")

    def test_fixed_split_is_the_only_configured_source_and_hashable(self):
        expected = self.root / "data/processed/P0_Physics_Audit_v1/splits_500/nyha_3class_sex_stratified_group_5fold.csv"
        self.assertEqual(self.cfg.split_csv, expected)
        self.assertEqual(len(sha256_file(expected)), 64)

    def test_real_fixed_split_structure_and_integrity(self):
        split = pd.read_csv(self.cfg.split_csv, dtype={"ID": str, "patient_group_id": str})
        self.assertEqual(list(split.columns), ["ID", "patient_group_id", "SEX", "sex_name", "NYHA", "label_3class", "label_3class_name", "stratum", "fold", "binary_label", "binary_name"])
        self.assertEqual(len(split), 500)
        self.assertEqual(split["ID"].nunique(), 500)
        self.assertEqual(sorted(split["fold"].astype(int).unique().tolist()), [0, 1, 2, 3, 4])
        self.assertFalse((split.groupby("patient_group_id")["fold"].nunique() > 1).any())

    def test_label_mapping_preserves_three_class(self):
        split = pd.read_csv(self.cfg.split_csv)
        derived = split["NYHA"].map(lambda x: 0 if int(x) == 0 else 1)
        self.assertTrue((derived == split["binary_label"]).all())
        self.assertIn("label_3class", split.columns)

    def test_config_does_not_start_deca_or_training_or_resplit(self):
        lines = (self.root / "config/p1/p1_component_audit_v1.yaml").read_text(encoding="utf-8").lower().splitlines()
        self.assertFalse(any(line.startswith("deca_root:") for line in lines))
        self.assertFalse(any(line.startswith("training") or line.startswith("train_") for line in lines))
        self.assertFalse(any("regenerate_folds" in line for line in lines))

    def test_manifest_outputs_after_audit_if_present(self):
        manifest = self.cfg.output_root / "manifests/p1_master_manifest.csv"
        decision = self.cfg.output_root / "metadata/readiness_decision.json"
        if manifest.is_file():
            df = pd.read_csv(manifest, dtype={"case_id": str})
            self.assertEqual(len(df), 500)
            self.assertEqual(df["case_id"].nunique(), 500)
            self.assertFalse(df["has_specular_like"].any())
        if decision.is_file():
            data = decision.read_text(encoding="utf-8")
            self.assertIn('"deca_inference_started": false', data)
            self.assertIn('"classification_started": false', data)
            self.assertIn('"folds_regenerated": false', data)


if __name__ == "__main__":
    unittest.main()
