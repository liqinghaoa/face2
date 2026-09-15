from __future__ import annotations

import numpy as np
import pandas as pd

from src.skin_optics_hsi.s1_formal_test import evaluate_formal_test_gates


GATES = {
    "minimum_primary_region_fraction": 0.75,
    "minimum_primary_regions_per_subject": 1,
    "improved_subject_fraction_vs_B0S_min": 2.0 / 3.0,
    "median_shape_log_rmse_ratio_vs_B0S_max": 0.90,
    "max_absolute_median_band_shape_residual": 0.15,
    "maximum_selected_model_fit_failure_fraction": 0.0,
    "require_finite_parameters": True,
    "require_full_design_rank": True,
}


def _inputs(usable_count: int = 8, improved_subjects: int = 4):
    subjects = ["p1", "p2", "p3", "p4"]
    spectrum_rows = []
    fit_rows = []
    residual_rows = []
    count = 0
    for subject_index, subject in enumerate(subjects):
        for side in ("left_cheek", "right_cheek"):
            usable = count < usable_count
            spectrum_rows.append({
                "sample_id": f"{subject}_neutral_front",
                "subject_id": subject,
                "region": side,
                "extraction_status": "USABLE" if usable else "UNAVAILABLE",
            })
            if usable:
                selected_error = 0.04 if subject_index < improved_subjects else 0.11
                for model, error, dimension in (("D2-MH", selected_error, 2), ("B0-S", 0.10, 0)):
                    fit_rows.append({
                        "sample_id": f"{subject}_neutral_front",
                        "subject_id": subject,
                        "region": side,
                        "model_id": model,
                        "shape_log_rmse": error,
                        "raw_log_rmse": error,
                        "raw_sam": error,
                        "parameters_finite": True,
                        "design_rank": dimension,
                        "design_dimension": dimension,
                    })
                residual_rows.extend({
                    "subject_id": subject,
                    "region": side,
                    "model_id": "D2-MH",
                    "wavelength_nm": float(wavelength),
                    "shape_residual": 0.02,
                } for wavelength in np.arange(400, 701, 10))
            count += 1
    return pd.DataFrame(spectrum_rows), pd.DataFrame(fit_rows), pd.DataFrame(residual_rows)


def test_formal_test_gate_passes_complete_reproducible_representation() -> None:
    spectra, fits, residuals = _inputs()
    summary, subjects = evaluate_formal_test_gates(
        spectra, fits, residuals, pd.DataFrame(), GATES,
        expected_subjects=4, expected_primary_regions=8,
    )
    assert summary["passed"] is True
    assert summary["usable_primary_regions"] == 8
    assert len(subjects) == 8


def test_formal_test_gate_allows_six_of_eight_but_requires_every_subject() -> None:
    spectra, fits, residuals = _inputs(usable_count=6)
    summary, _ = evaluate_formal_test_gates(
        spectra, fits, residuals, pd.DataFrame(), GATES,
        expected_subjects=4, expected_primary_regions=8,
    )
    assert summary["checks"]["primary_region_fraction"] is True
    assert summary["checks"]["minimum_one_primary_region_per_subject"] is False
    assert summary["passed"] is False


def test_formal_test_gate_rejects_weak_subject_level_improvement() -> None:
    spectra, fits, residuals = _inputs(improved_subjects=2)
    summary, _ = evaluate_formal_test_gates(
        spectra, fits, residuals, pd.DataFrame(), GATES,
        expected_subjects=4, expected_primary_regions=8,
    )
    assert summary["checks"]["improved_subject_fraction_vs_B0S"] is False
    assert summary["passed"] is False
