from __future__ import annotations

from src.skin_optics_hsi.s1_validation_freeze import select_representation


def _candidate(dimension: int, error: float, eligible: bool = True) -> dict[str, object]:
    return {"dimension": dimension, "median_subject_shape_log_rmse": error, "eligible": eligible}


def test_selects_lower_dimension_within_frozen_tolerance() -> None:
    result = select_representation(
        {
            "D1-M": _candidate(1, 0.0325),
            "D2-MH": _candidate(2, 0.0310),
            "D3-MHG": _candidate(3, 0.0300),
            "B2-PCA": _candidate(2, 0.0305),
        },
        tolerance=0.10,
        preference=["D1-M", "D2-MH", "D3-MHG", "B2-PCA"],
    )
    assert result["selected_model"] == "D1-M"
    assert result["selected_dimension"] == 1


def test_same_dimension_prefers_registered_physics_informed_candidate() -> None:
    result = select_representation(
        {
            "D2-MH": _candidate(2, 0.0305),
            "B2-PCA": _candidate(2, 0.0300),
            "D3-MHG": _candidate(3, 0.0298),
        },
        tolerance=0.10,
        preference=["D1-M", "D2-MH", "D3-MHG", "B2-PCA"],
    )
    assert result["selected_model"] == "D2-MH"


def test_ineligible_candidates_are_never_selected() -> None:
    result = select_representation(
        {
            "D1-M": _candidate(1, 0.0100, eligible=False),
            "D2-MH": _candidate(2, 0.0300, eligible=True),
        },
        tolerance=0.10,
        preference=["D1-M", "D2-MH"],
    )
    assert result["selected_model"] == "D2-MH"


def test_no_eligible_candidate_returns_no_selection() -> None:
    result = select_representation(
        {"D1-M": _candidate(1, 0.0100, eligible=False)},
        tolerance=0.10,
        preference=["D1-M"],
    )
    assert result["selected_model"] is None
    assert result["eligible_models"] == []

