"""Multilight audit placeholder with concrete summary helpers."""

from __future__ import annotations

from collections import defaultdict

from skin_optics.numpy_backend.colorchecker import CalibrationRecord


def loocv_by_light(records: list[CalibrationRecord]) -> dict[str, float]:
    """Return median LOOCV p95 by light."""

    values: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        values[rec.illuminant_name].append(rec.loocv_deltae00_p95)
    return {key: float(sorted(vals)[len(vals) // 2]) for key, vals in values.items()}
