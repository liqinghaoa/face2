"""Multicamera audit helpers."""

from __future__ import annotations

from collections import Counter

from skin_optics.numpy_backend.colorchecker import CalibrationRecord


def colorchecker_global_status(records: list[CalibrationRecord]) -> dict[str, object]:
    """Summarise ColorChecker qualification status."""

    total = len(records)
    qualified = [r for r in records if r.qualification_status == "PASS"]
    by_light = Counter(r.illuminant_name for r in qualified)
    canon = any(r.camera_name == "Canon 5DMarkII" and r.illuminant_name == "D65" and r.qualification_status == "PASS" for r in records)
    frac = len(qualified) / total if total else 0.0
    if frac >= 0.90 and all(by_light.get(light, 0) >= 20 for light in ["D65", "A", "FL2", "FL11"]) and canon:
        status = "PASS"
    elif frac >= 0.60 and all(by_light.get(light, 0) >= 12 for light in ["D65", "A", "FL2", "FL11"]) and canon:
        status = "PASS_WITH_LIMITS"
    else:
        status = "FAIL"
    return {"status": status, "qualified": len(qualified), "total": total, "qualified_fraction": frac, "qualified_by_light": dict(by_light)}
