"""Conservative automatic Grade A/B/C suggestion; human review remains required."""
from __future__ import annotations
from typing import Any

def suggest_grade(summary: dict[str, Any]) -> str:
    """Return a non-final Grade based on predeclared pilot feasibility summaries."""
    success=float(summary.get("success_count",0)); geometry=float(summary.get("geometry_usable",0)); albedo=float(summary.get("albedo_usable",0)); relight=float(summary.get("relight_success_fraction",0)); group_fail=bool(summary.get("any_group_all_failed",True)); direction=bool(summary.get("sh_direction_passed",False))
    if success>=11 and geometry>=10 and albedo>=9 and relight>=.85 and not group_fail and direction: return "A"
    if success>=9 and geometry>=8 and relight>=.70 and not group_fail and direction: return "B"
    return "C"
