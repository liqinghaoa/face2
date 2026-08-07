"""Anonymous P0-B review-table creation."""
from __future__ import annotations
from pathlib import Path
import csv
def create_blind_review(path: Path, audit_ids: list[str]) -> None:
    """Create a label-free manual review sheet containing only audit IDs."""
    fields=['audit_id','geometry_quality','albedo_shadow_leakage','albedo_highlight_leakage','skin_color_stability','shading_plausibility','reconstruction_quality','relight_direction_correctness','relight_identity_preservation','relight_artifacts','physics_core_usability','overall_usability','failure_codes','reviewer_notes']
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8-sig') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields); writer.writeheader(); writer.writerows({'audit_id':value} for value in audit_ids)
