from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from p2_counterfactual.a2_v2_assets import (
    DATA_ROOT,
    MANIFEST_PATH as A2_MANIFEST_PATH,
    OUTPUT_DIR as A2_ASSET_DIR,
    build_a2_v2_assets,
)
from p2_counterfactual.assets import PRESET_NAMES, ROOT
from p2_counterfactual.io_utils import json_safe, sha256_file
from p2_counterfactual.path_utils import resolve_project_path


DISPLAY_EXPERIMENT_ID = "P2-A3-v2_R3DPR_Full6Consistency_MeanBG"
EXPERIMENT_ID = "p2_a3_v2_r3dpr_full6_consistency_meanbg"
ASSET_DIR = DATA_ROOT / "p2_a3_v2_assets"
MANIFEST_PATH = ASSET_DIR / "p2_multiview_manifest.csv"
AUDIT_PATH = ASSET_DIR / "dataset_audit.json"
PRESET_MAPPING_PATH = ASSET_DIR / "preset_mapping.json"
CODE_AUDIT_PATH = ASSET_DIR / "code_audit.md"
TEST_RESULTS_PATH = ASSET_DIR / "test_results.txt"


def _image_audit(path: Path) -> dict[str, Any]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        extrema = rgb.getextrema()
        flat = [value for pair in extrema for value in pair]
        return {
            "path": str(path),
            "mode": rgb.mode,
            "size": list(rgb.size),
            "non_empty": path.stat().st_size > 0,
            "not_constant": len(set(flat)) > 1,
        }


def build_a3_v2_assets() -> dict[str, Any]:
    if not A2_MANIFEST_PATH.is_file():
        build_a2_v2_assets()
    ASSET_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(A2_MANIFEST_PATH, dtype={"case_id": str, "patient_group_id": str})
    frame.to_csv(MANIFEST_PATH, index=False, encoding="utf-8-sig")
    invalid_rows = []
    for row in frame.itertuples(index=False):
        case_id = str(row.case_id)
        paths = [resolve_project_path(getattr(row, "original_path"), ROOT, require_exists=True)]
        paths.extend(resolve_project_path(getattr(row, f"relight_{preset}_path"), ROOT, require_exists=True) for preset in PRESET_NAMES)
        for path in paths:
            audit = _image_audit(path)
            if audit["mode"] != "RGB" or audit["size"] != [224, 224] or not audit["non_empty"] or not audit["not_constant"]:
                invalid_rows.append({"case_id": case_id, **audit})
    patient_fold_violations = int(frame.groupby("patient_group_id")["fold"].nunique().gt(1).sum())
    preset_mapping_source = A2_ASSET_DIR / "preset_mapping.json"
    if preset_mapping_source.is_file():
        shutil.copy2(preset_mapping_source, PRESET_MAPPING_PATH)
    else:
        PRESET_MAPPING_PATH.write_text(
            json.dumps(
                {
                    "preset_order": [
                        {"preset_index": index, "canonical_preset_id": preset, "filename": f"relight_{preset}.png"}
                        for index, preset in enumerate(PRESET_NAMES)
                    ]
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    code_audit_source = ROOT / "reports/p2_a3_v2_r3dpr_full6_consistency_code_audit.md"
    if code_audit_source.is_file():
        shutil.copy2(code_audit_source, CODE_AUDIT_PATH)
    audit = {
        "display_experiment_id": DISPLAY_EXPERIMENT_ID,
        "experiment_id": EXPERIMENT_ID,
        "manifest_path": str(MANIFEST_PATH),
        "manifest_sha256": sha256_file(MANIFEST_PATH),
        "source_manifest_path": str(A2_MANIFEST_PATH),
        "source_manifest_sha256": sha256_file(A2_MANIFEST_PATH),
        "case_count": int(len(frame)),
        "unique_case_count": int(frame["case_id"].nunique()),
        "view_count_per_case": 7,
        "total_image_count": int(len(frame) * 7),
        "fold_counts": {str(int(k)): int(v) for k, v in frame["fold"].astype(int).value_counts().sort_index().to_dict().items()},
        "label_counts": {str(int(k)): int(v) for k, v in frame["binary_label"].astype(int).value_counts().sort_index().to_dict().items()},
        "patient_group_fold_violations": patient_fold_violations,
        "invalid_image_rows": invalid_rows,
        "old_deca_npz_columns_used": False,
        "preset_order": list(PRESET_NAMES),
        "status": "passed" if len(frame) == 500 and frame["case_id"].nunique() == 500 and not invalid_rows and patient_fold_violations == 0 else "failed",
    }
    AUDIT_PATH.write_text(json.dumps(json_safe(audit), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if audit["status"] != "passed":
        raise RuntimeError(f"A3-v2 asset audit failed: {AUDIT_PATH}")
    return audit


def main() -> None:
    print(json.dumps(json_safe(build_a3_v2_assets()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
