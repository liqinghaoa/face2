"""Add freshly extracted EXIF capture times to the group split CSV."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = ROOT / "data/processed/global_face_R3DPR/nyha_2class_sex_stratified_group_5fold.csv"
EXIF_JSON = ROOT / ".codex_tmp/group_5fold_raw_scene_exif.json"
OUTPUT_CSV = ROOT / "data/processed/global_face_R3DPR/nyha_2class_sex_stratified_group_5fold_with_exif_time.csv"
TIME_COLUMN = "EXIF_DateTimeOriginal"


def main() -> None:
    payload = json.loads(EXIF_JSON.read_text(encoding="utf-8"))
    headers = payload["main_headers"]
    id_index = headers.index("ID")
    original_time_index = headers.index("原始拍摄时间")
    exif_times = {
        str(row[id_index]).strip(): row[original_time_index]
        for row in payload["main_rows"]
    }

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        source_headers = list(rows[0]) if rows else []
    if not rows or "ID" not in source_headers:
        raise ValueError("Input CSV must contain rows and an ID column.")

    missing = [row["ID"].strip() for row in rows if row["ID"].strip() not in exif_times]
    if missing:
        raise ValueError(f"EXIF time missing for {len(missing)} sample IDs: {missing[:10]}")

    output_headers = source_headers + ([TIME_COLUMN] if TIME_COLUMN not in source_headers else [])
    for row in rows:
        row[TIME_COLUMN] = exif_times[row["ID"].strip()]

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_headers, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUTPUT_CSV}")
    print(f"Rows: {len(rows)}; EXIF times written: {len(rows)}")


if __name__ == "__main__":
    main()
