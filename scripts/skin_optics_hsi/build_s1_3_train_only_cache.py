"""Materialize an auditable Train-only S1-3 cache from the mixed release cache."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
import pyarrow


def digest(path: Path) -> str:
    value = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    source, output = Path(args.source).resolve(), Path(args.output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite Train-only cache: {output}")
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.read_parquet(source, filters=[("split", "==", "train")])
    if set(frame.split.astype(str).unique()) != {"train"}:
        raise RuntimeError("Filtered cache contains a non-Train row")
    target = output / "region_spectra_train.parquet"
    frame.to_parquet(target, index=False, row_group_size=256)
    parquet = pq.ParquetFile(target)
    split_index = parquet.schema_arrow.names.index("split")
    row_groups = []
    for index in range(parquet.num_row_groups):
        metadata = parquet.metadata.row_group(index)
        stats = metadata.column(split_index).statistics
        if stats is None or stats.min != "train" or stats.max != "train":
            raise RuntimeError("Train-only output has an unverifiable split row group")
        row_groups.append({"index": index, "rows": metadata.num_rows, "split_min": stats.min, "split_max": stats.max})
    provenance = {
        "schema_version": 1, "stage": "S1-3-derived-cache", "status": "TRAIN_ONLY_CACHE_READY",
        "source_path": str(source), "source_sha256": digest(source),
        "filter": "split == train", "rows_materialized": len(frame),
        "subjects": int(frame.subject_id.nunique()), "split_values": ["train"],
        "output_path": str(target), "output_sha256": digest(target), "row_groups": row_groups,
        "validation_rows_materialized": 0, "test_rows_materialized": 0,
        "script_path": str(Path(__file__).resolve()), "script_sha256": digest(Path(__file__).resolve()),
        "pandas": pd.__version__, "pyarrow": pyarrow.__version__,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "train_only_cache_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": provenance["status"], "rows": len(frame), "output": str(target)}, indent=2))


if __name__ == "__main__":
    main()
