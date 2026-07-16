"""Generate and audit the shared patient-group 5x5 nested CV protocol."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


CSV_ENCODING = "utf-8-sig"
REQUIRED_COLUMNS = {
    "ID",
    "patient_group_id",
    "SEX",
    "NYHA",
    "label_3class",
    "label_3class_name",
    "fold",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_split(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    frame = pd.read_csv(
        path,
        dtype={"ID": "string", "patient_group_id": "string"},
        encoding=CSV_ENCODING,
    )
    missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing:
        raise ValueError(f"split is missing columns {missing}: {path}")
    frame["ID"] = frame["ID"].astype("string").str.strip()
    frame["patient_group_id"] = frame["patient_group_id"].astype("string").str.strip()
    if frame["ID"].isna().any() or frame["ID"].duplicated().any():
        raise ValueError(f"split contains missing or duplicate IDs: {path}")
    return frame


def _assert_disjoint(
    left: pd.DataFrame, right: pd.DataFrame, column: str, context: str
) -> None:
    overlap = set(left[column]).intersection(right[column])
    if overlap:
        raise ValueError(f"{context}: {column} overlap count={len(overlap)}")


def _class_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame["label_3class"].astype(int).value_counts().sort_index()
    return {str(label): int(counts.get(label, 0)) for label in (0, 1, 2)}


def _write_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_protected_input_manifest(
    outer_split_dir: str | Path, protocol_dir: str | Path
) -> dict[str, Any]:
    """Record immutable-source hashes before training and verify on later runs."""
    outer_split_dir = Path(outer_split_dir).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    project_root = outer_split_dir.parents[2]
    label_path = project_root / "data/raw/label_raw.csv"
    baseline_config = (
        project_root
        / "config/train/preprocess_ablation_resnet18/nyha_3class_resnet18_preproc_hybrid_imagenet_meanbg.yaml"
    )
    image_dir = (
        project_root
        / "data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images"
    )
    image_digest = hashlib.sha256()
    image_paths = sorted(image_dir.glob("*.png"), key=lambda path: path.name)
    for path in image_paths:
        image_digest.update(path.name.encode("utf-8"))
        image_digest.update(sha256_file(path).encode("ascii"))
    current: dict[str, Any] = {
        "label_raw.csv": sha256_file(label_path),
        "baseline_config": sha256_file(baseline_config),
        "outer_split_files": {
            path.name: sha256_file(path) for path in sorted(outer_split_dir.glob("*")) if path.is_file()
        },
        "meanbg_images": {
            "file_count": len(image_paths),
            "aggregate_name_and_sha256": image_digest.hexdigest(),
        },
    }
    manifest_path = protocol_dir / "protected_input_hashes.json"
    if manifest_path.is_file():
        stored = json.loads(manifest_path.read_text(encoding="utf-8"))
        if stored != current:
            raise RuntimeError("protected input hash mismatch; formal training must not continue")
    else:
        _write_json(current, manifest_path)
    return current


def generate_shared_protocol(
    outer_split_dir: str | Path,
    protocol_dir: str | Path,
    *,
    base_seed: int = 2026,
    n_outer: int = 5,
    n_inner: int = 5,
) -> pd.DataFrame:
    """Create shared inner folds once, then audit and hash every file."""
    outer_split_dir = Path(outer_split_dir).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    protocol_dir.mkdir(parents=True, exist_ok=True)
    existing_hash_path = protocol_dir / "split_hashes.json"
    if existing_hash_path.is_file():
        audit = audit_shared_protocol(
            outer_split_dir, protocol_dir, n_outer=n_outer, n_inner=n_inner
        )
        stored = json.loads(existing_hash_path.read_text(encoding="utf-8"))
        for relative, expected in stored.items():
            path = protocol_dir / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RuntimeError(f"existing shared protocol hash mismatch: {path}")
        ensure_protected_input_manifest(outer_split_dir, protocol_dir)
        return audit

    audit_rows: list[dict[str, Any]] = []
    split_hashes: dict[str, str] = {}
    all_outer_test_ids: list[str] = []
    for outer_fold in range(n_outer):
        outer_train_path = outer_split_dir / f"fold_{outer_fold}_train.csv"
        outer_test_path = outer_split_dir / f"fold_{outer_fold}_val.csv"
        outer_train = read_split(outer_train_path)
        outer_test = read_split(outer_test_path)
        _assert_disjoint(outer_train, outer_test, "ID", f"outer fold {outer_fold}")
        _assert_disjoint(
            outer_train, outer_test, "patient_group_id", f"outer fold {outer_fold}"
        )
        all_outer_test_ids.extend(outer_test["ID"].astype(str).tolist())
        stratum = (
            outer_train["label_3class"].astype(int).astype(str)
            + "_"
            + outer_train["SEX"].astype(int).astype(str)
        )
        splitter = StratifiedGroupKFold(
            n_splits=n_inner,
            shuffle=True,
            random_state=base_seed + outer_fold,
        )
        outer_dir = protocol_dir / f"outer_fold_{outer_fold}"
        outer_dir.mkdir(parents=True, exist_ok=False)
        val_occurrences: list[str] = []
        per_outer: list[dict[str, Any]] = []
        for inner_fold, (train_indices, val_indices) in enumerate(
            splitter.split(
                outer_train.index.to_numpy(),
                y=stratum,
                groups=outer_train["patient_group_id"],
            )
        ):
            inner_train = outer_train.iloc[train_indices].copy()
            inner_val = outer_train.iloc[val_indices].copy()
            _assert_disjoint(
                inner_train, inner_val, "ID", f"outer {outer_fold} inner {inner_fold}"
            )
            _assert_disjoint(
                inner_train,
                inner_val,
                "patient_group_id",
                f"outer {outer_fold} inner {inner_fold}",
            )
            if set(inner_train["ID"]).union(inner_val["ID"]) != set(outer_train["ID"]):
                raise ValueError(f"inner partition does not equal outer train: {outer_fold}/{inner_fold}")
            if set(inner_train["ID"]).intersection(outer_test["ID"]) or set(
                inner_val["ID"]
            ).intersection(outer_test["ID"]):
                raise ValueError(f"outer test entered inner split: {outer_fold}/{inner_fold}")
            if set(inner_val["label_3class"].astype(int)) != {0, 1, 2}:
                raise ValueError(
                    f"inner validation missing class: outer={outer_fold}, inner={inner_fold}, "
                    f"counts={_class_counts(inner_val)}"
                )
            train_path = outer_dir / f"inner_fold_{inner_fold}_train.csv"
            val_path = outer_dir / f"inner_fold_{inner_fold}_val.csv"
            inner_train.to_csv(train_path, index=False, encoding=CSV_ENCODING)
            inner_val.to_csv(val_path, index=False, encoding=CSV_ENCODING)
            for path in (train_path, val_path):
                split_hashes[str(path.relative_to(protocol_dir)).replace("\\", "/")] = sha256_file(path)
            val_occurrences.extend(inner_val["ID"].astype(str).tolist())
            row = {
                "outer_fold": outer_fold,
                "inner_fold": inner_fold,
                "outer_train_n": len(outer_train),
                "outer_test_n": len(outer_test),
                "inner_train_n": len(inner_train),
                "inner_val_n": len(inner_val),
                "inner_train_groups": inner_train["patient_group_id"].nunique(),
                "inner_val_groups": inner_val["patient_group_id"].nunique(),
                "inner_train_class_counts": json.dumps(_class_counts(inner_train)),
                "inner_val_class_counts": json.dumps(_class_counts(inner_val)),
                "id_overlap": 0,
                "group_overlap": 0,
                "passed": True,
            }
            audit_rows.append(row)
            per_outer.append(row)
        occurrences = pd.Series(val_occurrences).value_counts()
        if set(val_occurrences) != set(outer_train["ID"]) or not occurrences.eq(1).all():
            raise ValueError(f"inner validation coverage invalid for outer fold {outer_fold}")
        _write_json(
            {
                "outer_fold": outer_fold,
                "random_state": base_seed + outer_fold,
                "outer_train_n": len(outer_train),
                "outer_test_n": len(outer_test),
                "outer_train_groups": outer_train["patient_group_id"].nunique(),
                "outer_test_groups": outer_test["patient_group_id"].nunique(),
                "outer_train_class_counts": _class_counts(outer_train),
                "outer_test_class_counts": _class_counts(outer_test),
                "inner_val_complete_coverage": True,
                "inner_val_each_id_once": True,
                "folds": per_outer,
            },
            outer_dir / "split_audit.json",
        )
    outer_counts = pd.Series(all_outer_test_ids).value_counts()
    if len(all_outer_test_ids) != 500 or len(outer_counts) != 500 or not outer_counts.eq(1).all():
        raise ValueError("five outer tests do not form exactly 500 unique IDs")
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(protocol_dir / "nested_split_audit.csv", index=False, encoding=CSV_ENCODING)
    table_columns = [
        "outer_fold",
        "inner_fold",
        "outer_train_n",
        "outer_test_n",
        "inner_train_n",
        "inner_val_n",
        "inner_train_groups",
        "inner_val_groups",
        "inner_train_class_counts",
        "inner_val_class_counts",
        "passed",
    ]
    table_lines = [
        "| " + " | ".join(table_columns) + " |",
        "| " + " | ".join(["---"] * len(table_columns)) + " |",
    ]
    for _, audit_row in audit[table_columns].iterrows():
        table_lines.append(
            "| " + " | ".join(str(audit_row[column]) for column in table_columns) + " |"
        )
    lines = [
        "# Shared nested 5x5 split audit",
        "",
        f"- Outer folds: {n_outer}",
        f"- Inner folds per outer fold: {n_inner}",
        "- Stratification: label_3class + '_' + SEX",
        "- Group: patient_group_id",
        f"- Base seed: {base_seed}",
        "- Five outer tests: 500 unique IDs, each exactly once",
        "- All ID/group leakage and inner coverage checks: passed",
        "",
        *table_lines,
    ]
    (protocol_dir / "nested_split_audit.md").write_text("\n".join(lines), encoding="utf-8")
    _write_json(split_hashes, existing_hash_path)
    protected = {
        str(path.relative_to(outer_split_dir)).replace("\\", "/"): sha256_file(path)
        for path in sorted(outer_split_dir.glob("*.csv"))
    }
    _write_json(protected, protocol_dir / "source_split_hashes.json")
    ensure_protected_input_manifest(outer_split_dir, protocol_dir)
    return audit_shared_protocol(
        outer_split_dir, protocol_dir, n_outer=n_outer, n_inner=n_inner
    )


def audit_shared_protocol(
    outer_split_dir: str | Path,
    protocol_dir: str | Path,
    *,
    n_outer: int = 5,
    n_inner: int = 5,
) -> pd.DataFrame:
    outer_split_dir = Path(outer_split_dir).resolve()
    protocol_dir = Path(protocol_dir).resolve()
    rows: list[dict[str, Any]] = []
    all_outer_tests: list[str] = []
    for outer_fold in range(n_outer):
        outer_train = read_split(outer_split_dir / f"fold_{outer_fold}_train.csv")
        outer_test = read_split(outer_split_dir / f"fold_{outer_fold}_val.csv")
        _assert_disjoint(outer_train, outer_test, "ID", f"outer {outer_fold}")
        _assert_disjoint(outer_train, outer_test, "patient_group_id", f"outer {outer_fold}")
        all_outer_tests.extend(outer_test["ID"].astype(str).tolist())
        val_ids: list[str] = []
        for inner_fold in range(n_inner):
            train = read_split(protocol_dir / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}_train.csv")
            val = read_split(protocol_dir / f"outer_fold_{outer_fold}" / f"inner_fold_{inner_fold}_val.csv")
            _assert_disjoint(train, val, "ID", f"outer {outer_fold} inner {inner_fold}")
            _assert_disjoint(train, val, "patient_group_id", f"outer {outer_fold} inner {inner_fold}")
            if set(train["ID"]).union(val["ID"]) != set(outer_train["ID"]):
                raise ValueError(f"inner union mismatch: outer {outer_fold} inner {inner_fold}")
            if set(train["ID"]).intersection(outer_test["ID"]) or set(val["ID"]).intersection(outer_test["ID"]):
                raise ValueError(f"outer test leakage: outer {outer_fold} inner {inner_fold}")
            if set(val["label_3class"].astype(int)) != {0, 1, 2}:
                raise ValueError(f"inner val missing class: outer {outer_fold} inner {inner_fold}")
            val_ids.extend(val["ID"].astype(str).tolist())
            rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": inner_fold,
                    "inner_train_n": len(train),
                    "inner_val_n": len(val),
                    "id_overlap": 0,
                    "group_overlap": 0,
                    "all_classes_in_val": True,
                    "passed": True,
                }
            )
        counts = pd.Series(val_ids).value_counts()
        if set(val_ids) != set(outer_train["ID"]) or not counts.eq(1).all():
            raise ValueError(f"inner val coverage mismatch: outer {outer_fold}")
    outer_counts = pd.Series(all_outer_tests).value_counts()
    if len(outer_counts) != 500 or not outer_counts.eq(1).all():
        raise ValueError("outer test coverage is not exactly 500 unique IDs")
    return pd.DataFrame(rows)
