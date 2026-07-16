"""Build and audit the fixed-fold S2 425-case training cohort."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from losses.classification_losses import compute_class_weights  # noqa: E402


CLASS_NAMES = {0: "normal", 1: "mild", 2: "severe"}
NYHA_MAP = {0: 0, 1: 1, 2: 1, 3: 2, 4: 2}
EXPECTED = {"original": 522, "retained": 425, "excluded": 97, "normal": 112, "mild": 202, "severe": 111}


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def load_csv(path: Path, id_column: str) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype={id_column: "string", "patient_group_id": "string"}, encoding="utf-8-sig")


def add_check(checks: list[dict[str, Any]], name: str, passed: bool, detail: str) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "detail": detail})


def image_audit(paths: pd.Series) -> tuple[list[str], dict[str, int]]:
    failures: list[str] = []
    mode_counts: dict[str, int] = {}
    for patient_id, value in paths.items():
        path = Path(str(value))
        if not path.is_file():
            failures.append(f"{patient_id}: missing {path}")
            continue
        if path.stat().st_size <= 0:
            failures.append(f"{patient_id}: zero-byte {path}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                mode_counts[image.mode] = mode_counts.get(image.mode, 0) + 1
                converted = image.convert("RGB")
                if converted.size[0] <= 0 or converted.size[1] <= 0:
                    failures.append(f"{patient_id}: invalid dimensions {converted.size}")
        except Exception as exc:
            failures.append(f"{patient_id}: unreadable {path}: {exc}")
    return failures, mode_counts


def build(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    master = load_csv(args.source_master_split.resolve(), "ID")
    retained = load_csv(args.retained_ids.resolve(), "patient_id")
    excluded = load_csv(args.excluded_ids.resolve(), "patient_id")
    joint = load_csv(args.joint_error_table.resolve(), "patient_id")
    labels = load_csv(args.label_csv.resolve(), "ID")
    oofs = {
        "resnet18": load_csv(args.resnet18_oof.resolve(), "ID"),
        "resnet34": load_csv(args.resnet34_oof.resolve(), "ID"),
        "resnet50": load_csv(args.resnet50_oof.resolve(), "ID"),
    }

    for frame, column in ((master, "ID"), (retained, "patient_id"), (excluded, "patient_id"), (joint, "patient_id"), (labels, "ID")):
        frame[column] = frame[column].astype("string").str.strip()

    master_ids = set(master["ID"])
    retained_ids = set(retained["patient_id"])
    excluded_ids = set(excluded["patient_id"])
    add_check(checks, "master_522_unique_ids", len(master) == EXPECTED["original"] and master["ID"].nunique() == EXPECTED["original"], f"rows={len(master)}, unique={master['ID'].nunique()}")
    add_check(checks, "retained_425_unique_ids", len(retained) == EXPECTED["retained"] and retained["patient_id"].nunique() == EXPECTED["retained"], f"rows={len(retained)}, unique={retained['patient_id'].nunique()}")
    add_check(checks, "excluded_97_unique_ids", len(excluded) == EXPECTED["excluded"] and excluded["patient_id"].nunique() == EXPECTED["excluded"], f"rows={len(excluded)}, unique={excluded['patient_id'].nunique()}")
    add_check(checks, "retained_excluded_disjoint", retained_ids.isdisjoint(excluded_ids), f"intersection={len(retained_ids & excluded_ids)}")
    add_check(checks, "retained_excluded_close_master", retained_ids | excluded_ids == master_ids, f"union={len(retained_ids | excluded_ids)}, master={len(master_ids)}")

    required_master = {"ID", "patient_group_id", "SEX", "sex_name", "NYHA", "label_3class", "label_3class_name", "fold"}
    missing = sorted(required_master - set(master.columns))
    add_check(checks, "master_required_columns", not missing, f"missing={missing}")
    if missing:
        write_audit(output_dir / "s2_data_audit.md", checks, {}, [])
        raise RuntimeError(f"Master split missing required columns: {missing}")

    master["NYHA"] = pd.to_numeric(master["NYHA"], errors="coerce").astype("Int64")
    master["label_3class"] = pd.to_numeric(master["label_3class"], errors="coerce").astype("Int64")
    master["fold"] = pd.to_numeric(master["fold"], errors="coerce").astype("Int64")
    expected_labels = master["NYHA"].map(NYHA_MAP).astype("Int64")
    bad_mapping = master.loc[master["label_3class"] != expected_labels, "ID"].tolist()
    add_check(checks, "nyha_mapping", not bad_mapping, f"mismatch_ids={bad_mapping[:20]}")

    label_view = labels[["ID", "NYHA", "SEX"]].copy()
    label_view["NYHA"] = pd.to_numeric(label_view["NYHA"], errors="coerce").astype("Int64")
    label_view["SEX"] = pd.to_numeric(label_view["SEX"], errors="coerce").astype("Int64")
    cross = master[["ID", "NYHA", "SEX"]].merge(label_view, on="ID", how="left", suffixes=("_split", "_label"), validate="one_to_one")
    label_missing = cross.loc[cross["NYHA_label"].isna(), "ID"].tolist()
    label_conflicts = cross.loc[(cross["NYHA_split"] != cross["NYHA_label"]) | (cross["SEX_split"] != cross["SEX_label"]), "ID"].tolist()
    add_check(checks, "raw_label_coverage", not label_missing, f"missing_ids={label_missing[:20]}")
    add_check(checks, "raw_label_agreement", not label_conflicts, f"conflict_ids={label_conflicts[:20]}")

    fold_conflicts: dict[str, list[str]] = {}
    for model, oof in oofs.items():
        aligned = master[["ID", "fold", "label_3class"]].merge(
            oof[["ID", "fold", "label_3class"]], on="ID", how="outer", suffixes=("_split", "_oof"), indicator=True
        )
        bad = aligned.loc[
            (aligned["_merge"] != "both")
            | (aligned["fold_split"] != aligned["fold_oof"])
            | (aligned["label_3class_split"] != aligned["label_3class_oof"]),
            "ID",
        ].astype(str).tolist()
        fold_conflicts[model] = bad
        add_check(checks, f"{model}_fold_and_label_agreement", not bad, f"conflict_ids={bad[:20]}")

    retained_cross = master[["ID", "fold", "label_3class"]].merge(
        retained[["patient_id", "fold", "true_label"]], left_on="ID", right_on="patient_id", how="inner", suffixes=("_split", "_retained"), validate="one_to_one"
    )
    retained_conflicts = retained_cross.loc[
        (retained_cross["fold_split"] != retained_cross["fold_retained"])
        | (retained_cross["label_3class"] != retained_cross["true_label"]), "ID"
    ].tolist()
    add_check(checks, "retained_fold_and_label_agreement", len(retained_cross) == 425 and not retained_conflicts, f"matched={len(retained_cross)}, conflicts={retained_conflicts[:20]}")

    joint_index = joint.set_index("patient_id")
    excluded_rule_bad = [patient_id for patient_id in excluded_ids if patient_id not in joint_index.index or not bool(joint_index.loc[patient_id, "all_three_same_wrong"])]
    retained_rule_bad = [patient_id for patient_id in retained_ids if patient_id in joint_index.index and bool(joint_index.loc[patient_id, "all_three_same_wrong"])]
    add_check(checks, "excluded_matches_s2_rule", not excluded_rule_bad, f"bad_ids={excluded_rule_bad[:20]}")
    add_check(checks, "retained_contains_no_s2_exclusions", not retained_rule_bad, f"bad_ids={retained_rule_bad[:20]}")

    s2 = master[master["ID"].isin(retained_ids)].copy()
    s2 = s2.sort_values(["fold", "ID"], kind="stable")
    class_counts = {
        int(key): int(value)
        for key, value in s2["label_3class"].value_counts().sort_index().to_dict().items()
    }
    add_check(checks, "s2_class_counts", class_counts == {0: 112, 1: 202, 2: 111}, f"counts={class_counts}")
    add_check(checks, "s2_each_id_one_fold", s2["ID"].nunique() == len(s2) and s2.groupby("ID")["fold"].nunique().max() == 1, f"rows={len(s2)}, unique={s2['ID'].nunique()}")
    group_folds = s2.groupby("patient_group_id")["fold"].nunique()
    leaking_groups = group_folds[group_folds > 1].index.astype(str).tolist()
    add_check(checks, "group_no_cross_fold_leakage", not leaking_groups, f"leaking_groups={leaking_groups[:20]}")
    fold_class = s2.groupby(["fold", "label_3class"]).size().unstack(fill_value=0).reindex(columns=list(CLASS_NAMES), fill_value=0)
    add_check(checks, "every_fold_has_three_classes", bool((fold_class > 0).all().all()), f"counts={fold_class.to_dict()}")

    image_root = args.image_root.resolve()
    s2["image_path"] = s2["ID"].map(lambda value: str(image_root / f"{value}.png"))
    failures, modes = image_audit(s2.set_index("ID")["image_path"])
    add_check(checks, "all_images_readable_nonzero", not failures, f"failures={failures[:20]}, modes={modes}")
    duplicate_paths = s2.groupby("image_path").filter(lambda group: group["label_3class"].nunique() > 1)["ID"].tolist()
    add_check(checks, "image_path_label_compatibility", not duplicate_paths, f"bad_ids={duplicate_paths[:20]}")

    if any(item["status"] == "FAIL" for item in checks):
        write_audit(output_dir / "s2_data_audit.md", checks, {"class_counts": class_counts}, failures)
        raise RuntimeError("S2 audit failed; see s2_data_audit.md")

    manifest = pd.DataFrame(
        {
            "patient_id": s2["ID"].astype(str),
            "ID": s2["ID"].astype(str),
            "group_id": s2["patient_group_id"].astype(str),
            "patient_group_id": s2["patient_group_id"].astype(str),
            "sex": s2["SEX"].astype(int),
            "SEX": s2["SEX"].astype(int),
            "sex_name": s2["sex_name"].astype(str),
            "original_nyha": s2["NYHA"].astype(int),
            "NYHA": s2["NYHA"].astype(int),
            "class_label": s2["label_3class"].astype(int),
            "label_3class": s2["label_3class"].astype(int),
            "class_name": s2["label_3class_name"].astype(str),
            "label_3class_name": s2["label_3class_name"].astype(str),
            "fold": s2["fold"].astype(int),
            "image_path": s2["image_path"].astype(str),
            "retained_reason": "S2 retained: not all three models same-class wrong",
            "source_label_file": str(args.label_csv.resolve()),
            "source_split_file": str(args.source_master_split.resolve()),
        }
    )
    save_csv(manifest, output_dir / "s2_manifest.csv")

    excluded_manifest = master[master["ID"].isin(excluded_ids)].merge(
        joint, left_on="ID", right_on="patient_id", how="left", suffixes=("", "_audit"), validate="one_to_one"
    )
    excluded_manifest["excluded_reason"] = "S2 excluded: ResNet18/34/50 all wrong with identical wrong class"
    excluded_manifest["source_label_file"] = str(args.label_csv.resolve())
    excluded_manifest["source_split_file"] = str(args.source_master_split.resolve())
    save_csv(excluded_manifest, output_dir / "s2_excluded_manifest.csv")

    fold_details: dict[str, Any] = {}
    fold_sets: list[set[str]] = []
    for fold in range(5):
        val = manifest[manifest["fold"] == fold].copy()
        train = manifest[manifest["fold"] != fold].copy()
        fold_sets.append(set(val["ID"]))
        save_csv(val, output_dir / f"fold_{fold}.csv")
        save_csv(val, output_dir / f"fold_{fold}_val.csv")
        save_csv(train, output_dir / f"fold_{fold}_train.csv")
        train_counts = train["label_3class"].value_counts().reindex(CLASS_NAMES, fill_value=0).astype(int)
        val_counts = val["label_3class"].value_counts().reindex(CLASS_NAMES, fill_value=0).astype(int)
        weights = compute_class_weights(train["label_3class"].tolist(), 3).tolist()
        fold_details[str(fold)] = {
            "train_n": len(train),
            "validation_n": len(val),
            "train_class_counts": {CLASS_NAMES[i]: int(train_counts[i]) for i in CLASS_NAMES},
            "validation_class_counts": {CLASS_NAMES[i]: int(val_counts[i]) for i in CLASS_NAMES},
            "train_sex_counts": train["sex_name"].value_counts().to_dict(),
            "validation_sex_counts": val["sex_name"].value_counts().to_dict(),
            "weighted_ce_weights_from_train_only": {CLASS_NAMES[i]: float(weights[i]) for i in CLASS_NAMES},
        }
    pairwise_overlap = sum(len(fold_sets[i] & fold_sets[j]) for i in range(5) for j in range(i + 1, 5))
    fold_union = set().union(*fold_sets)
    add_check(checks, "folds_pairwise_disjoint", pairwise_overlap == 0, f"pairwise_overlap={pairwise_overlap}")
    add_check(checks, "fold_union_equals_s2", fold_union == retained_ids, f"union={len(fold_union)}, expected={len(retained_ids)}")

    write_audit(output_dir / "s2_data_audit.md", checks, {"class_counts": class_counts, "folds": fold_details, "image_modes": modes}, failures)
    data_manifest = {
        "name": "S2_425_fixed_original_folds",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "interpretation": "post-hoc S2 cohort; not independent or unbiased validation",
        "expected_counts": EXPECTED,
        "actual_counts": {"original": len(master), "retained": len(manifest), "excluded": len(excluded_manifest), **{CLASS_NAMES[k]: int(v) for k, v in class_counts.items()}},
        "authoritative_fold_source": str(args.source_master_split.resolve()),
        "source_label_file": str(args.label_csv.resolve()),
        "image_root": str(image_root),
        "source_files": {
            "retained_ids": {"path": str(args.retained_ids.resolve()), "sha256": file_hash(args.retained_ids.resolve())},
            "excluded_ids": {"path": str(args.excluded_ids.resolve()), "sha256": file_hash(args.excluded_ids.resolve())},
            "joint_error_table": {"path": str(args.joint_error_table.resolve()), "sha256": file_hash(args.joint_error_table.resolve())},
            "master_split": {"path": str(args.source_master_split.resolve()), "sha256": file_hash(args.source_master_split.resolve())},
            "label_csv": {"path": str(args.label_csv.resolve()), "sha256": file_hash(args.label_csv.resolve())},
        },
        "fold_details": fold_details,
        "checks": checks,
        "all_checks_pass": all(item["status"] == "PASS" for item in checks),
    }
    (output_dir / "s2_data_manifest.json").write_text(json.dumps(data_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return data_manifest


def write_audit(path: Path, checks: list[dict[str, Any]], details: dict[str, Any], image_failures: list[str]) -> None:
    all_pass = all(item["status"] == "PASS" for item in checks)
    lines = [
        "# S2 425-case data audit",
        "",
        "> 本队列由既有ResNet18/34/50 OOF共同错误事后筛选得到，仅用于内部方法比较和数据敏感性分析。",
        "",
        f"**Overall status: {'PASS' if all_pass else 'FAIL'}**",
        "",
        "## Checks",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    for item in checks:
        lines.append(f"| {item['name']} | {item['status']} | {str(item['detail']).replace('|', '/')} |")
    if details:
        lines += ["", "## Structured details", "", "```json", json.dumps(details, ensure_ascii=False, indent=2), "```"]
    if image_failures:
        lines += ["", "## Image failures", "", *[f"- {item}" for item in image_failures]]
    lines += [
        "", "## Fold interpretation", "",
        "Each fold CSV is the held-out evaluation fold. Training fold k uses the other four folds and uses fold k both for checkpoint selection and evaluation. This is internal five-fold OOF evaluation, not an independent test or external validation.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    report_dir = PROJECT_ROOT / "reports" / "posthoc_oracle_data_adjustment_522"
    experiment_root = PROJECT_ROOT / "experiments"
    parser = argparse.ArgumentParser()
    parser.add_argument("--retained-ids", type=Path, default=report_dir / "retained_ids_S2.csv")
    parser.add_argument("--excluded-ids", type=Path, default=report_dir / "excluded_ids_S2.csv")
    parser.add_argument("--joint-error-table", type=Path, default=report_dir / "joint_oof_error_table.csv")
    parser.add_argument("--source-master-split", type=Path, default=PROJECT_ROOT / "data" / "processed" / "splits" / "nyha_3class_sex_stratified_group_5fold.csv")
    parser.add_argument("--label-csv", type=Path, default=PROJECT_ROOT / "data" / "raw" / "label_raw.csv")
    parser.add_argument("--image-root", type=Path, default=PROJECT_ROOT / "data" / "processed" / "global_face" / "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict" / "images")
    parser.add_argument("--resnet18-oof", type=Path, default=experiment_root / "Global224_ImageNetResNet18_NYHA3Class_WeightedCE_5Fold" / "summary" / "oof_predictions.csv")
    parser.add_argument("--resnet34-oof", type=Path, default=experiment_root / "Global224_ImageNetResNet34_NYHA3Class_WeightedCE_5Fold" / "summary" / "oof_predictions.csv")
    parser.add_argument("--resnet50-oof", type=Path, default=experiment_root / "Global224_ImageNetResNet50_NYHA3Class_WeightedCE_5Fold" / "summary" / "oof_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "data" / "processed" / "s2_425")
    return parser.parse_args()


if __name__ == "__main__":
    result = build(parse_args())
    print(json.dumps({"all_checks_pass": result["all_checks_pass"], "actual_counts": result["actual_counts"], "fold_details": result["fold_details"]}, ensure_ascii=False, indent=2))
