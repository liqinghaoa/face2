"""Score and rank OOF Grad-CAM examples using the SH093 face-valid masks."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.evaluate.R3DPR.generate_r3dpr_binary_gradcam import (
    GradCAM,
    load_model,
    make_dataset_row,
    target_from_row,
)
from scripts.evaluate.R3DPR.summarize_r3dpr_resnet18_binary_5fold import (
    expected_table,
    validate_oof,
)
from utils.experiment_utils import load_yaml


CLASS_NAMES = {0: "Control", 1: "Patient"}
DEFAULT_OUTPUT_SUBDIR = "figures/gradcam_oof_screening"
CAM_REPLAY_ATOL = 1e-3
TOP_PIXEL_QUANTILE = 0.90
FACE_BORDER_WIDTH_PIXELS = 8
CONTACT_SHEET_CANDIDATES = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--output-subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Render contact sheets and report from existing scoring CSV files without recomputing CAMs.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--no-save-raw-cams",
        action="store_true",
        help="Do not save fold-level compressed raw CAM arrays.",
    )
    return parser.parse_args()


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def output_path(experiment_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else experiment_dir / path


def configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 150,
        }
    )


def load_face_mask(mask_root: Path, sample_id: str, expected_shape: tuple[int, int]) -> np.ndarray:
    path = mask_root / f"{sample_id}.png"
    if not path.is_file():
        raise FileNotFoundError(f"Missing face_valid_mask for ID {sample_id}: {path}")
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"), dtype=np.uint8) > 0
    if tuple(mask.shape) != expected_shape:
        raise ValueError(
            f"face_valid_mask shape mismatch for ID {sample_id}: {mask.shape} != {expected_shape}"
        )
    if not mask.any():
        raise ValueError(f"face_valid_mask is empty for ID {sample_id}")
    return mask


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    kernel_size = 2 * int(radius) + 1
    image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    return np.asarray(image.filter(ImageFilter.MinFilter(kernel_size)), dtype=np.uint8) > 0


def attention_metrics(cam: np.ndarray, face_mask: np.ndarray) -> dict[str, float | bool]:
    if cam.shape != face_mask.shape:
        raise ValueError(f"CAM shape {cam.shape} does not match face mask shape {face_mask.shape}")
    total_energy = float(cam.sum())
    if not np.isfinite(total_energy) or total_energy <= 0.0:
        raise ValueError("Grad-CAM contains no positive energy")
    core_mask = erode_mask(face_mask, FACE_BORDER_WIDTH_PIXELS)
    border_mask = face_mask & ~core_mask
    face_energy = float(cam[face_mask].sum())
    core_energy = float(cam[core_mask].sum())
    border_energy = float(cam[border_mask].sum())
    threshold = float(np.quantile(cam, TOP_PIXEL_QUANTILE))
    top_mask = cam >= threshold
    peak_index = np.unravel_index(int(np.argmax(cam)), cam.shape)
    return {
        "face_area_fraction": float(face_mask.mean()),
        "face_cam_fraction": face_energy / total_energy,
        "background_cam_fraction": 1.0 - face_energy / total_energy,
        "face_core_cam_fraction": core_energy / total_energy,
        "face_border_cam_fraction": border_energy / total_energy,
        "top10_pixel_face_fraction": float(face_mask[top_mask].mean()),
        "peak_in_face": bool(face_mask[peak_index]),
        "peak_y": int(peak_index[0]),
        "peak_x": int(peak_index[1]),
        "cam_total_energy": total_energy,
    }


def replay_probability(logits: np.ndarray) -> np.ndarray:
    return torch.softmax(torch.from_numpy(logits), dim=0).numpy()


def attention_quality_score(metrics: dict[str, float | bool]) -> float:
    return float(
        0.50 * float(metrics["face_cam_fraction"])
        + 0.20 * float(metrics["face_core_cam_fraction"])
        + 0.20 * float(metrics["top10_pixel_face_fraction"])
        + 0.10 * float(bool(metrics["peak_in_face"]))
    )


def build_scores(
    args: argparse.Namespace, experiment_dir: Path, output_dir: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    config = load_yaml(experiment_dir / "config_snapshot.yaml")
    data = config["data"]
    image_root = project_path(data["image_root"])
    mask_root = image_root.parent / "face_valid_mask"
    manifest_path = experiment_dir / "figures" / "gradcam_oof" / "gradcam_manifest.csv"
    oof_path = experiment_dir / "oof_predictions.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source Grad-CAM manifest: {manifest_path}")
    if not mask_root.is_dir():
        raise FileNotFoundError(f"Missing face_valid_mask directory: {mask_root}")
    oof = pd.read_csv(oof_path, dtype={"sample_id": "string", "patient_group_id": "string"})
    n_folds = int(data["n_folds"])
    validate_oof(oof, expected_table(config), n_folds)
    manifest = pd.read_csv(manifest_path, dtype={"sample_id": "string"})
    required_manifest = {
        "sample_id",
        "fold",
        "true_class",
        "predicted_class",
        "prob_control",
        "prob_patient",
        "output_png",
    }
    missing = sorted(required_manifest.difference(manifest.columns))
    if missing:
        raise ValueError(f"Grad-CAM manifest lacks required columns: {missing}")
    if manifest["sample_id"].duplicated().any() or len(manifest) != len(oof):
        raise ValueError("Grad-CAM manifest must contain exactly one row per OOF sample")
    oof_join = oof.set_index("sample_id")[
        ["patient_group_id", "fold", "binary_label", "pred_class", "prob_control", "prob_patient"]
    ].rename(
        columns={
            "patient_group_id": "oof_patient_group_id",
            "fold": "oof_fold",
            "binary_label": "oof_binary_label",
            "pred_class": "oof_pred_class",
            "prob_control": "oof_prob_control",
            "prob_patient": "oof_prob_patient",
        }
    )
    joined = manifest.set_index("sample_id").join(oof_join, how="inner")
    if len(joined) != len(oof):
        raise ValueError("Grad-CAM manifest IDs do not exactly match OOF IDs")
    for column in ("fold", "binary_label", "pred_class"):
        manifest_column = "true_class" if column == "binary_label" else ("predicted_class" if column == "pred_class" else column)
        oof_column = {"fold": "oof_fold", "binary_label": "oof_binary_label", "pred_class": "oof_pred_class"}[column]
        if not np.array_equal(joined[manifest_column].astype(int), joined[oof_column].astype(int)):
            raise ValueError(f"Grad-CAM manifest {manifest_column} does not match OOF {column}")

    available_mask_ids = {path.stem for path in mask_root.glob("*.png")}
    missing_mask_ids = sorted(set(joined.index.astype(str)).difference(available_mask_ids))
    excluded = (
        joined.loc[missing_mask_ids]
        .reset_index()
        .loc[:, ["sample_id", "oof_patient_group_id", "oof_fold", "oof_binary_label", "oof_pred_class", "output_png"]]
        .rename(
            columns={
                "oof_patient_group_id": "patient_group_id",
                "oof_fold": "fold",
                "oof_binary_label": "true_class",
                "oof_pred_class": "predicted_class",
                "output_png": "source_gradcam_png",
            }
        )
    )
    excluded["exclusion_reason"] = "matching face_valid_mask PNG is unavailable for this OOF sample ID"
    joined = joined.drop(index=missing_mask_ids)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    raw_cam_dir = output_dir / "raw_cam_maps"
    if not args.no_save_raw_cams:
        raw_cam_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    expected_shape = (int(data["image_height"]), int(data["image_width"]))

    for fold in range(n_folds):
        fold_rows = joined.loc[joined["oof_fold"].astype(int) == fold].reset_index().sort_values("sample_id")
        checkpoint_path = experiment_dir / f"fold_{fold}" / "checkpoints" / "best_macro_auc.pth"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for fold {fold}: {checkpoint_path}")
        model = load_model(config, checkpoint_path, device)
        gradcam = GradCAM(model)
        fold_cams: list[np.ndarray] = []
        fold_ids: list[str] = []
        try:
            for row in fold_rows.itertuples(index=False):
                series = pd.Series(row._asdict())
                series["fold"] = int(series["oof_fold"])
                series["patient_group_id"] = str(series["oof_patient_group_id"])
                series["binary_label"] = int(series["oof_binary_label"])
                series["pred_class"] = int(series["oof_pred_class"])
                sample = make_dataset_row(series, config)
                image = sample["image"].unsqueeze(0).to(device)
                target_class = target_from_row(series, "predicted")
                logits, cam = gradcam.calculate(image, target_class)
                probability = replay_probability(logits)
                stored_probability = series[["oof_prob_control", "oof_prob_patient"]].to_numpy(dtype=float)
                predicted_class = int(probability.argmax())
                replay_difference = float(np.max(np.abs(probability - stored_probability)))
                if predicted_class != int(series["oof_pred_class"]) or replay_difference > CAM_REPLAY_ATOL:
                    raise RuntimeError(
                        f"OOF replay mismatch for ID {series['sample_id']} in fold {fold}: "
                        f"max_abs_difference={replay_difference:.6g}"
                    )
                face_mask = load_face_mask(mask_root, str(series["sample_id"]), expected_shape)
                metrics = attention_metrics(cam, face_mask)
                attention_score = attention_quality_score(metrics)
                prediction_margin = float(abs(probability[1] - 0.5) * 2.0)
                combined_score = float(0.85 * attention_score + 0.15 * prediction_margin)
                attention_pass = bool(
                    metrics["face_cam_fraction"] >= 0.80
                    and metrics["top10_pixel_face_fraction"] >= 0.80
                    and metrics["peak_in_face"]
                )
                records.append(
                    {
                        "sample_id": str(series["sample_id"]),
                        "patient_group_id": str(series["patient_group_id"]),
                        "fold": fold,
                        "true_class": int(series["oof_binary_label"]),
                        "true_class_name": CLASS_NAMES[int(series["oof_binary_label"])],
                        "predicted_class": predicted_class,
                        "predicted_class_name": CLASS_NAMES[predicted_class],
                        "is_correct": bool(predicted_class == int(series["oof_binary_label"])),
                        "prob_control": float(probability[0]),
                        "prob_patient": float(probability[1]),
                        "prediction_margin": prediction_margin,
                        "replay_max_abs_probability_difference": replay_difference,
                        "gradcam_target_class": target_class,
                        "gradcam_target_name": CLASS_NAMES[target_class],
                        "source_gradcam_png": str(series["output_png"]),
                        "input_image_path": str(sample["image_path"]),
                        "face_valid_mask_path": str(mask_root / f"{series['sample_id']}.png"),
                        "attention_quality_score": attention_score,
                        "combined_candidate_score": combined_score,
                        "attention_pass": attention_pass,
                        **metrics,
                    }
                )
                if not args.no_save_raw_cams:
                    fold_ids.append(str(series["sample_id"]))
                    fold_cams.append(cam.astype(np.float32, copy=False))
        finally:
            gradcam.close()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not args.no_save_raw_cams:
            np.savez_compressed(
                raw_cam_dir / f"fold_{fold}_raw_cams.npz",
                sample_id=np.asarray(fold_ids, dtype=str),
                cam=np.stack(fold_cams, axis=0),
            )
    return (
        pd.DataFrame(records).sort_values(["fold", "sample_id"]).reset_index(drop=True),
        excluded.sort_values(["fold", "sample_id"]).reset_index(drop=True),
    )


def rank_candidates(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    correct = scores.loc[scores["is_correct"]].copy()
    correct = correct.sort_values(
        ["fold", "true_class", "attention_pass", "combined_candidate_score", "sample_id"],
        ascending=[True, True, False, False, True],
    )
    correct["rank_within_fold_true_class"] = correct.groupby(["fold", "true_class"]).cumcount() + 1
    correct["selection_tier"] = np.where(
        correct["attention_pass"], "attention_pass", "fallback_attention_review")
    per_fold_top = correct.loc[correct["rank_within_fold_true_class"] == 1].copy()
    per_fold_top["rank_among_fold_winners"] = per_fold_top.groupby("true_class")["combined_candidate_score"].rank(
        method="first", ascending=False
    ).astype(int)
    recommended = per_fold_top.loc[per_fold_top["rank_among_fold_winners"] <= 3].copy()
    recommended["recommendation"] = "Top attention-ranked correct example from a distinct fold"
    return correct, recommended.sort_values(["true_class", "rank_among_fold_winners"])


def _plot_sheet(rows: pd.DataFrame, title: str, output_path: Path) -> None:
    columns = 3
    count = min(len(rows), CONTACT_SHEET_CANDIDATES)
    rows_needed = max(1, int(np.ceil(count / columns)))
    figure, axes = plt.subplots(rows_needed, columns, figsize=(10.5, 3.25 * rows_needed), constrained_layout=True)
    axes_array = np.asarray(axes).reshape(-1)
    figure.suptitle(title, fontsize=10, fontweight="bold")
    for axis, row in zip(axes_array, rows.head(count).itertuples(index=False)):
        axis.imshow(plt.imread(row.source_gradcam_png))
        axis.set_title(
            f"ID {row.sample_id} | score {row.combined_candidate_score:.3f}\n"
            f"face CAM {row.face_cam_fraction:.1%} | top10 face {row.top10_pixel_face_fraction:.1%} | P(patient) {row.prob_patient:.3f}",
            fontsize=6.7,
        )
        axis.set_axis_off()
    for axis in axes_array[count:]:
        axis.set_axis_off()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def write_contact_sheets(ranking: pd.DataFrame, recommended: pd.DataFrame, output_dir: Path) -> None:
    sheet_dir = output_dir / "contact_sheets"
    for fold in range(5):
        for true_class in (0, 1):
            rows = ranking.loc[(ranking["fold"] == fold) & (ranking["true_class"] == true_class)]
            _plot_sheet(
                rows,
                f"Fold {fold} | Correct {CLASS_NAMES[true_class]} candidates | ranked by face-constrained Grad-CAM score",
                sheet_dir / f"fold_{fold}_{CLASS_NAMES[true_class].lower()}_candidates.png",
            )
    _plot_sheet(
        recommended,
        "Recommended main-figure candidates | correct predictions from distinct folds",
        output_dir / "recommended_main_figure_candidates.png",
    )


def write_report(
    output_dir: Path,
    scores: pd.DataFrame,
    excluded: pd.DataFrame,
    ranking: pd.DataFrame,
    recommended: pd.DataFrame,
) -> None:
    correct_count = int(scores["is_correct"].sum())
    attention_pass_count = int((scores["is_correct"] & scores["attention_pass"]).sum())
    lines = [
        "# OOF Grad-CAM Candidate Screening",
        "",
        "## Scope",
        "",
        f"- Scored samples: {len(scores)} OOF predictions.",
        f"- Excluded from mask-constrained scoring: {len(excluded)}; see `excluded_samples_missing_face_mask.csv`.",
        f"- Correctly classified samples: {correct_count}.",
        f"- Correct samples passing face-attention criteria: {attention_pass_count}.",
        "- Grad-CAM was recomputed from the best macro-AUC checkpoint of the sample's own held-out fold.",
        "- Raw CAM arrays are retained per fold unless the script was run with `--no-save-raw-cams`.",
        "",
        "## Attention Metrics",
        "",
        "- `face_cam_fraction`: fraction of non-negative CAM energy inside the binary face-valid mask.",
        "- `background_cam_fraction`: complementary CAM energy outside the face-valid mask.",
        f"- `face_core_cam_fraction`: CAM energy in the face mask after an {FACE_BORDER_WIDTH_PIXELS}-pixel erosion.",
        "- `top10_pixel_face_fraction`: fraction of pixels in the highest 10% of CAM values that lie inside the face-valid mask.",
        "- `peak_in_face`: whether the maximum CAM response lies inside the face-valid mask.",
        "- `combined_candidate_score`: 85% face-constrained CAM quality plus 15% prediction margin from the 0.5 decision boundary.",
        "",
        "## Selection Rules",
        "",
        "- Only correctly classified samples are candidates for the main explanatory figure.",
        "- Candidates are ranked separately within each fold and true class; face-attention passes rank ahead of fallbacks.",
        "- The recommended file contains the top three fold winners per true class, ensuring fold diversity.",
        "- Final paper panels require manual review to exclude visually unrepresentative cases or residual preprocessing artefacts.",
        "- Grad-CAM identifies model-used regions for a prediction and must not be interpreted as a causal cardiac biomarker map.",
    ]
    (output_dir / "screening_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def screen(args: argparse.Namespace) -> Path:
    experiment_dir = project_path(args.experiment_dir)
    if not (experiment_dir / "config_snapshot.yaml").is_file():
        raise FileNotFoundError(f"Missing experiment config snapshot: {experiment_dir}")
    output_dir = output_path(experiment_dir, args.output_subdir)
    if args.render_only:
        required = {
            "scores": output_dir / "gradcam_attention_scores.csv",
            "excluded": output_dir / "excluded_samples_missing_face_mask.csv",
            "ranking": output_dir / "candidate_ranking_by_fold_and_class.csv",
            "recommended": output_dir / "recommended_main_figure_candidates.csv",
        }
        missing = [name for name, path in required.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"render-only requires existing screening files: {missing}")
        configure_matplotlib()
        scores = pd.read_csv(required["scores"])
        excluded = pd.read_csv(required["excluded"])
        ranking = pd.read_csv(required["ranking"])
        recommended = pd.read_csv(required["recommended"])
        write_contact_sheets(ranking, recommended, output_dir)
        write_report(output_dir, scores, excluded, ranking, recommended)
        return output_dir
    if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(f"Output directory is not empty: {output_dir}; use --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    scores, excluded = build_scores(args, experiment_dir, output_dir)
    ranking, recommended = rank_candidates(scores)
    scores.to_csv(output_dir / "gradcam_attention_scores.csv", index=False, encoding="utf-8-sig")
    excluded.to_csv(output_dir / "excluded_samples_missing_face_mask.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(output_dir / "candidate_ranking_by_fold_and_class.csv", index=False, encoding="utf-8-sig")
    recommended.to_csv(output_dir / "recommended_main_figure_candidates.csv", index=False, encoding="utf-8-sig")
    write_contact_sheets(ranking, recommended, output_dir)
    write_report(output_dir, scores, excluded, ranking, recommended)
    return output_dir


def main() -> Path:
    args = parse_args()
    output_dir = screen(args)
    print(f"GRADCAM_SCREENING_OUTPUT_DIR={output_dir}")
    return output_dir


if __name__ == "__main__":
    main()
