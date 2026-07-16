from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


# This is a read-only audit of project inputs. Every generated file is written
# beside this script, which is the report directory requested for this task.
OUTPUT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]

INTERMEDIATE_DIR = PROJECT_ROOT / (
    "data/processed/global_face/"
    "global_face_parsing_hybrid_foreheadrepair_blackbg_224_png_strict_intermediates"
)
ALIGNED_DIR = INTERMEDIATE_DIR / "aligned_rgb"
PARSING_DIR = INTERMEDIATE_DIR / "parsing_label"
FINAL_MASK_DIR = INTERMEDIATE_DIR / "final_mask"
MEANBG_DIR = PROJECT_ROOT / "data/processed/global_face/preprocess_ablation/hybrid_imagenet_meanbg/images"
ROI_OUTPUT_DIR = PROJECT_ROOT / (
    "data/processed/roi_dataset/"
    "global_aligned_face_parsing_roi_final5_224_canvas_500"
)
BBOX_LOG_PATH = ROI_OUTPUT_DIR / "logs/roi_metadata.csv"
EXIF_AUDIT_PATH = PROJECT_ROOT / "reports/exif_parameter_audit/image_parameter_audit.csv"

SELECTION_PATH = OUTPUT_DIR / "bbox_overlay_20_selection.csv"
REVIEW_PATH = OUTPUT_DIR / "bbox_overlay_20_review.csv"
REPORT_PATH = OUTPUT_DIR / "bbox_overlay_20_report.md"
BBOX_ONLY_DIR = OUTPUT_DIR / "bbox_only"
BBOX_SKIN_DIR = OUTPUT_DIR / "bbox_skin_overlay"

EXPECTED_DEVICES = ("HONOR/BVL-AN00", "Xiaomi/M2006J10C")
SKIN_LABEL = 1  # Defined in build_global_face_parsing_regularmask_blackbg_224_png_strict.py:112.
IMAGE_SIZE = (224, 224)
COLORS = {
    "forehead": (0, 200, 0),
    "cheek_image_left": (0, 102, 255),
    "cheek_image_right": (255, 0, 0),
}


# Visual review results, kept here so the final CSV is reproducible. Values are:
# coordinate_registration, forehead_content, cheek_left_content,
# cheek_right_content, issue_type, review_notes.
REVIEW_MAP: dict[str, tuple[str, str, str, str, str, str]] = {
    "100643124": ("PASS", "PASS", "PASS", "PASS", "NONE", "三组框均落在预期面部区域；未见整体偏移或缩放。"),
    "203874389": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部以暴露皮肤为主，双颊位置合理；坐标配准正常。"),
    "9300518956": ("PASS", "FAIL", "PASS", "PASS", "ROI_CONTENT_ONLY", "三组框的坐标关系正常，但额部框主要覆盖可见头发；皮肤交集叠加仍保留部分可见头发，属于额部内容/修复掩膜问题，不是坐标系错位。"),
    "A001083428": ("PASS", "PASS", "PASS", "PASS", "NONE", "三组框与额部、图像左颊和图像右颊位置一致。"),
    "A001808459": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部框包含少量发际边缘但以额部皮肤为主；双颊位置正常。"),
    "A001892094": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部和双颊框均与预期区域配准；未见系统性偏移。"),
    "A001938232": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部框下半部为清晰额部皮肤，双颊框位置合理；坐标一致。"),
    "A002038079": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部及双颊框落点正确，边界与224坐标系一致。"),
    "A002156070": ("PASS", "PASS", "PASS", "PASS", "NONE", "三组框均覆盖预期解剖区域；未见整体平移或缩放。"),
    "A002366013": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部和双颊位置合理；坐标配准正常。"),
    "100037382": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部以皮肤为主，双颊框位于鼻翼外侧；坐标一致。"),
    "203659997": ("PASS", "PASS", "PASS", "PASS", "NONE", "三组框均落在预期区域；未见系统性错位。"),
    "206664979": ("PASS", "FAIL", "PASS", "PASS", "ROI_CONTENT_ONLY", "三组框的坐标关系正常，但额部框主要覆盖可见头发；皮肤交集叠加仍保留可见头发，属于额部内容/修复掩膜问题，不是坐标系错位。"),
    "A000454833": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部与双颊框均在预期位置；背景与姿态未造成坐标偏移。"),
    "A001471135": ("PASS", "QUESTIONABLE", "PASS", "PASS", "ROI_CONTENT_ONLY", "坐标配准正常；额部框同时包含暴露额部和较多发丝，属于ROI内容边界可疑。"),
    "A001610296": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部和双颊框位置正确；未见整体偏移。"),
    "A001628261": ("PASS", "PASS", "PASS", "PASS", "NONE", "三组框与预期区域一致，额部有效区域清楚。"),
    "A001646267": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部及双颊框均正确配准；未见缩放或平移。"),
    "A001688703": ("PASS", "PASS", "PASS", "PASS", "NONE", "三组框落点符合规则；坐标一致。"),
    "A001794983": ("PASS", "PASS", "PASS", "PASS", "NONE", "额部和双颊框均在预期位置；坐标配准正常。"),
}
FINAL_COORDINATE_ALIGNMENT = "PASS"
FINAL_REUSE_DECISION = "SAFE_WITH_ROI_MASKING"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalized_id(value: object) -> str:
    return str(value).strip()


def read_current_ids() -> list[str]:
    ids = sorted(path.stem for path in MEANBG_DIR.glob("*.png"))
    require(len(ids) == 500, f"Meanbg ID count is {len(ids)}, expected 500")
    require(len(set(ids)) == 500, "Meanbg IDs are not unique")
    return ids


def load_exif(current_ids: list[str]) -> pd.DataFrame:
    # Deliberately load only non-clinical columns needed by the prompt.
    frame = pd.read_csv(
        EXIF_AUDIT_PATH,
        usecols=["ID", "Make", "Model"],
        dtype={"ID": str, "Make": str, "Model": str},
    )
    frame["ID"] = frame["ID"].map(normalized_id)
    frame = frame[frame["ID"].isin(current_ids)].copy()
    require(len(frame) == 500, f"EXIF matched rows: {len(frame)}, expected 500")
    require(frame["ID"].nunique() == 500, "EXIF ID match is not one-to-one")
    require(not frame[["Make", "Model"]].isna().any().any(), "Make/Model contains missing values")
    frame["Make"] = frame["Make"].str.strip()
    frame["Model"] = frame["Model"].str.strip()
    frame["camera_id"] = frame["Make"] + "/" + frame["Model"]
    unexpected = sorted(set(frame["camera_id"]) - set(EXPECTED_DEVICES))
    require(not unexpected, f"Unexpected device(s) among current 500 IDs: {unexpected}")
    for camera_id in EXPECTED_DEVICES:
        n = frame.loc[frame["camera_id"].eq(camera_id), "ID"].nunique()
        require(n >= 10, f"Device {camera_id} has only {n} matched IDs")
    return frame


def load_bbox_log(current_ids: list[str]) -> pd.DataFrame:
    # Do not load clinical columns that may coexist in this historical log.
    columns = [
        "ID",
        "roi_type",
        "roi_success",
        "bbox_x1",
        "bbox_y1",
        "bbox_x2",
        "bbox_y2",
        "left_cheek_bbox_x1",
        "left_cheek_bbox_y1",
        "left_cheek_bbox_x2",
        "left_cheek_bbox_y2",
        "right_cheek_bbox_x1",
        "right_cheek_bbox_y1",
        "right_cheek_bbox_x2",
        "right_cheek_bbox_y2",
    ]
    frame = pd.read_csv(BBOX_LOG_PATH, usecols=columns, dtype={"ID": str, "roi_type": str})
    frame["ID"] = frame["ID"].map(normalized_id)
    frame = frame[frame["ID"].isin(current_ids)].copy()
    require(frame["ID"].nunique() == 500, "BBox log does not match all 500 meanbg IDs")
    for roi_type in ("forehead_roi", "cheek_roi"):
        subset = frame[frame["roi_type"].eq(roi_type)]
        require(len(subset) == 500, f"{roi_type} row count is {len(subset)}, expected 500")
        require(subset["ID"].nunique() == 500, f"{roi_type} is not one row per ID")
        success = subset["roi_success"].astype(str).str.strip().str.lower().eq("true")
        require(bool(success.all()), f"{roi_type} contains unsuccessful rows")
    return frame


def quantile_indices(n: int, count: int = 10) -> list[int]:
    require(n >= count, f"Cannot select {count} unique positions from {n} records")
    positions = np.linspace(0, n - 1, count)
    indices = np.floor(positions + 0.5).astype(int).tolist()
    require(len(set(indices)) == count, f"Uniform quantile indices are not unique: {indices}")
    return indices


def bbox_from_values(values: list[object], label: str) -> tuple[int, int, int, int]:
    numeric = np.asarray(values, dtype=float)
    require(bool(np.isfinite(numeric).all()), f"{label} has non-finite coordinates: {values}")
    rounded = np.rint(numeric)
    require(bool(np.equal(numeric, rounded).all()), f"{label} has non-integer coordinates: {values}")
    x1, y1, x2, y2 = (int(v) for v in rounded)
    return x1, y1, x2, y2


def in_bounds(bbox: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = bbox
    return 0 <= x1 <= x2 < IMAGE_SIZE[0] and 0 <= y1 <= y2 < IMAGE_SIZE[1]


def collect_bboxes(bbox_log: pd.DataFrame, case_id: str) -> dict[str, tuple[int, int, int, int]]:
    forehead = bbox_log[(bbox_log["ID"].eq(case_id)) & (bbox_log["roi_type"].eq("forehead_roi"))].iloc[0]
    cheek = bbox_log[(bbox_log["ID"].eq(case_id)) & (bbox_log["roi_type"].eq("cheek_roi"))].iloc[0]
    result = {
        "forehead": bbox_from_values(
            [forehead["bbox_x1"], forehead["bbox_y1"], forehead["bbox_x2"], forehead["bbox_y2"]],
            f"{case_id} forehead",
        ),
        "cheek_image_left": bbox_from_values(
            [
                cheek["left_cheek_bbox_x1"],
                cheek["left_cheek_bbox_y1"],
                cheek["left_cheek_bbox_x2"],
                cheek["left_cheek_bbox_y2"],
            ],
            f"{case_id} image-left cheek",
        ),
        "cheek_image_right": bbox_from_values(
            [
                cheek["right_cheek_bbox_x1"],
                cheek["right_cheek_bbox_y1"],
                cheek["right_cheek_bbox_x2"],
                cheek["right_cheek_bbox_y2"],
            ],
            f"{case_id} image-right cheek",
        ),
    }
    return result


def select_cases(exif: pd.DataFrame, bbox_log: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for camera_id in EXPECTED_DEVICES:
        group = exif[exif["camera_id"].eq(camera_id)].sort_values("ID", kind="stable").reset_index(drop=True)
        indices = quantile_indices(len(group), 10)
        for index in indices:
            source = group.iloc[index]
            case_id = str(source["ID"])
            boxes = collect_bboxes(bbox_log, case_id)
            rows.append(
                {
                    "ID": case_id,
                    "Make": source["Make"],
                    "Model": source["Model"],
                    "camera_id": camera_id,
                    "aligned_rgb_path": str((ALIGNED_DIR / f"{case_id}.png").resolve()),
                    "bbox_log_path": str(BBOX_LOG_PATH.resolve()),
                    "forehead_bbox": str(list(boxes["forehead"])),
                    "cheek_image_left_bbox": str(list(boxes["cheek_image_left"])),
                    "cheek_image_right_bbox": str(list(boxes["cheek_image_right"])),
                    "selection_method": (
                        "within_camera_sorted_ID_uniform_quantile_10; "
                        f"index=floor(linspace(0,n-1,10)+0.5); rank={index + 1}/{len(group)}"
                    ),
                }
            )
    result = pd.DataFrame(rows)
    require(len(result) == 20 and result["ID"].nunique() == 20, "Selected cases are not 20 unique IDs")
    return result


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    font_name = "arialbd.ttf" if bold else "arial.ttf"
    font_path = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts" / font_name
    try:
        return ImageFont.truetype(str(font_path), size=size)
    except OSError:
        return ImageFont.load_default()


FONT = load_font(13)
BOLD_FONT = load_font(14, bold=True)
SMALL_FONT = load_font(12)


def load_case_images(case_id: str) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    aligned_path = ALIGNED_DIR / f"{case_id}.png"
    parsing_path = PARSING_DIR / f"{case_id}.png"
    final_path = FINAL_MASK_DIR / f"{case_id}.png"
    for path in (aligned_path, parsing_path, final_path):
        require(path.exists(), f"Missing required file: {path}")

    aligned = Image.open(aligned_path)
    require(aligned.mode == "RGB", f"{case_id} aligned mode is {aligned.mode}, expected RGB")
    require(aligned.size == IMAGE_SIZE, f"{case_id} aligned size is {aligned.size}, expected {IMAGE_SIZE}")
    aligned_array = np.asarray(aligned)
    require(aligned_array.dtype == np.uint8, f"{case_id} aligned dtype is {aligned_array.dtype}, expected uint8")

    parsing = np.asarray(Image.open(parsing_path))
    final_mask = np.asarray(Image.open(final_path))
    require(parsing.shape == (224, 224), f"{case_id} parsing shape is {parsing.shape}")
    require(final_mask.shape == (224, 224), f"{case_id} final mask shape is {final_mask.shape}")
    return aligned.copy(), parsing, final_mask


def overlay_skin(
    aligned: Image.Image,
    base_skin: np.ndarray,
    boxes: dict[str, tuple[int, int, int, int]],
    alpha: float = 0.43,
) -> Image.Image:
    array = np.asarray(aligned).copy()
    for roi_name, bbox in boxes.items():
        x1, y1, x2, y2 = bbox
        local_mask = base_skin[y1 : y2 + 1, x1 : x2 + 1]
        region = array[y1 : y2 + 1, x1 : x2 + 1]
        color = np.asarray(COLORS[roi_name], dtype=np.float32)
        blended = np.rint(region.astype(np.float32) * (1.0 - alpha) + color * alpha).astype(np.uint8)
        region[local_mask] = blended[local_mask]
    return Image.fromarray(array, mode="RGB")


def annotate_panel(
    image: Image.Image,
    case_id: str,
    camera_id: str,
    boxes: dict[str, tuple[int, int, int, int]],
    panel_name: str,
) -> Image.Image:
    canvas = Image.new("RGB", (560, 280), "white")
    image_origin = (8, 36)
    canvas.paste(image, image_origin)
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), f"{panel_name} | ID {case_id}", fill=(0, 0, 0), font=BOLD_FONT)
    draw.text((244, 38), f"camera: {camera_id}", fill=(0, 0, 0), font=FONT)
    draw.text((244, 62), "CSV endpoints: inclusive", fill=(60, 60, 60), font=SMALL_FONT)

    y = 92
    labels = [
        ("forehead", "forehead"),
        ("cheek_image_left", "image-left cheek"),
        ("cheek_image_right", "image-right cheek"),
    ]
    for key, label in labels:
        x1, y1, x2, y2 = boxes[key]
        color = COLORS[key]
        draw.rectangle(
            (image_origin[0] + x1, image_origin[1] + y1, image_origin[0] + x2, image_origin[1] + y2),
            outline=color,
            width=2,
        )
        draw.rectangle((244, y + 2, 256, y + 14), fill=color)
        draw.text((264, y), label, fill=(0, 0, 0), font=FONT)
        draw.text((264, y + 19), f"({x1}, {y1}, {x2}, {y2})", fill=(40, 40, 40), font=FONT)
        y += 54
    return canvas


def create_case_panels(
    selection: pd.DataFrame,
    bbox_log: pd.DataFrame,
) -> tuple[list[dict[str, object]], dict[str, tuple[Path, Path]]]:
    review_rows: list[dict[str, object]] = []
    panel_paths: dict[str, tuple[Path, Path]] = {}
    for folder in (BBOX_ONLY_DIR, BBOX_SKIN_DIR):
        folder.mkdir(parents=True, exist_ok=True)
        for old in folder.glob("*.png"):
            old.unlink()

    for row in selection.itertuples(index=False):
        case_id = str(row.ID)
        boxes = collect_bboxes(bbox_log, case_id)
        all_in_bounds = all(in_bounds(box) for box in boxes.values())
        require(all_in_bounds, f"Out-of-bounds bbox found for selected ID {case_id}: {boxes}")
        aligned, parsing, final_mask = load_case_images(case_id)
        base_skin = (parsing == SKIN_LABEL) & (final_mask > 0)

        bbox_only = annotate_panel(aligned, case_id, row.camera_id, boxes, "bbox only")
        skin_image = overlay_skin(aligned, base_skin, boxes)
        bbox_skin = annotate_panel(skin_image, case_id, row.camera_id, boxes, "bbox + valid skin")

        bbox_only_path = BBOX_ONLY_DIR / f"{case_id}.png"
        bbox_skin_path = BBOX_SKIN_DIR / f"{case_id}.png"
        bbox_only.save(bbox_only_path, format="PNG")
        bbox_skin.save(bbox_skin_path, format="PNG")
        panel_paths[case_id] = (bbox_only_path, bbox_skin_path)

        manual = REVIEW_MAP.get(
            case_id,
            ("PENDING", "PENDING", "PENDING", "PENDING", "PENDING", "Awaiting visual review"),
        )
        review_rows.append(
            {
                "ID": case_id,
                "camera_id": row.camera_id,
                "image_shape_valid": "TRUE",
                "bbox_in_bounds": "TRUE",
                "coordinate_registration": manual[0],
                "forehead_content": manual[1],
                "cheek_left_content": manual[2],
                "cheek_right_content": manual[3],
                "issue_type": manual[4],
                "review_notes": manual[5],
            }
        )
    return review_rows, panel_paths


def combined_case_tile(paths: tuple[Path, Path]) -> Image.Image:
    left = Image.open(paths[0]).convert("RGB")
    right = Image.open(paths[1]).convert("RGB")
    require(left.size == right.size, "Case panel sizes differ")
    tile = Image.new("RGB", (left.width + right.width, left.height), "white")
    tile.paste(left, (0, 0))
    tile.paste(right, (left.width, 0))
    return tile


def save_contact_sheet(case_ids: list[str], panel_paths: dict[str, tuple[Path, Path]], path: Path, columns: int) -> None:
    tiles = [combined_case_tile(panel_paths[case_id]) for case_id in case_ids]
    require(bool(tiles), f"No tiles for {path.name}")
    tile_width, tile_height = tiles[0].size
    rows = math.ceil(len(tiles) / columns)
    sheet = Image.new("RGB", (tile_width * columns, tile_height * rows), (230, 230, 230))
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * tile_width, (i // columns) * tile_height))
    sheet.save(path, format="PNG")


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = ["| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in view.itertuples(index=False, name=None)]
    return "\n".join([header, rule, *body])


def write_report(selection: pd.DataFrame, review: pd.DataFrame, exif: pd.DataFrame) -> None:
    camera_counts = exif.groupby("camera_id")["ID"].nunique().to_dict()
    status_counts = review["coordinate_registration"].value_counts().to_dict()
    flagged = review.loc[
        review["coordinate_registration"].isin(["QUESTIONABLE", "FAIL"])
        | review[["forehead_content", "cheek_left_content", "cheek_right_content"]]
        .isin(["QUESTIONABLE", "FAIL"])
        .any(axis=1),
        "ID",
    ].astype(str).tolist()
    complete = not (review[["coordinate_registration", "forehead_content", "cheek_left_content", "cheek_right_content"]] == "PENDING").any().any()
    completion_text = "完成：20例均已生成并逐例复核。" if complete else "图像生成完成；逐例视觉复核尚待写入。"
    continue_500 = (
        "从 bbox 坐标复用角度可以继续，但不建议立即无复核地批量落盘。正式构建必须将 bbox 与 parsing skin 和 final mask 相交；同时应先确认额部修复掩膜是否允许覆盖可见头发，因为 `9300518956` 和 `206664979` 的辅助叠加仍保留了这类区域。该问题不影响本次坐标一致性结论，但会影响正式额部 mask 的内容纯度。"
        if FINAL_REUSE_DECISION == "SAFE_WITH_ROI_MASKING"
        else "暂不建议，需先解决或澄清坐标一致性证据。"
    )
    lines = [
        "# 20例 ROI bbox 与 saved aligned RGB 坐标一致性审计",
        "",
        "## 1. 完成状态",
        "",
        completion_text,
        "本任务未修改任何已有代码或源图像，未重新执行人脸对齐/解析，未生成正式 ROI mask 或 ROI RGB 数据集，也未读取临床标签。",
        "",
        "## 2. bbox 数据来源与字段",
        "",
        f"- 实际 bbox 日志：`{BBOX_LOG_PATH}`",
        "- 日志与 `roi_qc_metrics.csv` 的 SHA-256 完全一致；本审计采用语义更明确的 `roi_metadata.csv`。",
        "- 每个当前研究 ID 在日志中有5条 ROI 记录；本任务只读取 `forehead_roi` 与 `cheek_roi`，各500条且全部 `roi_success=True`。",
        "- forehead：`bbox_x1, bbox_y1, bbox_x2, bbox_y2`（`roi_type=forehead_roi`）。",
        "- image-left cheek：`left_cheek_bbox_x1, left_cheek_bbox_y1, left_cheek_bbox_x2, left_cheek_bbox_y2`（`roi_type=cheek_roi`）。",
        "- image-right cheek：`right_cheek_bbox_x1, right_cheek_bbox_y1, right_cheek_bbox_x2, right_cheek_bbox_y2`（`roi_type=cheek_roi`）。",
        "",
        "## 3. 坐标格式与源码依据",
        "",
        "- 格式为 `(x1, y1, x2, y2)`，不是 `(x, y, w, h)`。",
        "- ROI 源码内部 `BBox` 使用 Python 切片式右开区间；写日志时以 `bbox.x2 - 1`、`bbox.y2 - 1` 保存，因此 CSV 的 `x2/y2` 是包含端点。绘图使用包含端点；如用于 NumPy/Python 切片，需恢复为 `y1:y2+1, x1:x2+1`。",
        "- 证据位于 `preprocessing/preprocess_global_aligned_face_parsing_roi_dataset_224_canvas.py`：内部 BBox 定义（约137–156行）、left/right 为 224-space 图像坐标的说明（约880行）、双颊日志写入（1477–1490行）、通用 bbox 日志写入（1561–1564行）。",
        "- `left` 是图像坐标中较小 x 的一侧，`right` 是较大 x 的一侧；不是患者解剖学左右。",
        "- 临时有效皮肤按项目定义计算为 `(parsing_label == 1) AND (final_mask > 0)`；皮肤标签 `1` 来自 `preprocessing/build_global_face_parsing_regularmask_blackbg_224_png_strict.py:112`。未进行腐蚀、膨胀或其他形态学处理。",
        "",
        "## 4. 20例选择方法",
        "",
        "只从当前 meanbg 目录的500个 ID 中选择。设备信息直接读取既有 EXIF 审计结果的 `Make` 与 `Model`，没有重新提取图像 EXIF；仅加载 `ID/Make/Model` 三列。每台设备内部按 ID 字符串稳定排序，在 `linspace(0, n-1, 10)` 上用 `floor(position+0.5)` 确定10个均匀分位位置。",
        "",
        f"- HONOR/BVL-AN00：当前500例中 {camera_counts.get('HONOR/BVL-AN00', 0)} 例，抽取10例。",
        f"- Xiaomi/M2006J10C：当前500例中 {camera_counts.get('Xiaomi/M2006J10C', 0)} 例，抽取10例。",
        "- 选择结果保存在 `bbox_overlay_20_selection.csv`，包含确切源路径、bbox 和排序位置，可完全复现。",
        "",
        "## 5. ID、文件与边界检查",
        "",
        "- meanbg：500个 PNG、500个唯一 ID。",
        "- EXIF 审计：500/500 ID 一一匹配。",
        "- bbox 日志：500/500 ID 匹配；额部和双颊所需行均为500/500。",
        "- saved aligned RGB、parsing label、final mask：三类文件均为500/500存在。",
        f"- 入选图像：{int((review['image_shape_valid'] == 'TRUE').sum())}/20 为 224×224、RGB、uint8。",
        f"- bbox：{int((review['bbox_in_bounds'] == 'TRUE').sum())}/20 例的三组坐标全部位于 224×224 范围内。",
        "",
        "## 6. 逐例视觉判断汇总",
        "",
        f"coordinate_registration：PASS {status_counts.get('PASS', 0)}；QUESTIONABLE {status_counts.get('QUESTIONABLE', 0)}；FAIL {status_counts.get('FAIL', 0)}。",
        "",
        markdown_table(
            review,
            ["ID", "camera_id", "coordinate_registration", "forehead_content", "cheek_left_content", "cheek_right_content", "issue_type"],
        ),
        "",
        "## 7. 坐标系统问题与 ROI 内容问题",
        "",
        "坐标系统不一致应表现为三个 bbox 在 saved aligned RGB 上出现一致方向的整体平移、缩放或落入错误区域。ROI 内容问题则是坐标落点与构造规则相符，但原始矩形可能纳入头发、背景、鼻部或其他非目标内容。后者不能据此判定坐标系统失败；正式使用时应以 parsing skin 与 final mask 的交集约束有效像素。",
        "",
        "## 8. 最终判定",
        "",
        f"- `BBOX_COORDINATE_ALIGNMENT={FINAL_COORDINATE_ALIGNMENT}`",
        f"- `BBOX_REUSE_DECISION={FINAL_REUSE_DECISION}`",
        f"- 可疑或失败 ID：{', '.join(flagged) if flagged else '无'}",
        "",
        "## 9. 是否可继续构建500例正式 mask",
        "",
        continue_500,
        "",
        "## 10. 下一步最小建议",
        "",
        "先用 `9300518956`、`206664979` 和 `A001471135` 做额部修复掩膜策略的最小复核；确认可见头发是否应被保留后，再新增一个单独、可追溯的 mask 构建步骤：按包含式 bbox 转为右开切片，并与 `parsing_skin AND final_mask` 相交。不要重新对齐或重新解析源图像。",
        "",
    ]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    current_ids = read_current_ids()
    exif = load_exif(current_ids)
    bbox_log = load_bbox_log(current_ids)
    selection = select_cases(exif, bbox_log)
    selection.to_csv(SELECTION_PATH, index=False, encoding="utf-8-sig")

    review_rows, panel_paths = create_case_panels(selection, bbox_log)
    review = pd.DataFrame(review_rows)
    review.to_csv(REVIEW_PATH, index=False, encoding="utf-8-sig")

    honor_ids = selection.loc[selection["camera_id"].eq("HONOR/BVL-AN00"), "ID"].astype(str).tolist()
    xiaomi_ids = selection.loc[selection["camera_id"].eq("Xiaomi/M2006J10C"), "ID"].astype(str).tolist()
    all_ids = selection["ID"].astype(str).tolist()
    save_contact_sheet(honor_ids, panel_paths, OUTPUT_DIR / "contact_sheet_honor.png", columns=1)
    save_contact_sheet(xiaomi_ids, panel_paths, OUTPUT_DIR / "contact_sheet_xiaomi.png", columns=1)
    save_contact_sheet(all_ids, panel_paths, OUTPUT_DIR / "contact_sheet_all.png", columns=2)
    write_report(selection, review, exif)

    coordinate_counts = review["coordinate_registration"].value_counts().to_dict()
    flagged = review.loc[
        review["coordinate_registration"].isin(["QUESTIONABLE", "FAIL"])
        | review[["forehead_content", "cheek_left_content", "cheek_right_content"]]
        .isin(["QUESTIONABLE", "FAIL"])
        .any(axis=1),
        "ID",
    ].astype(str).tolist()
    print(f"AUDIT_STATUS={'COMPLETE' if FINAL_COORDINATE_ALIGNMENT != 'PENDING' else 'VISUAL_REVIEW_PENDING'}")
    print(f"BBOX_LOG_PATH={BBOX_LOG_PATH}")
    print(f"SELECTED_CASES_TOTAL={len(selection)}")
    print(f"HONOR_CASES={len(honor_ids)}")
    print(f"XIAOMI_CASES={len(xiaomi_ids)}")
    print(f"BBOX_IN_BOUNDS_COUNT={(review['bbox_in_bounds'] == 'TRUE').sum()}")
    print(f"COORDINATE_PASS_COUNT={coordinate_counts.get('PASS', 0)}")
    print(f"COORDINATE_QUESTIONABLE_COUNT={coordinate_counts.get('QUESTIONABLE', 0)}")
    print(f"COORDINATE_FAIL_COUNT={coordinate_counts.get('FAIL', 0)}")
    print(f"BBOX_COORDINATE_ALIGNMENT={FINAL_COORDINATE_ALIGNMENT}")
    print(f"BBOX_REUSE_DECISION={FINAL_REUSE_DECISION}")
    print(f"QUESTIONABLE_OR_FAIL_IDS={','.join(flagged) if flagged else 'NONE'}")
    print(f"REPORT_PATH={REPORT_PATH}")
    print(f"CONTACT_SHEET_ALL_PATH={OUTPUT_DIR / 'contact_sheet_all.png'}")


if __name__ == "__main__":
    main()
