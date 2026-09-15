"""Frozen-protocol RGB inference worker for the single S1-7 Test run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import mediapipe as mp
import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from preprocessing import build_global_face_oval_blackbg_png_simalign_strict as alignment  # noqa: E402
from src.skin_optics_real_preprocess import face_parser  # noqa: E402


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.ascontiguousarray(np.asarray(image.convert("RGB"), dtype=np.uint8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    runtime = config["runtime"]
    checkpoint = Path(runtime["parser_checkpoint"])
    checkpoint = (checkpoint if checkpoint.is_absolute() else PROJECT_ROOT / checkpoint).resolve()
    device = face_parser.resolve_device(str(runtime["parser_device"]))
    model = face_parser.load_model("bisenet", checkpoint, device)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=int(runtime["face_detection_model_selection"]),
        min_detection_confidence=float(runtime["face_detection_min_confidence"]),
    )
    mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=float(runtime["face_mesh_min_confidence"]),
        min_tracking_confidence=0.5,
    )
    output = args.output_root.resolve()
    test_root = output / "test"
    test_root.mkdir(parents=True, exist_ok=False)
    with args.manifest.open("r", encoding="utf-8-sig", newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        sample_id = row["sample_id"]
        sample_dir = test_root / sample_id
        sample_dir.mkdir(parents=False, exist_ok=False)
        record: dict[str, Any] = {
            "sample_id": sample_id,
            "split": "test",
            "status": "FAIL",
            "failure_code": "",
            "failure_detail": "",
        }
        try:
            rgb = _load_rgb(Path(row["rgb_path"]))
            detections = alignment.detect_faces(rgb, detector)
            selected_face = alignment.select_face(detections, rgb.shape)
            if selected_face is None:
                crop_x, crop_y, crop_w, crop_h = 0, 0, rgb.shape[1], rgb.shape[0]
                detection_confidence = None
                detection_bbox = None
                detection_route = "full_image_facemesh_fallback"
            else:
                crop_x, crop_y, crop_w, crop_h = alignment.expand_bbox(
                    selected_face.bbox,
                    rgb.shape,
                    top_ratio=float(runtime["bbox_expand_top_ratio"]),
                    bottom_ratio=float(runtime["bbox_expand_bottom_ratio"]),
                    side_ratio=float(runtime["bbox_expand_side_ratio"]),
                )
                detection_confidence = float(selected_face.confidence)
                detection_bbox = list(selected_face.bbox)
                detection_route = "detector_expanded_crop"
            crop_rgb = np.ascontiguousarray(rgb[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w])
            landmarks = alignment.run_facemesh(crop_rgb, mesh)
            if landmarks is None:
                raise RuntimeError("facemesh_failed")
            points = np.asarray(
                [[float(item.x) * crop_w + crop_x, float(item.y) * crop_h + crop_y] for item in landmarks],
                dtype=np.float32,
            )
            if points.ndim != 2 or points.shape[0] < 468 or points.shape[1] != 2 or not np.isfinite(points).all():
                raise RuntimeError(f"invalid_landmarks:{points.shape}")
            labels = face_parser.run(np.ascontiguousarray(rgb), model, device)
            if labels.shape != rgb.shape[:2]:
                raise RuntimeError(f"parser_shape_mismatch:{labels.shape}")
            np.save(sample_dir / "parsing_label.npy", labels.astype(np.uint8), allow_pickle=False)
            np.save(sample_dir / "landmarks.npy", points, allow_pickle=False)
            metadata = {
                "sample_id": sample_id,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "rgb_path": row["rgb_path"],
                "rgb_sha256_expected": row["rgb_sha256"],
                "image_shape": list(rgb.shape),
                "detections": len(detections),
                "detection_route": detection_route,
                "selected_detection_confidence": detection_confidence,
                "selected_detection_bbox": detection_bbox,
                "expanded_bbox": [crop_x, crop_y, crop_w, crop_h],
                "landmark_count": int(points.shape[0]),
                "parser_checkpoint": str(checkpoint),
                "parser_checkpoint_sha256": _sha256(checkpoint),
            }
            (sample_dir / "rgb_inference.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
            )
            record.update({
                "status": "PASS",
                "detection_confidence": detection_confidence,
                "detection_route": detection_route,
                "landmark_count": int(points.shape[0]),
                "parser_checkpoint_sha256": metadata["parser_checkpoint_sha256"],
            })
        except Exception as error:
            record["failure_code"] = type(error).__name__
            record["failure_detail"] = repr(error)
        rows.append(record)
        print(f"S1-7 RGB Test: {index}/{len(selected)}", flush=True)
    detector.close()
    mesh.close()
    manifest_path = output / "rgb_worker_test.csv"
    fields = sorted({key for row in rows for key in row})
    with manifest_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    failures = sum(row["status"] != "PASS" for row in rows)
    summary = {
        "split": "test",
        "sample_count": len(rows),
        "pass_count": len(rows) - failures,
        "failure_count": failures,
        "parser_checkpoint_sha256": _sha256(checkpoint),
    }
    (output / "rgb_worker_test_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
