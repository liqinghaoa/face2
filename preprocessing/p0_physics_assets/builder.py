"""P0-A orchestration: preflight, immutable-source copies, processing, audit logs."""

from __future__ import annotations

import hashlib, json, random, shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .alignment_adapter import align_original_scene_once
from .asset_io import atomic_copy2, atomic_write_text, save_npz, save_png, verify_assets, write_csv, write_dataframe_csv
from .config import config_as_dict
from .mask_builder import build_base_masks, build_parsing_artifacts
from .metadata import build_sample_records, load_exif_500, validate_split
from .qc_visualization import save_qc
from .regression import evaluate_regression
from .roi_builder import build_roi_masks
from .types import P0Config, SampleBuildResult, SampleRecord

CORE_FILES = ("images/raw_scene", "images/aligned_scene_224", "images/aligned_blackbg_224", "images/e0b_meanbg_224", "parsing/parsing_label_224", "masks/source_valid_224", "masks/final_face_mask_224", "masks/face_valid_224", "masks/skin_strict_224", "transforms/alignment")
STATUS_VALUES = {"success", "success_with_roi_warnings", "failed_missing_asset", "failed_read_image", "failed_no_face", "failed_landmark_incomplete", "failed_alignment", "failed_parsing", "failed_empty_final_mask", "failed_empty_skin_mask", "failed_save", "failed_regression", "failed_unexpected_error"}


def preflight(config: P0Config) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Path]], list[SampleRecord]]:
    """Validate fixed split, EXIF mapping, all source assets, and checkpoint before writes."""
    split = validate_split(pd.read_csv(config.inputs.split_csv, dtype={"ID":str,"patient_group_id":str}))
    exif = load_exif_500(config, split)
    assets = verify_assets(set(split.ID), {"raw":config.inputs.raw_image_dir, "blackbg":config.inputs.existing_blackbg_dir, "meanbg":config.inputs.existing_meanbg_dir})
    if not config.inputs.parsing_checkpoint.is_file(): raise FileNotFoundError(f"parsing checkpoint missing: {config.inputs.parsing_checkpoint}")
    return split, exif, assets, build_sample_records(config, split, exif, assets)


def _dirs(root: Path) -> dict[str, Path]:
    names = [*CORE_FILES, "parsing/selected_semantic_mask_224", "parsing/semantic_regularized_mask_224", "parsing/candidate_envelope_224", "masks/roi_geometry_224", "masks/roi_effective_224", "metadata", "splits_500", "qc_preview", "qc_preview/random_success", "qc_preview/forehead_unavailable", "qc_preview/non_normal_orientation", "qc_preview/high_source_padding", "qc_preview/roi_warning", "qc_preview/processing_failure", "qc_preview/regression_mismatch"]
    for roi in ("left_cheek","right_cheek","combined_cheek","forehead","lip","eye","chin"):
        names += [f"masks/roi_geometry_224/{roi}", f"masks/roi_effective_224/{roi}"]
    values={name:root/name for name in names}
    for path in values.values(): path.mkdir(parents=True,exist_ok=True)
    return values


def _rel(root: Path, path: Path) -> str: return path.relative_to(root).as_posix()


def _output_complete(root: Path, image_id: str, raw_suffix: str) -> bool:
    required=[root/f"images/raw_scene/{image_id}{raw_suffix}",root/f"images/aligned_scene_224/{image_id}.png",root/f"images/aligned_blackbg_224/{image_id}.png",root/f"images/e0b_meanbg_224/{image_id}.png",root/f"parsing/parsing_label_224/{image_id}.png",root/f"masks/source_valid_224/{image_id}.png",root/f"masks/final_face_mask_224/{image_id}.png",root/f"masks/face_valid_224/{image_id}.png",root/f"masks/skin_strict_224/{image_id}.png",root/f"transforms/alignment/{image_id}.npz"]
    return all(path.is_file() for path in required)


def _base_fields(record: SampleRecord) -> dict[str, Any]:
    meta=record.metadata
    return {"ID":record.image_id,"patient_group_id":record.patient_group_id,"SEX":record.sex,"sex_name":record.sex_name,"NYHA":record.nyha,"label_3class":record.label_3class,"label_3class_name":record.label_3class_name,"binary_label":int(meta["binary_label"]),"binary_name":str(meta["binary_name"]),"fold":record.fold,"source_path":str(record.source_path),"source_sha256":meta.get("source_sha256","")}


def _map_failure(exc: BaseException) -> tuple[str,str]:
    text=f"{type(exc).__name__}: {exc}"; status=getattr(exc,"status","")
    mapping={"failed_no_image":"failed_missing_asset","failed_read_image":"failed_read_image","failed_no_face":"failed_no_face","failed_landmark_incomplete":"failed_landmark_incomplete","failed_alignment":"failed_alignment","failed_parsing_model":"failed_parsing","failed_empty_selected_mask":"failed_empty_final_mask"}
    if isinstance(exc, OSError): return "failed_save", text
    return mapping.get(status,"failed_unexpected_error"),text


def _process(record: SampleRecord, config: P0Config, root: Path, dirs: dict[str,Path], detector: Any, mesh: Any, parsing_model: Any, device: Any) -> SampleBuildResult:
    """Build one record. ROI errors remain traceable warnings, not parsing failures."""
    fields=_base_fields(record); image_id=record.image_id; warnings: list[str]=[]
    raw_target=root/f"images/raw_scene/{image_id}{record.source_path.suffix}"; black_target=root/f"images/aligned_blackbg_224/{image_id}.png"; mean_target=root/f"images/e0b_meanbg_224/{image_id}.png"
    try:
        atomic_copy2(record.source_path,raw_target); atomic_copy2(record.blackbg_path,black_target); atomic_copy2(record.meanbg_path,mean_target)
        from preprocessing import build_global_face_oval_blackbg_png_simalign_strict as legacy
        image=legacy.read_image_rgb(record.source_path)
        if image is None: raise legacy.SampleFailure("failed_read_image","cannot_decode_image")
        selected=legacy.select_face(legacy.detect_faces(image,detector),image.shape)
        if selected is None: raise legacy.SampleFailure("failed_no_face","no_primary_face")
        alignment=align_original_scene_once(image,selected,mesh,config.runtime.image_size)
        aligned_path=root/f"images/aligned_scene_224/{image_id}.png"; save_png(aligned_path,alignment.aligned_rgb); save_png(root/f"masks/source_valid_224/{image_id}.png",alignment.source_valid)
        try:
            parsing_artifacts, preliminary=build_parsing_artifacts(alignment.aligned_rgb,parsing_model,device,config.global_mask)
        except Exception as exc:
            status = "failed_empty_final_mask" if getattr(exc, "status", "") == "failed_empty_selected_mask" else "failed_parsing"
            return SampleBuildResult(image_id,status,status,f"{type(exc).__name__}: {exc}",warnings,fields)
        masks=build_base_masks(parsing_artifacts.label_map,alignment.source_valid,preliminary.final_face_mask,preliminary.feather_alpha)
        if not (masks.final_face_mask>0).any(): return SampleBuildResult(image_id,"failed_empty_final_mask","failed_empty_final_mask","final face mask empty",warnings,fields)
        if not (masks.skin_strict_mask>0).any(): return SampleBuildResult(image_id,"failed_empty_skin_mask","failed_empty_skin_mask","strict skin mask empty",warnings,fields)
        save_png(root/f"parsing/parsing_label_224/{image_id}.png",parsing_artifacts.label_map); save_png(root/f"parsing/selected_semantic_mask_224/{image_id}.png",parsing_artifacts.selected_semantic_mask); save_png(root/f"parsing/semantic_regularized_mask_224/{image_id}.png",parsing_artifacts.semantic_regularized_mask); save_png(root/f"parsing/candidate_envelope_224/{image_id}.png",parsing_artifacts.candidate_envelope_mask)
        for name, array in {"final_face_mask_224":masks.final_face_mask,"face_valid_224":masks.face_valid_mask,"skin_strict_224":masks.skin_strict_mask}.items(): save_png(root/f"masks/{name}/{image_id}.png",array)
        save_npz(root/f"transforms/alignment/{image_id}.npz",matrix_2x3=alignment.matrix_2x3,original_keypoints_5=alignment.original_keypoints_5,aligned_keypoints_5=alignment.aligned_keypoints_5,original_landmarks_468=alignment.original_landmarks_468,aligned_landmarks_468=alignment.aligned_landmarks_468,expanded_bbox_xywh=np.asarray(alignment.expanded_bbox_xywh),source_image_hw=np.asarray(alignment.source_image_hw),alignment_parameters=np.asarray(json.dumps(alignment.parameters,ensure_ascii=False)))
        try:
            geometry,effects,metrics=build_roi_masks(parsing_artifacts.label_map,masks.final_face_mask,alignment.source_valid,masks.face_valid_mask,masks.skin_strict_mask,alignment.aligned_landmarks_468,config.roi)
            for name in geometry: save_png(root/f"masks/roi_geometry_224/{name}/{image_id}.png",geometry[name]); save_png(root/f"masks/roi_effective_224/{name}/{image_id}.png",effects[name]); fields.update({f"{name}_geometry_relpath":_rel(root,root/f"masks/roi_geometry_224/{name}/{image_id}.png"),f"{name}_effective_relpath":_rel(root,root/f"masks/roi_effective_224/{name}/{image_id}.png")}); fields.update({f"{name}_{key}":value for name,value_map in metrics.items() for key,value in value_map.items()})
        except Exception as exc:
            warnings.append(f"roi_failure:{type(exc).__name__}:{exc}")
        alpha=masks.feather_alpha; reconstructed=legacy.apply_black_background(alignment.aligned_rgb,alpha); reference=legacy.read_image_rgb(record.blackbg_path)
        if reference is None: return SampleBuildResult(image_id,"failed_missing_asset","failed_missing_asset","cannot read old black background",warnings,fields)
        regression=evaluate_regression(reference,reconstructed,config.regression)
        fields.update({"regression_status":regression.status,"regression_mae":regression.mae,"regression_rmse":regression.rmse,"regression_max_abs_diff":regression.max_abs_diff,"regression_p99_abs_diff":regression.p99_abs_diff,"regression_rgb_ssim":regression.rgb_ssim,"regression_large_diff_fraction":regression.large_diff_fraction})
        if regression.status != "passed":
            save_png(root/f"qc_preview/regression_mismatch/{image_id}.png",regression.difference_heatmap); return SampleBuildResult(image_id,"failed_regression","failed_regression","black-background regression failed",warnings,fields)
        if config.qc.save_qc:
            cheeks=np.maximum(geometry.get("left_cheek",np.zeros_like(masks.final_face_mask)),geometry.get("right_cheek",np.zeros_like(masks.final_face_mask))) if 'geometry' in locals() else np.zeros_like(masks.final_face_mask); forehead_lip=np.maximum(geometry.get("forehead",np.zeros_like(cheeks)),geometry.get("lip",np.zeros_like(cheeks))) if 'geometry' in locals() else cheeks
            payload={"raw":image,"aligned":alignment.aligned_rgb,"labels":parsing_artifacts.label_map,"final":masks.final_face_mask,"skin":masks.skin_strict_mask,"cheeks":cheeks,"forehead_lip":forehead_lip,"existing":reference,"reconstructed":reconstructed,"heat":regression.difference_heatmap}
            targets=[]
            qc_key = int.from_bytes(hashlib.sha256(image_id.encode("utf-8")).digest()[:8], "big")
            if qc_key % max(1,500//max(1,config.qc.num_random_qc)) == 0: targets.append("random_success")
            if fields.get("forehead_forehead_available") is False: targets.append("forehead_unavailable")
            if str(record.metadata.get("orientation_value", "")) not in {"", "1", "1.0"}: targets.append("non_normal_orientation")
            if alignment.parameters["top_invalid_area_ratio"] > .20: targets.append("high_source_padding")
            if warnings: targets.append("roi_warning")
            for target in targets: save_qc(root/f"qc_preview/{target}/{image_id}.png",payload)
        fields.update({"camera_make":record.metadata.get("camera_make",""),"camera_model":record.metadata.get("camera_model",""),"datetime_original":record.metadata.get("datetime_original",""),"exposure_time_s":record.metadata.get("exposure_time_s",""),"f_number":record.metadata.get("f_number",""),"iso":record.metadata.get("iso",""),"brightness_value":record.metadata.get("brightness_value",""),"orientation_value":record.metadata.get("orientation_value",""),"orientation_description":record.metadata.get("orientation_description",""),"raw_scene_relpath":_rel(root,raw_target),"aligned_scene_relpath":_rel(root,aligned_path),"aligned_blackbg_relpath":_rel(root,black_target),"e0b_meanbg_relpath":_rel(root,mean_target),"parsing_label_relpath":_rel(root,root/f"parsing/parsing_label_224/{image_id}.png"),"source_valid_mask_relpath":_rel(root,root/f"masks/source_valid_224/{image_id}.png"),"final_face_mask_relpath":_rel(root,root/f"masks/final_face_mask_224/{image_id}.png"),"face_valid_mask_relpath":_rel(root,root/f"masks/face_valid_224/{image_id}.png"),"skin_strict_mask_relpath":_rel(root,root/f"masks/skin_strict_224/{image_id}.png"),"alignment_npz_relpath":_rel(root,root/f"transforms/alignment/{image_id}.npz"),"forehead_available":fields.get("forehead_forehead_available",False)})
        orientation=str(record.metadata.get("orientation_value", ""));
        if orientation not in {"", "1", "1.0"}: warnings.append("warning_non_normal_exif_orientation")
        status="success_with_roi_warnings" if warnings else "success"; return SampleBuildResult(image_id,"success",status,"",warnings,fields)
    except Exception as exc:
        core, reason=_map_failure(exc)
        if config.qc.save_qc and "image" in locals():
            try: save_png(root/f"qc_preview/processing_failure/{image_id}.png", image)
            except Exception: pass
        return SampleBuildResult(image_id,core,core,reason,warnings,fields)


def build(config: P0Config, validate_only: bool = False) -> dict[str, Any]:
    """Run P0-A without changing legacy preprocessors or source data directories."""
    split,exif,_,records=preflight(config)
    if validate_only: return {"preflight":"passed","rows":len(split),"exif_rows":len(exif),"records":len(records)}
    root=config.inputs.output_dir
    expected_root = (config.inputs.project_root / "data" / "processed" / "P0_Physics_Audit_v1").resolve()
    if root != expected_root:
        raise ValueError(f"P0 output_dir must be the isolated directory: {expected_root}")
    if root.exists() and config.runtime.overwrite: shutil.rmtree(root)
    if root.exists() and not config.runtime.resume: raise FileExistsError(f"P0 output exists: {root}; use --resume or --overwrite")
    existing_rows: dict[str, dict[str, Any]] = {}
    existing_index = root / "metadata/master_index.csv"
    if config.runtime.resume and existing_index.is_file():
        existing_rows = {str(row["ID"]): row for row in pd.read_csv(existing_index, dtype={"ID":str}).fillna("").to_dict("records")}
    dirs=_dirs(root); atomic_copy2(config.inputs.exif_workbook,root/"metadata/Image_Metadata_All.xlsx"); write_dataframe_csv(root/"splits_500/nyha_3class_sex_stratified_group_5fold.csv",split); write_dataframe_csv(root/"metadata/exif_500.csv",exif)
    atomic_write_text(root/"metadata/effective_config.yaml",yaml.safe_dump(config_as_dict(config),allow_unicode=True,sort_keys=False))
    selected=[record for record in records if (not config.runtime.sample_ids or record.image_id in config.runtime.sample_ids)]; selected=selected[:config.runtime.max_samples] if config.runtime.max_samples else selected
    import mediapipe as mp
    from preprocessing import build_global_face_parsing_regularmask_blackbg_224_png_strict as parsing
    device=parsing.resolve_parsing_device(config.runtime.parsing_device); model=parsing.load_face_parsing_model("bisenet",config.inputs.parsing_checkpoint,device); detector=mp.solutions.face_detection.FaceDetection(model_selection=1,min_detection_confidence=config.runtime.min_detection_confidence); mesh=mp.solutions.face_mesh.FaceMesh(static_image_mode=True,max_num_faces=1,refine_landmarks=True,min_detection_confidence=config.runtime.min_detection_confidence,min_tracking_confidence=.5)
    results=[]
    try:
        for record in selected:
            previous = existing_rows.get(record.image_id)
            if config.runtime.resume and previous and previous.get("core_status") == "success" and _output_complete(root,record.image_id,record.source_path.suffix):
                results.append(SampleBuildResult(record.image_id,"success",str(previous.get("overall_status") or "success"),"",str(previous.get("warning_flags") or "").split(";") if previous.get("warning_flags") else [],previous)); continue
            results.append(_process(record,config,root,dirs,detector,mesh,model,device))
    finally: detector.close(); mesh.close()
    rows=[]
    for result in results:
        row=dict(result.fields); row.update({"core_status":result.core_status,"overall_status":result.overall_status,"failure_reason":result.failure_reason,"warning_flags":";".join(result.warnings)}); rows.append(row)
    write_csv(root/"metadata/build_status.csv",rows); write_csv(root/"metadata/master_index.csv",rows)
    manifest={"selected":len(selected),"core_success":sum(r.core_status=="success" for r in results),"regression_passed":sum(r.fields.get("regression_status")=="passed" for r in results),"status_counts":pd.Series([r.overall_status for r in results]).value_counts().to_dict()}
    atomic_write_text(root/"metadata/build_manifest.json",json.dumps(manifest,ensure_ascii=False,indent=2))
    return manifest
