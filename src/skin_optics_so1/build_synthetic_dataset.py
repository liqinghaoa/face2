"""CLI entry point for SO-1 synthetic dataset generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import shutil
import sys
import time
from dataclasses import asdict, fields as dataclass_fields
from pathlib import Path

import numpy as np
import torch

from skin_optics_so1.camera_light_split import build_camera_light_split, save_camera_light_split, shuffled_balanced_pairs
from skin_optics_so1.color_input import make_target_mhsp_chw, srgb_hwc_to_linear_chw
from skin_optics_so1.random_fields import generate_mh_field, generate_p_field, generate_s_field, stable_seed
from skin_optics_so1.retry_pairs import assert_train_final_pair_invariants, select_retry_pair, train_final_pair_collisions
from skin_optics_so1.so0_batch_renderer import SO0CameraRenderer, clipping_stats, ensure_so0_import_path, render_with_oom_fallback
from skin_optics_so1.synthetic_config import assert_so0_range_compatible, load_config, to_plain_dict
from skin_optics_so1.synthetic_manifest import ManifestRow, build_manifest, validate_manifest, write_manifest
from skin_optics_so1.synthetic_masks import generate_valid_mask
from skin_optics_so1.synthetic_qc import rows_as_dicts, write_preview_png, write_qc_outputs
from skin_optics_so1.synthetic_storage import prepare_split_storage, write_metadata, write_run_hashes, write_sample


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build SO-1 synthetic dataset from frozen SO-0 camera renderer.")
    p.add_argument("--config", required=True)
    p.add_argument("--output-root", required=True)
    p.add_argument("--validate-only", action="store_true")
    p.add_argument("--smoke-only", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--overwrite-confirmed", action="store_true")
    p.add_argument("--repair-base-latent-ids", default="", help="Comma-separated train base_latent_id values for audited targeted repair.")
    return p.parse_args()


def preflight_environment() -> dict:
    info = {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpus": [],
        "cwd": os.getcwd(),
    }
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            info["gpus"].append({"index": i, "name": prop.name, "total_memory_bytes": int(prop.total_memory)})
    return info


def check_disk_space(output_root: str | Path, estimated_bytes: int, full_mode: bool) -> dict:
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(root)
    free = int(usage.free)
    min_full = 40 * 1024**3
    recommended = 55 * 1024**3
    if full_mode and free < min_full:
        raise RuntimeError(f"Refusing full generation: free space {free/1024**3:.1f} GiB < 40 GiB")
    if not full_mode and free < estimated_bytes:
        raise RuntimeError(f"Refusing smoke: free space {free} bytes < estimated {estimated_bytes} bytes")
    return {
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": free,
        "full_space_warning": bool(full_mode and min_full <= free < recommended),
    }


def _smoke_rows(rows: list[ManifestRow]) -> list[ManifestRow]:
    selected: list[ManifestRow] = []
    wanted = {"train": 24, "validation": 8, "id_test": 8, "camera_ood": 8, "light_ood": 8, "joint_ood": 8}
    for split, n in wanted.items():
        selected.extend([r for r in rows if r.split == split][:n])
    for i, r in enumerate(selected):
        r.row_index = i
        r.split_index = sum(1 for x in selected[:i] if x.split == r.split)
        r.sample_id = f"{r.split}_{r.split_index:06d}"
    return selected


def _fields_for_row(row: ManifestRow, size: int, min_valid_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    m, ms = generate_mh_field(row.m_base, row.m_seed, size)
    h, hs = generate_mh_field(row.h_base, row.h_seed, size)
    s, ss = generate_s_field(row.s_seed, size)
    p, ps = generate_p_field(row.p_seed, size)
    mask, valid_fraction, _ = generate_valid_mask(row.mask_seed, size, row.valid_mask_type, min_valid_fraction)
    stats = {"m": ms, "h": hs, "s": ss, "p": ps, "valid_fraction": valid_fraction}
    return m, h, s, p, mask, stats


def _clip_accepts(cfg, clip: dict) -> bool:
    return clip["total_clip_fraction"] <= cfg.render.max_total_clip_fraction and max(clip["per_channel_clip_fraction"]) <= cfg.render.max_channel_clip_fraction


def _retry_clipped_sample(
    cfg,
    split_cfg,
    row: ManifestRow,
    m: np.ndarray,
    h: np.ndarray,
    mask: np.ndarray,
    device: str,
    forbidden_pairs: set[str] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray, object, object, int, str, str]:
    """Retry a clipped sample with fixed M/H/mask and new S/P/acquisition pair."""

    allowed = split_cfg.allowed_pairs(row.split)
    dynamic_forbidden = set(forbidden_pairs or set())
    dynamic_forbidden.add(row.camera_light_pair)
    last_out: dict | None = None
    last_s: np.ndarray | None = None
    last_p: np.ndarray | None = None
    last_ss = None
    last_ps = None
    last_pair = None
    history: list[dict] = []
    for retry in range(1, cfg.render.max_retries + 1):
        pair = select_retry_pair(
            split=row.split,
            base_latent_id=row.base_latent_id,
            acquisition_variant_id=row.acquisition_variant_id,
            retry_index=retry,
            allowed_pairs=allowed,
            forbidden_pairs=dynamic_forbidden,
            acquisition_seed=row.acquisition_seed,
        )
        s_seed = stable_seed(cfg.global_seed, row.split, f"{row.base_latent_id}_retry_{retry}", row.acquisition_variant_id, "s_field")
        p_seed = stable_seed(cfg.global_seed, row.split, f"{row.base_latent_id}_retry_{retry}", row.acquisition_variant_id, "p_field")
        s, ss = generate_s_field(s_seed, cfg.patch.size)
        p, ps = generate_p_field(p_seed, cfg.patch.size)
        renderer = SO0CameraRenderer(pair.camera_name, pair.light_name, device=device, dtype=torch.float32)
        try:
            out = renderer.render_batch(m[None], h[None], s[None], p[None], exposure=cfg.controls.exposure)
        finally:
            renderer.close()
        sample_out = {k: v[0] for k, v in out.items()}
        last_out, last_s, last_p, last_ss, last_ps, last_pair = sample_out, s, p, ss, ps, pair
        clip = clipping_stats(sample_out["srgb_unclipped"])
        accepted = _clip_accepts(cfg, clip)
        history.append(
            {
                "retry_index": retry,
                "camera_light_pair": pair.key,
                "s_seed": s_seed,
                "p_seed": p_seed,
                "total_clip_fraction": float(clip["total_clip_fraction"]),
                "per_channel_clip_fraction": clip["per_channel_clip_fraction"],
                "accepted": bool(accepted),
            }
        )
        if accepted:
            row.s_seed = s_seed
            row.p_seed = p_seed
            row.camera_name = pair.camera_name
            row.light_name = pair.light_name
            row.camera_light_pair = pair.key
            row.final_camera_name = pair.camera_name
            row.final_light_name = pair.light_name
            row.final_camera_light_pair = pair.key
            row.retry_pair_history = json.dumps(history, sort_keys=True)
            return sample_out, s, p, ss, ps, retry, pair.camera_name, pair.light_name
        dynamic_forbidden.add(pair.key)
    assert last_out is not None and last_s is not None and last_p is not None and last_ss is not None and last_ps is not None
    assert last_pair is not None
    row.camera_name = last_pair.camera_name
    row.light_name = last_pair.light_name
    row.camera_light_pair = last_pair.key
    row.final_camera_name = last_pair.camera_name
    row.final_light_name = last_pair.light_name
    row.final_camera_light_pair = last_pair.key
    row.retry_pair_history = json.dumps(history, sort_keys=True)
    return last_out, last_s, last_p, last_ss, last_ps, cfg.render.max_retries, last_pair.camera_name, last_pair.light_name


def render_rows(cfg, rows: list[ManifestRow], output_root: str | Path, device: str, resume: bool, overwrite: bool, split_cfg) -> tuple[dict, dict]:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but PyTorch reports cuda_available=False. GPU SO-0 smoke/full generation cannot run on this host.")
    split_counts: dict[str, int] = {}
    for r in rows:
        split_counts[r.split] = max(split_counts.get(r.split, 0), r.split_index + 1)
    stores = prepare_split_storage(output_root, split_counts, cfg.patch.size, overwrite=overwrite and not resume)
    metadata_by_split: dict[str, list[dict]] = {s: [] for s in split_counts}
    timings: dict[str, dict] = {}
    audit_ids = {r.sample_id for r in rows[:32]}
    audit_baselines: dict[str, dict] = {}
    same_context_results: list[dict] = []
    t_all = time.perf_counter()
    preview_count = 0
    by_pair: dict[tuple[str, str], list[ManifestRow]] = {}
    for r in rows:
        if not r.initial_camera_light_pair:
            r.initial_camera_name = r.camera_name
            r.initial_light_name = r.light_name
            r.initial_camera_light_pair = r.camera_light_pair
        if not r.final_camera_light_pair:
            r.final_camera_name = r.camera_name
            r.final_light_name = r.light_name
            r.final_camera_light_pair = r.camera_light_pair
        by_pair.setdefault((r.camera_name, r.light_name), []).append(r)
    initial_pair_by_base_variant = {
        (int(r.base_latent_id), int(r.acquisition_variant_id)): r.initial_camera_light_pair or r.camera_light_pair
        for r in rows
        if r.split == "train"
    }
    final_pair_by_base_variant: dict[tuple[int, int], str] = {}

    batch_sizes = [cfg.render.batch_size] + [b for b in cfg.render.oom_fallback_batch_sizes if b != cfg.render.batch_size]
    for (camera, light), group in by_pair.items():
        renderer = SO0CameraRenderer(camera, light, device=device, dtype=torch.float32)
        try:
            prepared = []
            aux = []
            for r in group:
                m, h, s, p, mask, stats = _fields_for_row(r, cfg.patch.size, cfg.mask.min_valid_fraction)
                prepared.append((m, h, s, p))
                aux.append((r, mask, stats))
            rendered, timing = render_with_oom_fallback(renderer, prepared, batch_sizes, cfg.controls.exposure)
            timings[f"{camera} / {light}"] = timing
            for out, (r, mask, stats), fields in zip(rendered, aux, prepared):
                clip = clipping_stats(out["srgb_unclipped"])
                retry_count = 0
                status = "SUCCESS"
                if not _clip_accepts(cfg, clip):
                    forbidden_pairs = {r.camera_light_pair}
                    if r.split == "train":
                        sibling_variant = 1 - int(r.acquisition_variant_id)
                        sibling_key = (int(r.base_latent_id), sibling_variant)
                        sibling_initial = initial_pair_by_base_variant.get(sibling_key)
                        sibling_final = final_pair_by_base_variant.get(sibling_key)
                        if sibling_initial:
                            forbidden_pairs.add(sibling_initial)
                        if sibling_final:
                            forbidden_pairs.add(sibling_final)
                    retry_out, retry_s, retry_p, retry_ss, retry_ps, retry_count, _, _ = _retry_clipped_sample(
                        cfg,
                        split_cfg,
                        r,
                        fields[0],
                        fields[1],
                        mask,
                        device,
                        forbidden_pairs=forbidden_pairs,
                    )
                    out = retry_out
                    fields = (fields[0], fields[1], retry_s, retry_p)
                    stats["s"] = retry_ss
                    stats["p"] = retry_ps
                    clip = clipping_stats(out["srgb_unclipped"])
                    status = "SUCCESS" if _clip_accepts(cfg, clip) else "FAILED"
                if not r.final_camera_light_pair:
                    r.final_camera_name = r.camera_name
                    r.final_light_name = r.light_name
                    r.final_camera_light_pair = r.camera_light_pair
                if not r.retry_pair_history:
                    r.retry_pair_history = "[]"
                srgb = out["srgb_display_clipped"]
                linear = srgb_hwc_to_linear_chw(srgb, mask)
                target = make_target_mhsp_chw(*fields, mask)
                if status == "SUCCESS":
                    write_sample(stores, r.split, r.split_index, linear, target, mask, resume=resume)
                if r.split == "train":
                    final_pair_by_base_variant[(int(r.base_latent_id), int(r.acquisition_variant_id))] = r.final_camera_light_pair or r.camera_light_pair
                    sibling_key = (int(r.base_latent_id), 1 - int(r.acquisition_variant_id))
                    if sibling_key in final_pair_by_base_variant:
                        if final_pair_by_base_variant[(int(r.base_latent_id), int(r.acquisition_variant_id))] == final_pair_by_base_variant[sibling_key]:
                            raise RuntimeError(
                                f"Train paired variants resolved to same final camera-light pair after retry: "
                                f"base_latent_id={r.base_latent_id} pair={r.final_camera_light_pair}"
                            )
                r.actual_valid_fraction = float(stats["valid_fraction"])
                r.low_clip_fraction = float(clip["low_clip_fraction"])
                r.high_clip_fraction = float(clip["high_clip_fraction"])
                r.total_clip_fraction = float(clip["total_clip_fraction"])
                r.per_channel_clip_fraction = json.dumps(clip["per_channel_clip_fraction"])
                r.retry_count = retry_count
                r.generation_status = status
                r.rgb_min = float(linear.min())
                r.rgb_max = float(linear.max())
                r.rgb_mean = float(linear.mean())
                r.rgb_std = float(linear.std())
                r.m_mean = float(stats["m"].mean)
                r.m_std = float(stats["m"].std)
                r.h_mean = float(stats["h"].mean)
                r.h_std = float(stats["h"].std)
                r.s_mean = float(stats["s"].mean)
                r.s_std = float(stats["s"].std)
                r.p_mean = float(stats["p"].mean)
                r.p_std = float(stats["p"].std)
                r.p_nonzero_fraction = float(stats["p"].nonzero_fraction)
                r.p_blob_count = int(stats["p"].blob_count)
                metadata_by_split[r.split].append(asdict(r))
                if r.sample_id in audit_ids and status == "SUCCESS":
                    audit_baselines[r.sample_id] = {
                        "row": r,
                        "linear_float32": linear.astype(np.float32, copy=True),
                        "target_float16": target.astype(np.float16, copy=True),
                        "mask_uint8": mask.astype(np.uint8, copy=True),
                    }
                if preview_count < 16:
                    write_preview_png(Path(output_root) / "preview_panels" / f"{r.sample_id}.png", linear)
                    preview_count += 1
            selected_indices = [
                idx
                for idx, (r, _mask, _stats) in enumerate(aux)
                if r.sample_id in audit_ids and r.generation_status == "SUCCESS" and int(r.retry_count) == 0
            ]
            if selected_indices:
                replayed, _ = render_with_oom_fallback(renderer, prepared, batch_sizes, cfg.controls.exposure)
                for idx in selected_indices:
                    r, mask, _stats = aux[idx]
                    fields = prepared[idx]
                    baseline = audit_baselines[r.sample_id]
                    replay_linear = srgb_hwc_to_linear_chw(replayed[idx]["srgb_display_clipped"], mask)
                    replay_target = make_target_mhsp_chw(*fields, mask)
                    same_context_results.append(
                        _compare_same_context(
                            r,
                            baseline["linear_float32"],
                            replay_linear,
                            baseline["target_float16"],
                            replay_target.astype(np.float16),
                            baseline["mask_uint8"],
                            mask,
                        )
                    )
        finally:
            renderer.close()
    missing_same_context = [
        r
        for r in rows[:32]
        if r.sample_id in audit_baselines and r.sample_id not in {x["sample_id"] for x in same_context_results}
    ]
    for r in missing_same_context:
        m, h, s, p, mask, _ = _fields_for_row(r, cfg.patch.size, cfg.mask.min_valid_fraction)
        renderer = SO0CameraRenderer(r.camera_name, r.light_name, device=device, dtype=torch.float32)
        try:
            out1 = renderer.render_batch(m[None], h[None], s[None], p[None], exposure=cfg.controls.exposure)
            out2 = renderer.render_batch(m[None], h[None], s[None], p[None], exposure=cfg.controls.exposure)
        finally:
            renderer.close()
        linear1 = srgb_hwc_to_linear_chw(out1["srgb_display_clipped"][0], mask)
        linear2 = srgb_hwc_to_linear_chw(out2["srgb_display_clipped"][0], mask)
        target = make_target_mhsp_chw(m, h, s, p, mask)
        same_context_results.append(
            _compare_same_context(
                r,
                linear1,
                linear2,
                target.astype(np.float16),
                target.astype(np.float16),
                mask,
                mask,
            )
        )
    for split, md in metadata_by_split.items():
        write_metadata(output_root, split, md)
    assert_train_final_pair_invariants(rows)
    elapsed = time.perf_counter() - t_all
    timing = {"total_seconds": elapsed, "seconds_per_sample": elapsed / max(1, len(rows)), "pair_timings": timings}
    audit_context = {"baselines": audit_baselines, "same_context": same_context_results}
    return timing, audit_context


def _sha256_array(arr: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _float16_ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float16)
    b = np.asarray(b, dtype=np.float16)
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError("Cannot compute float16 ULP distance for non-finite values")
    if np.any(np.signbit(a)) or np.any(np.signbit(b)):
        raise ValueError("SO-1 RGB float16 ULP audit expects non-negative [0,1] values")
    return np.abs(a.view(np.uint16).astype(np.int32) - b.view(np.uint16).astype(np.int32))


def _channel_error_json(diff: np.ndarray) -> str:
    # diff is [3,H,W] independent - reference.
    payload = []
    for idx, name in enumerate(("R", "G", "B")):
        ch = diff[idx]
        payload.append(
            {
                "channel": name,
                "mean_signed_error": float(ch.mean()),
                "mean_abs_error": float(np.abs(ch).mean()),
                "p95_abs_error": float(np.percentile(np.abs(ch), 95)),
                "max_abs_error": float(np.abs(ch).max()),
                "positive_fraction": float(np.mean(ch > 0)),
                "negative_fraction": float(np.mean(ch < 0)),
            }
        )
    return json.dumps(payload, sort_keys=True)


def _systematic_bias(diff: np.ndarray) -> str:
    mean_signed = float(diff.mean())
    pos = float(np.mean(diff > 0))
    neg = float(np.mean(diff < 0))
    if abs(mean_signed) <= 1e-8 and abs(pos - neg) <= 0.05:
        return "none_detected"
    return "positive_bias" if mean_signed > 0 else "negative_bias"


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_file_integrity_hashes(output_root: Path, config_path: str | Path) -> list[dict]:
    rows: list[dict] = []
    split_files = [
        "linear_rgb.f16.npy",
        "target_mhsp.f16.npy",
        "valid_mask.u8.npy",
        "written.u8.npy",
        "metadata.csv",
    ]
    for path in [Path(config_path), output_root / "synthetic_manifest.csv", output_root / "synthetic_manifest.jsonl", output_root / "camera_light_split.json"]:
        if path.exists():
            rows.append({"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    for split_dir in sorted(p for p in output_root.iterdir() if p.is_dir()):
        if split_dir.name == "preview_panels":
            continue
        for name in split_files:
            path = split_dir / name
            if path.exists():
                rows.append({"path": str(path), "sha256": _sha256_file(path), "bytes": path.stat().st_size})
    _write_csv(output_root / "file_integrity_sha256.csv", rows)
    _write_csv(output_root / "reproducibility_hashes.csv", rows)
    return rows


def _compare_same_context(
    row: ManifestRow,
    reference_linear: np.ndarray,
    replay_linear: np.ndarray,
    reference_target: np.ndarray,
    replay_target: np.ndarray,
    reference_mask: np.ndarray,
    replay_mask: np.ndarray,
) -> dict:
    rgb_float32_exact = bool(np.array_equal(reference_linear.astype(np.float32), replay_linear.astype(np.float32)))
    rgb_float16_exact = bool(np.array_equal(reference_linear.astype(np.float16), replay_linear.astype(np.float16)))
    target_exact = bool(np.array_equal(reference_target.astype(np.float16), replay_target.astype(np.float16)))
    mask_exact = bool(np.array_equal(reference_mask.astype(np.uint8), replay_mask.astype(np.uint8)))
    reasons = []
    if not rgb_float32_exact:
        reasons.append("same_context_rgb_float32_not_exact")
    if not rgb_float16_exact:
        reasons.append("same_context_rgb_float16_not_exact")
    if not target_exact:
        reasons.append("same_context_target_not_exact")
    if not mask_exact:
        reasons.append("same_context_mask_not_exact")
    return {
        "sample_id": row.sample_id,
        "split": row.split,
        "split_index": row.split_index,
        "camera_light_pair": row.camera_light_pair,
        "same_context_rgb_exact": rgb_float32_exact and rgb_float16_exact,
        "same_context_rgb_float32_exact": rgb_float32_exact,
        "same_context_rgb_float16_exact": rgb_float16_exact,
        "same_context_target_exact": target_exact,
        "same_context_mask_exact": mask_exact,
        "same_context_rgb_max_abs_float32": float(np.max(np.abs(reference_linear.astype(np.float32) - replay_linear.astype(np.float32)))),
        "same_context_rgb_max_abs_float16": float(np.max(np.abs(reference_linear.astype(np.float16).astype(np.float32) - replay_linear.astype(np.float16).astype(np.float32)))),
        "status": "PASS" if not reasons else "FAIL",
        "failure_reason": ";".join(reasons),
    }


def run_reproducibility_audit(
    cfg,
    rows: list[ManifestRow],
    output_root: str | Path,
    device: str,
    audit_context: dict,
    n: int = 32,
    audit_sample_ids: list[str] | None = None,
) -> dict:
    """Run same-context exact and independent-regeneration tolerance audits."""

    root = Path(output_root)
    success_rows = [r for r in rows if r.generation_status == "SUCCESS"]
    if audit_sample_ids:
        by_id = {r.sample_id: r for r in success_rows}
        missing = [sid for sid in audit_sample_ids if sid not in by_id]
        if missing:
            raise RuntimeError(f"Requested reproducibility audit sample_ids are missing or not SUCCESS: {missing}")
        selected = [by_id[sid] for sid in audit_sample_ids]
        selected.extend([r for r in success_rows if r.sample_id not in set(audit_sample_ids)])
        selected = selected[: min(n, len(selected))]
    else:
        selected = success_rows[: min(n, len(rows))]
    baselines: dict[str, dict] = audit_context.get("baselines", {})
    same_context_rows: list[dict] = list(audit_context.get("same_context", []))
    same_by_id = {r["sample_id"]: r for r in same_context_rows}
    independent_rows: list[dict] = []

    for r in selected:
        if r.sample_id not in baselines:
            raise RuntimeError(f"Missing reproducibility baseline for {r.sample_id}")
        baseline = baselines[r.sample_id]
        m, h, s, p, mask, _ = _fields_for_row(r, cfg.patch.size, cfg.mask.min_valid_fraction)
        target = make_target_mhsp_chw(m, h, s, p, mask).astype(np.float16)
        disk_rgb = np.asarray(np.load(root / r.split / "linear_rgb.f16.npy", mmap_mode="r")[r.split_index])
        disk_target = np.asarray(np.load(root / r.split / "target_mhsp.f16.npy", mmap_mode="r")[r.split_index])
        disk_mask = np.asarray(np.load(root / r.split / "valid_mask.u8.npy", mmap_mode="r")[r.split_index, 0])
        target_exact = bool(np.array_equal(target, disk_target))
        mask_exact = bool(np.array_equal(mask.astype(np.uint8), disk_mask))

        renderer = SO0CameraRenderer(r.camera_name, r.light_name, device=device, dtype=torch.float32)
        try:
            out = renderer.render_batch(m[None], h[None], s[None], p[None], exposure=cfg.controls.exposure)
        finally:
            renderer.close()
        independent_linear = srgb_hwc_to_linear_chw(out["srgb_display_clipped"][0], mask).astype(np.float32)
        baseline_linear = np.asarray(baseline["linear_float32"], dtype=np.float32)
        independent_f16 = independent_linear.astype(np.float16)
        diff32 = independent_linear - baseline_linear
        abs32 = np.abs(diff32)
        diff16 = independent_f16.astype(np.float32) - disk_rgb.astype(np.float32)
        abs16 = np.abs(diff16)
        ulp = _float16_ulp_distance(independent_f16, disk_rgb)
        f32_allclose = bool(np.allclose(independent_linear, baseline_linear, rtol=1e-6, atol=1e-6))
        f32_max_abs = float(abs32.max())
        f16_max_abs = float(abs16.max())
        max_ulp = int(ulp.max())
        mismatch_count = int(np.count_nonzero(independent_f16 != disk_rgb))
        mismatch_fraction = float(mismatch_count / independent_f16.size)
        reasons = []
        if not target_exact:
            reasons.append("independent_target_not_exact")
        if not mask_exact:
            reasons.append("independent_mask_not_exact")
        if not f32_allclose or f32_max_abs > 1e-6:
            reasons.append("independent_rgb_float32_tolerance_failed")
        if max_ulp > 1:
            reasons.append("independent_rgb_float16_ulp_gt_1")
        if f16_max_abs > 5e-4:
            reasons.append("independent_rgb_float16_abs_gt_5e-4")
        same = same_by_id.get(r.sample_id)
        if same is None or same["status"] != "PASS":
            reasons.append("same_context_replay_failed_or_missing")
        independent_rows.append(
            {
                "sample_id": r.sample_id,
                "split": r.split,
                "base_latent_id": r.base_latent_id,
                "acquisition_variant_id": r.acquisition_variant_id,
                "m_seed": r.m_seed,
                "h_seed": r.h_seed,
                "s_seed": r.s_seed,
                "p_seed": r.p_seed,
                "mask_seed": r.mask_seed,
                "acquisition_seed": r.acquisition_seed,
                "camera_light_seed": r.camera_light_seed,
                "camera_name": r.camera_name,
                "light_name": r.light_name,
                "exposure": r.exposure,
                "same_context_rgb_exact": bool(same and same["same_context_rgb_exact"]),
                "independent_target_exact": target_exact,
                "independent_mask_exact": mask_exact,
                "independent_rgb_float32_allclose": f32_allclose,
                "independent_rgb_max_abs_float32": f32_max_abs,
                "independent_rgb_max_abs_float16": f16_max_abs,
                "independent_rgb_mean_abs": float(abs16.mean()),
                "independent_rgb_p95_abs": float(np.percentile(abs16, 95)),
                "independent_rgb_max_ulp": max_ulp,
                "independent_rgb_mismatch_count": mismatch_count,
                "independent_rgb_mismatch_fraction": mismatch_fraction,
                "independent_rgb_channel_errors": _channel_error_json(diff16),
                "independent_rgb_systematic_bias": _systematic_bias(diff16),
                "status": "PASS" if not reasons else "FAIL",
                "failure_reason": ";".join(reasons),
            }
        )

    _write_csv(root / "same_context_replay_audit.csv", same_context_rows)
    _write_csv(root / "reproducibility_audit.csv", independent_rows)
    failures = [r for r in independent_rows if r["status"] != "PASS"]
    if failures:
        raise RuntimeError(f"Reproducibility audit failed for {len(failures)} samples")
    return {
        "same_context": same_context_rows,
        "independent": independent_rows,
        "summary": {
            "same_context_count": len(same_context_rows),
            "same_context_pass": sum(1 for r in same_context_rows if r["status"] == "PASS"),
            "independent_count": len(independent_rows),
            "independent_pass": sum(1 for r in independent_rows if r["status"] == "PASS"),
            "independent_target_exact": sum(1 for r in independent_rows if r["independent_target_exact"]),
            "independent_mask_exact": sum(1 for r in independent_rows if r["independent_mask_exact"]),
            "independent_rgb_float32_max_abs": max((float(r["independent_rgb_max_abs_float32"]) for r in independent_rows), default=0.0),
            "independent_rgb_float16_max_abs": max((float(r["independent_rgb_max_abs_float16"]) for r in independent_rows), default=0.0),
            "independent_rgb_max_ulp": max((int(r["independent_rgb_max_ulp"]) for r in independent_rows), default=0),
            "independent_rgb_mismatch_count": sum(int(r["independent_rgb_mismatch_count"]) for r in independent_rows),
            "independent_rgb_mismatch_fraction": float(
                sum(int(r["independent_rgb_mismatch_count"]) for r in independent_rows)
                / max(1, len(independent_rows) * 3 * cfg.patch.size * cfg.patch.size)
            ),
        },
    }


def _coerce_manifest_value(name: str, value: str, fallback):
    if value == "":
        return fallback
    if isinstance(fallback, bool):
        return str(value).lower() in {"1", "true", "yes"}
    if isinstance(fallback, int) and not isinstance(fallback, bool):
        return int(value)
    if isinstance(fallback, float):
        return float(value)
    return value


def _load_existing_manifest_rows(path: Path, template_rows: list[ManifestRow]) -> list[ManifestRow]:
    template_by_id = {r.sample_id: r for r in template_rows}
    result: list[ManifestRow] = []
    field_names = [f.name for f in dataclass_fields(ManifestRow)]
    with path.open("r", encoding="utf-8", newline="") as f:
        for raw in csv.DictReader(f):
            sample_id = raw["sample_id"]
            if sample_id not in template_by_id:
                raise ValueError(f"Existing manifest row is not in deterministic manifest: {sample_id}")
            template = template_by_id[sample_id]
            values = {}
            for name in field_names:
                fallback = getattr(template, name)
                values[name] = _coerce_manifest_value(name, raw.get(name, ""), fallback)
            if not values["initial_camera_light_pair"]:
                values["initial_camera_name"] = template.initial_camera_name or template.camera_name
                values["initial_light_name"] = template.initial_light_name or template.light_name
                values["initial_camera_light_pair"] = template.initial_camera_light_pair or template.camera_light_pair
            if not values["final_camera_light_pair"]:
                values["final_camera_name"] = values["camera_name"]
                values["final_light_name"] = values["light_name"]
                values["final_camera_light_pair"] = values["camera_light_pair"]
            result.append(ManifestRow(**values))
    result.sort(key=lambda r: int(r.row_index))
    return result


def _render_row_once(cfg, row: ManifestRow, device: str) -> tuple[dict, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray, dict]:
    m, h, s, p, mask, stats = _fields_for_row(row, cfg.patch.size, cfg.mask.min_valid_fraction)
    renderer = SO0CameraRenderer(row.camera_name, row.light_name, device=device, dtype=torch.float32)
    try:
        out = renderer.render_batch(m[None], h[None], s[None], p[None], exposure=cfg.controls.exposure)
    finally:
        renderer.close()
    return {k: v[0] for k, v in out.items()}, (m, h, s, p), mask, stats


def _finalize_row_metadata(row: ManifestRow, linear: np.ndarray, stats: dict, clip: dict, retry_count: int, status: str) -> None:
    row.actual_valid_fraction = float(stats["valid_fraction"])
    row.low_clip_fraction = float(clip["low_clip_fraction"])
    row.high_clip_fraction = float(clip["high_clip_fraction"])
    row.total_clip_fraction = float(clip["total_clip_fraction"])
    row.per_channel_clip_fraction = json.dumps(clip["per_channel_clip_fraction"])
    row.retry_count = int(retry_count)
    row.generation_status = status
    row.rgb_min = float(linear.min())
    row.rgb_max = float(linear.max())
    row.rgb_mean = float(linear.mean())
    row.rgb_std = float(linear.std())
    row.m_mean = float(stats["m"].mean)
    row.m_std = float(stats["m"].std)
    row.h_mean = float(stats["h"].mean)
    row.h_std = float(stats["h"].std)
    row.s_mean = float(stats["s"].mean)
    row.s_std = float(stats["s"].std)
    row.p_mean = float(stats["p"].mean)
    row.p_std = float(stats["p"].std)
    row.p_nonzero_fraction = float(stats["p"].nonzero_fraction)
    row.p_blob_count = int(stats["p"].blob_count)
    row.final_camera_name = row.camera_name
    row.final_light_name = row.light_name
    row.final_camera_light_pair = row.camera_light_pair
    if not row.retry_pair_history:
        row.retry_pair_history = "[]"


def _row_content_hash(root: Path, split: str, split_index: int) -> str:
    h = hashlib.sha256()
    for name in ("linear_rgb.f16.npy", "target_mhsp.f16.npy", "valid_mask.u8.npy"):
        arr = np.load(root / split / name, mmap_mode="r")
        h.update(name.encode("utf-8"))
        h.update(np.ascontiguousarray(arr[split_index]).tobytes())
    return h.hexdigest()


def _build_repro_audit_context_for_rows(cfg, selected: list[ManifestRow], device: str) -> dict:
    baselines: dict[str, dict] = {}
    same_context: list[dict] = []
    for row in selected:
        m, h, s, p, mask, _ = _fields_for_row(row, cfg.patch.size, cfg.mask.min_valid_fraction)
        renderer = SO0CameraRenderer(row.camera_name, row.light_name, device=device, dtype=torch.float32)
        try:
            out1 = renderer.render_batch(m[None], h[None], s[None], p[None], exposure=cfg.controls.exposure)
            out2 = renderer.render_batch(m[None], h[None], s[None], p[None], exposure=cfg.controls.exposure)
        finally:
            renderer.close()
        linear1 = srgb_hwc_to_linear_chw(out1["srgb_display_clipped"][0], mask).astype(np.float32)
        linear2 = srgb_hwc_to_linear_chw(out2["srgb_display_clipped"][0], mask).astype(np.float32)
        target = make_target_mhsp_chw(m, h, s, p, mask)
        baselines[row.sample_id] = {
            "row": row,
            "linear_float32": linear1.copy(),
            "target_float16": target.astype(np.float16, copy=True),
            "mask_uint8": mask.astype(np.uint8, copy=True),
        }
        same_context.append(
            _compare_same_context(
                row,
                linear1,
                linear2,
                target.astype(np.float16),
                target.astype(np.float16),
                mask,
                mask,
            )
        )
    return {"baselines": baselines, "same_context": same_context}


def _sync_formal_layout(output_root: Path, config_path: str | Path) -> None:
    (output_root / "config").mkdir(exist_ok=True)
    (output_root / "manifests").mkdir(exist_ok=True)
    (output_root / "qc").mkdir(exist_ok=True)
    (output_root / "synthetic_data").mkdir(exist_ok=True)
    shutil.copy2(config_path, output_root / "config" / Path(config_path).name)
    for name in ("synthetic_manifest.csv", "synthetic_manifest.jsonl"):
        shutil.copy2(output_root / name, output_root / "manifests" / name)
    for name in (
        "dataset_statistics.json",
        "pair_coverage.csv",
        "clipping_statistics.csv",
        "reproducibility_hashes.csv",
        "reproducibility_audit.csv",
        "same_context_replay_audit.csv",
        "file_integrity_sha256.csv",
        "repair_reproducibility_audit.csv",
        "repair_unchanged_sample_hash_audit.csv",
        "synthetic_generation_qc_report.md",
    ):
        if (output_root / name).exists():
            shutil.copy2(output_root / name, output_root / "qc" / name)
    if (output_root / "preview_panels").exists() and not (output_root / "qc" / "preview_panels").exists():
        (output_root / "qc" / "preview_panels").symlink_to("../preview_panels")
    for split in ("train", "validation", "id_test", "camera_ood", "light_ood", "joint_ood"):
        link = output_root / "synthetic_data" / split
        if not link.exists():
            link.symlink_to(f"../{split}")
    if (output_root / "run_hashes.json").exists():
        shutil.copy2(output_root / "run_hashes.json", output_root / "run_manifest.json")


def _scan_array_hard_gates(output_root: Path, cfg) -> dict:
    expected_counts = {
        "train": 20000,
        "validation": 2000,
        "id_test": 2000,
        "camera_ood": 1000,
        "light_ood": 1000,
        "joint_ood": 1000,
    }
    expected_shapes = {
        "linear_rgb.f16.npy": lambda n: (n, 3, cfg.patch.size, cfg.patch.size),
        "target_mhsp.f16.npy": lambda n: (n, 4, cfg.patch.size, cfg.patch.size),
        "valid_mask.u8.npy": lambda n: (n, 1, cfg.patch.size, cfg.patch.size),
    }
    expected_dtypes = {
        "linear_rgb.f16.npy": np.float16,
        "target_mhsp.f16.npy": np.float16,
        "valid_mask.u8.npy": np.uint8,
    }
    checks: dict[str, dict] = {}
    for split, n in expected_counts.items():
        written = np.load(output_root / split / "written.u8.npy", mmap_mode="r")
        if written.shape != (n,) or written.dtype != np.uint8 or int(written.sum()) != n or not np.all(written == 1):
            raise RuntimeError(f"{split} written gate failed")
        for name, shape_fn in expected_shapes.items():
            arr = np.load(output_root / split / name, mmap_mode="r")
            if arr.shape != shape_fn(n) or arr.dtype != expected_dtypes[name]:
                raise RuntimeError(f"{split}/{name} shape or dtype gate failed: {arr.shape} {arr.dtype}")
            finite = True
            mn = float("inf")
            mx = float("-inf")
            values_ok = True
            for start in range(0, n, 256):
                block = arr[start : start + 256]
                finite = finite and bool(np.isfinite(block).all())
                mn = min(mn, float(block.min()))
                mx = max(mx, float(block.max()))
                if name == "valid_mask.u8.npy":
                    values_ok = values_ok and set(map(int, np.unique(block).tolist())).issubset({0, 1})
            if not finite:
                raise RuntimeError(f"{split}/{name} finite gate failed")
            if name == "valid_mask.u8.npy":
                if not values_ok:
                    raise RuntimeError(f"{split}/{name} binary mask gate failed")
            elif mn < 0.0 or mx > 1.0:
                raise RuntimeError(f"{split}/{name} range gate failed: {mn} {mx}")
            checks[f"{split}/{name}"] = {"shape": arr.shape, "dtype": str(arr.dtype), "min": mn, "max": mx, "finite": finite}
    return checks


def repair_base_latents(args, cfg, split_cfg, env: dict, disk: dict, split_hash: str) -> None:
    output_root = Path(args.output_root)
    repair_ids = [int(x) for x in args.repair_base_latent_ids.split(",") if x.strip()]
    if not repair_ids:
        raise ValueError("--repair-base-latent-ids was provided but empty")
    if any(x < 0 for x in repair_ids):
        raise ValueError("repair base_latent_id must be non-negative train ids")
    if not (output_root / "synthetic_manifest.csv").exists():
        raise FileNotFoundError("Targeted repair requires an existing completed dataset manifest")

    invalidated = bool(list(output_root.glob("COMPLETED.invalid_before_pair_retry_repair*.json")))
    completed_path = output_root / "COMPLETED.json"
    if completed_path.exists():
        invalid_path = output_root / "COMPLETED.invalid_before_pair_retry_repair.json"
        if invalid_path.exists():
            invalid_path = output_root / f"COMPLETED.invalid_before_pair_retry_repair.{int(time.time())}.json"
        completed_path.rename(invalid_path)
        invalidated = True

    initial_rows = build_manifest(cfg, split_cfg)
    current_rows = _load_existing_manifest_rows(output_root / "synthetic_manifest.csv", initial_rows)
    current_by_sample = {r.sample_id: r for r in current_rows}
    initial_by_base_variant = {
        (int(r.base_latent_id), int(r.acquisition_variant_id)): r
        for r in initial_rows
        if r.split == "train" and int(r.base_latent_id) in set(repair_ids)
    }
    repair_rows = []
    for base_id in repair_ids:
        for variant in (0, 1):
            key = (base_id, variant)
            if key not in initial_by_base_variant:
                raise ValueError(f"Repair target missing from deterministic initial manifest: {key}")
            row = initial_by_base_variant[key]
            row.retry_pair_history = "[]"
            repair_rows.append(row)

    repair_sample_ids = [r.sample_id for r in repair_rows]
    unexpected = [sid for sid in repair_sample_ids if not sid.startswith("train_")]
    if unexpected:
        raise RuntimeError(f"Repair is restricted to train split, got {unexpected}")

    repair_dir = output_root / "repair_audit"
    repair_dir.mkdir(exist_ok=True)
    backup_path = repair_dir / "repair_metadata_before.csv"
    if backup_path.exists():
        with backup_path.open("r", encoding="utf-8", newline="") as f:
            backup_rows = list(csv.DictReader(f))
    else:
        backup_rows = [asdict(current_by_sample[sid]) for sid in repair_sample_ids]
        _write_csv(backup_path, backup_rows)

    stores = prepare_split_storage(output_root, {"train": cfg.counts.split_counts()["train"]}, cfg.patch.size, overwrite=False)
    train_store = stores["train"]
    pre_target = {}
    pre_mask = {}
    for row in repair_rows:
        pre_target[row.sample_id] = np.asarray(train_store["target_mhsp"][row.split_index]).copy()
        pre_mask[row.sample_id] = np.asarray(train_store["valid_mask"][row.split_index, 0]).copy()
        train_store["written"][row.split_index] = np.uint8(0)
    train_store["written"].flush()

    sample_hash_indices = sorted(
        set([0, 1, 2, 333, 999, 10033, 10036, 10669, 10672, 10897, 10900, 18321, 18324, 19999])
        - {int(r.split_index) for r in repair_rows}
    )
    unchanged_before = {str(idx): _row_content_hash(output_root, "train", idx) for idx in sample_hash_indices}

    repaired_audit_rows: list[dict] = []
    final_pair_by_base_variant: dict[tuple[int, int], str] = {}
    previous_by_sample = {r["sample_id"]: r for r in backup_rows}
    previous_final_by_base_variant = {}
    for r in repair_rows:
        previous = previous_by_sample.get(r.sample_id)
        if previous:
            previous_pair = previous.get("final_camera_light_pair") or previous.get("camera_light_pair")
        else:
            previous_pair = current_by_sample[r.sample_id].final_camera_light_pair or current_by_sample[r.sample_id].camera_light_pair
        previous_final_by_base_variant[(int(r.base_latent_id), int(r.acquisition_variant_id))] = previous_pair

    for row in sorted(repair_rows, key=lambda r: (int(r.base_latent_id), int(r.acquisition_variant_id))):
        sibling_variant = 1 - int(row.acquisition_variant_id)
        sibling_key = (int(row.base_latent_id), sibling_variant)
        sibling_initial = initial_by_base_variant[sibling_key].initial_camera_light_pair
        sibling_previous_final = previous_final_by_base_variant.get(sibling_key)
        sibling_repaired_final = final_pair_by_base_variant.get(sibling_key)
        forbidden = {row.initial_camera_light_pair, sibling_initial}
        if sibling_previous_final:
            forbidden.add(sibling_previous_final)
        if sibling_repaired_final:
            forbidden.add(sibling_repaired_final)

        out, fields, mask, stats = _render_row_once(cfg, row, args.device)
        clip = clipping_stats(out["srgb_unclipped"])
        retry_count = 0
        status = "SUCCESS"
        if not _clip_accepts(cfg, clip):
            retry_out, retry_s, retry_p, retry_ss, retry_ps, retry_count, _, _ = _retry_clipped_sample(
                cfg,
                split_cfg,
                row,
                fields[0],
                fields[1],
                mask,
                args.device,
                forbidden_pairs=forbidden,
            )
            out = retry_out
            fields = (fields[0], fields[1], retry_s, retry_p)
            stats["s"] = retry_ss
            stats["p"] = retry_ps
            clip = clipping_stats(out["srgb_unclipped"])
            status = "SUCCESS" if _clip_accepts(cfg, clip) else "FAILED"
        linear = srgb_hwc_to_linear_chw(out["srgb_display_clipped"], mask)
        target = make_target_mhsp_chw(*fields, mask)
        _finalize_row_metadata(row, linear, stats, clip, retry_count, status)
        if status != "SUCCESS":
            raise RuntimeError(f"Repair target failed clipping gate: {row.sample_id}")
        write_sample(stores, row.split, row.split_index, linear, target, mask, resume=False)
        final_pair_by_base_variant[(int(row.base_latent_id), int(row.acquisition_variant_id))] = row.final_camera_light_pair
        target_disk = np.asarray(train_store["target_mhsp"][row.split_index])
        mask_disk = np.asarray(train_store["valid_mask"][row.split_index, 0])
        mh_exact = bool(np.array_equal(pre_target[row.sample_id][:2], target_disk[:2]))
        mask_exact_old = bool(np.array_equal(pre_mask[row.sample_id], mask_disk))
        repaired_audit_rows.append(
            {
                "sample_id": row.sample_id,
                "base_latent_id": row.base_latent_id,
                "initial_pair": row.initial_camera_light_pair,
                "previous_final_pair": previous_final_by_base_variant[(int(row.base_latent_id), int(row.acquisition_variant_id))],
                "repaired_final_pair": row.final_camera_light_pair,
                "sibling_final_pair": sibling_repaired_final or sibling_previous_final or sibling_initial,
                "retry_count": row.retry_count,
                "mh_exact_vs_pre_repair": mh_exact,
                "mask_exact_vs_pre_repair": mask_exact_old,
                "low_clip_fraction": row.low_clip_fraction,
                "high_clip_fraction": row.high_clip_fraction,
                "total_clip_fraction": row.total_clip_fraction,
                "clipping_status": status,
                "status": "PASS" if mh_exact and mask_exact_old and status == "SUCCESS" else "FAIL",
            }
        )

    repaired_by_id = {r.sample_id: r for r in repair_rows}
    merged_rows = []
    for row in current_rows:
        merged_rows.append(repaired_by_id.get(row.sample_id, row))
    assert_train_final_pair_invariants(merged_rows)
    if any(r.generation_status != "SUCCESS" for r in merged_rows):
        raise RuntimeError("Repair merged manifest contains non-SUCCESS rows")

    unchanged_after = {str(idx): _row_content_hash(output_root, "train", idx) for idx in sample_hash_indices}
    unchanged_ok = unchanged_before == unchanged_after
    if not unchanged_ok:
        raise RuntimeError("Non-repair row sample hash check failed")

    write_manifest(merged_rows, output_root)
    write_metadata(output_root, "train", [asdict(r) for r in merged_rows if r.split == "train"])
    write_qc_outputs(
        output_root,
        merged_rows,
        smoke=False,
        timing={"targeted_pair_retry_repair": True, "repaired_sample_count": len(repair_rows), "repaired_base_latent_ids": repair_ids},
    )
    _write_csv(output_root / "repair_reproducibility_audit.csv", repaired_audit_rows)

    max_repair_split_index = max(int(r.split_index) for r in repair_rows)
    supplemental_rows = [
        r
        for r in merged_rows
        if r.split == "train"
        and int(r.split_index) > max_repair_split_index
        and r.sample_id not in set(repair_sample_ids)
        and r.generation_status == "SUCCESS"
    ][:24]
    if len(supplemental_rows) < 24:
        supplemental_rows.extend(
            [
                r
                for r in merged_rows
                if r.sample_id not in set(repair_sample_ids)
                and r not in supplemental_rows
                and r.generation_status == "SUCCESS"
            ][: 24 - len(supplemental_rows)]
        )
    if len(supplemental_rows) < 24:
        raise RuntimeError("Unable to select 24 supplemental reproducibility audit samples")
    selected_rows = repair_rows + supplemental_rows
    audit_context = _build_repro_audit_context_for_rows(cfg, selected_rows, args.device)
    repro = run_reproducibility_audit(
        cfg,
        merged_rows,
        output_root,
        args.device,
        audit_context=audit_context,
        n=32,
        audit_sample_ids=[r.sample_id for r in selected_rows],
    )
    repro_by_id = {r["sample_id"]: r for r in repro["independent"]}
    for row in repaired_audit_rows:
        rr = repro_by_id[row["sample_id"]]
        row.update(
            {
                "target_exact": rr["independent_target_exact"],
                "mask_exact": rr["independent_mask_exact"],
                "rgb_float32_max_abs": rr["independent_rgb_max_abs_float32"],
                "rgb_float16_max_abs": rr["independent_rgb_max_abs_float16"],
                "rgb_max_ulp": rr["independent_rgb_max_ulp"],
                "status": "PASS"
                if row["status"] == "PASS"
                and rr["status"] == "PASS"
                and row["repaired_final_pair"] != row["sibling_final_pair"]
                else "FAIL",
            }
        )
    if any(r["status"] != "PASS" for r in repaired_audit_rows):
        raise RuntimeError("Repair reproducibility audit failed")
    _write_csv(output_root / "repair_reproducibility_audit.csv", repaired_audit_rows)

    array_gates = _scan_array_hard_gates(output_root, cfg)
    file_hashes = _write_file_integrity_hashes(output_root, args.config)
    _write_csv(output_root / "repair_unchanged_sample_hash_audit.csv", [{"split": "train", "split_index": k, "sha256": v} for k, v in unchanged_after.items()])

    link_targets = {}
    for split in ("train", "validation", "id_test", "camera_ood", "light_ood", "joint_ood"):
        link = output_root / "synthetic_data" / split
        if link.exists():
            link_targets[str(link.relative_to(output_root))] = str(link.resolve())
    run_payload = {
        "generator_config_hash": cfg.canonical_hash(),
        "camera_light_split_hash": split_hash,
        "environment": env,
        "disk": disk,
        "so0_frozen_so1_authorized_note": "FROZEN.json reports so1_authorized=false; SO-0 was not modified.",
        "previous_completion_invalidated": invalidated,
        "repair_reason": "train_variant_final_pair_collision_after_clipping_retry",
        "repair_base_latent_ids": repair_ids,
        "repair_sample_ids": repair_sample_ids,
        "repair_nonrepair_sample_hash_check": unchanged_ok,
        "synthetic_data_symlink_targets": link_targets,
    }
    write_run_hashes(output_root, run_payload)
    summary = repro["summary"]
    with (output_root / "synthetic_generation_qc_report.md").open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            "## Targeted pair-retry repair\n\n"
            f"- previous_completion_invalidated: {invalidated}\n"
            "- repair_reason: train_variant_final_pair_collision_after_clipping_retry\n"
            f"- repaired_base_latent_ids: {repair_ids}\n"
            f"- repaired_sample_count: {len(repair_rows)}\n"
            "- train_final_pair_collisions_after_repair: 0\n\n"
            "## Reproducibility audit\n\n"
            "same_context_replay:\n"
            f"  rgb_bitwise_exact: {'PASS' if summary['same_context_pass'] == summary['same_context_count'] else 'FAIL'}\n"
            f"  target_bitwise_exact: {'PASS' if all(r['same_context_target_exact'] for r in repro['same_context']) else 'FAIL'}\n"
            f"  mask_bitwise_exact: {'PASS' if all(r['same_context_mask_exact'] for r in repro['same_context']) else 'FAIL'}\n\n"
            "independent_regeneration:\n"
            f"  target_exact: {'PASS' if summary['independent_target_exact'] == summary['independent_count'] else 'FAIL'}\n"
            f"  mask_exact: {'PASS' if summary['independent_mask_exact'] == summary['independent_count'] else 'FAIL'}\n"
            f"  rgb_float32_tolerance: {'PASS' if summary['independent_rgb_float32_max_abs'] <= 1e-6 else 'FAIL'}\n"
            f"  rgb_float16_max_ulp: {summary['independent_rgb_max_ulp']}\n"
            f"  rgb_float16_tolerance: {'PASS' if summary['independent_rgb_float16_max_abs'] <= 5e-4 and summary['independent_rgb_max_ulp'] <= 1 else 'FAIL'}\n"
        )
    completed = {
        "status": "COMPLETED",
        "rows": len(merged_rows),
        "smoke": False,
        "full_generation_started": True,
        "previous_completion_invalidated": invalidated,
        "repair_reason": "train_variant_final_pair_collision_after_clipping_retry",
        "repair_base_latent_ids": repair_ids,
        "repair_sample_ids": repair_sample_ids,
        "train_final_pair_collision_count": len(train_final_pair_collisions(merged_rows)),
        "same_context_replay": {
            "count": summary["same_context_count"],
            "pass": summary["same_context_pass"],
            "rgb_bitwise_exact": summary["same_context_pass"] == summary["same_context_count"],
        },
        "independent_regeneration": summary,
        "file_hash_count": len(file_hashes),
        "array_hard_gate_count": len(array_gates),
        "repair_nonrepair_sample_hash_check": unchanged_ok,
        "output_root": str(output_root),
    }
    (output_root / "COMPLETED.json").write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _sync_formal_layout(output_root, args.config)
    print(json.dumps({"status": "REPAIRED", "completed": completed}, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    output_root = Path(args.output_root)
    full_mode = not args.smoke_only and not args.validate_only
    env = preflight_environment()
    disk = check_disk_space(output_root, cfg.total_samples * 7 * cfg.patch.size * cfg.patch.size * 2, full_mode=full_mode)

    ensure_so0_import_path()
    from skin_optics.config import load_so0_config

    so0_cfg = load_so0_config()
    range_notes = assert_so0_range_compatible(cfg, so0_cfg)
    split = build_camera_light_split(cfg.global_seed)
    split_hash = save_camera_light_split(
        split,
        output_root / "camera_light_split.json",
        overwrite=args.overwrite_confirmed or args.validate_only or args.smoke_only,
        allow_existing_identical=args.resume or bool(args.repair_base_latent_ids),
    )
    rows = build_manifest(cfg, split)
    if args.smoke_only:
        rows = _smoke_rows(rows)
    validate_manifest(rows, cfg if not args.smoke_only else _SmokeCountProxy(cfg, rows), split)
    if args.repair_base_latent_ids:
        if args.validate_only or args.smoke_only:
            raise ValueError("--repair-base-latent-ids cannot be combined with validate-only or smoke-only")
        repair_base_latents(args, cfg, split, env, disk, split_hash)
        return
    write_manifest(rows, output_root)
    write_run_hashes(
        output_root,
        {
            "generator_config_hash": cfg.canonical_hash(),
            "camera_light_split_hash": split_hash,
            "environment": env,
            "disk": disk,
            "so0_range_notes": range_notes,
            "so0_frozen_so1_authorized_note": "FROZEN.json reports so1_authorized=false; SO-0 was not modified.",
        },
    )
    if args.validate_only:
        write_qc_outputs(output_root, rows, smoke=False, timing={"validate_only": True})
        print(json.dumps({"status": "VALIDATED", "rows": len(rows), "output_root": str(output_root)}, indent=2))
        return
    timing, audit_context = render_rows(cfg, rows, output_root, args.device, resume=args.resume, overwrite=args.overwrite_confirmed or args.smoke_only, split_cfg=split)
    write_manifest(rows, output_root)
    write_qc_outputs(output_root, rows, smoke=args.smoke_only, timing=timing)
    failures = [r.sample_id for r in rows if r.generation_status != "SUCCESS"]
    if failures:
        raise RuntimeError(f"Generation completed with failed samples: {failures[:20]} total={len(failures)}")
    repro = run_reproducibility_audit(cfg, rows, output_root, args.device, audit_context=audit_context, n=32)
    file_hashes = _write_file_integrity_hashes(output_root, args.config)
    summary = repro["summary"]
    report_path = output_root / "synthetic_generation_qc_report.md"
    with report_path.open("a", encoding="utf-8") as f:
        f.write(
            "\n"
            "## Reproducibility audit\n\n"
            "same_context_replay:\n"
            f"  rgb_bitwise_exact: {'PASS' if summary['same_context_pass'] == summary['same_context_count'] else 'FAIL'}\n"
            f"  target_bitwise_exact: {'PASS' if all(r['same_context_target_exact'] for r in repro['same_context']) else 'FAIL'}\n"
            f"  mask_bitwise_exact: {'PASS' if all(r['same_context_mask_exact'] for r in repro['same_context']) else 'FAIL'}\n\n"
            "independent_regeneration:\n"
            f"  target_exact: {'PASS' if summary['independent_target_exact'] == summary['independent_count'] else 'FAIL'}\n"
            f"  mask_exact: {'PASS' if summary['independent_mask_exact'] == summary['independent_count'] else 'FAIL'}\n"
            f"  rgb_float32_tolerance: {'PASS' if summary['independent_rgb_float32_max_abs'] <= 1e-6 else 'FAIL'}\n"
            f"  rgb_float16_max_ulp: {summary['independent_rgb_max_ulp']}\n"
            f"  rgb_float16_tolerance: {'PASS' if summary['independent_rgb_float16_max_abs'] <= 5e-4 and summary['independent_rgb_max_ulp'] <= 1 else 'FAIL'}\n"
        )
    completed = {
        "status": "COMPLETED",
        "rows": len(rows),
        "smoke": args.smoke_only,
        "full_generation_started": full_mode,
        "same_context_replay": {
            "count": summary["same_context_count"],
            "pass": summary["same_context_pass"],
            "rgb_bitwise_exact": summary["same_context_pass"] == summary["same_context_count"],
        },
        "independent_regeneration": summary,
        "file_hash_count": len(file_hashes),
        "output_root": str(output_root),
    }
    (output_root / "COMPLETED.json").write_text(json.dumps(completed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": "SUCCESS",
                "rows": len(rows),
                "same_context_replay": f"{summary['same_context_pass']}/{summary['same_context_count']}",
                "independent_regeneration": f"{summary['independent_pass']}/{summary['independent_count']}",
                "independent_rgb_float32_max_abs": summary["independent_rgb_float32_max_abs"],
                "independent_rgb_float16_max_abs": summary["independent_rgb_float16_max_abs"],
                "independent_rgb_max_ulp": summary["independent_rgb_max_ulp"],
                "timing": timing,
                "output_root": str(output_root),
            },
            indent=2,
        )
    )


class _SmokeCountProxy:
    """Minimal config proxy so manifest validation can validate selected smoke rows."""

    def __init__(self, cfg, rows: list[ManifestRow]) -> None:
        self.patch = cfg.patch
        self.mask = cfg.mask
        self.controls = cfg.controls
        self.render = cfg.render
        self.storage = cfg.storage
        counts = {s: 0 for s in ("train", "validation", "id_test", "camera_ood", "light_ood", "joint_ood")}
        for r in rows:
            counts[r.split] += 1

        class C:
            train_base_latents = counts["train"] // 2
            train_variants_per_latent = 2

            def split_counts(self_inner):
                return counts

        self.counts = C()

    @property
    def total_samples(self) -> int:
        return sum(self.counts.split_counts().values())


if __name__ == "__main__":
    main()
