"""S1-4 model-contract audit and immutable artifact writer."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from .model_registry import DEFAULT_MODEL_REGISTRY_PATH, Stage1ModelRegistry, load_model_registry
from .s1_spectral_sensitivity import band_mask, gaussian_srf_reflectance
from .skin_forward import RegisteredSkinForwardModel


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _require_new_directory(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty S1-4 output: {path}")
    path.mkdir(parents=True, exist_ok=True)


def _split_reference(model: RegisteredSkinForwardModel) -> tuple[np.ndarray | None, np.ndarray | None]:
    bio = (
        np.asarray([model.registry.parameter(name).reference_value for name in model.bio_parameter_names])
        if model.bio_parameter_names
        else None
    )
    nuisance = (
        np.asarray([model.registry.parameter(name).reference_value for name in model.nuisance_parameter_names])
        if model.nuisance_parameter_names
        else None
    )
    return bio, nuisance


def _gradient_audit(model: RegisteredSkinForwardModel, wavelength: np.ndarray) -> dict[str, Any]:
    theta = np.asarray(model.registry.reference_values_for(model.model_id), dtype=np.float64)
    theta_t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    prediction_t = model.forward_flat_torch(theta_t, wavelength)
    torch.sum(prediction_t).backward()
    if theta_t.grad is None:
        raise RuntimeError(f"Missing Torch gradient for {model.model_id}")
    analytic = theta_t.grad.detach().cpu().numpy()
    numeric = np.zeros_like(theta)
    for index, (lower, upper) in enumerate(model.bounds):
        step = (upper - lower) * 1e-6
        lo = theta.copy()
        hi = theta.copy()
        lo[index] -= step
        hi[index] += step
        numeric[index] = (
            np.sum(model.forward_flat_numpy(hi, wavelength))
            - np.sum(model.forward_flat_numpy(lo, wavelength))
        ) / (2.0 * step)
    absolute = np.abs(analytic - numeric)
    relative = absolute / np.maximum(np.maximum(np.abs(analytic), np.abs(numeric)), 1e-12)
    return {
        "parameter_names": list(model.parameter_names),
        "torch_gradient": analytic.tolist(),
        "finite_difference_gradient": numeric.tolist(),
        "max_absolute_error": float(np.max(absolute)),
        "max_relative_error": float(np.max(relative)),
        "pass": bool(np.all(np.isfinite(analytic)) and np.max(relative) <= 2e-5),
    }


def _physical_model_audit(model: RegisteredSkinForwardModel, wavelength: np.ndarray) -> dict[str, Any]:
    bio, nuisance = _split_reference(model)
    numpy_prediction, components = model.forward_numpy(
        bio, wavelength, nuisance, return_components=True
    )
    bio_t = None if bio is None else torch.tensor(bio, dtype=torch.float64, requires_grad=True)
    nuisance_t = None if nuisance is None else torch.tensor(nuisance, dtype=torch.float64, requires_grad=True)
    torch_prediction = model.forward_torch(bio_t, wavelength, nuisance_t)
    parity = float(np.max(np.abs(numpy_prediction - torch_prediction.detach().cpu().numpy())))

    lower = np.asarray([bound[0] for bound in model.bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in model.bounds], dtype=np.float64)
    lower_prediction = np.asarray(model.forward_flat_numpy(lower, wavelength))
    upper_prediction = np.asarray(model.forward_flat_numpy(upper, wavelength))
    component_balance = float(
        np.max(
            np.abs(
                components.total_dermal_absorption_mm1
                - components.baseline_absorption_mm1
                - components.hemoglobin_absorption_mm1
            )
        )
    )
    gradient = _gradient_audit(model, wavelength)
    finite_and_nonnegative = bool(
        np.all(np.isfinite(numpy_prediction))
        and np.all(numpy_prediction >= 0)
        and np.all(np.isfinite(lower_prediction))
        and np.all(lower_prediction >= 0)
        and np.all(np.isfinite(upper_prediction))
        and np.all(upper_prediction >= 0)
    )
    passed = finite_and_nonnegative and parity <= 1e-10 and component_balance <= 1e-12 and gradient["pass"]
    return {
        "model_id": model.model_id,
        "theta_bio": list(model.bio_parameter_names),
        "theta_nuisance": list(model.nuisance_parameter_names),
        "status": model.model_spec.status,
        "activation": model.model_spec.activation,
        "reference_reflectance_min": float(np.min(numpy_prediction)),
        "reference_reflectance_max": float(np.max(numpy_prediction)),
        "numpy_torch_max_abs_error": parity,
        "component_balance_max_abs_error": component_balance,
        "boundary_outputs_finite_nonnegative": finite_and_nonnegative,
        "gradient": gradient,
        "pass": passed,
    }


def _b0_audit(model: RegisteredSkinForwardModel, wavelength: np.ndarray) -> dict[str, Any]:
    reference = np.linspace(0.2, 0.6, len(wavelength), dtype=np.float64)
    globals_ = {"train_mean_reflectance": reference}
    numpy_prediction, components = model.forward_numpy(
        None, wavelength, global_params=globals_, return_components=True
    )
    torch_prediction = model.forward_torch(None, wavelength, global_params=globals_)
    parity = float(np.max(np.abs(numpy_prediction - torch_prediction.detach().cpu().numpy())))
    requires_reference = False
    try:
        model.forward_numpy(None, wavelength)
    except ValueError as exc:
        requires_reference = "Train-only" in str(exc)
    passed = bool(
        requires_reference
        and components.physical_model_applied is False
        and np.array_equal(numpy_prediction, reference)
        and parity <= 1e-12
    )
    return {
        "model_id": "B0",
        "theta_bio": [],
        "theta_nuisance": [],
        "status": model.model_spec.status,
        "activation": model.model_spec.activation,
        "requires_explicit_train_only_reference": requires_reference,
        "numpy_torch_max_abs_error": parity,
        "pass": passed,
    }


def _model_cards(registry: Stage1ModelRegistry, audits: list[dict[str, Any]]) -> str:
    audit_by_id = {row["model_id"]: row for row in audits}
    lines = [
        "# S1-4 候选模型卡",
        "",
        "这些模型用于真实 HSI 上的候选筛查，不是 Jonasson 三层 inverse Monte Carlo 的复现，也不输出绝对生理浓度。",
        "",
        "## 共同物理骨架",
        "",
        "- 黑色素形状：`m(λ)=(λ/570)^(-4.3)`；表皮双程传输为 `exp[-2 M m(λ)]`。",
        "- 血红蛋白形状：审计过的 OMLC HbO2/Hb 光谱按 `sO2` 混合，并在 570 nm 归一化。",
        "- 基线吸收：`[0.244+85.3 exp(-(λ-154)/66.2)]/10 mm^-1`。",
        "- 约化散射：`S_amp[(1-γ)(λ/600)^(-β)+γ(λ/600)^(-4)]`，默认 `S_amp=1.99 mm^-1, β=0.82, γ=0.31`。",
        "- 真皮反射：半无限 Kubelka–Munk 近似；最终反射为表皮传输乘真皮反射。",
        "- 不允许逐图自由曝光尺度；左右侧接口是 Train-only 一次估计后冻结的全局 log-gain，默认均为 0。",
        "",
        "## 注册模型",
        "",
        "| ID | theta_bio | theta_nuisance | 当前状态 | 激活规则 | 单测 |",
        "|---|---|---|---|---|---|",
    ]
    for model_id, model in registry.models.items():
        audit = audit_by_id[model_id]
        bio = ", ".join(model.theta_bio) or "无"
        nuisance = ", ".join(model.theta_nuisance) or "无"
        lines.append(
            f"| `{model_id}` | {bio} | {nuisance} | `{model.status}` | `{model.activation}` | "
            f"{'PASS' if audit['pass'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            "`B0` 的 Train 平均光谱不在 S1-4 读取或计算；由 S1-5 仅从 Train 主域生成并哈希。",
            "`P3-O/P4` 虽已具备可测试代码路径，但当前不激活；只有 S1-5 Train 残差满足注册触发条件才可运行。",
            "",
            "## 参数证据与限制",
            "",
            "M、H、S_amp 和 sO2 的范围都是开发期宽范围，不是 Hyper-Skin 个体真值。Jonasson 研究使用前臂、双源探距离和三层 inverse Monte Carlo；Hyper-Skin 是单视角人脸 HSI，因此只借用谱形、量级和参数分层，不声称方法等价。",
            "发布版 31 波段的有效 SRF 仍缺失。S1-5 必须运行预注册的全 31 波段、去端点、去高曲率候选、中心偏移和假定 Gaussian FWHM 敏感性；这些设置不能由 Validation/Test 新增。",
            "",
            "## 参考资料",
            "",
        ]
    )
    lines.extend(f"- {reference}" for reference in registry.raw["references"])
    lines.append("")
    return "\n".join(lines)


def build_s1_4_model_audit(
    s1_3_final_decision_path: str | Path,
    data_contract_path: str | Path,
    output_dir: str | Path,
    registry_path: str | Path = DEFAULT_MODEL_REGISTRY_PATH,
    supersedes_decision_path: str | Path | None = None,
) -> dict[str, Any]:
    """Audit all registered models without opening Train/Validation/Test spectra."""

    s1_3_path = Path(s1_3_final_decision_path).resolve()
    contract_path = Path(data_contract_path).resolve()
    destination = Path(output_dir).resolve()
    supersedes_path = Path(supersedes_decision_path).resolve() if supersedes_decision_path is not None else None
    registry = load_model_registry(registry_path)
    s1_3 = _read_json(s1_3_path)
    contract = _read_json(contract_path)
    if s1_3.get("status") != "PASS_FOR_S1_4" or not bool(s1_3.get("next_stage_allowed")):
        raise RuntimeError("S1-4 is not authorized by the S1-3 final decision")
    if s1_3.get("authorized_next_stage") != "S1-4" or int(s1_3.get("test_access_count", -1)) != 0:
        raise RuntimeError("S1-3 authorization or Test-isolation gate failed")
    centers = contract["wavelength"]["centers_nm"]
    if centers["status"] != "confirmed_from_official_code":
        raise RuntimeError("S1-4 requires officially confirmed wavelength centers")
    if tuple(float(value) for value in centers["value"]) != registry.wavelength_nm:
        raise RuntimeError("Registry wavelengths do not match the frozen data contract")
    if contract["wavelength"]["release_effective_bandwidth_or_srf"]["status"] != "missing":
        raise RuntimeError("S1-4 must retain the effective-SRF evidence state")
    if supersedes_path is not None and not supersedes_path.is_file():
        raise FileNotFoundError(f"Missing superseded S1-4 decision: {supersedes_path}")

    wavelength = np.asarray(registry.wavelength_nm, dtype=np.float64)
    audits: list[dict[str, Any]] = []
    for model_id in registry.models:
        model = RegisteredSkinForwardModel(model_id, registry)
        audits.append(_b0_audit(model, wavelength) if model_id == "B0" else _physical_model_audit(model, wavelength))

    p2 = RegisteredSkinForwardModel("P2", registry)
    p2_reference = np.asarray(registry.reference_values_for("P2"), dtype=np.float64)
    band_sensitivity = {
        name: int(band_mask(wavelength, name, registry).sum())
        for name in registry.raw["sensitivity"]["band_sets"]
    }
    srf_sensitivity = {
        str(float(fwhm)): bool(
            np.all(
                np.isfinite(
                    gaussian_srf_reflectance(p2, p2_reference, wavelength, float(fwhm))
                )
            )
        )
        for fwhm in registry.raw["sensitivity"]["assumed_gaussian_fwhm_nm"]
    }
    sensitivity_pass = (
        band_sensitivity == {"full_31": 31, "remove_endpoints": 29, "remove_high_curvature_candidates": 24}
        and all(srf_sensitivity.values())
    )
    forbidden_scale_rejected = False
    try:
        p2.forward_flat_numpy(p2_reference, wavelength, global_params={"per_image_scale": 1.1})
    except ValueError as exc:
        forbidden_scale_rejected = "forbidden" in str(exc)

    all_models_pass = all(bool(row["pass"]) for row in audits)
    final_pass = all_models_pass and sensitivity_pass and forbidden_scale_rejected
    _require_new_directory(destination)

    parameter_path = destination / "parameter_contract.yaml"
    with parameter_path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(registry.parameter_contract_payload(), stream, sort_keys=False, allow_unicode=True)
    registry_output_path = destination / "model_registry.json"
    _write_json(registry_output_path, registry.model_registry_payload())
    audit_path = destination / "model_unit_audit.json"
    audit_payload = {
        "schema_version": 1,
        "stage": "S1-4",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "models": audits,
        "band_sensitivity_selected_counts": band_sensitivity,
        "assumed_gaussian_fwhm_outputs_finite": srf_sensitivity,
        "forbidden_per_image_scale_rejected": forbidden_scale_rejected,
        "test_access_count": 0,
        "real_region_spectrum_access_count": 0,
        "pass": final_pass,
    }
    _write_json(audit_path, audit_payload)
    cards_path = destination / "MODEL_CARDS.md"
    cards_path.write_text(_model_cards(registry, audits), encoding="utf-8", newline="\n")

    output_files = {
        "parameter_contract": parameter_path,
        "model_registry": registry_output_path,
        "model_unit_audit": audit_path,
        "model_cards": cards_path,
    }
    decision = {
        "schema_version": 1,
        "stage": "S1-4",
        "status": "PASS_FOR_S1_5" if final_pass else "STOP_MODEL_IMPLEMENTATION",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "s1_3_final_decision_path": str(s1_3_path),
        "s1_3_final_decision_sha256": _sha256(s1_3_path),
        "data_contract_path": str(contract_path),
        "data_contract_sha256": _sha256(contract_path),
        "source_registry_path": str(registry.path),
        "source_registry_sha256": _sha256(registry.path),
        "supersedes": (
            {
                "decision_path": str(supersedes_path),
                "decision_sha256": _sha256(supersedes_path),
                "reason": "YAML_reference_with_colon_was_parsed_as_a_mapping;_numeric_model_audit_unchanged",
            }
            if supersedes_path is not None
            else None
        ),
        "model_ids": list(registry.models),
        "activated_for_s1_5_entry": ["B0", "B1", "P2", "P3-S"],
        "conditional_not_activated": ["P3-O", "P4"],
        "effective_srf_status": "missing",
        "effective_srf_policy": "pre_registered_sensitivity_required_before_formal_test",
        "side_nuisance_policy": "separate_and_symmetric_reporting;_optional_gain_train_only_global_then_frozen",
        "per_spectrum_exposure_scale_allowed": False,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in output_files.items()
        },
        "automatic_checks": {
            "all_registered_models_pass": all_models_pass,
            "sensitivity_interfaces_pass": sensitivity_pass,
            "forbidden_per_image_scale_rejected": forbidden_scale_rejected,
            "official_wavelength_contract_matches": True,
            "effective_srf_missing_retained": True,
            "test_access_count_zero": True,
            "real_region_spectrum_access_count_zero": True,
        },
        "next_stage_allowed": final_pass,
        "authorized_next_stage": "S1-5" if final_pass else None,
        "test_access_count": 0,
    }
    _write_json(destination / "s1_4_decision.json", decision)
    return decision
