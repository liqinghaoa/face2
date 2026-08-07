"""State serialization and resume protection for the P1 component sweep."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from utils.p1_rgb_audit import sha


EXPERIMENT_STATUS_VALUES = (
    "NOT_IMPLEMENTED",
    "FRAMEWORK_VALIDATED",
    "P1_COMPONENT_FRAMEWORK_READY_FOR_PHASE2",
    "PENDING",
    "SMOKE_PASSED",
    "RUNNING",
    "FOLDS_COMPLETE",
    "OOF_COMPLETE",
    "COMPLETED",
    "FAILED",
    "SKIPPED_UNAVAILABLE",
)


@dataclass(frozen=True)
class P1ComponentSweepState:
    experiment_key: str
    status: str
    contract_version: str
    config_sha256: str | None = None
    split_sha256: str | None = None
    manifest_sha256: str | None = None
    approval_sha256: str | None = None
    completed_folds: tuple[int, ...] = ()
    failed_folds: tuple[int, ...] = ()
    oof_rows: int = 0
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_status(self, status: str) -> "P1ComponentSweepState":
        return replace(self, status=status)


def _validate_status(status: str) -> str:
    if status not in EXPERIMENT_STATUS_VALUES:
        raise ValueError(f"invalid experiment status: {status!r}")
    return status


def save_state(state: P1ComponentSweepState, path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = state.to_dict()
    _validate_status(payload["status"])
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output


def load_state(path: str | Path) -> P1ComponentSweepState:
    state_path = Path(path)
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    _validate_status(str(payload["status"]))
    return P1ComponentSweepState(
        experiment_key=str(payload["experiment_key"]),
        status=str(payload["status"]),
        contract_version=str(payload["contract_version"]),
        config_sha256=payload.get("config_sha256"),
        split_sha256=payload.get("split_sha256"),
        manifest_sha256=payload.get("manifest_sha256"),
        approval_sha256=payload.get("approval_sha256"),
        completed_folds=tuple(int(x) for x in payload.get("completed_folds", [])),
        failed_folds=tuple(int(x) for x in payload.get("failed_folds", [])),
        oof_rows=int(payload.get("oof_rows", 0)),
        note=str(payload.get("note", "")),
    )


def hash_file(path: str | Path) -> str:
    return sha(Path(path))


def validate_resume_contract(
    state: P1ComponentSweepState,
    *,
    current_config_sha256: str | None = None,
    current_split_sha256: str | None = None,
    current_manifest_sha256: str | None = None,
    current_contract_version: str | None = None,
) -> None:
    if current_contract_version is not None and state.contract_version != current_contract_version:
        raise ValueError("code protocol version changed")
    if current_config_sha256 is not None and state.config_sha256 not in {None, current_config_sha256}:
        raise ValueError("config sha256 mismatch")
    if current_split_sha256 is not None and state.split_sha256 not in {None, current_split_sha256}:
        raise ValueError("split sha256 mismatch")
    if current_manifest_sha256 is not None and state.manifest_sha256 not in {None, current_manifest_sha256}:
        raise ValueError("manifest sha256 mismatch")


def state_from_status(
    experiment_key: str,
    status: str,
    contract_version: str,
    **kwargs: Any,
) -> P1ComponentSweepState:
    _validate_status(status)
    return P1ComponentSweepState(
        experiment_key=experiment_key,
        status=status,
        contract_version=contract_version,
        **kwargs,
    )
