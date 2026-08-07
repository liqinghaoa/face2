from __future__ import annotations

import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, Iterable


DEFAULT_KNOWN_PROJECT_ROOTS = (
    "/mnt/e/projects/face2",
    "E:/projects/face2",
    r"E:\projects\face2",
)


def _as_posix_text(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def _looks_like_windows_absolute(path: str) -> bool:
    return len(path) >= 3 and path[1] == ":" and path[2] in {"/", "\\"}


def _wsl_drive_to_windows(path: str) -> str | None:
    normalized = path.replace("\\", "/")
    parts = PurePosixPath(normalized).parts
    if len(parts) >= 3 and parts[0] == "/" and parts[1] == "mnt" and len(parts[2]) == 1:
        drive = parts[2].upper()
        rest = "/".join(parts[3:])
        return f"{drive}:/{rest}" if rest else f"{drive}:/"
    return None


def _relative_to_known_root(stored: str, known_roots: Iterable[str | Path]) -> str | None:
    stored_norm = _as_posix_text(stored)
    stored_lower = stored_norm.lower()
    for root in known_roots:
        root_norm = _as_posix_text(root)
        root_lower = root_norm.lower()
        if stored_lower == root_lower:
            return ""
        if stored_lower.startswith(root_lower + "/"):
            return stored_norm[len(root_norm) + 1 :]
    return None


def resolve_project_path(
    stored_path: str | Path,
    project_root: str | Path,
    *,
    known_project_roots: Iterable[str | Path] | None = None,
    require_exists: bool = False,
    case_id: str | None = None,
    column_name: str | None = None,
) -> Path:
    """Resolve manifest-stored project paths without rewriting the manifest.

    Supports WSL paths, Windows paths, and project-relative paths. Known old project
    roots are mapped to the supplied current ``project_root`` only by exact prefix.
    """

    stored_text = str(stored_path)
    roots = tuple(known_project_roots or DEFAULT_KNOWN_PROJECT_ROOTS)
    project = Path(project_root).expanduser().resolve()
    relative = _relative_to_known_root(stored_text, roots)
    if relative is not None:
        resolved = project / PurePosixPath(relative)
    else:
        wsl_as_windows = _wsl_drive_to_windows(stored_text)
        if wsl_as_windows is not None:
            resolved = Path(wsl_as_windows)
        elif _looks_like_windows_absolute(stored_text):
            resolved = Path(PureWindowsPath(stored_text))
        else:
            path = Path(stored_text).expanduser()
            resolved = path if path.is_absolute() else project / path
    resolved = resolved.resolve()
    if require_exists and not resolved.exists():
        raise FileNotFoundError(
            "P2 manifest path could not be resolved to an existing file: "
            f"case_id={case_id!r}, column_name={column_name!r}, "
            f"stored_path={stored_text!r}, resolved_path={str(resolved)!r}"
        )
    return resolved


def normalize_path_for_comparison(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
    known_project_roots: Iterable[str | Path] | None = None,
) -> str:
    """Normalize only for path-tail comparisons, not for filesystem access."""

    roots = tuple(known_project_roots or DEFAULT_KNOWN_PROJECT_ROOTS)
    if project_root is not None:
        roots = (str(project_root), *roots)
    relative = _relative_to_known_root(str(path), roots)
    if relative is not None:
        return str(PurePosixPath(relative)).lower()
    wsl_as_windows = _wsl_drive_to_windows(str(path))
    text = _as_posix_text(wsl_as_windows or path)
    if _looks_like_windows_absolute(text):
        win = PureWindowsPath(text)
        return win.as_posix().lower()
    return text.lower()


def path_resolution_options(config: dict[str, Any] | None) -> dict[str, Any]:
    cfg = dict(config or {})
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "known_project_roots": tuple(cfg.get("known_project_roots", DEFAULT_KNOWN_PROJECT_ROOTS)),
        "require_exists": bool(cfg.get("require_exists", True)),
        "rewrite_manifest": bool(cfg.get("rewrite_manifest", False)),
    }
