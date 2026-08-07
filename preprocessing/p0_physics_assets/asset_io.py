"""Atomic raster/metadata writing and byte-preserving source copies."""

from __future__ import annotations

import csv, io, os, shutil, tempfile
from pathlib import Path
from typing import Any

import numpy as np


def asset_index(directory: Path, suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")) -> dict[str, Path]:
    """Index common raster extensions by stem, refusing ambiguous duplicate stems."""
    found: dict[str, Path] = {}
    for path in directory.iterdir():
        if path.is_file() and path.suffix in suffixes:
            if path.stem in found: raise ValueError(f"ambiguous asset ID {path.stem} in {directory}")
            found[path.stem] = path
    return found


def verify_assets(ids: set[str], directories: dict[str, Path]) -> dict[str, dict[str, Path]]:
    """Require every split ID in every asset input; extras are permitted."""
    indexed = {name: asset_index(path) for name, path in directories.items()}
    for name, values in indexed.items():
        missing = sorted(ids - set(values))
        if missing: raise ValueError(f"{name} lacks {len(missing)} split assets: {missing[:10]}")
    return {name: {image_id: values[image_id] for image_id in ids} for name, values in indexed.items()}


def atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle: handle.write(content)
        os.replace(name, path)
    except BaseException:
        Path(name).unlink(missing_ok=True); raise


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write UTF-8 text metadata."""
    atomic_write_bytes(path, content.encode("utf-8"))


def atomic_copy2(source: Path, destination: Path) -> None:
    """Copy an existing source byte-for-byte while preserving stat metadata."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    shutil.copy2(source, temporary); os.replace(temporary, destination)


def save_png(path: Path, array: np.ndarray) -> None:
    """Atomically encode a uint8 image using OpenCV only at processing time."""
    import cv2
    if array.dtype != np.uint8: raise ValueError("PNG arrays must be uint8")
    encode = cv2.imencode(".png", cv2.cvtColor(array, cv2.COLOR_RGB2BGR) if array.ndim == 3 else array)[1]
    atomic_write_bytes(path, encode.tobytes())


def save_npz(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    np.savez_compressed(temporary, **arrays); os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row}) if rows else []
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore"); writer.writeheader(); writer.writerows(rows)
    os.replace(temporary, path)


def write_dataframe_csv(path: Path, frame: Any) -> None:
    """Atomically serialize a pandas-like table without exposing a partial CSV."""
    buffer = io.StringIO()
    frame.to_csv(buffer, index=False, encoding="utf-8-sig")
    atomic_write_bytes(path, buffer.getvalue().encode("utf-8-sig"))
