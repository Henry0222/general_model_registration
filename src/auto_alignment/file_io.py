"""Small, dependency-free helpers for writing files without torn output."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import uuid

import numpy as np


def json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"无法 JSON 序列化：{type(value).__name__}")


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8") -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{uuid.uuid4().hex}.tmp"
    try:
        temporary.write_text(text, encoding=encoding)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def write_json(path: str | Path, payload: Any) -> Path:
    return atomic_write_text(
        path,
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default),
    )
