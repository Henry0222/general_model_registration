"""Read exported numeric results without importing the geometry stack."""
from __future__ import annotations

import json
from pathlib import Path


def load_numeric_candidates(results_path: Path) -> list[tuple[str, dict]]:
    """Keep unavailable candidates visible; restrict links to this result folder."""
    root = results_path.resolve(strict=True)
    payload = json.loads(root.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("结果文件格式无效。")
    records = [("主结果", payload)]
    links = payload.get("candidate_results") or {}
    if not isinstance(links, dict):
        raise ValueError("候选结果索引格式无效。")
    for name, relative in links.items():
        try:
            path = (root.parent / str(relative)).resolve()
            if not path.is_relative_to(root.parent):
                raise ValueError("候选路径超出结果目录。")
            candidate = json.loads(path.read_text(encoding="utf-8-sig"))
            if not isinstance(candidate, dict):
                raise ValueError("候选结果格式无效。")
        except (OSError, ValueError) as error:
            candidate = {"read_error": str(error)}
        records.append((str(name), candidate))
    return records
