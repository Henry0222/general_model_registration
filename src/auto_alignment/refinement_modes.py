"""Lightweight mode names shared by the GUI and registration API."""

REFINEMENT_MODES = {
    "baseline": ("原版配准 · 2.0", "沿用 2.0 配准，不运行 A/B 后级。人工选区始终优先。"),
    "A": ("A · 局部共同表面", "适合存在未改变共同表面的模型；保留原版供对比，候选需复核。"),
    "B": ("B · 连续表面偏差", "适合连续表面偏差；可能使正确位姿退化，保留原版供对比。"),
    "compare": ("A/B 对比（保留原版）", "计算 A、B 候选，主结果保持原版；查看结果时可切换候选。"),
    "auto": ("自动选择 · 保守实验", "优先保住原版，可能漏掉 B 修正，也可能使正确位姿退化；B 可手动选择，请复核候选。"),
}

# The desktop defaults to auto at the operator's request. Existing SDK calls
# retain AlignmentConfig's baseline default for compatibility.
GUI_DEFAULT_REFINEMENT_MODE = "auto"


class RegistrationCancelled(RuntimeError):
    """Cooperative cancellation, checked between computation stages."""


def check_cancelled(cancel):
    if cancel is not None and cancel():
        raise RegistrationCancelled("配准已取消。")
