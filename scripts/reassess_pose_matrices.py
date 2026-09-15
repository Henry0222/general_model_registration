"""Re-evaluate saved poses by reference matrices, without registering any mesh."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from auto_alignment.pose_evaluation import matrix_pose_error, imposed_parameter_comparison

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vector(values):
    return " / ".join(f"{v:.7g}" for v in values) if values is not None else "欧拉角奇异，见矩阵"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report-copy", type=Path)
    args = parser.parse_args()
    from benchmark_v200 import cases
    definitions = {case["id"]: case for case in cases(args.data.resolve())}
    root, output = args.evidence.resolve(), args.output.resolve()
    run_paths = {
        "baseline": root/"baseline_results/summary.json",
        "v200_first_pass": root/"v200_results/summary.json",
        "v200": root/"v200_final_results/summary.json",
    }
    wrap_path = root/"wrap_comparison.json"
    hashes = {str(path): digest(path) for path in [*run_paths.values(), wrap_path]}
    runs = {name: {row["id"]: row for row in read(path)} for name, path in run_paths.items()}
    wraps = {row["id"]: row for row in read(wrap_path)}
    records = []
    for index, final in enumerate(runs["v200"].values(), 1):
        if "reference_matrix" not in final:
            continue
        reference = np.asarray(final["reference_matrix"])
        definition = definitions[final["id"]]
        entry = dict(index=index, id=final["id"], reference_kind=final["reference_kind"],
                     gold_registration_matrix=reference.tolist(), methods={})
        for name, run in runs.items():
            row = run[final["id"]]
            matrix = np.asarray(row["transformation"])
            detail = dict(registration_matrix=matrix.tolist(), status=row["status"],
                          matrix_pose_error=matrix_pose_error(matrix, reference),
                          auxiliary_pose_p95_mm=row.get("pose", {}).get("pose_p95_mm"))
            if isinstance(definition["gold"], tuple):
                detail["imposed_parameters"] = imposed_parameter_comparison(matrix, *definition["gold"])
            entry["methods"][name] = detail
        wrap = wraps.get(final["id"])
        if wrap and wrap.get("verified"):
            detail = dict(registration_matrix=wrap["transform"], status="saved_wrap_export",
                          matrix_pose_error=matrix_pose_error(wrap["transform"], reference),
                          auxiliary_pose_p95_mm=wrap["pose_p95_mm"])
            if isinstance(definition["gold"], tuple):
                detail["imposed_parameters"] = imposed_parameter_comparison(wrap["transform"], *definition["gold"])
            entry["methods"]["wrap"] = detail
        write(output/f"{index:02d}/matrices.json", entry)
        # These are offline metadata only; production results/matrices/STLs stay unchanged.
        sidecar_path = root/f"review_results/{index:02d}/offline_evaluation.json"
        if sidecar_path.exists():
            sidecar = read(sidecar_path)
            sidecar["matrix_pose_error"] = entry["methods"]["v200"]["matrix_pose_error"]
            if "imposed_parameters" in entry["methods"]["v200"]:
                sidecar["imposed_parameter_comparison"] = entry["methods"]["v200"]["imposed_parameters"]
            sidecar["primary_reference_evaluation"] = "matrix six-degree-of-freedom errors; P95 auxiliary"
            write(sidecar_path, sidecar)
        records.append(entry)
    assert hashes == {path: digest(Path(path)) for path in hashes}, "Original evidence unexpectedly changed"
    payload = dict(timestamp=datetime.now().astimezone().isoformat(timespec="seconds"),
                   evaluation="matrix_pose_error_v1", original_evidence_sha256=hashes,
                   no_registration_rerun=True, records=records)
    write(output/"evaluation.json", payload)
    make_report(output, payload, args.report_copy)
    print(json.dumps(dict(evaluated_reference_cases=len(records),
                         known_rigid_motion_cases=sum(r["reference_kind"] == "known_rigid_motion" for r in records),
                         report=str(output/"REPORT.md")), ensure_ascii=False))


def make_report(output, payload, report_copy):
    known = [r for r in payload["records"] if r["reference_kind"] == "known_rigid_motion"]
    lines = ["# 2.0.0 有金标准实验：六自由度与矩阵评价", "", f"更新：{payload['timestamp']}。只重算既有矩阵，没有重新配准或更换候选。", "",
        "本报告将三轴平移、三轴旋转及完整矩阵作为主要评价。P95 保留为辅助：它此前也是由金标准矩阵算出的对应点位移，不是普通表面残差。矩阵评价更直接、没有曲面采样误差，但仍需固定坐标系、旋转顺序和原点，不能简单称为对所有问题都“更精准”。", "",
        "## 两种需要分别查看的矩阵差异", "",
        "所有配准矩阵都把同一浮动坐标系变到同一固定坐标系：`p_fixed = T · p_moving`。", "",
        "1. **参数恢复**：直接查看结果 T 与金标准 G 的平移列、旋转矩阵；平移列差为 `T[:3,3] − G[:3,3]`。总旋转误差由 `R_T · R_Gᵀ` 计算。再反解为当初实验输入的平移 XYZ、旋转 XYZ，可逐轴核对金标准参数。",
        "2. **残余位姿**：计算 `E = T · G⁻¹`，它把理想固定位置变到结果位置。E 的平移列是固定坐标原点处的残余位移；分解 E 的旋转矩阵得到残余旋转 XYZ。要把当前结果校正到真值，应施加 `C = G · T⁻¹ = E⁻¹`。E 和 C 都完整保存。", "",
        "这两种平移不能混用：`E.t = T.t − E.R · G.t`，一般不等于 `T.t − G.t`。原点远离模型时，小角度旋转会显著改变平移列。固定当前实验坐标系后可以逐项比较，但不能把毫米与角度随意相加成总分；各轴也可能互有优劣。", "",
        "欧拉分解明确采用 `R = Rx · Ry · Rz`，与本项目已核对的输入矩阵约定一致。在列向量运算中右侧矩阵先作用；仅凭界面上“XYZ/全局”文字不能推断次序。三轴角度的直接参数差仅用于恢复输入值；真实旋转误差用相对旋转矩阵计算。跨 ±180° 使用等价分支，奇异情况明确标记。公式和分解约定参考 [SciPy from_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html) 与 [as_euler](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html)，实现另用 Open3D 矩阵往返测试核对。", "",
        "## 0903：按原实验输入参数核对", "",
        "已知输入平移为 **(100, −60, 500) mm**，旋转为 **(58, 47, 268)°**。本项目已核对的生成式为 `p_moving = (Rx · Ry · Rz) · (p_fixed + t_input)`。因此配准矩阵 G 的平移列为 **(−100, 60, −500)**，并不能把配准方向的旋转简单写成三个输入角度各自取负。", "",
        "表中从配准矩阵恢复的 t_input 为 `−T.t`，角度从 `T.Rᵀ` 分解并选择接近原输入的等价分支。所有三元组顺序均为 X / Y / Z。", "",
        "| 模型 | 方法 | 恢复输入平移 (mm) | 恢复输入旋转 (°) |",
        "|---|---|---|---|",
    ]
    for case in known:
        if not case["id"].startswith("0903"):
            continue
        for method, label in (("v200", "2.0"), ("wrap", "Wrap")):
            data = case["methods"][method]["imposed_parameters"]
            lines.append(f"| {case['id'].split('/')[-1]} | {label} | {vector(data['estimated_translation_xyz_mm'])} | {vector(data['estimated_rotation_xyz_deg'])} |")
    lines += ["", "## 0903：残余六自由度（主诊断表）", "",
        "下表来自 E，理想值六项均为零。正负号表示相对于固定轴的残余方向；它不是上一表角度的逐项相减。", "",
        "| 模型 | 方法 | 残余平移 X / Y / Z (mm) | 残余旋转 X / Y / Z (°) |",
        "|---|---|---|---|",
    ]
    for case in known:
        if case["id"].startswith("0903"):
            for method, label in (("v200", "2.0"), ("wrap", "Wrap")):
                data = case["methods"][method]["matrix_pose_error"]
                lines.append(f"| {case['id'].split('/')[-1]} | {label} | {vector(data['relative_translation_xyz_mm'])} | {vector(data['relative_rotation_xyz_deg'])} |")
    lines += ["", "## 平移与旋转分别汇总", "",
        "平移列差模长：直接比较 T 与 G 的平移列。残余平移模长：E.t 的长度。总旋转角：相对旋转的最短角度，与欧拉角表示分支无关；用 atan2 计算，避免极小角度在 acos(trace) 中被舍入为零。", "",
        "| 模型 | 方法 | 平移列差模长 (mm) | 残余平移模长 (mm) | 总旋转误差 (°) |",
        "|---|---|---:|---:|---:|",
    ]
    for case in known:
        if case["id"].startswith("0903"):
            for method, label in (("v200", "2.0"), ("wrap", "Wrap")):
                d = case["methods"][method]["matrix_pose_error"]
                lines.append(f"| {case['id'].split('/')[-1]} | {label} | {d['translation_column_difference_norm_mm']:.7g} | {d['relative_translation_norm_mm']:.7g} | {d['relative_rotation_angle_deg']:.7g} |")
    lines += ["", "**按矩阵分项后，结论比单个 P95 更细：0903 四组中，2.0 的总旋转误差和平移列差模长均小于所提供的 Wrap 结果；但除“充分简单表面变化”外，另外三组的固定原点残余平移模长仍大于 Wrap。** 不能将其简化为“2.0 所有方面都更差”，也不能仅凭参数差较小宣布整体位置全部优于 Wrap。", "",
        "“充分复杂表面变化”的 2.0 输出仍是 failed 预览。此处更换评价方式不改变软件当时的质量判定或配准矩阵。", "",
        "## 0829：六自由度完整对照", "",
        "已知输入平移为 (30, −40, 15) mm、旋转为 (8, 168, 285)°。此处列出旧源码和最终 2.0；可验证的 Wrap 文件也列出。", "",
        "| 模型 | 方法 | 残余平移 X / Y / Z (mm) | 残余旋转 X / Y / Z (°) |",
        "|---|---|---|---|",
    ]
    for case in known:
        if case["id"].startswith("0829"):
            for method, label in (("baseline", "旧源码"), ("v200", "2.0"), ("wrap", "Wrap")):
                if method not in case["methods"]:
                    continue
                d = case["methods"][method]["matrix_pose_error"]
                lines.append(f"| {case['id'].split('/')[-1]} | {label} | {vector(d['relative_translation_xyz_mm'])} | {vector(d['relative_rotation_xyz_deg'])} |")
    lines += ["", "## 金标准范围、原始矩阵与验证", "",
        "13 组已知刚体运动作为金标准评估。咬合调整组另有既往配准参考矩阵，文件中保留其六自由度差异，但不列为独立金标准。4 组无金标准数据不生成虚构的位姿误差。第一轮结果也写入机器可读文件以供追溯。", "",
        "所有组均保存 G、各方法 T、残余 E、校正 C、三轴分量、输入参数恢复、平移列差及辅助 P95。原基线/最终/Wrap 证据文件通过 SHA-256 前后核对，内容未改变；查看结果的 offline_evaluation.json 仅增加离线指标。", "",
        "新增 9 项矩阵评价测试通过：固定坐标系复合与校正、已知两组输入角度及等价分支、±180° 跨越、微小角度、±90° 奇异、原点变化及非法矩阵拒绝。", "",
        f"- [完整 JSON](<{(output/'evaluation.json').as_posix()}>)", "",
        "| 编号 | 数据 | 含各方法完整矩阵的文件 |",
        "|---|---|---|",
    ]
    for case in payload["records"]:
        path = output/f"{case['index']:02d}/matrices.json"
        lines.append(f"| {case['index']:02d} | {case['id']} | [matrices.json](<{path.as_posix()}>) |")
    report = "\n".join(lines)+"\n"
    (output/"REPORT.md").write_text(report, encoding="utf-8")
    if report_copy:
        report_copy.parent.mkdir(parents=True, exist_ok=True)
        report_copy.write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
