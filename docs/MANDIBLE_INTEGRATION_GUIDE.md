# 下颌软件调用通用配准公共接口指南

核对日期：2026-09-07。面向维护 `mandible_registration` 的 LLM；依据当前两个项目的实际源码与已安装验证的 wheel 编写。

## 1. 交给下游 LLM 的任务说明

请在 `E:\MyScript\Auto_alignment\mandible_registration` 中，将现有通用配准依赖迁移到本文的公共接口。先核对当前代码与本文是否有后续差异，保留他人已有修改。

本次工作包括：安装明确版本的依赖、移除相邻源码目录注入、迁移核心和查看器导入、以公共多区域组件替换私有选区继承，以及完成下游回归测试。六文件工作流、CT 阶段已有多次尝试/一致性检查、质量放行规则、髁突中心定义、历史项目兼容性继续由下颌软件维护。迁移接口本身不应改变这些算法或业务规则。

通用端源码：`E:\MyScript\Auto_alignment\general_model_registration`。公共模块只有以下三个业务入口：

| 入口 | 用途 |
|---|---|
| `auto_alignment.integration` | 公共 API 版本、应用版本 |
| `auto_alignment.integration.core` | 配置、网格加载、配准、结果和质量类型 |
| `auto_alignment.integration.review` | 阶段复核清单、彩虹图查看器、公共字体配置 |
| `auto_alignment.integration.selection` | 多区域选区、完整快照、公开面片编码和状态工具 |

本轮机器学习实验尚未接入生产求解器。不要导入 `analysis_output` 内的引擎快照、训练脚本或学习权重作为正式依赖。当前任务不包含 EXE 构建。

## 2. 安装与版本核验

本次已验证交付物：

- wheel：`E:\MyScript\Auto_alignment\analysis_output\integration_collaboration_20260907\wheel\general_model_registration-2.0.0-py3-none-any.whl`
- SHA-256：`93eff461b4c4a44ec315c0787efd5fbca8ec7f63343be7efd89ded4b53b38c34`
- 应用版本：`2.0.0`，当前为未发布源码验证版。
- 公共 API：`INTEGRATION_API_VERSION == 1`。
- 声明的 Python 范围：`>=3.11,<3.14`；以下示例使用 Python 3.12。

先确认下颌程序实际使用的 Python 解释器。不要假定已有 `mandible_registration\.venv`，也不要只在另一个终端环境安装后便认定 GUI 子进程可用。可使用现有合适环境，或明确新建独立环境；以下命令是新建环境的示例，供下游执行：

```powershell
$mandibleRoot = 'E:\MyScript\Auto_alignment\mandible_registration'
$wheelPath = 'E:\MyScript\Auto_alignment\analysis_output\integration_collaboration_20260907\wheel\general_model_registration-2.0.0-py3-none-any.whl'
Get-FileHash -LiteralPath $wheelPath -Algorithm SHA256
py -3.12 -m venv "$mandibleRoot\.venv-integration"
$mandiblePython = "$mandibleRoot\.venv-integration\Scripts\python.exe"
& $mandiblePython -m pip install $wheelPath
& $mandiblePython -m pip install -e $mandibleRoot
& $mandiblePython -m pip check
```

在下颌项目 `pyproject.toml` 的依赖列表追加 `"general-model-registration==2.0.0"`。该版本目前通过本地 wheel 提供，不假定它已发布到 PyPI。先安装上方 wheel，再安装下颌项目；自动化部署应提供同一 wheel 或受控 wheelhouse。

如果某环境已安装另一个同名 `2.0.0`，仅看版本号不足以确认内容。核对 wheel 哈希，必要时对这个明确路径执行 `pip install --force-reinstall --no-deps`，随后正常解析下颌依赖并运行 `pip check`。不要将 `--no-deps` 用作新环境的完整安装方法。

以下核验应在源码仓库之外、没有通用项目 `PYTHONPATH` 注入的进程中执行。打印出的模块必须来自所选环境的 `site-packages`，不能来自相邻项目的 `src`。

```python
import sys
from importlib.metadata import version
import auto_alignment.integration as integration

assert integration.INTEGRATION_API_VERSION == 1
assert integration.GENERAL_MODEL_REGISTRATION_VERSION == "2.0.0"
assert version("general-model-registration") == "2.0.0"
print(sys.executable)
print(integration.__file__)
```

### 替换 `core_bridge.py`

现有 `ensure_registration_core()` 会向 `sys.path` 注入相邻 `general_model_registration/src`。可保留函数名称，改为明确的依赖检查。当前调用方未使用返回值；如以后使用，下面返回的是已安装公共包目录，不再是源码根目录。

```python
from pathlib import Path


def ensure_registration_core() -> Path:
    try:
        import auto_alignment.integration as integration
    except ModuleNotFoundError as exc:
        if exc.name not in {"auto_alignment", "auto_alignment.integration"}:
            raise  # 不掩盖依赖自身缺失等其他导入故障。
        raise RuntimeError("请在当前解释器中安装已交付的通用配准 2.0.0 wheel。") from exc
    if (integration.INTEGRATION_API_VERSION != 1
            or integration.GENERAL_MODEL_REGISTRATION_VERSION != "2.0.0"):
        raise RuntimeError("通用配准版本与本次已验证接口不一致，请核对安装包。")
    return Path(integration.__file__).resolve().parent
```

只导入版本不会加载 Open3D。不要在该检查函数中初始化 GUI。今后升级版本应更新依赖与兼容性测试，而非直接删除版本核验。

## 3. 当前下游文件的迁移位置

| 下游文件或位置 | 当前依赖 | 迁移方式 |
|---|---|---|
| `core_bridge.py` | 相邻目录与 `sys.path` 注入 | 改为上方版本检查；审计其他注入点 |
| `workflow.py` | `auto_alignment.config/mesh_io/registration` | 改用 `integration.core`；应用版本来自 `integration` |
| `viewer.py` | 原 `load_mesh` 导入 | 改用 `integration.core.load_mesh` |
| `stage_review.py` | 手拼清单、原 `result_viewer` | 用公共清单构造/校验；保留下游附加字段和历史路径解析 |
| `condyle_selection.py` | 继承 `ModelSelectionViewer`，访问私有状态/控件 | 使用 `MultiRegionSelectionViewer` 或独立运行入口，通过快照和回调适配 |
| `condyles.py` | `_encode_ranges` | 改用 `integration.selection.encode_face_ranges`；保留面积加权中心计算 |
| `scene_data.py` 等 | 仅调用 bridge | 使用新检查函数；不再触发目录注入 |
| `scripts/check_stage_viewer.py` | 原查看器与 `_configure_font` | 使用 `integration.review` 的公开名称 |
| `tests/test_workflow.py`、`test_consensus.py` 等 | 原结果/指标类型导入 | 改用 `integration.core` |
| `tests/test_condyles.py`、`test_stage_review.py` 等 | 原选区工具、旧 monkeypatch 目标 | 改用公开类型；patch 实际使用该符号的绑定位置 |

迁移后搜索 `auto_alignment`、`sys.path`、`_configure_font`、`_encode_ranges`、`_decode_ranges` 和对旧查看器私有成员的访问。公共组件内部仍有私有实现是正常的；下游不继续依赖这些实现。

## 4. 核心配准调用

```python
from auto_alignment.integration import (
    GENERAL_MODEL_REGISTRATION_VERSION as registration_core_version,
)
from auto_alignment.integration.core import (
    AlignmentConfig, MeshFacts, MeshValidationError, load_mesh,
    RegistrationMetrics, RegistrationResult, RegistrationQualityReport,
    PositionConfidence, register_meshes,
)


def register_pair(target_path, source_path, config, progress=None):
    target, target_facts = load_mesh(target_path)
    source, source_facts = load_mesh(source_path)
    return register_meshes(
        target, source, target_facts, source_facts, config, progress,
        target_priority_faces=None,
        source_priority_faces=None,
    )
```

`config` 传入当前阶段已有的配置，不要在迁移时统一替换为默认 `AlignmentConfig()`。`AlignmentConfig`、`MeshFacts` 继续支持 `dataclasses.replace`。对已经变换的网格继续沿用下游 `_transformed_facts` 更新包围盒等描述，不能配准新网格却传入失效的旧 facts。

### 参数与结果约定

- **固定模型 target 在前，移动模型 source 在后**；`load_mesh(path, *, flip_normals=False)` 返回 `(mesh, facts)`，支持有效的 STL/PLY 三角网格。长度按 mm 使用，接口不自动推断或转换文件单位。
- `progress(fraction, message)` 的 fraction 为有限的 0～1；回调异常向上传播。在计算线程中通过 Qt signal/队列通知界面，不直接操作 GUI。
- 优先面 mask 为 `None` 或严格的一维 `bool` 数组，长度等于对应基准网格面数。索引属于同一 `load_mesh` 得到、尚未删面的基准网格，不保证等于 STL 磁盘记录顺序。重建网格、独立清理或删面后的副本需要明确映射；仅面数相等不足以复用 mask。
- `result.transformation` 是 4×4 刚体矩阵。`result.status` 只允许 `success / warning / failed`；`result.succeeded` 对 `warning` 也为真。**保留下游 `_require_success` 及已有 warning 放行条件，不能用 `.succeeded` 代替原门槛。**
- `result.confidence` 是展示文本。程序判断使用 `result.quality.position_confidence` 枚举，并先判断 `quality is not None`；枚举 `.value` 为 `high / medium / low / failed`。这些字段不会自动替下游决定能否传递矩阵。
- `RegistrationResult` 保留 `transformation/status/confidence/metrics/warnings/elapsed_seconds/quality` 字段，`quality` 可以为空。完整指标与配置字段见文末签名参考，不要推测未导出的属性。
- `metrics.rotation_degrees`、`metrics.translation_mm` 描述本次变换量，**不是相对金标准的位姿误差**。SCR、P90/P95 等残差也不能在没有金标准的项目中直接解释为髁突测量误差。

### 矩阵方向与现有三阶段工作流

采用齐次列向量：`p_target = T @ p_source`。对 N×3 行向量数组：`points @ T[:3, :3].T + T[:3, 3]`。矩阵平移列已包含绕任意中心旋转所需的补偿；将同一矩阵作用于同坐标系颌骨时，不再围绕颌骨自身中心重做一次旋转。

| 阶段 | 固定 target | 移动 source | 输出矩阵含义 |
|---|---|---|---|
| `T_CT` | 第一次口扫下颌 | CT 牙列 | CT 原始坐标 → 第一次口扫坐标 |
| `T_UPPER` | 第一次口扫上颌 | 第二次口扫上颌 | 第二次口扫坐标 → 第一次口扫坐标 |
| `T_DELTA` | 已施加 `T_UPPER` 的第二次口扫下颌 | 第一次口扫下颌 | 在统一上颌参考中，下颌从初始到重新定位后的刚体运动 |

`T_UPPER` 同时用于第二次口扫上、下颌，保持导出的咬合关系。对原始 CT 下颌骨：

```python
import numpy as np


def mandible_frames(T_CT, T_DELTA, original_ct_points):
    T_MANDIBLE_T0 = T_CT
    T_MANDIBLE_T1 = T_DELTA @ T_CT
    points = np.asarray(original_ct_points, dtype=float)
    p0 = points @ T_MANDIBLE_T0[:3, :3].T + T_MANDIBLE_T0[:3, 3]
    p1 = points @ T_MANDIBLE_T1[:3, :3].T + T_MANDIBLE_T1[:3, 3]
    return p0, p1
```

这里的 `T_DELTA` 已由完成上颌归一化的目标模型求得，不再额外乘一次 `T_UPPER`。现有刚体矩阵合法性检查继续保留。Open3D 的 `.transform(T)` 会原地修改网格，传递到多个模型前使用副本，避免覆盖原始 CT 或重复施加变换。

## 5. 阶段彩虹图与复核清单

### 新清单导出：保留 `stage_key` 与矩阵

`RegistrationReviewSpec` 没有 `stage_key`、`transformation` 字段；这些仍属于下游项目记录。下面是供 `stage_review.py` 使用的适配示例，保留当前字段和目录结构。示例中的 `write_mesh/write_json/sha256_file/transformed_mesh` 是**下颌项目已有函数**，不是新增通用 API。

```python
import os
from pathlib import Path
import numpy as np
from auto_alignment.integration.review import (
    RegistrationReviewSpec, build_review_manifest,
    validate_review_manifest, load_review_manifest,
)
from .exporter import sha256_file, write_json, write_mesh
from .transforms import transformed_mesh


def export_stage_review(directory, stage_key, target_path, source_mesh, result,
                        *, accepted=True, extra_warnings=()):
    if stage_key not in {"T_CT", "T_UPPER", "T_DELTA"}:
        raise ValueError("未知配准阶段")
    root = Path(directory).resolve()
    folder = root / "stage_views" / stage_key
    aligned = write_mesh(folder / "aligned.stl",
                         transformed_mesh(source_mesh, result.transformation))
    target = Path(target_path).resolve(strict=True)
    confidence = (result.quality.position_confidence.value
                  if result.quality is not None else None)
    spec = RegistrationReviewSpec(
        target_path=target,
        aligned_path=aligned.name,  # 相对于 results.json 所在目录。
        status=result.status if accepted else "failed",
        position_confidence=confidence,
        confidence_display=result.confidence,
        warnings=tuple(result.warnings) + tuple(extra_warnings),
        review_only=(not accepted or result.status == "failed"),
        target_sha256=sha256_file(target),
        aligned_sha256=sha256_file(aligned),
        annotations_path="viewer_annotations.json",
        minimum_nominal_mm=-0.05,
        maximum_nominal_mm=0.05,
    )
    payload = build_review_manifest(spec)
    payload["stage_key"] = stage_key
    payload["transformation"] = np.asarray(result.transformation).tolist()
    if target.is_relative_to(root):
        payload["target_mesh"]["archived_path"] = os.path.relpath(target, folder)
    validate_review_manifest(payload)  # 校验结构和值；此函数不读模型文件。
    manifest = write_json(folder / "results.json", payload)  # 下游原子写入。
    load_review_manifest(manifest)  # 核验实际文件、相对路径、SHA-256。
    return manifest
```

仅有标准字段的场景可直接 `write_review_manifest(path, spec)`；其第二参数是 **spec，不是 dict**，且会在写入前校验文件。需要下游附加字段时采用上方构造方式。不要将已有下游清单 `load` 成 spec 后直接 `write` 回去，误以为未知字段会自动保留。

对已写出的模型计算新清单哈希；打开历史结果时，按历史记录中的哈希核对，不能重新生成哈希来掩盖文件变化。检查失败应明确报错。

### 打开清单与旧项目兼容

新清单可在查看器子进程主线程调用 `run_registration_review(manifest_path)`。现有 `show_stage_review()` 可先保留原路径解析逻辑，仅将查看器导入改成：

```python
from auto_alignment.integration.review import run_general_result_viewer


def open_resolved_stage(manifest_path, resolved_target, resolved_aligned):
    run_general_result_viewer(
        manifest_path,
        target_override=resolved_target,
        aligned_override=resolved_aligned,
    )
```

两种入口都读取已经对齐的模型，**不会根据清单内的 transformation 再执行配准或移动模型**。

必须保留/验证以下兼容行为：

1. 清单中的相对模型路径以清单父目录为基准；显式 `target_override/aligned_override` 的相对路径以进程工作目录为基准，因此下游宜先解析成绝对路径。
2. `target_mesh.archived_path` 存在且有效时优先使用它。下游旧 `T_DELTA` 的同目录 `target.stl` 兜底是下游特例，公共接口不会自动替你推断；保留现有逻辑，再传 overrides。
3. `prepare_project_stage()` 对旧项目只补复核适配文件，复用已存矩阵和网格，不重新配准。移动项目目录后继续正确找到已归档模型。
4. 新复核清单 `schema_version` 为整数 `1`；也支持已知旧外壳字符串 `1.4.0/1.4.1/1.4.2` 和未声明版本的旧清单，拒绝未知未来版本。不要批量改历史版本标记。
5. 默认标注文件总是清单旁的 `viewer_annotations.json`，与清单名称无关。当前一阶段一目录可沿用；同目录有多个清单时必须给不同 `annotations_path`，防止标注串用。标注路径不能覆盖输入模型或清单本身。
6. `review_only` 是查看器元数据，不会自动阻止业务代码传递矩阵；下游仍执行放行门槛。`direction_reversed` 控制偏差评价/显示方向，不代表取逆矩阵。
7. `unknown` 仅供缺少配准结果的查看元数据使用，不是核心求解器的第四种状态。

## 6. 左右髁突选区适配

### 两种 schema 1 文件不能混用

| 文件 | 下游髁突业务文件 | 通用多区域 UI 状态文件 |
|---|---|---|
| 用途 | 中心计算与位移分析 | 选区组件的保存/恢复 |
| 标识 | `kind: mandible_condyle_regions` | `indexing: load_mesh baseline triangles before edit deletions` |
| 面数 | `mesh_triangle_count` | `triangle_count` |
| 面片范围 | `regions.left/right.selected_ranges` | `masks.left/right` |
| 其他内容 | `center_ct_mm`、`area_mm2`、中心定义 | `active_region`、`allow_overlap` |

两者各自的 `schema_version == 1` 不表示格式相同。保留现有业务文件；另用 `原业务文件名.regions.v1.json` 保存 UI 状态。不能把业务 profile 的路径直接当成组件 `state_path`。

`condyles.build_profile()` 中先将私有编码替换为公开调用：

```python
import numpy as np
from auto_alignment.integration.selection import encode_face_ranges


def selected_ranges(mask):
    return encode_face_ranges(np.flatnonzero(mask))
```

编码函数接收整数面片索引，不接收 bool mask；范围为首尾均包含的 `[first, last]`。解码使用 `decode_face_ranges(ranges, triangle_count)`，返回新的 bool mask，非法范围会报错。

### 用回调连接现有业务计算

下面函数供 `condyle_selection.py` 使用，替换私有查看器子类的主要职责。前提是已按上文迁移 `build_profile()` 内的编码导入。业务 profile 的路径仍由下游既有 `profile_path()`/命令参数决定。

该示例自行管理查看器子进程中的 Open3D 生命周期，便于补齐自定义摘要文字的中文字形；它与便捷函数 `run_multi_region_selection_viewer()` 是二选一的启动方式。

```python
from pathlib import Path
from threading import current_thread, main_thread
import numpy as np
from open3d.visualization import gui
from auto_alignment.integration.core import load_mesh
from auto_alignment.integration.review import configure_open3d_font
from auto_alignment.integration.selection import (
    SelectionRegionSpec, MultiRegionSelectionViewer, decode_face_ranges,
)
from .condyles import build_profile, read_profile
from .exporter import sha256_file, write_json


def run_condyle_selection(mesh_path, selection_path):
    if current_thread() is not main_thread():
        raise RuntimeError("髁突查看器必须在子进程主线程启动")
    mesh_path = Path(mesh_path).resolve(strict=True)
    selection_path = Path(selection_path).resolve()
    ui_path = selection_path.with_name(selection_path.stem + ".regions.v1.json")
    digest = sha256_file(mesh_path)
    mesh, _ = load_mesh(mesh_path)
    if sha256_file(mesh_path) != digest:
        raise ValueError("读取期间模型文件发生变化")
    face_count = len(mesh.triangles)
    initial_masks = None
    if selection_path.is_file():
        saved = read_profile(selection_path, digest)
        if saved["mesh_triangle_count"] != face_count:
            raise ValueError("髁突选区的基准面数不匹配")
        initial_masks = {}
        for key in ("left", "right"):
            region = saved["regions"].get(key)
            initial_masks[key] = (
                decode_face_ranges(region["selected_ranges"], face_count)
                if region is not None else np.zeros(face_count, dtype=bool)
            )

    def profile_from_snapshot(snapshot):
        if snapshot.mesh_sha256 != digest or snapshot.triangle_count != face_count:
            raise ValueError("快照不属于缓存的 CT 基准网格")
        return build_profile(mesh, mesh_path, digest, snapshot.masks)

    def persist_profile(snapshot):
        # 仅写下游业务文件；不要在回调内修改 viewer/session。
        write_json(selection_path, profile_from_snapshot(snapshot))

    def summary(snapshot):
        profile = profile_from_snapshot(snapshot)
        rows = []
        for key, label in (("left", "左侧髁突"), ("right", "右侧髁突")):
            region = profile["regions"].get(key)
            if region is None:
                rows.append(f"{label}：未选择")
            else:
                x, y, z = region["center_ct_mm"]
                rows.append(f"{label} CT 中心 mm：{x:.3f}, {y:.3f}, {z:.3f}")
        return "\n".join(rows)

    regions = (
        SelectionRegionSpec("left", "左侧髁突", (1.0, 0.2, 0.1)),
        SelectionRegionSpec("right", "右侧髁突", (0.1, 0.3, 1.0)),
    )
    app = gui.Application.instance
    app.initialize()
    configure_open3d_font(app, extra_text=str(mesh_path) + "左侧右侧髁突未选择中心 CT mm")
    viewer = MultiRegionSelectionViewer(
        mesh_path, ui_path, regions,
        initial_masks=initial_masks,
        allow_overlap=False,
        on_change=None,
        on_save=persist_profile,
        summary_provider=summary,
    )
    app.run()
    return viewer.get_snapshot()
```

业务 profile 只应由 `on_save` 写入，以便操作者能够用标题栏关闭或“取消并关闭”放弃本次编辑。若业务 profile 已存在，以其 mask 为本例的恢复依据，默认激活首个区域；辅助 UI 文件不覆盖它。两者都不存在时初始化空选区。

这是一项明确的恢复优先级，不是两个文件的跨文件事务。公共 UI 状态先写入、再执行 `on_save`；若业务文件写失败，UI 文件可能已经较新，内存仍保留并报错。不能吞掉异常或显示“全部保存成功”。如果要采用其他冲突恢复策略，应显式设计并测试，而非静默选一个文件。

### 必须遵守的组件行为

- 公共组件至少定义一个区域；髁突适配器仍定义 `left/right` 两个区域，并允许任一侧为空。`left/right` 的解剖意义由下游和操作者确定，公共组件不会根据坐标自动识别左右。单区域业务可直接提供一个区域，并隐藏无意义的区域切换控件。
- mask 对应同一次加载政策下的原始 CT 基准面序。源文件 SHA-256、基准面数、面序约定共同核验；原始 STL 不被覆盖。
- 所有输出快照都包含全部区域、文件 SHA-256、面数及当前区域，mask 是副本。直接改快照不会修改查看器；修改通过公开方法提交。
- `allow_overlap=False` 时，一次编辑若产生左右重叠，整次操作拒绝，不接受其中部分，也不产生历史记录或回调。
- 区域切换也能撤销；撤销/重做同时恢复所有 mask 和当前区域。不要继续沿用旧子类“切换后清空历史”的做法。
- `on_change` 发生在编辑已提交之后，异常不会回滚该次编辑；`on_save` 发生在状态文件原子写入之后。失败时保留内存与 dirty 状态。回调内禁止重入修改会话；摘要回调只返回文字。
- 只有“保存并关闭”或公共 `close()` 方法会尝试保存，失败时留住窗口；标题栏关闭和“取消并关闭”直接放弃本次内存编辑。强制终止进程不保证保存。
- 选择表面面积加权中心、坐标系说明、`analyze_motion()` 和矩阵合法性仍由下游承担，不移入通用查看器。保留 `center_definition: selected_surface_area_weighted_centroid`。

### UI 对象与无界面会话的方法名不同

| 操作 | `MultiRegionSelectionViewer` | `RegionSelectionSession` |
|---|---|---|
| 读取完整快照 | `get_snapshot()` | `snapshot()` |
| 替换区域 mask | `set_region_mask(key, mask)` | `set_mask(key, mask)` |
| 切换区域 | `set_active_region(key)` | `select_region(key)` |
| 撤销/重做 | `undo()` / `redo()` | `undo()` / `redo()` |
| 保存 | `save()`，路径已在构造时给定 | `save(path)` |
| 恢复文件 | 构造时 `initial_masks=None` 自动检查已有文件 | `load(path)` |
| 关闭/视角 | `close()` / `reset_view()` | 无 GUI 方法 |

无界面会话适合测试状态和回调，不需要创建窗口。其他公开工具包括 `ModelEditState`、`clone_state_with_masks(state, selected, deleted)`、`bounded_component_faces(mesh, selected, deleted)`；前者的 selected 和 deleted 两个参数都必须提供。现有 `apply_edit_state` 并非此公共模块导出，不要自行编造 `integration.selection.apply_edit_state`。

## 7. GUI 生命周期与启动方式

保留下游已有 Qt `QProcess` 子进程方案，让每个查看器进程在主线程拥有一个 Open3D 循环。不要在正在运行的 Qt 主循环里嵌套 `app.run()`，也不要在普通工作线程初始化 Open3D 窗口。

现有下游命令入口保持兼容：

```text
python -m mandible_registration --view-stage <results.json>
python -m mandible_registration --view-stage-project <project.json> --stage-key T_DELTA
python -m mandible_registration --select-condyles <CT下颌骨.stl> --condyle-profile <业务选区.json>
python -m mandible_registration --dataset-dir <六文件目录> --output-dir <新输出目录>
```

启动子进程使用已安装依赖的同一解释器，例如当前程序的 `sys.executable`。不要依赖系统 PATH 中碰巧找到的另一个 `python`。

独立 `run_*` 入口负责初始化和运行；直接构造 `GeneralResultViewer` / `MultiRegionSelectionViewer` 时，宿主先在主线程 `initialize()`，然后 `configure_open3d_font(app, extra_text=...)`，最后创建窗口和运行一次循环。带自定义中文摘要的宿主应把相关字形加入 `extra_text`。仅导入模块或读取清单不应创建窗口。

## 8. 迁移顺序与验收

建议先运行下游原测试记录基线；再分别迁移依赖/核心导入、复核适配、选区适配，每步验证后继续。不要用接口迁移顺带修改配准阈值、候选排序或中心定义，否则无法区分结果变化来自哪里。

| 验收项 | 需要的证据 |
|---|---|
| 独立安装 | wheel SHA、实际解释器、模块路径、`pip check`；从仓库外导入成功，无相邻源码注入 |
| 核心兼容 | 同配置/同网格的接口前后矩阵在数值容差内一致；原 CT 一致性检查继续执行 |
| 位姿方向 | 用已知 4×4 变换验证 target/source、矩阵乘法顺序；不同几何中心的附属模型直接传矩阵仍正确 |
| 三阶段链 | `T_MANDIBLE_T1 == T_DELTA @ T_CT`；目标下颌已应用 `T_UPPER`；无重复变换 |
| 状态放行 | success、warning、failed 分别覆盖；拒绝候选可查看但不会因 `.succeeded` 或复核按钮被传到颌骨 |
| 复核文件 | 保留 `stage_key/transformation`；中文路径、相对路径、归档路径、移动项目目录、旧 T_DELTA 兜底均可用 |
| 完整性与标注 | 篡改目标/对齐模型后拒绝复核；未知 schema 拒绝；多阶段标注分别保存且重开后恢复 |
| 业务选区兼容 | 旧 profile 转换正确；一侧为空可保存；中心与原面积加权实现一致；CT SHA/面数不符拒绝 |
| 选区操作 | 区域切换后撤销/重做恢复完整状态；重叠整次拒绝；快照修改不污染组件；保存/回调失败不丢内存 |
| 真实 GUI | 首次显示、中文、套索、前表面/透选、有界组件、旋转/平移/缩放、保存和关闭失败均人工或原生交互核验 |
| 六文件流程 | 下游实际六文件从输入到阶段复核、矩阵传递和髁突输出走通；旧项目继续打开；结果输出到新目录 |

已有通用端证据：98 项测试通过；wheel 在独立环境中完成已知刚体配准、中文复核清单、选区保存和 `pip check`。已创建 Open3D 窗口对象并验证公开方法与渲染。后台验证时首次绘制布局由检查脚本显式触发，**不能据此声称原生首次显示或人工鼠标手势已验收**。下游迁移及完整六文件端到端验收尚需下游完成，本文示例也不能替代这些验收。

请下游 LLM 回交：修改文件列表；安装包/解释器/导入路径；实际运行的测试及结果；六文件测试输出路径与矩阵比较；GUI 已验证与未验证项；任何必须改变旧业务行为的原因。出现新的接口缺口时给出最小复现与期望契约，不再通过继承私有成员临时绕过。

## 9. 对照资料

- [公共接口契约](INTEGRATION_API.md)
- [全部公开签名与 dataclass 字段](INTEGRATION_API_REFERENCE.md)
- [本次通用端交付报告](../../analysis_output/integration_collaboration_20260907/DELIVERY.md)
- [机器可读接口清单](../../analysis_output/integration_collaboration_20260907/public_api.json)
- [独立安装验证](../../analysis_output/integration_collaboration_20260907/installed_verification.json)
- [GUI 自动验证范围](../../analysis_output/integration_collaboration_20260907/gui_verification.json)
- [本文示例检查记录](../../analysis_output/integration_collaboration_20260907/guide_examples_verification.json)

本文 8 段 Python 示例已通过语法编译和公开导入检查；在源码仓库外，使用已交付 wheel 验证了依赖桥接、配准调用参数绑定、矩阵组合、阶段清单附加字段与哈希读取、选区编码及无界面保存/撤销。本次文档检查没有重新执行求解器或启动 GUI，也没有执行完整髁突 GUI 适配和六文件工作流。

若实际代码已有后续更新，以核验后的版本和签名为准，并在迁移报告中说明差异。本文不授权下游静默放宽质量门槛或改写历史结果。
