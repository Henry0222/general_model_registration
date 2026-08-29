"""Open3D result viewer shared by the standalone general-registration app."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from auto_alignment.comparison import signed_point_to_mesh_distances
from auto_alignment.deviation_scale import DeviationScale
from auto_alignment.mesh_io import clone_mesh, load_mesh
from auto_alignment.version import __version__


_UI_TEXT = """
通用模型配准结果偏差色图固定参考移动浮动配准后叠加显示模式
水平正交投影左键拖动旋转模型世界固定光源滚轮缩放中键平移
偏差标注开启点击标注平均范围半径删除上一个清空全部
正偏差负偏差名义范围最大最小临界值应用刷新模型色段距离
已更新输入后回车或点击应用同步重绘关闭坐标映射术中法向
点击位置没有命中移动模型无法添加有效数据不足
红正偏差蓝负偏差绿名义范围黄线固定模型青线移动模型
彩虹图表示配准后浮动STL相对固定STL的表面偏差程度毫米
全部与按约的、（），。：；·±%/–～■+0123456789AFDXYZRimmHz"""


@dataclass(frozen=True)
class ViewerData:
    results_path: Path
    target: o3d.geometry.TriangleMesh
    aligned: o3d.geometry.TriangleMesh
    signed_distances_mm: np.ndarray
    deviation_scale: DeviationScale
    direction_reversed: bool


@dataclass(frozen=True)
class GeneralAnnotation:
    annotation_id: str
    anchor_mm: np.ndarray
    surface_normal: np.ndarray
    radius_mm: float
    mean_signed_deviation_mm: float
    sample_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "annotation_id": self.annotation_id,
            "anchor_mm": self.anchor_mm.tolist(),
            "surface_normal": self.surface_normal.tolist(),
            "radius_mm": self.radius_mm,
            "mean_signed_deviation_mm": self.mean_signed_deviation_mm,
            "sample_count": self.sample_count,
        }


def _stored_path(base: Path, value: object) -> Path:
    candidate = Path(str(value))
    return candidate if candidate.is_absolute() else (base / candidate).resolve()


def load_viewer_data(
    results_path: str | Path,
    *,
    target_override: str | Path | None = None,
    aligned_override: str | Path | None = None,
) -> ViewerData:
    path = Path(results_path).resolve(strict=True)
    payload = json.loads(path.read_text(encoding="utf-8"))
    target_payload = payload["target_mesh"]
    if target_override is not None:
        target_path = Path(target_override).resolve(strict=True)
    elif target_payload.get("archived_path"):
        target_path = _stored_path(path.parent, target_payload["archived_path"])
        if not target_path.is_file():
            target_path = Path(str(target_payload["path"]))
    else:
        target_path = Path(str(target_payload["path"]))
    if aligned_override is not None:
        aligned_path = Path(aligned_override).resolve(strict=True)
    else:
        aligned_path = _stored_path(
            path.parent,
            payload["outputs"]["aligned_current_stl"],
        )
    if not target_path.is_file():
        raise FileNotFoundError(f"找不到历史记录使用的固定 STL：{target_path}")
    if not aligned_path.is_file():
        raise FileNotFoundError(f"找不到历史记录中的已配准 STL：{aligned_path}")
    target, _ = load_mesh(target_path)
    aligned, _ = load_mesh(aligned_path)
    reversed_direction = bool(
        payload.get("distance_statistics", {}).get("direction_reversed", False)
    )
    signed = signed_point_to_mesh_distances(
        np.asarray(aligned.vertices), target, reversed_direction
    )
    mapping = payload.get("color_mapping", {})
    minimum_nominal = float(
        mapping.get("configured_minimum_nominal_mm", -0.05)
    )
    maximum_nominal = float(
        mapping.get("configured_maximum_nominal_mm", 0.05)
    )
    scale = DeviationScale.from_signed_distances(
        signed,
        minimum_nominal_mm=minimum_nominal,
        maximum_nominal_mm=maximum_nominal,
    )
    return ViewerData(path, target, aligned, signed, scale, reversed_direction)


def load_pair_viewer_data(
    target_path: str | Path,
    aligned_path: str | Path,
    *,
    minimum_nominal_mm: float = -0.05,
    maximum_nominal_mm: float = 0.05,
) -> ViewerData:
    target_file = Path(target_path).resolve(strict=True)
    aligned_file = Path(aligned_path).resolve(strict=True)
    target, _ = load_mesh(target_file)
    aligned, _ = load_mesh(aligned_file)
    signed = signed_point_to_mesh_distances(np.asarray(aligned.vertices), target, False)
    scale = DeviationScale.from_signed_distances(
        signed,
        minimum_nominal_mm=minimum_nominal_mm,
        maximum_nominal_mm=maximum_nominal_mm,
    )
    return ViewerData(aligned_file, target, aligned, signed, scale, False)


def _system_chinese_font() -> Path | None:
    root = Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts"
    for name in ("msyhbd.ttc", "msyh.ttc", "simsun.ttc"):
        candidate = root / name
        if candidate.is_file():
            return candidate
    return None


def _configure_font(app: gui.Application) -> None:
    path = _system_chinese_font()
    if path is None:
        return
    description = gui.FontDescription(
        gui.FontDescription.SANS_SERIF,
        gui.FontStyle.NORMAL,
        18,
    )
    description.add_typeface_for_code_points(
        str(path),
        sorted({ord(character) for character in _UI_TEXT if not character.isspace()}),
    )
    app.set_font(gui.Application.DEFAULT_FONT_ID, description)


def _surface_frame(
    surface_normal: np.ndarray,
    camera_up: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(surface_normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    up_hint = (
        np.asarray(camera_up, dtype=float)
        if camera_up is not None
        else np.array([0.0, 1.0, 0.0])
    )
    up = up_hint - float(np.dot(up_hint, normal)) * normal
    if float(np.linalg.norm(up)) <= 1e-8:
        alternate = np.array([1.0, 0.0, 0.0])
        up = alternate - float(np.dot(alternate, normal)) * normal
    up /= max(float(np.linalg.norm(up)), 1e-12)
    right = np.cross(up, normal)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    return normal, up, right


def _annotation_ring(
    center: np.ndarray,
    radius_mm: float,
    normal: np.ndarray,
    up: np.ndarray,
    right: np.ndarray,
    segments: int = 64,
) -> o3d.geometry.TriangleMesh:
    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    directions = np.cos(angles)[:, None] * right + np.sin(angles)[:, None] * up
    band_width = min(max(float(radius_mm) * 0.16, 0.02), 0.18)
    inner_radius = max(float(radius_mm) - band_width * 0.5, 0.0)
    outer_radius = float(radius_mm) + band_width * 0.5
    origin = np.asarray(center, dtype=float)
    vertices = np.vstack(
        (origin + outer_radius * directions, origin + inner_radius * directions)
    )
    outer = np.arange(segments, dtype=np.int32)
    inner = outer + segments
    next_outer = np.roll(outer, -1)
    next_inner = np.roll(inner, -1)
    front = np.vstack(
        (
            np.column_stack((outer, next_outer, next_inner)),
            np.column_stack((outer, next_inner, inner)),
        )
    )
    triangles = np.vstack((front, front[:, ::-1]))
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(triangles),
    )
    mesh.paint_uniform_color([0.0, 0.0, 0.0])
    mesh.compute_vertex_normals()
    return mesh


def _line_set(points: np.ndarray) -> o3d.geometry.LineSet:
    line = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(np.asarray(points, dtype=float)),
        o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32)),
    )
    line.colors = o3d.utility.Vector3dVector([[0.0, 0.0, 0.0]])
    return line


def _closest_surface(
    scene: o3d.t.geometry.RaycastingScene,
    point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    query = np.asarray(point, dtype=np.float32).reshape((1, 3))
    closest = scene.compute_closest_points(o3d.core.Tensor(query))
    anchor = closest["points"].numpy().astype(float)[0]
    normal = closest["primitive_normals"].numpy().astype(float)[0]
    length = float(np.linalg.norm(normal))
    if not np.isfinite(anchor).all() or not np.isfinite(normal).all() or length <= 1e-8:
        raise ValueError("无法将点击位置映射到配准后移动模型。")
    return anchor, normal / length


def _vertex_area_weights(mesh: o3d.geometry.TriangleMesh) -> np.ndarray:
    vertices = np.asarray(mesh.vertices, dtype=float)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    points = vertices[triangles]
    areas = np.linalg.norm(
        np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0]),
        axis=1,
    ) * 0.5
    weights = np.zeros(len(vertices), dtype=float)
    for corner in range(3):
        np.add.at(weights, triangles[:, corner], areas / 3.0)
    return np.maximum(weights, 1e-12)


def calculate_general_annotation(
    vertices: np.ndarray,
    signed_distances_mm: np.ndarray,
    area_weights: np.ndarray,
    *,
    anchor_mm: np.ndarray,
    surface_normal: np.ndarray,
    radius_mm: float,
    annotation_id: str,
) -> GeneralAnnotation:
    radius = float(radius_mm)
    if not 0.05 <= radius <= 3.0:
        raise ValueError("标注平均范围必须位于 0.05–3.00 mm。")
    points = np.asarray(vertices, dtype=float)
    values = np.asarray(signed_distances_mm, dtype=float)
    weights = np.asarray(area_weights, dtype=float)
    anchor = np.asarray(anchor_mm, dtype=float)
    normal = np.asarray(surface_normal, dtype=float)
    normal /= max(float(np.linalg.norm(normal)), 1e-12)
    offsets = points - anchor
    axial = offsets @ normal
    planar = offsets - np.outer(axial, normal)
    inside = (
        np.isfinite(values)
        & (np.linalg.norm(planar, axis=1) <= radius)
        & (np.abs(axial) <= max(0.15, min(radius, 0.75)))
    )
    if not np.any(inside):
        nearest = int(np.argmin(np.linalg.norm(offsets, axis=1)))
        if float(np.linalg.norm(offsets[nearest])) > max(0.60, radius * 3.0):
            raise ValueError("点击位置周围没有有效的移动模型偏差数据。")
        inside[nearest] = True
    local_weights = weights[inside]
    mean = float(np.average(values[inside], weights=local_weights))
    return GeneralAnnotation(
        annotation_id=str(annotation_id),
        anchor_mm=anchor.copy(),
        surface_normal=normal.copy(),
        radius_mm=radius,
        mean_signed_deviation_mm=mean,
        sample_count=int(np.count_nonzero(inside)),
    )


def _pan_look_at(
    view_matrix: np.ndarray,
    projection_matrix: np.ndarray,
    focus_point: np.ndarray,
    delta_x_pixels: float,
    delta_y_pixels: float,
    viewport_height_pixels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    view = np.asarray(view_matrix, dtype=float)
    inverse = np.linalg.inv(view)
    eye = inverse[:3, 3]
    right = inverse[:3, 0]
    up = inverse[:3, 1]
    forward = -inverse[:3, 2]
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up /= max(float(np.linalg.norm(up)), 1e-12)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    focus = np.asarray(focus_point, dtype=float)
    distance = float(np.dot(focus - eye, forward))
    projection_y = abs(float(np.asarray(projection_matrix)[1, 1]))
    world_per_pixel = 2.0 / max(
        projection_y * max(float(viewport_height_pixels), 1.0), 1e-12
    )
    shift = world_per_pixel * (
        -float(delta_x_pixels) * right + float(delta_y_pixels) * up
    )
    center = eye + max(distance, 1e-6) * forward
    return center + shift, eye + shift, up


def _rotation_about_point(
    center: np.ndarray,
    axis: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    normalized_axis = np.asarray(axis, dtype=float)
    length = float(np.linalg.norm(normalized_axis))
    if length <= 1e-12 or abs(float(angle_radians)) <= 1e-12:
        return np.eye(4, dtype=float)
    x, y, z = normalized_axis / length
    cosine = math.cos(float(angle_radians))
    sine = math.sin(float(angle_radians))
    one_minus_cosine = 1.0 - cosine
    rotation = np.array(
        [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ],
        dtype=float,
    )
    pivot = np.asarray(center, dtype=float).reshape(3)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = pivot - rotation @ pivot
    return transform


def _transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float).reshape((4, 4))
    homogeneous = np.append(np.asarray(point, dtype=float).reshape(3), 1.0)
    return (matrix @ homogeneous)[:3]


class GeneralResultViewer:
    _DEVIATION = "偏差色图（配准后移动模型）"
    _TARGET = "固定/参考模型"
    _SOURCE = "配准后移动模型"
    _OVERLAY = "固定/移动叠加"

    def __init__(self, data: ViewerData) -> None:
        self.data = data
        self.deviation_scale = data.deviation_scale
        self._minimum_critical_mm = self.deviation_scale.minimum_critical_mm
        self._maximum_critical_mm = self.deviation_scale.maximum_critical_mm
        self._mode = self._DEVIATION
        self._model_transform = np.eye(4, dtype=float)
        self._model_center_mm = np.zeros(3, dtype=float)
        self._model_rotate_active = False
        self._model_rotate_last: tuple[int, int] | None = None
        self._middle_pan_active = False
        self._middle_pan_last: tuple[int, int] | None = None
        self._annotation_click_start: tuple[int, int] | None = None
        self._annotation_enabled = False
        self._annotation_radius_mm = 0.10
        self._annotations: list[GeneralAnnotation] = []
        self._annotation_geometry_names: set[str] = set()
        self._annotation_labels: list[gui.Label3D] = []
        self._annotation_label_local_points: list[np.ndarray] = []

        app = gui.Application.instance
        self.window = app.create_window(f"通用模型配准结果 v{__version__}", 1420, 900)
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)
        self.scene_widget.scene.set_background([0.92, 0.92, 0.92, 1.0])
        self.scene_widget.scene.show_axes(True)
        self.scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.MED_SHADOWS,
            np.array([0.45, -0.65, -0.62], dtype=np.float32),
        )

        self._mesh_material = rendering.MaterialRecord()
        self._mesh_material.shader = "defaultLit"
        self._line_material = rendering.MaterialRecord()
        self._line_material.shader = "unlitLine"
        self._line_material.line_width = 12.0
        self._annotation_material = rendering.MaterialRecord()
        self._annotation_material.shader = "defaultUnlit"
        self._annotation_material.base_color = [0.0, 0.0, 0.0, 1.0]

        self._target = clone_mesh(data.target)
        self._target.paint_uniform_color([0.72, 0.72, 0.74])
        self._target.compute_vertex_normals()
        self._source = clone_mesh(data.aligned)
        self._source.paint_uniform_color([0.75, 0.77, 0.80])
        self._source.compute_vertex_normals()
        self._overlay_target = clone_mesh(data.target)
        self._overlay_target.paint_uniform_color([1.0, 0.72, 0.05])
        self._overlay_target.compute_vertex_normals()
        self._overlay_source = clone_mesh(data.aligned)
        self._overlay_source.paint_uniform_color([0.05, 0.82, 0.95])
        self._overlay_source.compute_vertex_normals()
        self._deviation = self._colored_source()

        self._aligned_vertices = np.asarray(data.aligned.vertices, dtype=float)
        self._vertex_weights = _vertex_area_weights(data.aligned)
        self._surface_scene = o3d.t.geometry.RaycastingScene()
        self._surface_scene.add_triangles(
            o3d.t.geometry.TriangleMesh.from_legacy(data.aligned)
        )
        self._load_annotations()

        self._add_geometries()
        self._build_panel()
        self._build_legend()
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.add_child(self.legend_image)
        self.window.add_child(self.legend_unit)
        for label in self._scale_labels:
            self.window.add_child(label)
        self.window.add_child(self.deviation_hint)
        self.window.set_on_layout(self._on_layout)
        self.scene_widget.set_on_mouse(self._on_mouse)

        minimum = np.minimum(
            data.target.get_axis_aligned_bounding_box().min_bound,
            data.aligned.get_axis_aligned_bounding_box().min_bound,
        )
        maximum = np.maximum(
            data.target.get_axis_aligned_bounding_box().max_bound,
            data.aligned.get_axis_aligned_bounding_box().max_bound,
        )
        bounds = o3d.geometry.AxisAlignedBoundingBox(minimum, maximum)
        self._model_center_mm = np.asarray(bounds.get_center(), dtype=float)
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
        extent = np.asarray(bounds.get_extent(), dtype=float)
        diagonal = max(float(np.linalg.norm(extent)), 1.0)
        self._ortho_initial_half_height = max(float(np.max(extent)) * 0.62, 1.0)
        self._ortho_half_height = self._ortho_initial_half_height
        self._camera_far = max(diagonal * 50.0, 1000.0)
        self._apply_projection()
        self._apply_mode()
        self._refresh_annotations()

    def _colored_source(self) -> o3d.geometry.TriangleMesh:
        mesh = clone_mesh(self.data.aligned)
        mesh.vertex_colors = o3d.utility.Vector3dVector(
            self.deviation_scale.map_colors(self.data.signed_distances_mm)
        )
        mesh.compute_vertex_normals()
        return mesh

    def _add_geometries(self) -> None:
        for name, mesh in self._base_geometries().items():
            self.scene_widget.scene.add_geometry(name, mesh, self._mesh_material)

    def _base_geometries(self) -> dict[str, o3d.geometry.TriangleMesh]:
        return {
            "deviation": self._deviation,
            "target": self._target,
            "source": self._source,
            "overlay_target": self._overlay_target,
            "overlay_source": self._overlay_source,
        }

    def _build_panel(self) -> None:
        em = self.window.theme.font_size
        self.panel = gui.ScrollableVert(
            0.65 * em,
            gui.Margins(0.6 * em, 0.6 * em, 0.6 * em, 0.6 * em),
        )
        self.panel.add_child(gui.Label("通用模型配准结果"))
        self.panel.add_child(gui.Label("显示模式"))
        self.mode_combo = gui.Combobox()
        for item in (self._DEVIATION, self._TARGET, self._SOURCE, self._OVERLAY):
            self.mode_combo.add_item(item)
        self.mode_combo.set_on_selection_changed(self._on_mode_changed)
        self.panel.add_child(self.mode_combo)

        self.panel.add_child(gui.Label("配准后移动模型偏差标注"))
        self.annotation_toggle = gui.ToggleSwitch("开启点击标注")
        self.annotation_toggle.is_on = False
        self.annotation_toggle.set_on_clicked(self._on_annotation_enabled)
        self.panel.add_child(self.annotation_toggle)
        radius_row = gui.Horiz(0.35 * em)
        radius_row.add_child(gui.Label("平均范围半径"))
        self.annotation_radius_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.annotation_radius_edit.set_limits(0.05, 3.0)
        self.annotation_radius_edit.decimal_precision = 2
        self.annotation_radius_edit.set_value(self._annotation_radius_mm)
        self.annotation_radius_edit.set_on_value_changed(self._on_radius_changed)
        radius_row.add_child(self.annotation_radius_edit)
        radius_row.add_child(gui.Label("mm"))
        self.panel.add_child(radius_row)
        annotation_row = gui.Horiz(0.35 * em)
        delete_button = gui.Button("删除上一个标注")
        delete_button.set_on_clicked(self._delete_last_annotation)
        annotation_row.add_child(delete_button)
        clear_button = gui.Button("清空全部")
        clear_button.set_on_clicked(self._clear_annotations)
        annotation_row.add_child(clear_button)
        self.panel.add_child(annotation_row)
        self.annotation_status = gui.Label("开启后短按左键可添加局部平均偏差。")
        self.panel.add_child(self.annotation_status)

        self.panel.add_child(gui.Label("偏差色阶临界值（mm）"))
        minimum_row = gui.Horiz(0.35 * em)
        minimum_row.add_child(gui.Label("负向临界"))
        self.minimum_critical_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.minimum_critical_edit.set_limits(-100.0, 0.0)
        self.minimum_critical_edit.decimal_precision = 3
        self.minimum_critical_edit.set_value(self._minimum_critical_mm)
        minimum_row.add_child(self.minimum_critical_edit)
        self.panel.add_child(minimum_row)
        maximum_row = gui.Horiz(0.35 * em)
        maximum_row.add_child(gui.Label("正向临界"))
        self.maximum_critical_edit = gui.NumberEdit(gui.NumberEdit.DOUBLE)
        self.maximum_critical_edit.set_limits(0.0, 100.0)
        self.maximum_critical_edit.decimal_precision = 3
        self.maximum_critical_edit.set_value(self._maximum_critical_mm)
        maximum_row.add_child(self.maximum_critical_edit)
        self.panel.add_child(maximum_row)
        self.minimum_critical_edit.set_on_value_changed(self._on_minimum_changed)
        self.maximum_critical_edit.set_on_value_changed(self._on_maximum_changed)
        apply_button = gui.Button("应用临界值 / 刷新模型")
        apply_button.set_on_clicked(self._apply_critical_limits)
        self.panel.add_child(apply_button)
        self.critical_status = gui.Label("输入后回车或点击应用，偏差色图将同步重绘。")
        self.panel.add_child(self.critical_status)

        self.panel.add_child(
            gui.Label(
                "红=正偏差  蓝=负偏差  绿=名义范围\n"
                "黄线=固定模型  青线=移动模型\n"
                "左键拖动=旋转  滚轮=缩放  中键拖动=平移"
            )
        )
        close_button = gui.Button("关闭")
        close_button.set_on_clicked(self.window.close)
        self.panel.add_child(close_button)

    def _legend_pixels(self, height: int = 700, width: int = 28) -> np.ndarray:
        height = max(2, int(height))
        width = max(1, int(width))
        ticks = np.asarray(self.deviation_scale.legend_ticks_mm, dtype=float)
        positions = np.linspace(0.0, len(ticks) - 1.0, height)
        values = np.interp(positions, np.arange(len(ticks), dtype=float), ticks)
        colors = np.clip(self.deviation_scale.map_colors(values), 0.0, 1.0)
        pixels = np.repeat(colors[:, None, :], width, axis=1)
        return np.ascontiguousarray(np.rint(pixels * 255.0).astype(np.uint8))

    def _build_legend(self) -> None:
        self._legend_image_size = (28, 700)
        self.legend_image = gui.ImageWidget(o3d.geometry.Image(self._legend_pixels()))
        self.legend_unit = gui.Label("mm")
        self.legend_unit.text_color = gui.Color(0.18, 0.18, 0.18)
        self.legend_unit.background_color = gui.Color(0.0, 0.0, 0.0, 0.0)
        self._scale_labels: list[gui.Label] = []
        for _ in range(14):
            label = gui.Label("")
            label.text_color = gui.Color(0.18, 0.18, 0.18)
            label.background_color = gui.Color(0.0, 0.0, 0.0, 0.0)
            self._scale_labels.append(label)
        self.deviation_hint = gui.Label(
            "彩虹图表示配准后浮动 STL 相对固定 STL 的表面偏差（mm）"
        )
        self.deviation_hint.text_color = gui.Color(0.28, 0.28, 0.28)
        self.deviation_hint.background_color = gui.Color(0.0, 0.0, 0.0, 0.0)
        self._refresh_scale_labels()

    def _refresh_scale_labels(self) -> None:
        for label, value in zip(
            self._scale_labels,
            self.deviation_scale.legend_ticks_mm,
        ):
            label.text = f"{value:+.4f}"
        if hasattr(self, "legend_image"):
            width, height = self._legend_image_size
            self.legend_image.update_image(
                o3d.geometry.Image(self._legend_pixels(height, width))
            )

    def _on_layout(self, _context: gui.LayoutContext) -> None:
        rect = self.window.content_rect
        panel_width = min(520, max(420, int(rect.width * 0.34)))
        scene_width = rect.width - panel_width
        self.scene_widget.frame = gui.Rect(rect.x, rect.y, scene_width, rect.height)
        self.panel.frame = gui.Rect(rect.get_right() - panel_width, rect.y, panel_width, rect.height)
        legend_height = min(700, max(390, rect.height - 150))
        legend_x = rect.x + 14
        legend_y = rect.y + 44
        self.legend_unit.frame = gui.Rect(legend_x, rect.y + 14, 48, 24)
        legend_image_width = 28
        image_size = (legend_image_width, legend_height)
        if image_size != self._legend_image_size:
            self._legend_image_size = image_size
            self.legend_image.update_image(
                o3d.geometry.Image(
                    self._legend_pixels(legend_height, legend_image_width)
                )
            )
        self.legend_image.frame = gui.Rect(
            legend_x,
            legend_y,
            legend_image_width,
            legend_height,
        )
        label_x = legend_x + 38
        label_width = max(78, min(108, scene_width - label_x - 8))
        label_height = 22
        last_tick = max(len(self._scale_labels) - 1, 1)
        for index, label in enumerate(self._scale_labels):
            tick_y = legend_y + int(round(index * legend_height / last_tick))
            label.frame = gui.Rect(
                label_x,
                tick_y - label_height // 2,
                label_width,
                label_height,
            )
        self.deviation_hint.frame = gui.Rect(
            rect.x + 14,
            rect.get_bottom() - 34,
            max(300, scene_width - 28),
            26,
        )
        self._apply_projection()

    def _apply_projection(self) -> None:
        height = max(float(self.scene_widget.frame.height), 1.0)
        aspect = max(float(self.scene_widget.frame.width) / height, 0.1)
        half_height = max(float(self._ortho_half_height), 1e-4)
        half_width = half_height * aspect
        self.scene_widget.scene.camera.set_projection(
            rendering.Camera.Projection.Ortho,
            -half_width,
            half_width,
            -half_height,
            half_height,
            0.01,
            self._camera_far,
        )

    def _on_mode_changed(self, text: str, _index: int) -> None:
        self._mode = text
        self._apply_mode()

    def _apply_mode(self) -> None:
        scene = self.scene_widget.scene
        scene.show_geometry("deviation", self._mode == self._DEVIATION)
        scene.show_geometry("target", self._mode == self._TARGET)
        scene.show_geometry("source", self._mode == self._SOURCE)
        scene.show_geometry("overlay_target", self._mode == self._OVERLAY)
        scene.show_geometry("overlay_source", self._mode == self._OVERLAY)
        for name in self._annotation_geometry_names:
            if scene.has_geometry(name):
                scene.show_geometry(name, True)

    def _replace_mesh(self, name: str, mesh: o3d.geometry.TriangleMesh) -> None:
        scene = self.scene_widget.scene
        if scene.has_geometry(name):
            scene.remove_geometry(name)
        if len(mesh.triangles):
            scene.add_geometry(name, mesh, self._mesh_material)
            scene.set_geometry_transform(name, self._model_transform)

    def _apply_model_transform(self) -> None:
        scene = self.scene_widget.scene
        for name in (*self._base_geometries(), *self._annotation_geometry_names):
            if scene.has_geometry(name):
                scene.set_geometry_transform(name, self._model_transform)
        for label, local_point in zip(
            self._annotation_labels,
            self._annotation_label_local_points,
        ):
            label.position = _transform_point(
                self._model_transform,
                local_point,
            ).astype(np.float32)

    def _rotate_model(self, delta_x: float, delta_y: float) -> None:
        distance = math.hypot(float(delta_x), float(delta_y))
        if distance <= 1e-9:
            return
        view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=float)
        inverse_view = np.linalg.inv(view)
        camera_right = inverse_view[:3, 0]
        camera_up = inverse_view[:3, 1]
        axis = float(delta_y) * camera_right + float(delta_x) * camera_up
        incremental = _rotation_about_point(
            self._model_center_mm,
            axis,
            math.radians(0.35) * distance,
        )
        self._model_transform = incremental @ self._model_transform
        self._apply_model_transform()
        self.scene_widget.force_redraw()

    def _on_minimum_changed(self, value: float) -> None:
        self._minimum_critical_mm = min(float(value), 0.0)

    def _on_maximum_changed(self, value: float) -> None:
        self._maximum_critical_mm = max(float(value), 0.0)

    def _apply_critical_limits(self) -> None:
        try:
            self.deviation_scale = self.deviation_scale.with_critical_limits(
                self._minimum_critical_mm,
                self._maximum_critical_mm,
            )
            self._deviation = self._colored_source()
            self._replace_mesh("deviation", self._deviation)
            self._refresh_scale_labels()
            self._apply_mode()
            self._refresh_annotations()
            self.critical_status.text = (
                f"已更新：负向 {self._minimum_critical_mm:+.3f} mm，"
                f"正向 {self._maximum_critical_mm:+.3f} mm。"
            )
        except Exception as exc:
            self.critical_status.text = str(exc)

    def _on_annotation_enabled(self, enabled: bool) -> None:
        self._annotation_enabled = bool(enabled)
        self._annotation_click_start = None
        self.annotation_status.text = (
            "已开启：短按左键可添加局部平均偏差。"
            if enabled
            else "已关闭点击标注。"
        )

    def _on_radius_changed(self, value: float) -> None:
        self._annotation_radius_mm = float(np.clip(value, 0.05, 3.0))
        if self._annotations:
            last = self._annotations[-1]
            self._create_annotation(last.anchor_mm, last.annotation_id, True)

    def _annotation_text(self, value: float) -> str:
        if value > self.deviation_scale.configured_maximum_nominal_mm:
            category = "正偏差"
        elif value < self.deviation_scale.configured_minimum_nominal_mm:
            category = "负偏差"
        else:
            category = "名义范围"
        return f"{category} {value:+.3f} mm"

    def _create_annotation(
        self,
        requested_anchor: np.ndarray,
        annotation_id: str | None = None,
        replace_last: bool = False,
    ) -> None:
        try:
            anchor, normal = _closest_surface(self._surface_scene, requested_anchor)
            annotation = calculate_general_annotation(
                self._aligned_vertices,
                self.data.signed_distances_mm,
                self._vertex_weights,
                anchor_mm=anchor,
                surface_normal=normal,
                radius_mm=self._annotation_radius_mm,
                annotation_id=annotation_id or f"A{len(self._annotations) + 1:03d}",
            )
        except Exception as exc:
            self.annotation_status.text = str(exc)
            return
        if replace_last and self._annotations:
            self._annotations[-1] = annotation
        else:
            self._annotations.append(annotation)
        self.annotation_status.text = self._annotation_text(
            annotation.mean_signed_deviation_mm
        )
        self._refresh_annotations()
        self._persist_annotations()

    def _clear_annotation_display(self) -> None:
        scene = self.scene_widget.scene
        for name in tuple(self._annotation_geometry_names):
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        self._annotation_geometry_names.clear()
        for label in self._annotation_labels:
            self.scene_widget.remove_3d_label(label)
        self._annotation_labels.clear()
        self._annotation_label_local_points.clear()

    def _refresh_annotations(self) -> None:
        self._clear_annotation_display()
        try:
            view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=float)
            camera_up = np.linalg.inv(view)[:3, 1]
        except Exception:
            camera_up = np.array([0.0, 1.0, 0.0])
        camera_up = self._model_transform[:3, :3].T @ camera_up
        for index, annotation in enumerate(self._annotations):
            normal, up, right = _surface_frame(annotation.surface_normal, camera_up)
            marker_radius = max(
                min(annotation.radius_mm * 0.24, self._ortho_half_height * 0.022),
                0.02,
            )
            surface_offset = normal * max(marker_radius * 0.2, 0.005)
            anchor = annotation.anchor_mm + surface_offset
            prefix = f"general_annotation_{index}"
            ring_name = prefix + "_ring"
            marker_name = prefix + "_marker"
            leader_name = prefix + "_leader"
            self.scene_widget.scene.add_geometry(
                ring_name,
                _annotation_ring(anchor, annotation.radius_mm, normal, up, right),
                self._annotation_material,
            )
            marker = o3d.geometry.TriangleMesh.create_sphere(
                radius=marker_radius, resolution=12
            )
            marker.translate(anchor)
            marker.compute_vertex_normals()
            self.scene_widget.scene.add_geometry(
                marker_name, marker, self._annotation_material
            )
            label_point = (
                anchor
                + right * max(annotation.radius_mm * 1.7, marker_radius * 5.0)
                + up * max(annotation.radius_mm * 1.25, marker_radius * 3.0)
            )
            self.scene_widget.scene.add_geometry(
                leader_name,
                _line_set(np.vstack((anchor, label_point))),
                self._line_material,
            )
            label = self.scene_widget.add_3d_label(
                _transform_point(self._model_transform, label_point).astype(np.float32),
                self._annotation_text(annotation.mean_signed_deviation_mm),
            )
            label.color = gui.Color(0.02, 0.02, 0.02)
            label.scale = 1.15
            self._annotation_labels.append(label)
            self._annotation_label_local_points.append(label_point.copy())
            self._annotation_geometry_names.update(
                (ring_name, marker_name, leader_name)
            )
        self._apply_model_transform()
        self.scene_widget.force_redraw()

    def _delete_last_annotation(self) -> None:
        if self._annotations:
            self._annotations.pop()
            self._refresh_annotations()
            self._persist_annotations()

    def _clear_annotations(self) -> None:
        self._annotations.clear()
        self._clear_annotation_display()
        self._persist_annotations()

    def _persist_annotations(self) -> None:
        path = self.data.results_path.parent / "viewer_annotations.json"
        path.write_text(
            json.dumps(
                {
                    "version": __version__,
                    "annotations": [item.as_dict() for item in self._annotations],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load_annotations(self) -> None:
        path = self.data.results_path.parent / "viewer_annotations.json"
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            loaded: list[GeneralAnnotation] = []
            for item in payload.get("annotations", []):
                loaded.append(
                    GeneralAnnotation(
                        annotation_id=str(item["annotation_id"]),
                        anchor_mm=np.asarray(item["anchor_mm"], dtype=float),
                        surface_normal=np.asarray(item["surface_normal"], dtype=float),
                        radius_mm=float(item["radius_mm"]),
                        mean_signed_deviation_mm=float(item["mean_signed_deviation_mm"]),
                        sample_count=int(item["sample_count"]),
                    )
                )
            self._annotations = loaded
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
            self._annotations = []

    def _request_pick(self, window_x: int, window_y: int) -> None:
        frame = self.scene_widget.frame
        local_x = int(window_x - frame.x)
        local_y = int(window_y - frame.y)
        width = max(int(frame.width), 1)
        height = max(int(frame.height), 1)
        if not (0 <= local_x < width and 0 <= local_y < height):
            return
        inverse_model_transform = np.linalg.inv(self._model_transform.copy())

        def on_depth(depth_image: o3d.geometry.Image) -> None:
            depth = np.asarray(depth_image)
            if depth.ndim != 2 or local_y >= depth.shape[0] or local_x >= depth.shape[1]:
                return
            value = float(depth[local_y, local_x])
            if not np.isfinite(value) or value >= 0.999999:
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: setattr(
                        self.annotation_status,
                        "text",
                        "点击位置没有命中模型。",
                    ),
                )
                return
            world = np.asarray(
                self.scene_widget.scene.camera.unproject(
                    float(local_x),
                    float(local_y),
                    value,
                    float(width),
                    float(height),
                ),
                dtype=float,
            ).reshape(3)
            local_model_point = _transform_point(inverse_model_transform, world)
            gui.Application.instance.post_to_main_thread(
                self.window,
                lambda point=local_model_point: self._create_annotation(point),
            )

        self.scene_widget.scene.scene.render_to_depth_image(on_depth)

    def _on_mouse(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        if event.type == gui.MouseEvent.WHEEL:
            factor = math.exp(-0.14 * float(event.wheel_dy))
            self._ortho_half_height = float(
                np.clip(
                    self._ortho_half_height * factor,
                    self._ortho_initial_half_height * 0.015,
                    self._ortho_initial_half_height * 25.0,
                )
            )
            self._apply_projection()
            self.scene_widget.force_redraw()
            return gui.Widget.EventCallbackResult.CONSUMED

        left_down = event.is_button_down(gui.MouseButton.LEFT)
        if event.type == gui.MouseEvent.BUTTON_DOWN and left_down:
            self._model_rotate_active = True
            self._model_rotate_last = (int(event.x), int(event.y))
            if self._annotation_enabled:
                self._annotation_click_start = (int(event.x), int(event.y))
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.DRAG and self._model_rotate_active:
            previous = self._model_rotate_last
            current = (int(event.x), int(event.y))
            self._model_rotate_last = current
            if previous is not None:
                self._rotate_model(current[0] - previous[0], current[1] - previous[1])
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.BUTTON_UP and self._model_rotate_active:
            self._model_rotate_active = False
            self._model_rotate_last = None
            if self._annotation_enabled and self._annotation_click_start:
                start = self._annotation_click_start
                self._annotation_click_start = None
                if math.hypot(float(event.x - start[0]), float(event.y - start[1])) <= 5.0:
                    self._request_pick(int(event.x), int(event.y))
            return gui.Widget.EventCallbackResult.CONSUMED

        middle_down = event.is_button_down(gui.MouseButton.MIDDLE)
        if event.type == gui.MouseEvent.BUTTON_DOWN and middle_down:
            self._middle_pan_active = True
            self._middle_pan_last = (int(event.x), int(event.y))
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.DRAG and self._middle_pan_active:
            previous = self._middle_pan_last
            current = (int(event.x), int(event.y))
            self._middle_pan_last = current
            if previous is not None:
                self._pan_camera(current[0] - previous[0], current[1] - previous[1])
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.BUTTON_UP and self._middle_pan_active:
            self._middle_pan_active = False
            self._middle_pan_last = None
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.HANDLED

    def _pan_camera(self, delta_x: float, delta_y: float) -> None:
        camera = self.scene_widget.scene.camera
        view = np.asarray(camera.get_view_matrix(), dtype=float)
        previous_eye = np.linalg.inv(view)[:3, 3]
        rotation_center = np.asarray(self.scene_widget.center_of_rotation, dtype=float)
        center, eye, up = _pan_look_at(
            view,
            camera.get_projection_matrix(),
            rotation_center,
            delta_x,
            delta_y,
            self.scene_widget.frame.height,
        )
        shift = eye - previous_eye
        camera.look_at(
            center.astype(np.float32),
            eye.astype(np.float32),
            up.astype(np.float32),
        )
        self.scene_widget.center_of_rotation = (rotation_center + shift).astype(np.float32)
        self._apply_projection()
        self.scene_widget.force_redraw()


def run_general_result_viewer(
    results_path: str | Path,
    *,
    target_override: str | Path | None = None,
    aligned_override: str | Path | None = None,
) -> None:
    data = load_viewer_data(
        results_path,
        target_override=target_override,
        aligned_override=aligned_override,
    )
    _run_viewer(data)


def run_pair_result_viewer(
    target_path: str | Path,
    aligned_path: str | Path,
) -> None:
    _run_viewer(load_pair_viewer_data(target_path, aligned_path))


def _run_viewer(data: ViewerData) -> None:
    app = gui.Application.instance
    app.initialize()
    _configure_font(app)
    GeneralResultViewer(data)
    app.run()
