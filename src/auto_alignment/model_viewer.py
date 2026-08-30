"""Standalone input-mesh viewer with freehand triangle selection editing."""

from __future__ import annotations

import math
from pathlib import Path
import threading

import numpy as np
import open3d as o3d
from open3d.visualization import gui, rendering

from .mesh_io import load_mesh
from .mesh_selection import (
    ModelEditState,
    apply_edit_state,
    bounded_component_faces,
    clone_state_with_masks,
    floating_component_faces,
    load_edit_state,
    save_edit_state,
)
from .result_viewer import (
    _configure_font,
    _pan_look_at,
    _rotation_about_point,
)
from .version import __version__


LASSO_LINE_WIDTH_PX = 3.0


_UI_TEXT = """
模型查看选区编辑自由套索透选已选删除工作副本
左键拖动划选Ctrl左键取消右键拖动旋转右键菜单中键平移滚轮缩放
全选取消选区反选有界组件删除所选面片去除浮动面片撤销重做保存关闭
主模型表面积最大连通组件原始STL不会被修改
选择短按重置视角并增加移除发现分离无法有效当前匹配尚未初始化重新
全部与按约的、（），。：；·±%/–～■+0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"""


def _mesh_from_faces(
    mesh: o3d.geometry.TriangleMesh,
    face_mask: np.ndarray,
) -> o3d.geometry.TriangleMesh:
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    mask = np.asarray(face_mask, dtype=bool).reshape(-1)
    if len(mask) != len(triangles) or not np.any(mask):
        return o3d.geometry.TriangleMesh()
    result = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(mesh.vertices, dtype=float).copy()),
        o3d.utility.Vector3iVector(triangles[mask].copy()),
    )
    result.remove_unreferenced_vertices()
    result.compute_triangle_normals()
    result.compute_vertex_normals()
    return result


def _selection_render_meshes(
    mesh: o3d.geometry.TriangleMesh,
    selected_faces: np.ndarray,
) -> tuple[o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh]:
    """Split selected and unselected faces into non-overlapping render meshes."""
    selected = np.asarray(selected_faces, dtype=bool).reshape(-1)
    if len(selected) != len(mesh.triangles):
        raise ValueError("选区三角面掩码与工作副本不匹配。")
    return _mesh_from_faces(mesh, ~selected), _mesh_from_faces(mesh, selected)


def _stroke_ribbon_screen_mesh(
    points: np.ndarray,
    width_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a solid screen-space ribbon for a freehand stroke.

    Open3D's Windows/Filament backend can clamp native line primitives to one
    pixel even when ``MaterialRecord.line_width`` is much larger.  Representing
    every stroke segment as an overlapping quad makes the requested pixel width
    independent of the driver's wide-line support.
    """
    path = np.asarray(points, dtype=float).reshape((-1, 2))
    half_width = max(float(width_px), 1.0) * 0.5
    vertices: list[np.ndarray] = []
    triangles: list[tuple[int, int, int]] = []
    for start, end in zip(path[:-1], path[1:]):
        delta = end - start
        length = float(np.linalg.norm(delta))
        if not np.isfinite(length) or length <= 1e-9:
            continue
        tangent = delta / length
        normal = np.array((-tangent[1], tangent[0]), dtype=float)
        # Extend each quad by half the stroke width. Adjacent segments overlap
        # at corners, so fast mouse movement cannot leave visible pinholes.
        extended_start = start - tangent * half_width
        extended_end = end + tangent * half_width
        base = len(vertices)
        vertices.extend(
            (
                extended_start + normal * half_width,
                extended_start - normal * half_width,
                extended_end + normal * half_width,
                extended_end - normal * half_width,
            )
        )
        triangles.extend(
            (
                (base, base + 1, base + 2),
                (base + 1, base + 3, base + 2),
                # Duplicate the reverse winding so the ribbon remains visible
                # regardless of the camera/backend culling convention.
                (base + 2, base + 1, base),
                (base + 2, base + 3, base + 1),
            )
        )
    return (
        np.asarray(vertices, dtype=float).reshape((-1, 2)),
        np.asarray(triangles, dtype=np.int32).reshape((-1, 3)),
    )


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape((-1, 2))
    polygon = np.asarray(polygon, dtype=float).reshape((-1, 2))
    inside = np.zeros(len(points), dtype=bool)
    if len(polygon) < 3 or not len(points):
        return inside
    x, y = points[:, 0], points[:, 1]
    previous = polygon[-1]
    for current in polygon:
        x1, y1 = previous
        x2, y2 = current
        crossing = ((y1 > y) != (y2 > y)) & (
            x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-15) + x1
        )
        inside ^= crossing
        previous = current
    return inside


def _cross_2d(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]


def _projected_triangles_intersect_polygon(
    projected_triangles: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """Return every projected triangle touched by a freehand polygon.

    A true through-selection is a 2-D selection prism. Testing only face
    centers and vertices leaves holes when a polygon crosses a triangle edge
    or lies completely inside a large triangle. This routine covers all four
    overlap cases: triangle vertex in polygon, polygon vertex in triangle,
    crossing edges, and full containment.
    """
    triangles = np.asarray(projected_triangles, dtype=float).reshape((-1, 3, 2))
    polygon = np.asarray(polygon, dtype=float).reshape((-1, 2))
    selected = np.zeros(len(triangles), dtype=bool)
    if len(polygon) < 3 or not len(triangles):
        return selected

    finite = np.all(np.isfinite(triangles), axis=(1, 2))
    polygon_min = np.min(polygon, axis=0)
    polygon_max = np.max(polygon, axis=0)
    triangle_min = np.min(np.where(np.isfinite(triangles), triangles, np.inf), axis=1)
    triangle_max = np.max(np.where(np.isfinite(triangles), triangles, -np.inf), axis=1)
    candidates = np.flatnonzero(
        finite
        & np.all(triangle_max >= polygon_min, axis=1)
        & np.all(triangle_min <= polygon_max, axis=1)
    )
    if not len(candidates):
        return selected

    candidate_triangles = triangles[candidates]
    vertex_inside = _points_in_polygon(
        candidate_triangles.reshape((-1, 2)), polygon
    ).reshape((-1, 3)).any(axis=1)
    selected[candidates[vertex_inside]] = True
    remaining = candidates[~vertex_inside]
    if not len(remaining):
        return selected

    epsilon = 1e-9
    # Work in chunks so a long freehand stroke and a dense STL do not create
    # an unbounded triangle-by-lasso broadcast allocation.
    chunk_size = 1_024
    polygon_next = np.roll(polygon, -1, axis=0)
    for start in range(0, len(remaining), chunk_size):
        ids = remaining[start : start + chunk_size]
        chunk = triangles[ids]

        points = polygon[None, :, :]
        first = chunk[:, None, 0, :]
        second = chunk[:, None, 1, :]
        third = chunk[:, None, 2, :]
        cross_first = _cross_2d(second - first, points - first)
        cross_second = _cross_2d(third - second, points - second)
        cross_third = _cross_2d(first - third, points - third)
        polygon_inside = (
            ((cross_first >= -epsilon) & (cross_second >= -epsilon) & (cross_third >= -epsilon))
            | ((cross_first <= epsilon) & (cross_second <= epsilon) & (cross_third <= epsilon))
        ).any(axis=1)

        not_containing = np.flatnonzero(~polygon_inside)
        crossing = np.zeros(len(chunk), dtype=bool)
        if len(not_containing):
            local = chunk[not_containing]
            edge_first = local[:, (0, 1, 2), :]
            edge_second = local[:, (1, 2, 0), :]
            polygon_first = polygon[None, None, :, :]
            polygon_second = polygon_next[None, None, :, :]
            triangle_first = edge_first[:, :, None, :]
            triangle_second = edge_second[:, :, None, :]
            orientation_1 = _cross_2d(
                triangle_second - triangle_first,
                polygon_first - triangle_first,
            )
            orientation_2 = _cross_2d(
                triangle_second - triangle_first,
                polygon_second - triangle_first,
            )
            orientation_3 = _cross_2d(
                polygon_second - polygon_first,
                triangle_first - polygon_first,
            )
            orientation_4 = _cross_2d(
                polygon_second - polygon_first,
                triangle_second - polygon_first,
            )
            segment_bbox_overlap = (
                np.maximum(
                    np.minimum(triangle_first[..., 0], triangle_second[..., 0]),
                    np.minimum(polygon_first[..., 0], polygon_second[..., 0]),
                )
                <= np.minimum(
                    np.maximum(triangle_first[..., 0], triangle_second[..., 0]),
                    np.maximum(polygon_first[..., 0], polygon_second[..., 0]),
                )
                + epsilon
            ) & (
                np.maximum(
                    np.minimum(triangle_first[..., 1], triangle_second[..., 1]),
                    np.minimum(polygon_first[..., 1], polygon_second[..., 1]),
                )
                <= np.minimum(
                    np.maximum(triangle_first[..., 1], triangle_second[..., 1]),
                    np.maximum(polygon_first[..., 1], polygon_second[..., 1]),
                )
                + epsilon
            )
            intersects = (
                (orientation_1 * orientation_2 <= epsilon)
                & (orientation_3 * orientation_4 <= epsilon)
                & segment_bbox_overlap
            )
            crossing[not_containing] = intersects.any(axis=(1, 2))
        selected[ids[polygon_inside | crossing]] = True
    return selected


def _frontmost_candidate_faces(
    mesh: o3d.geometry.TriangleMesh,
    candidate_faces: np.ndarray,
    ray_direction: np.ndarray,
) -> np.ndarray:
    """Keep candidates whose own centroid is the first surface on its view ray."""
    candidate = np.asarray(candidate_faces, dtype=bool).reshape(-1)
    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    selected = np.zeros(len(triangles), dtype=bool)
    if len(candidate) != len(triangles):
        raise ValueError("可见性候选掩码与工作副本不匹配。")
    ids = np.flatnonzero(candidate)
    if not len(ids):
        return selected

    direction = np.asarray(ray_direction, dtype=float).reshape(3)
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 1e-12:
        return selected
    direction /= length
    vertices = np.asarray(mesh.vertices, dtype=float)
    centroids = np.mean(vertices[triangles[ids]], axis=1)
    extent = np.asarray(mesh.get_axis_aligned_bounding_box().get_extent(), dtype=float)
    ray_distance = max(float(np.linalg.norm(extent)) * 4.0, 1.0)
    origins = centroids - direction * ray_distance
    directions = np.repeat(direction[None, :], len(ids), axis=0)
    rays = np.column_stack((origins, directions)).astype(np.float32)
    ray_scene = o3d.t.geometry.RaycastingScene()
    ray_scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    hits = ray_scene.cast_rays(o3d.core.Tensor(rays))["primitive_ids"].numpy()
    visible = hits.astype(np.int64, copy=False) == ids
    selected[ids[visible]] = True
    return selected


class ModelSelectionViewer:
    def __init__(
        self,
        mesh_path: str | Path,
        state_path: str | Path,
    ) -> None:
        self.mesh_path = Path(mesh_path).expanduser().resolve(strict=True)
        self.state_path = Path(state_path)
        self.original_mesh, _ = load_mesh(self.mesh_path)
        self.state = load_edit_state(
            self.state_path,
            self.mesh_path,
            len(self.original_mesh.triangles),
        )
        self._undo: list[ModelEditState] = []
        self._redo: list[ModelEditState] = []
        self._model_transform = np.eye(4, dtype=float)
        self._model_center = np.asarray(
            self.original_mesh.get_axis_aligned_bounding_box().get_center(), dtype=float
        )
        self._lasso: list[tuple[int, int]] = []
        self._lasso_subtract = False
        self._lasso_preview_generation = 0
        self._right_active = False
        self._right_dragged = False
        self._right_last: tuple[int, int] | None = None
        self._middle_active = False
        self._middle_last: tuple[int, int] | None = None

        app = gui.Application.instance
        self.window = app.create_window(
            f"模型查看 / 选区编辑 v{__version__} — {self.mesh_path.name}",
            1420,
            900,
        )
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([0.92, 0.92, 0.92, 1.0])
        self.scene_widget.scene.show_axes(True)
        self.scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.MED_SHADOWS,
            np.array([0.45, -0.65, -0.62], dtype=np.float32),
        )
        self.scene_widget.set_view_controls(gui.SceneWidget.Controls.ROTATE_MODEL)

        self.mesh_material = rendering.MaterialRecord()
        self.mesh_material.shader = "defaultLit"
        self.selection_material = rendering.MaterialRecord()
        self.selection_material.shader = "defaultLit"
        self.selection_material.base_color = [1.0, 0.24, 0.02, 1.0]
        self.lasso_material = rendering.MaterialRecord()
        self.lasso_material.shader = "defaultUnlit"
        self.lasso_material.base_color = [1.0, 0.02, 0.0, 1.0]

        self._build_panel()
        self._build_context_menu()
        self.window.add_child(self.scene_widget)
        self.window.add_child(self.panel)
        self.window.add_child(self.context_menu)
        self.window.set_on_layout(self._on_layout)
        self.scene_widget.set_on_mouse(self._on_mouse)

        bounds = self.original_mesh.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
        extent = np.asarray(bounds.get_extent(), dtype=float)
        diagonal = max(float(np.linalg.norm(extent)), 1.0)
        self._ortho_initial_half_height = max(float(np.max(extent)) * 0.62, 1.0)
        self._ortho_half_height = self._ortho_initial_half_height
        self._camera_far = max(diagonal * 50.0, 1000.0)
        self._apply_projection()
        self._refresh_mesh()

    def _build_panel(self) -> None:
        em = self.window.theme.font_size
        self.panel = gui.ScrollableVert(
            0.65 * em,
            gui.Margins(0.7 * em, 0.7 * em, 0.7 * em, 0.7 * em),
        )
        self.panel.add_child(gui.Label("模型查看 / 选区编辑"))
        self.path_label = gui.Label(str(self.mesh_path))
        self.panel.add_child(self.path_label)
        self.panel.add_child(
            gui.Label(
                "左键拖动：自由套索选择\n"
                "Ctrl + 左键：取消划过的选区\n"
                "右键拖动：旋转；右键短按：菜单\n"
                "中键：平移；滚轮：缩放"
            )
        )
        self.through_toggle = gui.ToggleSwitch("透选（选中前后所有面片）")
        self.through_toggle.is_on = False
        self.panel.add_child(self.through_toggle)
        self.count_label = gui.Label("")
        self.panel.add_child(self.count_label)
        self.status_label = gui.Label("编辑会自动保存，原始 STL 不会被修改。")
        self.panel.add_child(self.status_label)
        reset_button = gui.Button("重置视角")
        reset_button.set_on_clicked(self._reset_camera)
        self.panel.add_child(reset_button)
        close_button = gui.Button("保存并关闭")
        close_button.set_on_clicked(self._save_and_close)
        self.panel.add_child(close_button)

    def _menu_button(self, label: str, callback) -> gui.Button:
        button = gui.Button(label)
        button.set_on_clicked(callback)
        self.context_menu.add_child(button)
        return button

    def _build_context_menu(self) -> None:
        em = self.window.theme.font_size
        self.context_menu = gui.Vert(
            0.2 * em,
            gui.Margins(0.4 * em, 0.4 * em, 0.4 * em, 0.4 * em),
        )
        self.context_menu.visible = False
        self._menu_button("全选", self._select_all)
        self._menu_button("取消选区", self._clear_selection)
        self._menu_button("反选", self._invert_selection)
        self.bounded_button = self._menu_button("有界组件", self._bounded_components)
        self.delete_button = self._menu_button("删除所选面片", self._delete_selection)
        self._menu_button("去除所有浮动面片", self._remove_floating_components)
        self.undo_button = self._menu_button("撤销", self._undo_action)
        self.redo_button = self._menu_button("重做", self._redo_action)

    def _on_layout(self, _context: gui.LayoutContext) -> None:
        rect = self.window.content_rect
        panel_width = min(410, max(330, int(rect.width * 0.28)))
        self.scene_widget.frame = gui.Rect(
            rect.x, rect.y, rect.width - panel_width, rect.height
        )
        self.panel.frame = gui.Rect(
            rect.get_right() - panel_width, rect.y, panel_width, rect.height
        )
        if self.context_menu.visible:
            frame = self.context_menu.frame
            width, height = 250, 330
            x = min(max(frame.x, rect.x), rect.get_right() - panel_width - width)
            y = min(max(frame.y, rect.y), rect.get_bottom() - height)
            self.context_menu.frame = gui.Rect(x, y, width, height)
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

    def _refresh_mesh(self) -> None:
        applied = apply_edit_state(self.original_mesh, self.state)
        self.working = applied.mesh
        self.working_original_faces = applied.original_face_indices
        base, selected_mesh = _selection_render_meshes(
            self.working, applied.selected_faces
        )
        scene = self.scene_widget.scene
        for name in ("model", "selected"):
            if scene.has_geometry(name):
                scene.remove_geometry(name)
        if len(base.triangles):
            base.paint_uniform_color([0.72, 0.74, 0.78])
            base.compute_vertex_normals()
            scene.add_geometry("model", base, self.mesh_material)
        if len(selected_mesh.triangles):
            selected_mesh.paint_uniform_color([1.0, 0.24, 0.02])
            selected_mesh.compute_vertex_normals()
            scene.add_geometry("selected", selected_mesh, self.selection_material)
        self._apply_model_transform()
        available = self.state.triangle_count - int(np.count_nonzero(self.state.deleted))
        self.count_label.text = (
            f"当前有效面片：{available:,}\n"
            f"已选：{int(np.count_nonzero(self.state.selected)):,}\n"
            f"已删除（工作副本）：{int(np.count_nonzero(self.state.deleted)):,}"
        )
        has_selection = bool(np.any(self.state.selected))
        self.bounded_button.enabled = has_selection
        self.delete_button.enabled = has_selection
        self.undo_button.enabled = bool(self._undo)
        self.redo_button.enabled = bool(self._redo)
        self.scene_widget.force_redraw()

    def _apply_model_transform(self) -> None:
        scene = self.scene_widget.scene
        for name in ("model", "selected"):
            if scene.has_geometry(name):
                scene.set_geometry_transform(name, self._model_transform)

    def _record_change(self, next_state: ModelEditState, message: str) -> None:
        self._undo.append(self.state)
        if len(self._undo) > 50:
            self._undo.pop(0)
        self._redo.clear()
        self.state = next_state.normalized()
        save_edit_state(self.state_path, self.state)
        self.status_label.text = message
        self.context_menu.visible = False
        self._refresh_mesh()

    def _select_all(self) -> None:
        selected = ~self.state.deleted
        self._record_change(
            clone_state_with_masks(self.state, selected, self.state.deleted),
            "已选中所有有效面片。",
        )

    def _clear_selection(self) -> None:
        self._record_change(
            clone_state_with_masks(
                self.state, np.zeros(self.state.triangle_count, dtype=bool), self.state.deleted
            ),
            "已取消选区。",
        )

    def _invert_selection(self) -> None:
        selected = (~self.state.selected) & (~self.state.deleted)
        self._record_change(
            clone_state_with_masks(self.state, selected, self.state.deleted),
            "已反选。",
        )

    def _bounded_components(self) -> None:
        expanded = bounded_component_faces(
            self.original_mesh, self.state.selected, self.state.deleted
        )
        self._record_change(
            clone_state_with_masks(self.state, expanded, self.state.deleted),
            "已扩选至网格边界。",
        )

    def _delete_selection(self) -> None:
        deleted = self.state.deleted | self.state.selected
        selected = np.zeros(self.state.triangle_count, dtype=bool)
        next_state = clone_state_with_masks(self.state, selected, deleted)
        try:
            apply_edit_state(self.original_mesh, next_state)
        except ValueError as error:
            self.status_label.text = str(error)
            self.context_menu.visible = False
            return
        self._record_change(next_state, "已从工作副本删除所选面片。")

    def _remove_floating_components(self) -> None:
        remove = floating_component_faces(self.original_mesh, self.state.deleted)
        count = int(np.count_nonzero(remove))
        if not count:
            self.status_label.text = "未发现与主模型分离的浮动面片。"
            self.context_menu.visible = False
            return
        deleted = self.state.deleted | remove
        selected = self.state.selected & ~deleted
        next_state = clone_state_with_masks(self.state, selected, deleted)
        try:
            apply_edit_state(self.original_mesh, next_state)
        except ValueError as error:
            self.status_label.text = str(error)
            self.context_menu.visible = False
            return
        self._record_change(next_state, f"已删除 {count:,} 个非主连通组件面片。")

    def _undo_action(self) -> None:
        if not self._undo:
            return
        self._redo.append(self.state)
        self.state = self._undo.pop()
        save_edit_state(self.state_path, self.state)
        self.context_menu.visible = False
        self.status_label.text = "已撤销。"
        self._refresh_mesh()

    def _redo_action(self) -> None:
        if not self._redo:
            return
        self._undo.append(self.state)
        self.state = self._redo.pop()
        save_edit_state(self.state_path, self.state)
        self.context_menu.visible = False
        self.status_label.text = "已重做。"
        self._refresh_mesh()

    def _save_and_close(self) -> None:
        save_edit_state(self.state_path, self.state)
        self.window.close()

    def _show_context_menu(self, x: int, y: int) -> None:
        self.context_menu.frame = gui.Rect(int(x), int(y), 250, 330)
        self.context_menu.visible = True
        self.window.set_needs_layout()
        self._refresh_mesh()

    def _reset_camera(self) -> None:
        self._model_transform = np.eye(4, dtype=float)
        bounds = self.original_mesh.get_axis_aligned_bounding_box()
        self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())
        self._ortho_half_height = self._ortho_initial_half_height
        self._apply_projection()
        self._apply_model_transform()
        self.scene_widget.force_redraw()

    def _rotate_model(self, delta_x: float, delta_y: float) -> None:
        distance = math.hypot(float(delta_x), float(delta_y))
        if distance <= 1e-9:
            return
        view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=float)
        inverse_view = np.linalg.inv(view)
        axis = float(delta_y) * inverse_view[:3, 0] + float(delta_x) * inverse_view[:3, 1]
        incremental = _rotation_about_point(
            self._model_center, axis, math.radians(0.35) * distance
        )
        self._model_transform = incremental @ self._model_transform
        self._apply_model_transform()
        self.scene_widget.force_redraw()

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
        camera.look_at(center.astype(np.float32), eye.astype(np.float32), up.astype(np.float32))
        self.scene_widget.center_of_rotation = (rotation_center + shift).astype(np.float32)
        self._apply_projection()
        self.scene_widget.force_redraw()

    def _refresh_lasso(self) -> None:
        scene = self.scene_widget.scene
        if scene.has_geometry("lasso"):
            scene.remove_geometry("lasso")
        if len(self._lasso) < 2:
            return
        frame = self.scene_widget.frame
        width, height = max(int(frame.width), 1), max(int(frame.height), 1)
        camera = scene.camera
        screen_vertices, triangles = _stroke_ribbon_screen_mesh(
            np.asarray(self._lasso, dtype=float),
            LASSO_LINE_WIDTH_PX,
        )
        if not len(triangles):
            return
        points = np.asarray(
            [
                # Keep the ribbon just beyond the near plane so it is always
                # visible over the model.
                camera.unproject(
                    float(x), float(y), 0.001, float(width), float(height)
                )
                for x, y in screen_vertices
            ],
            dtype=float,
        ).reshape((-1, 3))
        ribbon = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector3iVector(triangles),
        )
        ribbon.paint_uniform_color([1.0, 0.12, 0.02])
        ribbon.compute_triangle_normals()
        scene.add_geometry("lasso", ribbon, self.lasso_material)
        self.scene_widget.force_redraw()

    def _clear_lasso_geometry(self) -> None:
        scene = self.scene_widget.scene
        if scene.has_geometry("lasso"):
            scene.remove_geometry("lasso")
            self.scene_widget.force_redraw()

    def _clear_lasso_after_preview(self, delay_seconds: float = 0.75) -> None:
        """Keep the completed stroke briefly visible, then clear it on the UI thread."""
        generation = self._lasso_preview_generation

        def enqueue_clear() -> None:
            try:
                gui.Application.instance.post_to_main_thread(
                    self.window,
                    lambda: (
                        self._clear_lasso_geometry()
                        if generation == self._lasso_preview_generation
                        else None
                    ),
                )
            except RuntimeError:
                pass

        timer = threading.Timer(max(0.0, float(delay_seconds)), enqueue_clear)
        timer.daemon = True
        timer.start()

    def _project_working_vertices(self) -> np.ndarray:
        vertices = np.asarray(self.working.vertices, dtype=float)
        homogeneous = np.column_stack((vertices, np.ones(len(vertices))))
        world = (self._model_transform @ homogeneous.T).T
        view = np.asarray(self.scene_widget.scene.camera.get_view_matrix(), dtype=float)
        projection = np.asarray(
            self.scene_widget.scene.camera.get_projection_matrix(), dtype=float
        )
        clip = (projection @ view @ world.T).T
        denominator = clip[:, 3]
        valid = np.abs(denominator) > 1e-12
        ndc = np.full((len(vertices), 3), np.nan, dtype=float)
        ndc[valid] = clip[valid, :3] / denominator[valid, None]
        width = max(float(self.scene_widget.frame.width), 1.0)
        height = max(float(self.scene_widget.frame.height), 1.0)
        return np.column_stack(
            ((ndc[:, 0] + 1.0) * 0.5 * width, (1.0 - ndc[:, 1]) * 0.5 * height)
        )

    def _through_lasso_faces(self, polygon: np.ndarray) -> np.ndarray:
        screen = self._project_working_vertices()
        triangles = np.asarray(self.working.triangles, dtype=np.int64)
        projected = screen[triangles]
        return _projected_triangles_intersect_polygon(projected, polygon)

    def _visible_lasso_faces(self, polygon: np.ndarray) -> np.ndarray:
        screen = self._project_working_vertices()
        triangles = np.asarray(self.working.triangles, dtype=np.int64)
        centroids = np.mean(screen[triangles], axis=1)
        candidates = _points_in_polygon(centroids, polygon)
        if not np.any(candidates):
            return np.zeros(len(triangles), dtype=bool)

        view = np.asarray(
            self.scene_widget.scene.camera.get_view_matrix(), dtype=float
        )
        inverse_view = np.linalg.inv(view)
        direction_world = -inverse_view[:3, 2]
        inverse_model = np.linalg.inv(self._model_transform)
        direction_local = inverse_model[:3, :3] @ direction_world
        return _frontmost_candidate_faces(
            self.working, candidates, direction_local
        )

    def _apply_lasso_selection(self) -> None:
        if len(self._lasso) < 3:
            return
        polygon = np.asarray(self._lasso, dtype=float)
        working_faces = (
            self._through_lasso_faces(polygon)
            if self.through_toggle.is_on
            else self._visible_lasso_faces(polygon)
        )
        original_faces = self.working_original_faces[np.flatnonzero(working_faces)]
        selected = self.state.selected.copy()
        if self._lasso_subtract:
            selected[original_faces] = False
            message = f"已从选区移除 {len(original_faces):,} 个面片。"
        else:
            selected[original_faces] = True
            message = f"已增加选中 {len(original_faces):,} 个面片。"
        selected[self.state.deleted] = False
        self._record_change(
            clone_state_with_masks(self.state, selected, self.state.deleted), message
        )

    @staticmethod
    def _ctrl_down(event: gui.MouseEvent) -> bool:
        try:
            return bool(event.is_modifier_down(gui.KeyModifier.CTRL))
        except (AttributeError, TypeError):
            try:
                return bool(event.modifiers & gui.KeyModifier.CTRL)
            except (AttributeError, TypeError):
                return False

    def _on_mouse(self, event: gui.MouseEvent) -> gui.Widget.EventCallbackResult:
        frame = self.scene_widget.frame
        local = (int(event.x - frame.x), int(event.y - frame.y))
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

        if event.type == gui.MouseEvent.BUTTON_DOWN and event.is_button_down(gui.MouseButton.LEFT):
            self.context_menu.visible = False
            self._lasso_preview_generation += 1
            self._clear_lasso_geometry()
            self._lasso = [local]
            self._lasso_subtract = self._ctrl_down(event)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.DRAG and self._lasso:
            if math.hypot(local[0] - self._lasso[-1][0], local[1] - self._lasso[-1][1]) >= 3.0:
                self._lasso.append(local)
                self._refresh_lasso()
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.BUTTON_UP and self._lasso:
            self._lasso.append(local)
            self._refresh_lasso()
            self._apply_lasso_selection()
            self._lasso.clear()
            self._clear_lasso_after_preview()
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.BUTTON_DOWN and event.is_button_down(gui.MouseButton.RIGHT):
            self._right_active = True
            self._right_dragged = False
            self._right_last = (int(event.x), int(event.y))
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.DRAG and self._right_active:
            current = (int(event.x), int(event.y))
            previous = self._right_last
            self._right_last = current
            if previous is not None:
                dx, dy = current[0] - previous[0], current[1] - previous[1]
                if math.hypot(dx, dy) > 0:
                    self._right_dragged = True
                    self._rotate_model(dx, dy)
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.BUTTON_UP and self._right_active:
            dragged = self._right_dragged
            self._right_active = False
            self._right_last = None
            if not dragged:
                self._show_context_menu(int(event.x), int(event.y))
            return gui.Widget.EventCallbackResult.CONSUMED

        if event.type == gui.MouseEvent.BUTTON_DOWN and event.is_button_down(gui.MouseButton.MIDDLE):
            self._middle_active = True
            self._middle_last = (int(event.x), int(event.y))
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.DRAG and self._middle_active:
            current = (int(event.x), int(event.y))
            previous = self._middle_last
            self._middle_last = current
            if previous is not None:
                self._pan_camera(current[0] - previous[0], current[1] - previous[1])
            return gui.Widget.EventCallbackResult.CONSUMED
        if event.type == gui.MouseEvent.BUTTON_UP and self._middle_active:
            self._middle_active = False
            self._middle_last = None
            return gui.Widget.EventCallbackResult.CONSUMED
        return gui.Widget.EventCallbackResult.HANDLED


def run_model_selection_viewer(
    mesh_path: str | Path,
    state_path: str | Path,
) -> None:
    app = gui.Application.instance
    app.initialize()
    _configure_font(app, f"{_UI_TEXT}\n{Path(mesh_path)}")
    ModelSelectionViewer(mesh_path, state_path)
    app.run()
