"""GUI adapter owned by the integration package; reuses the existing gestures."""
from __future__ import annotations
from pathlib import Path
import threading
import numpy as np
from open3d.visualization import gui, rendering

from ..model_viewer import ModelSelectionViewer, _mesh_from_faces
from ..mesh_selection import clone_state_with_masks
from .review import configure_open3d_font
from .selection import RegionSelectionSession


class MultiRegionSelectionViewer(ModelSelectionViewer):
    """Create a window in an ALREADY initialized Open3D application.

    Public methods are get_snapshot, set_region_mask, set_active_region, undo,
    redo, save, reset_view and close. Hosts need not access inherited internals.
    summary_provider(snapshot) returns text; on_change/on_save receive snapshots.
    """
    def __init__(self, mesh_path, state_path, regions, *, initial_masks=None,
                 active_region=None, allow_overlap=False, on_change=None,
                 on_save=None, summary_provider=None, preloaded_mesh=None,
                 preloaded_mesh_sha256=None):
        if threading.current_thread() is not threading.main_thread():
            raise RuntimeError("Open3D viewers must start on the main thread")
        self._region_specs = tuple(regions)
        self._initial_masks = initial_masks
        self._initial_active = active_region
        self._allow_overlap = allow_overlap
        self._change_callback, self._save_callback = on_change, on_save
        self._summary_provider = summary_provider
        self._region_geometry_names = []
        self._ui_sync = False
        super().__init__(
            mesh_path,
            state_path,
            preloaded_mesh=preloaded_mesh,
            preloaded_mesh_sha256=preloaded_mesh_sha256,
        )
        self.window.set_on_close(self._on_close)

    def _prepare_selection_state(self):
        self._session = RegionSelectionSession(self.mesh_path, len(self.original_mesh.triangles), self._region_specs,
            initial_masks=self._initial_masks, active_region=self._initial_active,
            allow_overlap=self._allow_overlap, on_change=self._change_callback,
            on_save=self._save_callback, mesh_sha256=self.state.mesh_sha256)
        if self._initial_masks is None and self.state_path.is_file():
            self._session.load(self.state_path)
        self._sync_state()

    def _sync_state(self):
        snap = self._session.snapshot()
        self.state = clone_state_with_masks(self.state, snap.masks[snap.active_region],
                                            np.zeros(snap.triangle_count, dtype=bool))

    def _build_panel(self):
        super()._build_panel()
        self.status_label.text = "多区域选区：保存并关闭时写入；原始网格保持不变。"
        self.region_selector = gui.Combobox()
        for region in self._region_specs: self.region_selector.add_item(region.label)
        self.region_selector.selected_index = [r.key for r in self._region_specs].index(self._session.snapshot().active_region)
        self.region_selector.set_on_selection_changed(self._switch_region)
        self.panel.add_child(gui.Label("当前编辑区域"))
        self.panel.add_child(self.region_selector)
        self.summary_label = gui.Label("")
        self.panel.add_child(self.summary_label)
        cancel_button = gui.Button("取消并关闭")
        cancel_button.set_on_clicked(self._cancel_and_close)
        self.panel.add_child(cancel_button)

    def _build_context_menu(self):
        em = self.window.theme.font_size
        self.context_menu = gui.Vert(.2*em, gui.Margins(.4*em, .4*em, .4*em, .4*em))
        self.context_menu.visible = False
        self._menu_button("全选当前区域", self._select_all)
        self._menu_button("清空当前区域", self._clear_selection)
        self._menu_button("反选当前区域", self._invert_selection)
        self.bounded_button = self._menu_button("有界组件", self._bounded_components)
        self.undo_button = self._menu_button("撤销", self._undo_action)
        self.redo_button = self._menu_button("重做", self._redo_action)
        self._menu_button("重置视角", self.reset_view)
        self._menu_button("保存并关闭", self._save_and_close)
        self._menu_button("取消并关闭", self._cancel_and_close)

    def _refresh_counts(self):
        snap = self._session.snapshot()
        self.count_label.text = "\n".join(f"{r.label}：{int(snap.masks[r.key].sum()):,} 面" for r in self._region_specs)
        self.bounded_button.enabled = bool(snap.masks[snap.active_region].any())
        self.undo_button.enabled = self._session.can_undo
        self.redo_button.enabled = self._session.can_redo
        self._ui_sync = True
        try:
            self.region_selector.selected_index = [r.key for r in self._region_specs].index(snap.active_region)
        finally:
            self._ui_sync = False
        if self._summary_provider is not None:
            try:
                self.summary_label.text = str(self._summary_provider(self._session.snapshot()))
            except Exception as exc:
                self.summary_label.text = f"区域摘要失败：{exc}"

    def _refresh_mesh(self):
        self._sync_state()
        snap = self._session.snapshot()
        # Multi-region selection never deletes faces. Reuse the validated input
        # mesh directly instead of rebuilding a full working copy and normals
        # before the first frame (costly on million-face CT meshes).
        if getattr(self, "working", None) is not self.original_mesh:
            self.working = self.original_mesh
            self.working_original_faces = np.arange(snap.triangle_count, dtype=np.int64)
            self._working_deleted = np.zeros(snap.triangle_count, dtype=bool)
            self._working_ray_scene = None
        scene = self.scene_widget.scene
        for name in ["model", "selected", *self._region_geometry_names]:
            if scene.has_geometry(name): scene.remove_geometry(name)
        self._region_geometry_names = []
        occupied = np.zeros(snap.triangle_count, bool)
        # When overlap is explicitly enabled, the active region wins rendering.
        order = sorted(range(len(self._region_specs)), key=lambda i: self._region_specs[i].key != snap.active_region)
        for index in order:
            region = self._region_specs[index]
            mask = snap.masks[region.key] & ~occupied
            occupied |= snap.masks[region.key]
            if not mask.any(): continue
            mesh = _mesh_from_faces(self.original_mesh, mask)
            mesh.paint_uniform_color(region.color)
            material = rendering.MaterialRecord()
            material.shader = "defaultLit"
            material.base_color = [*region.color, 1.]
            name = f"region_{index}"
            scene.add_geometry(name, mesh, material)
            self._region_geometry_names.append(name)
        if not occupied.any():
            base = self.original_mesh
        else:
            base = _mesh_from_faces(self.original_mesh, ~occupied)
        if len(base.triangles):
            base.paint_uniform_color([.72, .74, .78])
            scene.add_geometry("model", base, self.mesh_material)
        self._apply_model_transform()
        self._refresh_counts()
        self.scene_widget.force_redraw()

    def _apply_model_transform(self):
        super()._apply_model_transform()
        for name in self._region_geometry_names:
            if self.scene_widget.scene.has_geometry(name):
                self.scene_widget.scene.set_geometry_transform(name, self._model_transform)

    def _refresh_after(self, operation):
        try:
            operation()
        finally:
            self._refresh_mesh()

    def _ui_action(self, operation, message):
        try:
            self._refresh_after(operation)
            self.status_label.text = message
        except Exception as exc:
            self.status_label.text = str(exc)
        self.context_menu.visible = False

    def _record_change(self, next_state, message):
        active = self._session.snapshot().active_region
        self._ui_action(lambda: self._session.set_mask(active, next_state.selected), message)

    def _switch_region(self, _label, index):
        if not self._ui_sync:
            self._ui_action(lambda: self._session.select_region(self._region_specs[index].key), "已切换区域")

    def _undo_action(self): self._ui_action(self._session.undo, "已撤销")
    def _redo_action(self): self._ui_action(self._session.redo, "已重做")

    def get_snapshot(self): return self._session.snapshot()
    def set_region_mask(self, region, mask): self._refresh_after(lambda: self._session.set_mask(region, mask))
    def set_active_region(self, region): self._refresh_after(lambda: self._session.select_region(region))
    def undo(self): self._refresh_after(self._session.undo)
    def redo(self): self._refresh_after(self._session.redo)
    def reset_view(self): self._reset_camera()

    def save(self):
        result = self._session.save(self.state_path)
        self.status_label.text = "全部区域已保存"
        return result

    def _on_close(self):
        """The title-bar close action cancels this editing session."""
        if self._closing:
            return True
        if getattr(self, "_selection_busy", False):
            self._cancel_after_selection = True
            self.status_label.text = "正在结束当前划选，完成后直接关闭且不保存。"
            return False
        self._closing = True
        return True

    def _save_and_close(self):
        if self._closing:
            return
        if self._defer_close_for_selection():
            return
        try:
            self.save()
        except Exception as exc:
            self.status_label.text = f"保存失败，窗口与选区已保留：{exc}"
            return
        self._closing = True
        self.window.close()

    def _cancel_and_close(self):
        if self._closing:
            return
        if getattr(self, "_selection_busy", False):
            self._cancel_after_selection = True
            self.status_label.text = "正在结束当前划选，完成后直接关闭且不保存。"
            return
        self._closing = True
        self.window.close()

    def close(self):
        self._save_and_close()
        return self._closing


def run_multi_region_selection_viewer(mesh_path, state_path, regions, *, initial_masks=None,
                                     active_region=None, allow_overlap=False, on_change=None,
                                     on_save=None, summary_provider=None, preloaded_mesh=None,
                                     preloaded_mesh_sha256=None):
    """Standalone entry; embedded hosts construct MultiRegionSelectionViewer."""
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError("Open3D viewers must start on the main thread")
    regions = tuple(regions)
    app = gui.Application.instance
    app.initialize()
    configure_open3d_font(app, str(mesh_path) + "多区域当前保存失败未应用" + "".join(r.label for r in regions))
    viewer = MultiRegionSelectionViewer(mesh_path, state_path, regions, initial_masks=initial_masks,
        active_region=active_region, allow_overlap=allow_overlap, on_change=on_change,
        on_save=on_save, summary_provider=summary_provider, preloaded_mesh=preloaded_mesh,
        preloaded_mesh_sha256=preloaded_mesh_sha256)
    app.run()
    return viewer.get_snapshot()
