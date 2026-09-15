"""Release the mouse promptly without losing selection state or close requests."""

from queue import Queue
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from auto_alignment import model_viewer as module
from auto_alignment.mesh_selection import ModelEditState


@pytest.fixture
def viewer(monkeypatch, tmp_path):
    item = module.ModelSelectionViewer.__new__(module.ModelSelectionViewer)
    item._lasso = [(0, 0), (10, 0), (0, 10)]
    item._lasso_subtract = False
    item._selection_busy = item._close_after_selection = item._closing = False
    item._cancel_after_selection = False
    item._model_transform = np.eye(4)
    item._working_ray_scene = None
    item.working = object()
    item.working_original_faces = np.array([0, 2, 3])
    item.state = ModelEditState("mesh.stl", "hash", 4, np.zeros(4, bool),
                                np.array([False, True, False, False]), "")
    item.state_path = tmp_path / "selection.json"
    item.status_label = SimpleNamespace(text="")
    item.through_toggle = SimpleNamespace(is_on=True)
    item.scene_widget = SimpleNamespace(
        scene=SimpleNamespace(camera=SimpleNamespace(
            get_view_matrix=lambda: np.eye(4), get_projection_matrix=lambda: np.eye(4))),
        frame=SimpleNamespace(width=100, height=100))
    item.window = SimpleNamespace(post_redraw=lambda: None, close=lambda: None)
    callbacks = Queue()
    monkeypatch.setattr(module, "gui", SimpleNamespace(Application=SimpleNamespace(
        instance=SimpleNamespace(post_to_main_thread=lambda _window, callback: callbacks.put(callback)))))
    return item, callbacks


def test_compute_is_background_and_uses_release_time_camera_snapshot(viewer, monkeypatch):
    item, callbacks = viewer
    started, release = Event(), Event()
    worker_threads = []
    captured = []
    main_thread = get_ident()

    def compute(*args):
        worker_threads.append(get_ident())
        started.set()
        assert release.wait(5)
        captured.append(args)
        return np.array([True, False, True]), "cached_scene"

    changes = []
    item._record_change = lambda state, message: changes.append((get_ident(), state))
    monkeypatch.setattr(module, "_lasso_faces_from_snapshot", compute)
    try:
        item._apply_lasso_selection()
        assert started.wait(5)
        assert item._selection_busy and not changes
        item._model_transform[0, 3] = 99
        item._lasso.clear()
        item._lasso_subtract = True
    finally:
        release.set()
    callbacks.get(timeout=5)()
    assert worker_threads[0] != main_thread
    assert changes[0][0] == main_thread
    np.testing.assert_array_equal(changes[0][1].selected, [True, False, False, True])
    assert captured[0][2][0, 3] == 0
    assert len(captured[0][1]) == 3
    assert item._working_ray_scene == "cached_scene"
    assert not item._selection_busy


@pytest.mark.parametrize("subtract", [False, True])
def test_close_waits_for_selection_and_saves_final_mask(viewer, monkeypatch, subtract):
    item, callbacks = viewer
    item._lasso_subtract = subtract
    item.state.selected[:] = [True, False, True, True] if subtract else False
    monkeypatch.setattr(module, "_lasso_faces_from_snapshot",
                        lambda *args: (np.array([True, False, True]), None))
    item._record_change = lambda state, message: setattr(item, "state", state)
    closed = []
    item.window.close = lambda: closed.append(True)
    item._apply_lasso_selection()
    item._save_and_close()
    assert not closed and item._close_after_selection
    assert item._on_close() is False
    callbacks.get(timeout=5)()
    assert closed and item._closing
    import json
    saved = json.loads(item.state_path.read_text(encoding="utf-8"))
    assert saved["selected_ranges"] == ([[2, 2]] if subtract else [[0, 0], [3, 3]])


def test_changed_state_discards_old_result(viewer, monkeypatch):
    item, callbacks = viewer
    monkeypatch.setattr(module, "_lasso_faces_from_snapshot",
                        lambda *args: (np.ones(3, bool), None))
    changes = []
    item._record_change = lambda *args: changes.append(args)
    item._apply_lasso_selection()
    item.state = item.state.normalized()  # e.g. delete, undo or switch region
    callbacks.get(timeout=5)()
    assert not changes and not item._selection_busy
    assert "已改变" in item.status_label.text


def test_worker_error_keeps_window_open_and_allows_retry(viewer, monkeypatch):
    item, callbacks = viewer

    def fail(*args):
        raise ValueError("test failure")

    monkeypatch.setattr(module, "_lasso_faces_from_snapshot", fail)
    item._apply_lasso_selection()
    item._save_and_close()
    callbacks.get(timeout=5)()
    assert not item._selection_busy and not item._closing and not item._close_after_selection
    assert "test failure" in item.status_label.text


def test_multi_region_title_close_cancels_after_pending_selection():
    from auto_alignment.integration._selection_viewer import MultiRegionSelectionViewer
    item = MultiRegionSelectionViewer.__new__(MultiRegionSelectionViewer)
    item._closing = False
    item._selection_busy = True
    item.status_label = SimpleNamespace(text="")
    saves = []
    item.save = lambda: saves.append(True)
    assert item._on_close() is False
    assert item._cancel_after_selection and not saves


def test_multi_region_title_close_cancels_without_saving():
    from auto_alignment.integration._selection_viewer import MultiRegionSelectionViewer
    item = MultiRegionSelectionViewer.__new__(MultiRegionSelectionViewer)
    item._closing = False
    item._selection_busy = False
    item.status_label = SimpleNamespace(text="")
    saves = []
    item.save = lambda: saves.append(True)
    assert item._on_close() is True
    assert item._closing and not saves


def test_save_failure_does_not_mark_window_closed(viewer, monkeypatch):
    item, _callbacks = viewer

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(module, "save_edit_state", fail)
    assert item._on_close() is False
    assert not item._closing
    assert "disk full" in item.status_label.text
