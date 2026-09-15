# Public integration API v1

Application version: 3.0.0 (local source version). `INTEGRATION_API_VERSION = 1` is independent of application, review-manifest and edit-state versions. Exact signatures and dataclass fields are in [INTEGRATION_API_REFERENCE.md](INTEGRATION_API_REFERENCE.md).

## Install and import

```powershell
python -m pip install general_model_registration-3.0.0-py3-none-any.whl
```

```python
from auto_alignment.integration import INTEGRATION_API_VERSION
from auto_alignment.integration.core import AlignmentConfig, load_mesh, register_meshes

assert INTEGRATION_API_VERSION == 1
target, target_facts = load_mesh("固定模型.stl")
source, source_facts = load_mesh("移动模型.stl")
result = register_meshes(target, source, target_facts, source_facts, AlignmentConfig())
```

No parent-directory lookup or `sys.path` modification is required. In a development checkout an explicit editable installation is possible, but release validation must use the wheel from outside the repository. The wheel relies on its declared Open3D/NumPy/PySide6/SciPy dependencies; it is not an EXE bundle. This build targets Python 3.12.

The integration package's version import is lightweight and does not import Open3D. Review and selection modules do not initialize GUI applications on import. Accessing GUI classes may import the underlying visualization module; this still must not initialize an application or create a window.

## Registration contract

- Fixed/target mesh is first; moving/source is second. `transformation` is a 4×4 rigid homogeneous matrix acting on column vectors: `[target_x, target_y, target_z, 1] = T @ [source_x, source_y, source_z, 1]`.
- Coordinates and distances are millimetres; reported angles are degrees. The loader does not infer or convert physical units from file extensions.
- Rotations about an arbitrary pivot already include that pivot in the matrix's translation column. Do not apply the same transform again about another mesh centre.
- Progress is `callback(fraction: float, message: str)`, with finite fraction in [0, 1]. Callback exceptions propagate; callbacks should not invoke GUI functions from a worker thread.
- `target_priority_faces`/`source_priority_faces` are `None` or 1D boolean arrays matching the corresponding mesh triangle count. Invalid dimensions, dtype or length are rejected by the public wrapper before the solver runs.
- The three valid statuses are `success`, `warning`, `failed`. For these statuses, `succeeded` is true for success and warning. The public wrapper rejects unexpected solver statuses. The historical dataclass itself remains unchanged: manually constructing it with another string is outside the supported contract.
- `confidence` is display text. Use `quality.position_confidence` (`PositionConfidence.HIGH/MEDIUM/LOW/FAILED`) for machine logic, first checking that `quality is not None`.
- `AlignmentConfig` and `MeshFacts` retain `dataclasses.replace` support. Existing field order and defaults are preserved; new fields are appended with defaults. API v1 adds no required fields to existing dataclasses.
- `AlignmentConfig.refinement_mode` defaults to `"baseline"`. Optional values are `"A"`, `"B"`, `"compare"`, and `"auto"`; automatic routing is experimental. `algorithm_version="2.0"` still identifies the front end. See [3.0 refinement](V300_REFINEMENT.md).
- `RegistrationResult.alternatives` contains `(name, result)` pairs for other available candidates. `RegistrationMetrics.refinement` contains mode, chosen branch, runtime, fallback and failure details; it is `None` in baseline mode. All candidate transforms remain rigid.
- Optional `cancel: Callable[[], bool]` requests cooperative cancellation by returning true. `RegistrationCancelled` propagates at computation boundaries; native calls finish before cancellation is checked. Batch cancellations are recorded separately from quality failures.

The default mode delegates to the existing solver with the same meshes and configuration. Optional modes apply packaged geometry after the front end, preserve the original result, and reassess candidate quality. Operator selections and front-end failures bypass refinement. Finite floating-point repetition noise can exist; equivalence checks use numerical tolerances, not bitwise comparison of separately computed floats.

## Face indexing

Selection masks index the **baseline mesh returned by `load_mesh`, before applying edit deletions**. Loading can remove duplicate or degenerate triangles; this is not necessarily the raw STL record order. Source SHA-256, baseline triangle count and the documented load policy identify the selection frame. Existing `apply_edit_state(...).original_face_indices` maps edited working faces back to this baseline.

Do not reuse masks on a remeshed or independently cleaned mesh merely because its face count matches. Mapping to disk-record indices requires an explicit mapping, not a change to the existing loader.

`encode_face_ranges` accepts integer face indices and returns inclusive `[first, last]` ranges. `decode_face_ranges` returns a fresh boolean array. Both reuse the existing codec; the new public boundary rejects malformed input. Private legacy codecs retain their original lenient behavior for old callers.

## Review manifest

```python
from auto_alignment.integration.review import RegistrationReviewSpec, write_review_manifest

spec = RegistrationReviewSpec(
    target_path="target.stl", aligned_path="aligned.stl",
    status="warning", position_confidence="medium",
    confidence_display="需要复核", warnings=("检查共同区域",), review_only=True,
)
manifest = write_review_manifest("result/stage_review.json", spec)
```

The two paths above are relative to the manifest's directory. Absolute paths and Chinese paths are supported. Default nominal band is -0.05 to +0.05 mm. Optional SHA-256 values are validated against the actual resolved files on load/write, including overrides. `target_override`/`aligned_override` retain their historical explicit-path behavior (relative overrides resolve against the calling working directory).

New review manifests use schema version **1**, including `target_mesh`, `outputs`, `registration`, `distance_statistics` and `color_mapping`. Existing general-application result envelopes tagged "1.4.0", "1.4.1" or "1.4.2" are also accepted; these known legacy identifiers are not changed by this API. Legacy files without an explicit version are interpreted as legacy v1. Unsupported future versions are rejected. Existing `archived_path` is preferred when it exists; fallback paths are resolved against the manifest directory. Legacy `results.json` and old viewer function names remain supported.

Optional `viewer_annotations_path` defaults to `viewer_annotations.json` beside the manifest even if the manifest has another filename. Pair-only viewing retains its original behavior without automatic annotations beside the input model. Annotation paths cannot overwrite either input or the manifest.

`status`, display confidence, warnings, `review_only` and direction reversal are passed to the existing viewer. `unknown` is supported for review-only metadata that has no registration result; it is not a fourth solver status. Missing review_only defaults to true for a legacy failed result.

`run_registration_review(path)` starts the viewer. An in-memory `RegistrationReviewSpec` must include `manifest_path` so its annotation location is explicit; the entry writes that manifest before viewing. `load_review_manifest` performs no GUI initialization. In an embedded application, call `load_viewer_data` then create `GeneralResultViewer(data)` after the host has initialized Open3D.

## Multi-region selection

```python
from auto_alignment.integration.selection import SelectionRegionSpec, run_multi_region_selection_viewer

regions = [SelectionRegionSpec("left", "左侧", (1.0, 0.2, 0.1)),
           SelectionRegionSpec("right", "右侧", (0.1, 0.3, 1.0))]
snapshot = run_multi_region_selection_viewer(
    "mandible.stl", "result/regions.json", regions,
    on_save=lambda snapshot: print(snapshot.mesh_sha256),
    summary_provider=lambda snapshot: str({k: int(v.sum()) for k, v in snapshot.masks.items()}),
)
```

The host retains anatomical meaning and calculations (e.g. area-weighted condyle centre). The component manages one or more named surface regions. It reuses existing lasso/front-face/through-selection, bounded components, camera rotation, panning, zoom and reset behavior. Region meshes have distinct colors. With explicit allow_overlap=True, the active region takes visual precedence on shared triangles.

`initial_masks` imports a subset or all defined regions, with missing regions initialized empty; unknown keys or malformed masks are rejected. Without initial_masks, an existing state file is loaded strictly. A different mesh hash/count, region key set, incompatible schema or forbidden overlap raises an error rather than silently discarding the saved selection.

`MultiRegionSelectionViewer` creates a window in an **already initialized** Open3D application. Stable public methods: `get_snapshot`, `set_region_mask`, `set_active_region`, `undo`, `redo`, `save`, `reset_view`, `close`. The optional `preloaded_mesh` keyword lets an embedding host transfer an already validated Open3D mesh and avoid a duplicate large-file read. A host that computed the same file digest while loading may also pass `preloaded_mesh_sha256` to avoid hashing the large file again before the first frame; save still revalidates the source file. The title-bar close action and “取消并关闭” discard the current in-memory edit session, while `close()` and “保存并关闭” save explicitly. Downstream code should not use inherited private members.

`RegionSelectionSession` exposes the same non-UI state operations for headless use and testing. Every outward snapshot contains mask copies. Original mesh files are never overwritten. The multi-region state schema is independently numbered 1; `EDIT_STATE_SCHEMA = "1.4.2"` and its legacy compatibility remain unchanged.

### State and callback order

1. Validate the entire proposed edit. If overlap is prohibited and any overlap occurs, reject the entire operation, with no partial mask or history update and no callback.
2. Commit masks and active region, append undo history, clear redo history and mark dirty.
3. Invoke on_change with a copied full snapshot. Region switching is itself undoable; undo/redo restore both masks and the active region. No-op actions emit nothing.
4. Save writes the complete state atomically, then invokes on_save. A write or callback failure propagates and leaves the current in-memory state dirty and available. A callback cannot reenter mutation methods. An on_change exception occurs after a committed edit, so it does not roll that edit back.
5. The viewer catches action errors, displays them and redraws committed state. Both its save button and native close action save before accepting closure. Failure keeps the window open and `_closing` false. Native OS process termination cannot be made into a guaranteed save.

Snapshots and callbacks always include all masks, mesh hash, baseline triangle count and active region. The summary provider receives a separate copied snapshot and is for text display, not state mutation.

## GUI lifecycle and fonts

All GUI initialization/window creation must occur on the main thread. Standalone run functions own `initialize/run`; embedded hosts initialize once and create viewer objects themselves. Do not nest Open3D event loops inside an already running host loop. A host using Qt may choose a separate viewer process.

`configure_open3d_font(app, extra_text="")` includes the common characters from both standard viewers; callers only add their own labels. Existing `_configure_font` aliases remain. It uses an available Windows Chinese font and must be called after application initialization and before window creation.

## Compatibility and release checks

Application version, integration API version, viewer manifest schema and edit-state schema are independent. Existing constructors gain no required positional parameters; the ViewerData annotation field is appended with a default. Old module paths, old viewer calls, `_configure_font`, `_encode_ranges` and `_decode_ranges` remain.

Build with `python -m build --wheel`. Inspect archive contents and LICENSE, install into a fresh environment, clear PYTHONPATH, change outside the repository, confirm loaded module paths, test synthetic registration and manifest round-trips, then run `pip check`. GUI state tests and actual window rendering/interaction checks must be reported separately.

The mandible application must separately migrate its imports, install dependency and private selection subclass. This package does not modify its workflow, clinical interpretation, or packaging configuration.
