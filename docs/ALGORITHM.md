# Algorithm overview

General Model Registration estimates a rigid 4×4 transformation from a moving triangle mesh to a fixed target mesh. Coordinates are interpreted in millimetres.

## Default 2.0 pipeline

The objective is to automatically discover supported common geometry, and otherwise propose the best supported whole-shape pose. An unchanged patch need not be marked by the operator. The search is heuristic: neither the existence of a common patch nor a high internal score guarantees a unique or globally optimal pose.

1. Keep original triangle identities, area weights and barycentric sample coordinates. Build one locally centered closest-surface scene per mesh; cache samples, downsampled clouds and features within this registration call. Caches are not persistent across different pairs or application launches.
2. Generate identity/centroid, two-scale FPFH/FGR, repeated RANSAC, proper PCA permutations and supported-plane-frame candidates. Plane frames contribute orientation hypotheses, not an assumed ideal cube or a fitted manufacturing model. PCA remains enabled for unequal-area pairs. The optional exhaustive search adds further orientations.
3. Preserve each initial pose and its short ICP result. Select six geometrically distinct branches using fixed evaluation; preserve every multiscale ICP level. Explore residual-fraction common-surface fits (35% and 65% quantiles) and up to three GICP alternatives. Common-surface fitting adapts a robust cutoff from fit-sample distances and requires a rank-six point-to-plane system.
4. Evaluate all snapshots on the same 8,000 area samples per direction, generated separately from fit samples. Unmatched and high-error points remain in the denominator. Use full Euclidean nearest-surface distance, normal compatibility, and equal weight between area-average support and occupied spatial-block support. No experiment truth or file-name rules enter this evaluation.
5. Preserve a parent unless an update improves the fixed score by more than a numerical/block-variability tolerance. Rejected updates remain recorded for diagnosis; search can continue through them. Select the highest-scoring eligible retained pose, rather than unconditionally returning the last solver iterate.
6. Report distributed common-surface support versus shape fallback, insufficient six-degree-of-freedom constraints, and close-scoring distinct poses. Existing broad quality checks still contribute. Warning results remain available with explicit uncertainty; failed results are exported as inspection-only previews.

### Reusing feature matching work

RANSAC restarts share the nearest-neighbour feature correspondences at each scale. Each direction is searched once; mutual pairs retain Open3D 0.19's source-index order and its 10% small-set fallback (including integer truncation). Every configured restart still uses its original seed, convergence criteria and correspondence checkers. The cache belongs to one immutable feature pair in one registration call; weighted selections and strict selected meshes create their own caches. Already prepared point clouds are reused by matching refinement scales.

Candidate diagnostics expose `global_search.feature_matching_s`, `global_search.ransac_s` and `cache.feature_match_searches`. The two durations are components of `timings.global_hypotheses_s`, not additional stages. The UI reports each global matching restart.

### What is fixed, and what is learned from this pair?

The fit cutoff is estimated from residual quantiles. Evaluation distances use the smaller mesh diagonal times `[0.0003, 0.001, 0.003, 0.01, 0.03]` with weights `[4, 3, 2, 1, 1]/11`; a bounded whole-shape penalty discourages gross separation. If area ratio exceeds 1.8, the smaller-to-larger direction gets 0.8 weight and the reverse direction 0.2. Otherwise both directions have equal weight.

A near-exact precision component (diagonal × 1e-6, floor 1e-6 mm) adds support only when it covers at least 15% of fixed samples, at least four occupied spatial blocks, and constrains all six local degrees of freedom. This helps prevent broad-fit refinements from displacing an already near-exact common surface. It is not a calibrated sensor noise estimate.

These fractions, weights, normal-angle cutoff, block grid, rank tolerance and ambiguity tolerance are algorithmic priors. 2.0 is **not parameter-free**. The paired-block update tolerance is a heuristic, not a statistical confidence interval. Fixed evaluation is reused to select candidates: it is an internal geometry check, **not an independent spatial holdout or an accuracy certificate**. A rank-six local Hessian does not rule out global symmetry. No fixed score can by itself prove recovery of the imposed rigid motion when deformation changes the preferred fit.

The provided experimental data were used during development, including a second pass to address a remeshing regression. Results therefore measure development-set behavior. Further objective changes need a separate validation set, sensor-resolution calibration and repeated runs; the known matrices may be used offline for diagnosis, never to select production candidates.

### Frames and distances

Matrices map moving coordinates to fixed coordinates: `p_fixed = R @ p_moving + t`. Composition acts on the coordinates, not on the mesh's newly recomputed center. Local centered solvers expand their rotation center into the translation before returning the 4×4 matrix. Pose error measured on fixed-frame points uses `T_estimated @ inverse(T_reference)`. Translation-column magnitude alone is not a frame-invariant alignment quality check; catastrophic separation checks compare transformed geometry centers.

For experiments with known reference matrices, the primary offline evaluation now reports all three translation and rotation components of that relative matrix, its inverse correction, total rotation angle, and the direct translation-column difference separately. Rotation decomposition is explicitly `R = Rx @ Ry @ Rz`; angle wrapping and gimbal lock are handled. The imposed-motion input parameters are also reconstructed using their original convention. Point-displacement P95 remains auxiliary. Rotation and translation are not combined into an arbitrary scalar score, and reference metrics never enter production candidate selection. See `scripts/reassess_pose_matrices.py` and `docs/V200_MATRIX_REPORT.md`.

Color-map magnitude is the full Euclidean closest-surface distance. The nearest target triangle normal determines sign; at a tangential open boundary the sign uses a convention while the distance magnitude remains nonzero. Surface residuals and ground-truth corresponding-point pose errors are separate measurements.

### Compatibility and scope

`AlignmentConfig(algorithm_version="1.4")` retains the old solver path for comparison. Its old high-precision and stable-region options do not control the default 2.0 candidate-bank fit. The `high_precision_refinement_enabled` switch still controls common-surface refinement in 2.0. Manual selections retain their existing wrapper semantics below; unmarked benchmarks use only the new default pipeline. Shared mandible workflows are not replaced by this general-purpose application's changes.

The exported outer schema remains `1.4.2` for existing readers; application `version` is `2.0.0` and `registration.metrics.candidate_selection` is an additive diagnostics field. It contains candidate matrices, parent/stage, scores, fixed denominators, accepted/rejected reasons, ambiguity and cache timings. Do not interpret geometric confidence as independent ground-truth accuracy.

## Retained 1.4.x processing stages (historical)

The following describes the previous implementation, not the default 2.0 solver. Its adaptive holdout and vertex-mask machinery is bypassed by 2.0. Historical claims here are not evidence that all those legacy mechanisms were independently validated in this release.

1. Validate and clean both triangle meshes.
2. Sample deterministic, area-weighted surface point clouds.
3. Compute downsampled normals and FPFH features.
4. Generate global candidates using fast global registration and repeated RANSAC.
5. Rank candidates and refine the best candidates with multi-scale robust point-to-plane ICP.
6. Estimate registration quality from bidirectional surface coverage, residuals, normal consistency, spatial coverage and candidate agreement.
7. Estimate the noise level and a coherent stable region from the data, then refine with point-to-surface ICP restricted to that region (see below).
8. Accept the refined pose only when it validates on held-out surface that was not fitted; otherwise retain the multi-scale result.
9. Compute signed closest-surface deviations and export geometry, transformation and quality metadata.

## Legacy adaptive stable-region refinement (unreleased 1.4.x)

Two scans of the same object are never identical: soft tissue swells, a print shrinks by a fraction of a percent, a crown is replaced. A single fixed distance threshold cannot separate those changes from measurement noise, so the final refinement estimates the separation from the data:

1. **Noise calibration.** Signed point-to-plane residuals at the incoming pose are collected on both surfaces. A robust scale σ is estimated from the consensus set by iterated median-absolute-deviation, so surface that has genuinely moved falls out of the estimate rather than inflating it. All later thresholds are multiples of σ.
2. **Bias detection.** A broad, low-amplitude change (gingival swelling, shrinkage) hides inside the noise band per vertex, but its residuals share a sign across the whole patch. Averaging residuals over a few rings of neighbours shrinks the noise by roughly √n while leaving the offset intact; vertices whose local mean exceeds the expected spread are marked changed.
3. **Coherent patches.** Per-vertex stability is smoothed over face adjacency and small islands are dropped, so the stable region is made of contiguous surface rather than scattered coincidences.
4. **Held-out validation.** The stable region on the moving surface is split into two spatially interleaved halves. Only one half is fitted; the pose then alternates between re-estimating the region and re-fitting, and is finally judged on the other half and on the fixed surface: median, P90 and signed bias must not worsen, the consensus area must not shrink, and no stable point may move further than a few σ beyond the previous ICP resolution.
5. A failed validation keeps the multi-scale pose and records the reason.

The legacy fixed-threshold gate remains available via `stable_region_enabled=False`.

`scripts/synthetic_benchmark.py` applies a known rigid transform plus synthetic local, broad-low-amplitude, scaling and noise corruptions to any scan so pose error can be measured directly against ground truth.

## Operator-selected registration regions (v1.4.2)

When either mesh contains an operator selection, the selected surface is the primary registration objective rather than a soft hint:

1. Run the unchanged full-surface pipeline to obtain a coarse baseline.
2. Generate both a 70%-weighted selection candidate and a strict selected-surface global candidate.
3. Starting from the baseline, weighted candidate and strict candidate, run local multi-scale ICP using only selected faces on each selected side. If only one side is selected, its region is matched against the complete opposite mesh.
4. Rank every non-catastrophic candidate by selected-region coverage first, then selected-region median and P90 distance. Full-surface quality does not veto a transform merely because most unselected faces changed.
5. Use the complete meshes only to reject non-finite/non-rigid transforms, excessive translation and catastrophic bounding-box separation.
6. A selected-region coverage below the confidence threshold lowers status and confidence but does not force a return to the full-surface baseline. Too few faces or insufficient normal diversity still triggers an explicit safe fallback because the region cannot constrain all six rigid degrees of freedom.

With no operator selection, 2.0 uses the candidate-bank pipeline described above. Selecting `algorithm_version="1.4"` invokes the retained legacy solver.

## Determinism

Surface sampling uses local NumPy random generators with fixed seeds. The same inputs and configuration should therefore produce stable samples and candidate evaluation on the same supported software stack. Floating-point and third-party implementation differences can still produce small platform-dependent changes.

## Important limitations

- The transform contains rotation and translation only.
- Symmetric or repetitive geometry may admit multiple plausible poses.
- Low overlap or an insufficiently distributed common surface can make the result underconstrained.
- Signed deviation depends on the target mesh's triangle-normal orientation.
- Confidence metrics describe geometric evidence and are not an accuracy guarantee.
