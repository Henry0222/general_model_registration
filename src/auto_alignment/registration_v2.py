"""2.0 rigid registration: multiple geometric hypotheses, fixed evaluation,
common-surface refinement and retained intermediate poses.

Experiment references never enter this module. All caches are run-local.
"""
from __future__ import annotations

from dataclasses import replace
import itertools
import math
import time
import numpy as np
import open3d as o3d

from .candidate_selection import CandidateBank, FixedEvaluator, MeshCache, observability, rigid_valid
from .feature_matching import FeatureMatchCache
from .mesh_io import prepare_cloud
from .quality import CandidateDiagnostic, PositionConfidence, assess_registration_quality


def _matrix(rotation, source_center, target_center):
    result = np.eye(4)
    result[:3, :3] = rotation
    result[:3, 3] = target_center-rotation@source_center
    return result


def _plane_frame(cloud, diagonal, seed):
    """Use actual supported plane normals; never assume ideal cube dimensions."""
    working = o3d.geometry.PointCloud(cloud)
    normals = []
    minimum = max(40, int(len(working.points)*.035))
    for index in range(6):
        if len(working.points) < minimum:
            break
        o3d.utility.random.seed(seed+index)
        plane, indices = working.segment_plane(max(diagonal*.0015, .005), 3, 350)
        if len(indices) < minimum:
            break
        normal = np.asarray(plane[:3], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)
        if not any(abs(normal@other) > .85 for other in normals):
            normals.append(normal)
        working = working.select_by_index(indices, invert=True)
        if len(normals) >= 3:
            break
    if len(normals) < 2:
        return None
    first = normals[0]
    second = normals[1]-first*(normals[1]@first)
    if np.linalg.norm(second) < .25:
        return None
    second /= np.linalg.norm(second)
    return np.column_stack((first, second, np.cross(first, second)))


def _refine_common(source, target, samples, initial, fraction, config, bank, prefix):
    """Estimate weights on fit samples only; evaluate on separate fixed samples.

    Distance weights use full Euclidean distance (including boundary/tangent
    errors), not just the signed normal projection. Every sample retains its
    triangle identity even above the old high-precision vertex cap.
    """
    from .registration import _rotation_from_vector

    transform = initial.transformation.copy()
    incumbent = initial
    floor = max(1e-6, min(source.diagonal, target.diagonal)*1e-6)
    counts = np.bincount(samples.blocks, minlength=64)
    spatial_weights = 1/np.maximum(counts[samples.blocks], 1)
    for iteration in range(max(1, int(config.candidate_common_iterations))):
        points = samples.points@transform[:3, :3].T+transform[:3, 3]
        nearest, normals, _ = target.query(points)
        distances = np.linalg.norm(points-nearest, axis=1)
        agreement = np.abs(np.einsum("ij,ij->i", samples.normals@transform[:3, :3].T, normals))
        eligible = np.isfinite(distances) & (agreement >= .5)
        if eligible.sum() < 60:
            break
        cutoff = max(floor*4, float(np.quantile(distances[eligible], fraction))*1.6)
        residuals = np.einsum("ij,ij->i", points-nearest, normals)
        weights = np.maximum(0, 1-(distances/cutoff)**2)**2
        weights *= eligible*spatial_weights*np.clip((agreement-.5)*2, 0, 1)
        valid = weights > 1e-12
        if valid.sum() < 30:
            break
        rank, _ = observability(points, normals, weights)
        if rank < 6:
            break
        weight = weights[valid]
        center = np.average(points[valid], axis=0, weights=weight)
        centered = points[valid]-center
        radius = max(float(np.sqrt(np.average(np.sum(centered**2, axis=1), weights=weight))), 1e-6)
        jacobian = np.column_stack((np.cross(centered, normals[valid])/radius, normals[valid]))
        square = np.sqrt(weight/weight.sum())
        step, _, solved_rank, _ = np.linalg.lstsq(jacobian*square[:, None], -residuals[valid]*square, rcond=1e-8)
        if solved_rank < 6 or not np.isfinite(step).all():
            break
        rotation_vector = step[:3]/radius
        translation = step[3:]
        multiplier = min(1., .12/max(np.linalg.norm(rotation_vector), 1e-15),
                         min(source.diagonal, target.diagonal)*.02/max(np.linalg.norm(translation), 1e-15))
        rotation = _rotation_from_vector(rotation_vector*multiplier)
        delta = _matrix(rotation, center, center+translation*multiplier)
        transform = delta@transform
        if (iteration+1) % 4 == 0 or iteration+1 == config.candidate_common_iterations:
            item = bank.add(f"{prefix}_{iteration+1}", "common_surface", transform, incumbent, guarded=True)
            item.evaluation["fit"] = dict(count=int(valid.sum()), cutoff_mm=cutoff,
                                           requested_fraction=fraction, observability_rank=rank)
            if item.accepted:
                incumbent = item
        if np.linalg.norm(step) < floor*.05:
            bank.add(f"{prefix}_converged_{iteration+1}", "common_surface", transform, incumbent, guarded=True)
            break


def register_v2(target_mesh, source_mesh, target_facts, source_facts, config, progress=None,
                *, target_priority_faces=None, source_priority_faces=None):
    from . import registration as legacy

    started = time.perf_counter()
    timings = {}
    warnings = list(target_facts.warnings+source_facts.warnings)
    def notify(fraction, message):
        if progress:
            progress(fraction, message)

    notify(.03, "2.0：缓存曲面并建立固定评价点…")
    source_cache, target_cache = MeshCache(source_mesh), MeshCache(target_mesh)
    diagonal = min(source_cache.diagonal, target_cache.diagonal)
    voxel = config.effective_voxel(diagonal)
    seed = config.random_seed
    sample_count = max(2000, int(config.global_sample_points))
    source_samples = source_cache.sample(sample_count, seed, source_priority_faces)
    target_samples = target_cache.sample(sample_count, seed+1, target_priority_faces)
    source_raw, target_raw = source_samples.cloud(), target_samples.cloud()
    evaluator = FixedEvaluator(source_cache, target_cache, max(1000, config.candidate_evaluation_points), seed,
                               config.partial_registration_enabled)
    bank = CandidateBank(evaluator)
    timings["geometry_cache_s"] = time.perf_counter()-started

    # Multi-scale clouds and descriptors are shared by all hypotheses.
    stage_started = time.perf_counter()
    scales = sorted(set(max(voxel, min(diagonal/45, voxel*float(scale)))
                        for scale in config.candidate_feature_scales))
    clouds, features, feature_matches = {}, {}, {}
    global_work = dict(feature_matching_s=0., ransac_s=0.)
    hypotheses = [("identity", np.eye(4)),
                  ("centers", _matrix(np.eye(3), source_cache.origin, target_cache.origin))]
    errors = []
    for index, scale in enumerate(scales):
        notify(.12+.06*index, f"2.0：提取第 {index+1} 层几何特征…")
        sc = prepare_cloud(source_raw, scale, config.normal_radius_multiplier)
        tc = prepare_cloud(target_raw, scale, config.normal_radius_multiplier)
        clouds[scale] = (sc, tc)
        sf, tf = legacy._features(sc, scale, config), legacy._features(tc, scale, config)
        features[scale] = (sf, tf)
        feature_matches[scale] = FeatureMatchCache(sf, tf)
        try:
            result = legacy._fast_global_registration(sc, tc, sf, tf, scale)
            hypotheses.append((f"fgr_scale_{index}", result.transformation))
        except RuntimeError as exc:
            errors.append(f"FGR scale {index}: {exc}")
    restarts = max(1, config.global_registration_restarts)
    for restart in range(restarts):
        notify(.22+.07*restart/restarts, f"2.0：全局匹配 {restart+1}/{restarts}…")
        scale = scales[restart % len(scales)]
        sc, tc = clouds[scale]
        sf, tf = features[scale]
        o3d.utility.random.seed(seed+restart)
        try:
            matching_started = time.perf_counter()
            matches = feature_matches[scale].get(mutual_filter=restart % 3 != 0)
            global_work["feature_matching_s"] += time.perf_counter()-matching_started
            ransac_started = time.perf_counter()
            result = legacy._global_registration(sc, tc, sf, tf, scale, config,
                                                   mutual_filter=restart % 3 != 0,
                                                   distance_multiplier=max(1.6, config.ransac_distance_multiplier),
                                                   correspondences=matches)
            global_work["ransac_s"] += time.perf_counter()-ransac_started
            hypotheses.append((f"ransac_{restart+1}", result.transformation))
        except RuntimeError as exc:
            errors.append(f"RANSAC {restart}: {exc}")
    sc, tc = clouds[max(scales)]
    # Always generate proper PCA permutations, including unequal surface areas.
    for index, transform in enumerate(legacy._exhaustive_principal_axis_candidates(
        sc, tc, angle_step_degrees=config.exhaustive_orientation_angle_step_degrees,
        eigen_tolerance_ratio=config.exhaustive_orientation_eigen_tolerance_ratio,
        max_candidates=config.exhaustive_orientation_max_candidates if config.exhaustive_orientation_search else 24,
    )):
        hypotheses.append((f"pca_pose_{index}", transform))
    if config.candidate_primitive_enabled:
        source_frame = _plane_frame(sc, diagonal, seed+90)
        target_frame = _plane_frame(tc, diagonal, seed+190)
        if source_frame is not None and target_frame is not None:
            for index, permutation in enumerate(legacy._proper_signed_axis_permutations()):
                hypotheses.append((f"plane_pose_{index}", _matrix(target_frame@permutation@source_frame.T,
                                                                 source_cache.origin, target_cache.origin)))
    timings["global_hypotheses_s"] = time.perf_counter()-stage_started

    stage_started = time.perf_counter()
    notify(.30, "2.0：保留原始姿态，比较多种共同表面假设…")
    point_estimator = o3d.pipelines.registration.TransformationEstimationPointToPoint(False)
    for index, (name, transform) in enumerate(hypotheses):
        if not rigid_valid(transform):
            errors.append(name+": invalid rigid transform")
            continue
        item = bank.add(name, "global", transform)
        refined = o3d.pipelines.registration.registration_icp(
            sc, tc, max(scales)*4, transform, point_estimator,
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=25))
        if rigid_valid(refined.transformation):
            bank.add(name+"_short", "short_icp", refined.transformation, item)
        if index % 10 == 0:
            notify(.3+.18*(index+1)/len(hypotheses), f"2.0：已检查 {index+1}/{len(hypotheses)} 个初始姿态…")
    if not bank.candidates:
        raise RuntimeError("所有全局候选均无法生成合法刚体变换。")
    timings["short_icp_and_evaluation_s"] = time.perf_counter()-stage_started

    # Common-region hypotheses are explored BEFORE one final pose is chosen.
    stage_started = time.perf_counter()
    initial_finalists = bank.finalists(max(2, int(config.candidate_pool_finalists)), evaluator.scales[1]*2)
    fit_samples = source_cache.sample(min(sample_count, 20000), seed+501, source_priority_faces)
    levels = []
    for index, fraction in enumerate(config.voxel_fractions):
        local_voxel = voxel*fraction
        if local_voxel not in clouds:
            clouds[local_voxel] = (prepare_cloud(source_raw, local_voxel, config.normal_radius_multiplier),
                                   prepare_cloud(target_raw, local_voxel, config.normal_radius_multiplier))
        levels.append((*clouds[local_voxel],
                       local_voxel*config.correspondence_multipliers[index], config.icp_iterations[index]))
    estimator = legacy._robust_estimator(config)
    for index, candidate in enumerate(initial_finalists):
        notify(.5+.30*index/len(initial_finalists), f"2.0：精化第 {index+1}/{len(initial_finalists)} 条候选分支…")
        transform = candidate.transformation.copy()
        incumbent = candidate
        for level, (source_level, target_level, distance, iterations) in enumerate(levels):
            result = o3d.pipelines.registration.registration_icp(
                source_level, target_level, distance, transform, estimator,
                o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iterations,
                                                                 relative_fitness=1e-7, relative_rmse=1e-7))
            transform = np.asarray(result.transformation)
            item = bank.add(f"{candidate.name}_level_{level}", "multiscale_icp", transform, incumbent, guarded=True)
            if item.accepted:
                incumbent = item
        if config.high_precision_refinement_enabled:
            for fraction in config.candidate_common_fractions:
                _refine_common(source_cache, target_cache, fit_samples, incumbent, fraction, config, bank,
                               f"{candidate.name}_common_{fraction:g}")
        if config.candidate_gicp_enabled and index < 3:
            source_level, target_level, distance, _ = levels[0]
            try:
                result = o3d.pipelines.registration.registration_generalized_icp(
                    source_level, target_level, distance*2, incumbent.transformation,
                    o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
                    o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=30))
                bank.add(candidate.name+"_gicp", "gicp", result.transformation, incumbent, guarded=True)
            except RuntimeError as exc:
                errors.append(f"GICP {candidate.name}: {exc}")
    timings["local_hypotheses_s"] = time.perf_counter()-stage_started

    selected = bank.best()
    transform = selected.transformation
    notify(.86, "2.0：统一重评全部保留姿态并检查歧义…")
    diagnostics = tuple(CandidateDiagnostic(c.name, c.transformation, c.evaluation["directions"][0]["common_coverage"],
                                           c.evaluation["directions"][0]["rms_mm"]) for c in bank.candidates)
    alternatives = bank.finalists(4, evaluator.scales[2])
    ambiguity = bool(len(alternatives) > 1 and selected.evaluation["score"]-alternatives[1].evaluation["score"] < .002
                     and evaluator.displacement(selected.transformation, alternatives[1].transformation) > evaluator.scales[2])
    source_level, target_level, distance, _ = levels[-1]
    evaluated = o3d.pipelines.registration.evaluate_registration(source_level, target_level, distance, transform)
    overlap = legacy._aabb_overlap_ratio(source_raw, target_raw, transform)
    aligned = o3d.geometry.TriangleMesh(source_mesh)
    aligned.transform(transform)
    aligned.compute_vertex_normals()
    # Broad exploratory hypotheses are not repeated estimates of the final pose.
    quality_diagnostics = tuple(d for d in diagnostics if evaluator.displacement(d.transformation, transform) <= evaluator.scales[2])
    quality = assess_registration_quality(target_mesh, aligned, transform, overlap, quality_diagnostics,
                                           coverage_distance_mm=config.coverage_distance_mm,
                                           min_directed_overlap=config.partial_overlap_threshold)
    common = selected.evaluation["mode"] == "common_surface"
    plausible = max(d["support_by_scale"][-1] for d in selected.evaluation["directions"]) >= .5
    if common and quality.position_confidence is not PositionConfidence.FAILED:
        status = "success" if quality.position_confidence is PositionConfidence.HIGH and not ambiguity else "warning"
        confidence = "高" if status == "success" else "中"
    elif plausible:
        status, confidence = "warning", "低"
        warnings.append("已输出整体形状最佳候选；共同表面证据不足，不能据此确认真实位姿精度。")
    else:
        status, confidence = "failed", "失败"
        warnings.append("候选之间没有足够的共同表面或整体形状支持。")
    if ambiguity:
        warnings.append("存在分数接近但位置不同的候选，请检查重复或近对称结构。")
    if errors:
        warnings.append(f"部分候选生成器未完成（{len(errors)} 项），详情见候选诊断。")
    metadata = dict(
        algorithm="2.0.0", policy="fixed_multiscale_surface_support_v1", selected=selected.name,
        selected_mode=selected.evaluation["mode"], ambiguous=ambiguity,
        candidate_count=len(bank.candidates), accepted_count=sum(c.accepted for c in bank.candidates),
        rejected_update_count=sum(not c.accepted for c in bank.candidates),
        evaluation_points=list(selected.evaluation["fixed_denominators"]),
        evaluation_scales_mm=evaluator.scales.tolist(), direction_weights=evaluator.direction_weights.tolist(),
        validation="Fixed independent area samples; internal geometry check, not independent ground truth.",
        hypotheses=[c.as_dict() for c in bank.candidates], errors=errors, timings=timings,
        global_search=global_work,
        cache=dict(scene_builds=2, source_queries=source_cache.queries, target_queries=target_cache.queries,
                   feature_match_searches=sum(cache.searches for cache in feature_matches.values()),
                   sample_sets=len(source_cache._samples)+len(target_cache._samples)),
    )
    metrics = legacy.RegistrationMetrics(
        fitness=float(evaluated.fitness), inlier_rmse_mm=float(evaluated.inlier_rmse),
        correspondence_count=len(evaluated.correspondence_set), overlap_ratio=overlap,
        rotation_degrees=legacy._rotation_angle_degrees(transform), translation_mm=float(np.linalg.norm(transform[:3, 3])),
        candidate_diagnostics=diagnostics, candidate_selection=metadata,
        high_precision_decision=dict(enabled=config.high_precision_refinement_enabled,
                                     accepted=selected.stage == "common_surface", selected_stage=selected.stage,
                                     mode="candidate_bank", reasons=["由固定评价选择全部阶段候选，保留未通过更新的父姿态。"]),
    )
    quality = replace(quality, position_confidence={"success": PositionConfidence.HIGH, "warning": PositionConfidence.MEDIUM if common else PositionConfidence.LOW,
                                                   "failed": PositionConfidence.FAILED}[status])
    notify(.90, "2.0：候选选择完成，正在保存结果…")
    return legacy.RegistrationResult(transform, status, confidence, metrics, tuple(dict.fromkeys(warnings)),
                                     time.perf_counter()-started, quality)
