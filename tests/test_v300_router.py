import numpy as np
import open3d as o3d
import pytest

from auto_alignment.refinement import _router as router


def validation():
    def stats(median):
        return {direction: dict(median_mm=median, p90_mm=.12, p95_mm=.15,
                                median_normal_dot=.99)
                for direction in ('source_to_target', 'target_to_source')}
    return dict(initial=stats(.1), A=stats(.09), B=stats(.06),
                correction_mm=.1, correction_deg=.2)


def select(v, meta=None):
    return router.decide(dict(keep_initial=False), dict(keep_initial=False),
                         meta or dict(status='final_40', iterations=40), v)[0]


def test_routing_never_uses_rms_or_field_magnitude():
    v = validation()
    assert select(v) == 'B'
    for value in (0., 1e9, float('nan')):
        for name in ('initial', 'A', 'B'):
            v[name]['rmse'] = value
        assert select(v, dict(status='final_40', iterations=40, field_rms=value)) == 'B'


@pytest.mark.parametrize('direction', ['source_to_target', 'target_to_source'])
@pytest.mark.parametrize('metric,value', [('median_mm', .085), ('p90_mm', .121),
                                          ('p95_mm', .151), ('median_mm', float('nan'))])
def test_either_direction_or_tail_can_veto_b(direction, metric, value):
    v = validation()
    v['B'][direction][metric] = value
    assert select(v) == 'initial'


def test_near_zero_baseline_and_exact_support_are_preserved():
    v = validation()
    for name in ('initial', 'A', 'B'):
        for direction in ('source_to_target', 'target_to_source'):
            v[name][direction]['median_mm'] = 0.
    assert select(v) == 'initial'
    assert router.decide(dict(keep_initial=True), {}, {}, {})[0] == 'initial'
    assert router.decide(dict(keep_initial=False), dict(keep_initial=True), {}, {})[0] == 'A'


def test_holdout_scores_identical_rigid_poses_identically_without_a_fitted_field():
    mesh = o3d.geometry.TriangleMesh.create_box(8., 13., 21.)
    np.asarray(mesh.vertices)[0] += [.7, .3, .2]
    mesh.compute_vertex_normals()
    ctx = router.Context(mesh, mesh, np.eye(4))
    stats, cache = ctx.validate(ctx.T0, ctx.T0)
    assert stats['initial'] == stats['A'] == stats['B']
    # Any fitted field, including an unusable object, must be irrelevant.
    ctx.validation = object()
    again, _ = ctx.validate(ctx.T0, ctx.T0)
    assert again == stats
    for name in ('initial', 'A', 'B'):
        for direction in ('source_to_target', 'target_to_source'):
            prefix = f'validation_{name}_{direction}_'
            expected = np.quantile(np.linalg.norm(cache[prefix+'q'] - cache[prefix+'y'], axis=1), [.5, .9, .95])
            np.testing.assert_allclose(expected, [stats[name][direction][key]
                                                for key in ('median_mm', 'p90_mm', 'p95_mm')])
            assert stats[name][direction]['p95_mm'] < 1e-5
