from __future__ import annotations

import numpy as np
import open3d as o3d
import pytest

from auto_alignment.pose_evaluation import euler_xyz_degrees, imposed_parameter_comparison, matrix_pose_error


def transform(angles=(0, 0, 0), translation=(0, 0, 0)):
    value = np.eye(4)
    value[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(np.radians(angles))
    value[:3, 3] = translation
    return value


def test_relative_pose_uses_fixed_frame_and_correction_is_its_inverse():
    gold = transform((23, -47, 98), (100, -60, 500))
    error = transform((.3, -.7, 1.2), (.2, -.4, .6))
    result = error @ gold
    metrics = matrix_pose_error(result, gold)
    assert np.allclose(metrics["relative_error_fixed_matrix"], error)
    assert np.allclose(metrics["relative_translation_xyz_mm"], [.2, -.4, .6])
    assert np.allclose(metrics["relative_rotation_xyz_deg"], [.3, -.7, 1.2])
    assert np.allclose(np.asarray(metrics["correction_fixed_matrix"]) @ result, gold)
    assert not np.allclose(metrics["translation_column_difference_xyz_mm"], metrics["relative_translation_xyz_mm"])


@pytest.mark.parametrize("angles", [(58, 47, 268), (8, 168, 285), (2, -179, 359)])
def test_recovered_imposed_parameters_preserve_equivalent_euler_branch(angles):
    translation = np.array([100., -60., 500.])
    applied = transform(angles)
    applied[:3, 3] = applied[:3, :3] @ translation
    metrics = imposed_parameter_comparison(np.linalg.inv(applied), translation, angles)
    assert np.allclose(metrics["estimated_translation_xyz_mm"], translation)
    assert np.allclose(metrics["estimated_rotation_xyz_deg"], angles)
    assert np.allclose(metrics["rotation_parameter_difference_xyz_deg"], 0, atol=1e-12)


def test_small_angle_and_wrapped_angle_are_not_rounded_or_subtracted_wrongly():
    tiny = matrix_pose_error(transform((1e-7, 0, 0)), np.eye(4))
    assert tiny["relative_rotation_angle_deg"] == pytest.approx(1e-7)
    across = matrix_pose_error(transform((0, 0, -179)), transform((0, 0, 179)))
    assert across["relative_rotation_angle_deg"] == pytest.approx(2)
    assert across["relative_rotation_xyz_deg"][2] == pytest.approx(2)


@pytest.mark.parametrize("y", [-90, 90])
def test_gimbal_lock_is_flagged_and_rotation_still_round_trips(y):
    original = transform((20, y, -37))[:3, :3]
    angles, singular = euler_xyz_degrees(original)
    assert singular
    assert np.allclose(transform(angles)[:3, :3], original)


def test_translation_is_origin_dependent_but_total_rotation_is_not():
    error = transform((2, 0, 0), (.1, .2, .3))
    change = transform(translation=(100, -300, 800))
    original = matrix_pose_error(error, np.eye(4))
    shifted = matrix_pose_error(change @ error @ np.linalg.inv(change), np.eye(4))
    assert shifted["relative_rotation_angle_deg"] == pytest.approx(original["relative_rotation_angle_deg"])
    assert not np.allclose(shifted["relative_translation_xyz_mm"], original["relative_translation_xyz_mm"])


def test_nonrigid_reference_rejected():
    invalid = np.eye(4)
    invalid[0, 0] = 2
    with pytest.raises(ValueError):
        matrix_pose_error(np.eye(4), invalid)
