"""Offline, reference-based rigid-pose evaluation. Never used by pose selection.

Column vectors; both registrations map the same moving frame to the same fixed
frame. Euler XYZ means R = Rx @ Ry @ Rz, not elementwise angle subtraction.
"""
from __future__ import annotations

import numpy as np


def _rigid_matrix(value):
    matrix = np.asarray(value, dtype=float)
    if (matrix.shape != (4, 4) or not np.isfinite(matrix).all()
            or not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8)
            or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=1e-5)
            or abs(np.linalg.det(matrix[:3, :3])-1) > 1e-5):
        raise ValueError("Expected a finite proper rigid 4x4 transformation.")
    return matrix


def euler_xyz_degrees(rotation, near=None):
    """Decompose Rx @ Ry @ Rz; flag the non-unique gimbal-lock case.

    If near is supplied, choose the equivalent non-singular Euler branch and
    full turns nearest that reference. This is for displaying imposed input
    parameters, not for computing the relative rotation error.
    """
    rotation = np.asarray(rotation, dtype=float)
    cosine_y = float(np.hypot(rotation[0, 0], rotation[0, 1]))
    y = np.arctan2(rotation[0, 2], cosine_y)
    singular = cosine_y < 1e-8
    if singular:
        x = np.arctan2(np.sign(rotation[0, 2])*rotation[1, 0], rotation[1, 1])
        z = 0.0
    else:
        x = np.arctan2(-rotation[1, 2], rotation[2, 2])
        z = np.arctan2(-rotation[0, 1], rotation[0, 0])
    angles = np.degrees([x, y, z])
    if near is not None and not singular:
        near = np.asarray(near, dtype=float)
        candidates = np.array([angles, [angles[0]+180, 180-angles[1], angles[2]+180]])
        candidates += 360*np.round((near-candidates)/360)
        angles = candidates[np.argmin(np.linalg.norm(candidates-near, axis=1))]
    return angles, bool(singular)


def _angle_degrees(rotation):
    # atan2 resolves very small rotations that acos(trace) can round to zero.
    skew = np.array([rotation[2, 1]-rotation[1, 2], rotation[0, 2]-rotation[2, 0],
                     rotation[1, 0]-rotation[0, 1]])/2
    return float(np.degrees(np.arctan2(np.linalg.norm(skew),
                                      np.clip((np.trace(rotation)-1)/2, -1, 1))))


def matrix_pose_error(estimated, reference):
    estimated, reference = _rigid_matrix(estimated), _rigid_matrix(reference)
    error = estimated @ np.linalg.inv(reference)
    correction = reference @ np.linalg.inv(estimated)
    angles, singular = euler_xyz_degrees(error[:3, :3])
    correction_angles, correction_singular = euler_xyz_degrees(correction[:3, :3])
    column_difference = estimated[:3, 3]-reference[:3, 3]
    return dict(
        convention="Column vectors; fixed frame; Euler XYZ means R = Rx @ Ry @ Rz.",
        relative_error_fixed_matrix=error.tolist(),
        relative_translation_xyz_mm=error[:3, 3].tolist(),
        relative_translation_norm_mm=float(np.linalg.norm(error[:3, 3])),
        relative_rotation_xyz_deg=angles.tolist(),
        relative_rotation_angle_deg=_angle_degrees(error[:3, :3]),
        relative_euler_gimbal_lock=singular,
        correction_fixed_matrix=correction.tolist(),
        correction_translation_xyz_mm=correction[:3, 3].tolist(),
        correction_rotation_xyz_deg=correction_angles.tolist(),
        correction_euler_gimbal_lock=correction_singular,
        translation_column_difference_xyz_mm=column_difference.tolist(),
        translation_column_difference_norm_mm=float(np.linalg.norm(column_difference)),
        rotation_matrix_difference_frobenius=float(np.linalg.norm(estimated[:3, :3]-reference[:3, :3])),
    )


def imposed_parameter_comparison(estimated, translation, angles_degrees):
    """Display recovered inputs for p_moving = R_xyz @ (p_fixed + t).

    The user's t is NOT the final forward matrix's translation column R @ t.
    With this particular convention the recovered t equals -T_registration.t.
    """
    estimated = _rigid_matrix(estimated)
    angles, singular = euler_xyz_degrees(estimated[:3, :3].T, near=angles_degrees)
    recovered_translation = -estimated[:3, 3]
    return dict(
        convention="p_moving = (Rx @ Ry @ Rz) @ (p_fixed + input_translation)",
        reference_translation_xyz_mm=np.asarray(translation, dtype=float).tolist(),
        estimated_translation_xyz_mm=recovered_translation.tolist(),
        translation_difference_xyz_mm=(recovered_translation-np.asarray(translation)).tolist(),
        reference_rotation_xyz_deg=np.asarray(angles_degrees, dtype=float).tolist(),
        estimated_rotation_xyz_deg=angles.tolist(),
        rotation_parameter_difference_xyz_deg=None if singular else (angles-np.asarray(angles_degrees)).tolist(),
        euler_gimbal_lock=singular,
    )
