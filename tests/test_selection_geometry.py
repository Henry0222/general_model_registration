"""Selection geometry must stay exact while avoiding topology reconstruction."""

import numpy as np
import open3d as o3d
import pytest

from auto_alignment.model_viewer import (
    _mesh_from_faces,
    _points_in_polygon,
    _projected_triangles_intersect_polygon,
    _selection_render_meshes,
)


@pytest.mark.parametrize("selection", ["none", "some", "all"])
def test_render_partition_preserves_face_order_coordinates_and_smooth_normals(selection):
    mesh = o3d.geometry.TriangleMesh.create_sphere(resolution=12)
    mesh.compute_vertex_normals()
    chosen = np.zeros(len(mesh.triangles), dtype=bool)
    if selection == "some":
        chosen[::3] = True
    elif selection == "all":
        chosen[:] = True
    vertices = np.asarray(mesh.vertices).copy()
    faces = np.asarray(mesh.triangles).copy()
    normals = np.asarray(mesh.vertex_normals).copy()

    for part, mask in zip(_selection_render_meshes(mesh, chosen), (~chosen, chosen)):
        part_faces = np.asarray(part.triangles)
        np.testing.assert_array_equal(np.asarray(part.vertices)[part_faces], vertices[faces[mask]])
        np.testing.assert_array_equal(
            np.asarray(part.vertex_normals)[part_faces], normals[faces[mask]]
        )
        assert len(np.unique(part_faces)) == len(part.vertices)
    np.testing.assert_array_equal(np.asarray(mesh.vertices), vertices)
    np.testing.assert_array_equal(np.asarray(mesh.triangles), faces)
    np.testing.assert_array_equal(np.asarray(mesh.vertex_normals), normals)


def test_render_mesh_computes_normals_when_input_has_none():
    mesh = o3d.geometry.TriangleMesh.create_box()
    part = _mesh_from_faces(mesh, np.ones(len(mesh.triangles), dtype=bool))
    assert part.has_vertex_normals()
    assert not mesh.has_vertex_normals()


@pytest.mark.parametrize("reverse", [False, True])
def test_through_selection_concave_lasso_and_boundary_contacts(reverse):
    # An L-shaped lasso excludes the upper right square of its bounding box.
    polygon = np.array([[0, 0], [4, 0], [4, 1], [1, 1], [1, 4], [0, 4]], dtype=float)
    if reverse:
        polygon = polygon[::-1]
    triangles = np.array([
        [[2, 2], [3, 2], [2, 3]],       # inside bounding box, outside concave lasso
        [[2, .2], [3, .2], [2, .8]],    # inside lower arm
        [[-1, -1], [12, -1], [-1, 12]], # contains whole lasso, no triangle vertex inside
        [[-1, 2], [2, 2], [2, 2.1]],   # crosses narrow arm, all vertices outside
        [[4, 0], [5, -1], [5, 0]],     # exact corner contact
        [[4, .2], [5, .2], [4, .8]],    # collinear edge contact
        [[5, 2], [6, 2], [5, 3]],      # disjoint
        [[np.nan, 0], [1, 0], [0, 1]], # invalid projection
    ])
    np.testing.assert_array_equal(
        _projected_triangles_intersect_polygon(triangles, polygon),
        [False, True, True, True, True, True, False, False],
    )


def test_point_membership_with_duplicate_vertices_and_self_crossing_stroke():
    polygon = np.array([[0, 0], [4, 4], [0, 4], [4, 0], [4, 0], [0, 0]], dtype=float)
    points = np.array([[2, .5], [2, 3.5], [.2, 2], [3.8, 2], [5, 2], [np.nan, 2]])
    np.testing.assert_array_equal(
        _points_in_polygon(points, polygon), [True, True, False, False, False, False]
    )


def test_empty_and_short_strokes_do_not_select_faces():
    triangles = np.array([[[0, 0], [1, 0], [0, 1]]], dtype=float)
    for polygon in (np.empty((0, 2)), np.array([[0, 0], [1, 1]])):
        assert not _projected_triangles_intersect_polygon(triangles, polygon).any()
    assert _projected_triangles_intersect_polygon(
        np.empty((0, 3, 2)), np.array([[0, 0], [1, 0], [0, 1]])
    ).size == 0
