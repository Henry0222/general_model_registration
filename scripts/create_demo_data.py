from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("Open3D is required. Run install_windows.bat first.")


def tooth_mesh(center_x: float, center_y: float, radii: tuple[float, float, float]):
    mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=22)
    vertices = np.asarray(mesh.vertices)
    vertices *= np.asarray(radii)
    z = vertices[:, 2]
    vertices[:, 0] *= 1.0 + 0.10 * np.cos(z / radii[2] * np.pi)
    vertices[:, 1] *= 1.0 + 0.07 * np.sin(z / radii[2] * np.pi)
    vertices[:, 0] += center_x
    vertices[:, 1] += center_y
    vertices[:, 2] += radii[2] * 0.65
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals()
    return mesh


def build_arch(local_change: bool = False):
    combined = o3d.geometry.TriangleMesh()
    positions = [(-18, 2.0), (-9, 0.6), (0, 0), (9, 0.7), (18, 2.2)]
    for index, (x, y) in enumerate(positions):
        mesh = tooth_mesh(x, y, (4.6 + index * 0.12, 5.1, 6.8 - index * 0.08))
        if local_change and index == 2:
            vertices = np.asarray(mesh.vertices)
            crown = vertices[:, 2] > 5.8
            vertices[crown, 1] -= 0.65 * (vertices[crown, 2] - 5.8) / 2.5
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.compute_vertex_normals()
        combined += mesh
    combined.compute_vertex_normals()
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Create synthetic dental STL demo data")
    parser.add_argument("--output", type=Path, default=Path("demo_data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    target = build_arch(local_change=False)
    current = build_arch(local_change=True)
    angle = np.deg2rad(24.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]],
        dtype=float,
    )
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = [8.0, -5.0, 3.0]
    current.transform(transform)

    target_path = args.output / "demo_target.stl"
    current_path = args.output / "demo_current.stl"
    o3d.io.write_triangle_mesh(str(target_path), target)
    o3d.io.write_triangle_mesh(str(current_path), current)
    np.savetxt(args.output / "applied_target_to_current_transform.txt", transform)
    print(target_path.resolve())
    print(current_path.resolve())


if __name__ == "__main__":
    sys.exit(main())
