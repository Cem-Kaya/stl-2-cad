#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
from shapely import contains_xy
from shapely.affinity import translate
from shapely.geometry import Point, Polygon, box
from skimage.measure import marching_cubes


HERE = Path(__file__).resolve().parent
ASSETS_DIR = HERE / "assets"
EXAMPLES_DIR = HERE / "examples"
DEFAULT_STL = HERE.parent / "input" / "3d-enclosure-for-xiao-bus-servo-adapter-bottom.stl"
SAMPLE_SPEC = EXAMPLES_DIR / "3d-enclosure-for-xiao-bus-servo-adapter-bottom_fit.json"


def rounded_rectangle_polygon(width: float, height: float, radius: float) -> Polygon:
    radius = min(radius, width * 0.5, height * 0.5)
    if radius <= 1e-9:
        return box(-width * 0.5, -height * 0.5, width * 0.5, height * 0.5)
    core = box(
        -width * 0.5 + radius,
        -height * 0.5 + radius,
        width * 0.5 - radius,
        height * 0.5 - radius,
    )
    return core.buffer(radius, quad_segs=24, join_style=1)


def primitive_to_polygon(primitive: dict[str, Any]) -> Polygon:
    kind = primitive["kind"]
    params = primitive["params"]
    if kind == "circle":
        return Point(0.0, 0.0).buffer(params["radius"], quad_segs=64)
    if kind == "annulus":
        outer = Point(0.0, 0.0).buffer(params["outer_radius"], quad_segs=64)
        inner = Point(0.0, 0.0).buffer(params["inner_radius"], quad_segs=64)
        return outer.difference(inner)
    if kind == "rectangle":
        return box(
            -params["width"] * 0.5,
            -params["height"] * 0.5,
            params["width"] * 0.5,
            params["height"] * 0.5,
        )
    if kind == "rounded_rectangle":
        return rounded_rectangle_polygon(
            params["width"], params["height"], params["radius"]
        )
    if kind in {"polygon", "triangle"}:
        return Polygon(params["points"])
    if kind == "profile":
        return Polygon(params["exterior"], holes=params["holes"])
    raise ValueError(f"Unsupported primitive kind: {kind}")


def spec_to_voxel_occupancy(
    spec: dict[str, Any],
    pitch: float,
    min_corner: np.ndarray,
    max_corner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    size = max_corner - min_corner
    dims = np.ceil(size / pitch).astype(int) + 2
    occupancy = np.zeros((int(dims[0]), int(dims[1]), int(dims[2])), dtype=bool)
    center0 = min_corner + pitch * 0.5

    xs = center0[0] + np.arange(occupancy.shape[0]) * pitch
    ys = center0[1] + np.arange(occupancy.shape[1]) * pitch
    zs = center0[2] + np.arange(occupancy.shape[2]) * pitch

    for feature in spec["features"]:
        base_geom = primitive_to_polygon(feature["primitive"])
        z_lo = float(feature["z_start"])
        z_hi = z_lo + float(feature["height"])
        z_mask = (zs >= z_lo - 1e-9) & (zs < z_hi - 1e-9)
        z_indices = np.flatnonzero(z_mask)
        if z_indices.size == 0:
            continue

        for center in feature["centers"]:
            geom = translate(base_geom, xoff=float(center[0]), yoff=float(center[1]))
            minx, miny, maxx, maxy = geom.bounds
            x_idx = np.flatnonzero((xs >= minx - pitch) & (xs <= maxx + pitch))
            y_idx = np.flatnonzero((ys >= miny - pitch) & (ys <= maxy + pitch))
            if x_idx.size == 0 or y_idx.size == 0:
                continue

            xx, yy = np.meshgrid(xs[x_idx], ys[y_idx], indexing="xy")
            inside = contains_xy(geom.buffer(pitch * 0.02), xx, yy)
            if not inside.any():
                continue

            xy_mask = inside.T
            for z_index in z_indices:
                if feature["op"] == "add":
                    occupancy[np.ix_(x_idx, y_idx, [z_index])] |= xy_mask[:, :, None]
                else:
                    occupancy[np.ix_(x_idx, y_idx, [z_index])] &= ~xy_mask[:, :, None]

    return occupancy, center0


def occupancy_to_surface_mesh(
    occupancy: np.ndarray, center0: np.ndarray, pitch: float
) -> trimesh.Trimesh:
    padded = np.pad(occupancy.astype(np.float32), 1)
    vertices, faces, _normals, _values = marching_cubes(
        padded, level=0.5, spacing=(pitch, pitch, pitch)
    )
    vertices = vertices + (center0 - pitch)
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.remove_unreferenced_vertices()
    return mesh


def face_colors(mesh: trimesh.Trimesh, color: str) -> np.ndarray:
    rgb = np.asarray(to_rgb(color), dtype=float)
    normals = mesh.face_normals
    light_dir = np.array([0.45, -0.35, 0.82], dtype=float)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
    intensity = 0.38 + 0.62 * diffuse
    shaded = np.clip(rgb[None, :] * intensity[:, None], 0.0, 1.0)
    return np.concatenate([shaded, np.ones((len(shaded), 1), dtype=float)], axis=1)


def apply_shared_camera(ax: Any, bounds: np.ndarray) -> None:
    mins = bounds[0]
    maxs = bounds[1]
    extents = maxs - mins
    center = (mins + maxs) * 0.5
    radius = float(max(extents) * 0.62)

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_box_aspect(np.maximum(extents, 1e-6))
    ax.set_axis_off()
    ax.view_init(elev=24, azim=-48, roll=0)
    ax.dist = 8


def draw_mesh(ax: Any, mesh: trimesh.Trimesh, color: str, bounds: np.ndarray, title: str) -> None:
    triangles = mesh.vertices[mesh.faces]
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors(mesh, color),
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
    )
    ax.add_collection3d(collection)
    apply_shared_camera(ax, bounds)
    ax.set_title(title, fontsize=16, pad=16)


def save_single_view(mesh: trimesh.Trimesh, color: str, title: str, target: Path, bounds: np.ndarray) -> None:
    fig = plt.figure(figsize=(8.2, 6.2), dpi=220)
    ax = fig.add_subplot(111, projection="3d")
    draw_mesh(ax, mesh, color, bounds, title)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
    fig.savefig(target, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def save_comparison(
    before_path: Path,
    after_path: Path,
    target: Path,
) -> None:
    before = Image.open(before_path).convert("RGB")
    after = Image.open(after_path).convert("RGB")
    width = before.width + after.width + 48
    height = max(before.height, after.height) + 24
    canvas = Image.new("RGB", (width, height), color=(255, 255, 255))
    canvas.paste(before, (12, (height - before.height) // 2))
    canvas.paste(after, (before.width + 36, (height - after.height) // 2))
    canvas.save(target)


def build_reconstruction_mesh(spec: dict[str, Any], reference_mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    pitch = float(spec["grid_pitch"])
    min_corner = np.floor(reference_mesh.bounds[0] / pitch) * pitch
    max_corner = np.ceil(reference_mesh.bounds[1] / pitch) * pitch
    occupancy, center0 = spec_to_voxel_occupancy(spec, pitch, min_corner, max_corner)
    return occupancy_to_surface_mesh(occupancy, center0, pitch)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render README preview assets.")
    parser.add_argument("--stl", type=Path, default=DEFAULT_STL, help="Input STL to render.")
    parser.add_argument(
        "--spec",
        type=Path,
        default=SAMPLE_SPEC,
        help="Model spec JSON used to reconstruct the voxel-fit mesh.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ASSETS_DIR,
        help="Directory to write the preview PNG files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    original_mesh = trimesh.load_mesh(args.stl, force="mesh")
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    recon_mesh = build_reconstruction_mesh(spec, original_mesh)

    shared_bounds = np.array(
        [
            np.minimum(original_mesh.bounds[0], recon_mesh.bounds[0]),
            np.maximum(original_mesh.bounds[1], recon_mesh.bounds[1]),
        ],
        dtype=float,
    )

    save_single_view(
        original_mesh,
        color="#4f46e5",
        title="Original STL",
        target=output_dir / "before-original-stl.png",
        bounds=shared_bounds,
    )
    save_single_view(
        recon_mesh,
        color="#f97316",
        title="Voxel Reconstruction",
        target=output_dir / "after-voxel-fit.png",
        bounds=shared_bounds,
    )
    save_comparison(
        before_path=output_dir / "before-original-stl.png",
        after_path=output_dir / "after-voxel-fit.png",
        target=output_dir / "before-after-comparison.png",
    )

    print("Wrote", output_dir / "before-original-stl.png")
    print("Wrote", output_dir / "after-voxel-fit.png")
    print("Wrote", output_dir / "before-after-comparison.png")


if __name__ == "__main__":
    main()
