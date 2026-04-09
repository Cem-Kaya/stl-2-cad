#!/usr/bin/env python3
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import gradio as gr
import numpy as np
import plotly.graph_objects as go
import trimesh
from shapely import contains_xy
from shapely.affinity import translate
from shapely.geometry import Point, Polygon, box

from fit_stl_to_parametric import (
    export_step_from_model_spec,
    export_stl_from_model_spec,
    solve,
    spec_to_dict,
    write_rebuild_script,
    write_spec,
)


APP_DIR = Path(__file__).resolve().parent
UI_OUTPUT_DIR = APP_DIR / "ui-output"
UI_OUTPUT_DIR.mkdir(exist_ok=True)


def mesh_trace(mesh: trimesh.Trimesh, name: str, color: str, opacity: float, visible: bool) -> go.Mesh3d:
    vertices = mesh.vertices
    faces = mesh.faces
    return go.Mesh3d(
        x=vertices[:, 0],
        y=vertices[:, 1],
        z=vertices[:, 2],
        i=faces[:, 0],
        j=faces[:, 1],
        k=faces[:, 2],
        name=name,
        color=color,
        opacity=opacity,
        visible=visible,
        flatshading=True,
    )


def points_trace(points: np.ndarray, name: str, color: str, size: float, visible: bool) -> go.Scatter3d:
    if points.size == 0:
        points = np.zeros((0, 3), dtype=float)
    return go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode="markers",
        name=name,
        visible=visible,
        marker={"size": size, "color": color, "opacity": 0.85},
    )


def voxel_point_cloud(mesh: trimesh.Trimesh, pitch: float) -> tuple[np.ndarray, trimesh.voxel.base.VoxelGrid]:
    vox = mesh.voxelized(pitch).fill()
    if len(vox.sparse_indices) == 0:
        return np.zeros((0, 3), dtype=float), vox
    points = vox.points
    return np.asarray(points, dtype=float), vox


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
    return core.buffer(radius, quad_segs=16, join_style=1)


def primitive_to_polygon(primitive: dict[str, Any]) -> Polygon:
    kind = primitive["kind"]
    params = primitive["params"]
    if kind == "circle":
        return Point(0.0, 0.0).buffer(params["radius"], quad_segs=32)
    if kind == "annulus":
        outer = Point(0.0, 0.0).buffer(params["outer_radius"], quad_segs=32)
        inner = Point(0.0, 0.0).buffer(params["inner_radius"], quad_segs=32)
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
        outer = Polygon(params["exterior"])
        holes = [Polygon(hole) for hole in params["holes"]]
        geom = outer
        for hole in holes:
            geom = geom.difference(hole)
        return geom
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


def voxel_grid_to_mesh(occupancy: np.ndarray, center0: np.ndarray, pitch: float) -> trimesh.Trimesh:
    transform = np.array(
        [
            [pitch, 0.0, 0.0, center0[0]],
            [0.0, pitch, 0.0, center0[1]],
            [0.0, 0.0, pitch, center0[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    vox = trimesh.voxel.VoxelGrid(occupancy, transform=transform)
    mesh = vox.as_boxes()
    mesh.process(validate=True)
    return mesh


def occupancy_indices(points: np.ndarray, center0: np.ndarray, pitch: float) -> np.ndarray:
    if points.size == 0:
        return np.zeros((0, 3), dtype=np.int32)
    return np.rint((points - center0) / pitch).astype(np.int32)


def build_comparison_data(
    original_mesh: trimesh.Trimesh,
    spec: dict[str, Any],
    compare_pitch: float,
) -> dict[str, Any]:
    original_points, original_vox = voxel_point_cloud(original_mesh, compare_pitch)

    spec_min = np.asarray(original_mesh.bounds[0], dtype=float)
    spec_max = np.asarray(original_mesh.bounds[1], dtype=float)
    min_corner = np.floor(spec_min / compare_pitch) * compare_pitch
    max_corner = np.ceil(spec_max / compare_pitch) * compare_pitch
    recon_occ, center0 = spec_to_voxel_occupancy(spec, compare_pitch, min_corner, max_corner)
    recon_mesh = voxel_grid_to_mesh(recon_occ, center0, compare_pitch)
    recon_points = center0 + np.argwhere(recon_occ) * compare_pitch

    min_corner = np.floor(min_corner / compare_pitch) * compare_pitch
    center0 = min_corner + compare_pitch * 0.5

    orig_idx = occupancy_indices(original_points, center0, compare_pitch)
    recon_idx = occupancy_indices(recon_points, center0, compare_pitch)

    orig_set = {tuple(item) for item in orig_idx.tolist()}
    recon_set = {tuple(item) for item in recon_idx.tolist()}

    inter = orig_set & recon_set
    union = orig_set | recon_set
    missing = orig_set - recon_set
    extra = recon_set - orig_set

    def idx_to_points(idx_set: set[tuple[int, int, int]]) -> np.ndarray:
        if not idx_set:
            return np.zeros((0, 3), dtype=float)
        arr = np.asarray(sorted(idx_set), dtype=float)
        return center0 + arr * compare_pitch

    iou = float(len(inter) / len(union)) if union else 1.0

    return {
        "original_vertices": original_mesh.vertices.tolist(),
        "original_faces": original_mesh.faces.tolist(),
        "recon_vertices": recon_mesh.vertices.tolist(),
        "recon_faces": recon_mesh.faces.tolist(),
        "missing_points": idx_to_points(missing).tolist(),
        "extra_points": idx_to_points(extra).tolist(),
        "compare_pitch": compare_pitch,
        "iou": iou,
        "original_voxels": len(orig_set),
        "recon_voxels": len(recon_set),
        "intersection_voxels": len(inter),
        "union_voxels": len(union),
        "missing_voxels": len(missing),
        "extra_voxels": len(extra),
        "original_bounds": original_mesh.bounds.tolist(),
        "recon_bounds": recon_mesh.bounds.tolist(),
    }


def make_figure(
    comparison: dict[str, Any] | None,
    show_original: bool,
    show_recon: bool,
    show_missing: bool,
    show_extra: bool,
) -> go.Figure:
    fig = go.Figure()
    if not comparison:
        fig.update_layout(template="plotly_white", title="No comparison yet")
        return fig

    original_mesh = trimesh.Trimesh(
        vertices=np.asarray(comparison["original_vertices"], dtype=float),
        faces=np.asarray(comparison["original_faces"], dtype=np.int32),
        process=False,
    )
    recon_mesh = trimesh.Trimesh(
        vertices=np.asarray(comparison["recon_vertices"], dtype=float),
        faces=np.asarray(comparison["recon_faces"], dtype=np.int32),
        process=False,
    )
    missing_points = np.asarray(comparison["missing_points"], dtype=float)
    extra_points = np.asarray(comparison["extra_points"], dtype=float)

    fig.add_trace(mesh_trace(original_mesh, "Original STL", "#3b82f6", 0.28, show_original))
    fig.add_trace(mesh_trace(recon_mesh, "Reconstruction", "#111827", 0.50, show_recon))
    fig.add_trace(points_trace(missing_points, "Missing In Recon", "#ef4444", 3.0, show_missing))
    fig.add_trace(points_trace(extra_points, "Extra In Recon", "#10b981", 3.0, show_extra))

    bounds = np.vstack([original_mesh.bounds, recon_mesh.bounds])
    mins = bounds.min(axis=0)
    maxs = bounds.max(axis=0)
    extent = maxs - mins
    max_extent = float(max(extent.max(), 1.0))

    fig.update_layout(
        template="plotly_white",
        margin={"l": 0, "r": 0, "t": 30, "b": 0},
        legend={"orientation": "h"},
        scene={
            "aspectmode": "manual",
            "aspectratio": {
                "x": float(extent[0] / max_extent),
                "y": float(extent[1] / max_extent),
                "z": float(extent[2] / max_extent),
            },
            "xaxis_title": "X",
            "yaxis_title": "Y",
            "zaxis_title": "Z",
        },
    )
    return fig


def metrics_markdown(spec: dict[str, Any], comparison: dict[str, Any]) -> str:
    notes = spec.get("notes", [])
    note_text = "\n".join(f"- {note}" for note in notes) if notes else "- none"
    return "\n".join(
        [
            f"**Voxel IoU:** `{comparison['iou']:.4f}`",
            f"**Step Count:** `{spec['step_count']}` / `{spec['step_budget']}`",
            f"**Original Voxels:** `{comparison['original_voxels']}`",
            f"**Reconstruction Voxels:** `{comparison['recon_voxels']}`",
            f"**Missing Voxels:** `{comparison['missing_voxels']}`",
            f"**Extra Voxels:** `{comparison['extra_voxels']}`",
            f"**Fit Notes:**\n{note_text}",
        ]
    )


def run_fit(
    input_file: str,
    voxel_pitch: float,
    compare_pitch: float,
    max_steps: int,
) -> tuple[str, go.Figure, str, str, str, str, dict[str, Any]]:
    input_path = Path(input_file)
    session_dir = Path(tempfile.mkdtemp(prefix="gradio-fit-", dir=UI_OUTPUT_DIR))

    spec = solve(
        input_path=input_path,
        grid=320,
        pad=0.6,
        thin_axis="z",
        max_steps=max_steps,
        voxel_pitch=voxel_pitch,
    )
    spec_data = spec_to_dict(spec)

    stem = input_path.stem
    spec_path = session_dir / f"{stem}_fit.json"
    rebuild_path = session_dir / f"{stem}_rebuild.py"
    step_path = session_dir / f"{stem}_rebuild.step"
    recon_stl_path = session_dir / f"{stem}_rebuild.stl"

    write_spec(spec, spec_path)
    write_rebuild_script(spec, rebuild_path, spec_path)
    export_step_from_model_spec(spec, step_path)
    export_stl_from_model_spec(spec, recon_stl_path)

    original_mesh = trimesh.load(str(input_path), force="mesh")
    comparison = build_comparison_data(original_mesh, spec_data, compare_pitch)
    figure = make_figure(comparison, True, True, True, True)
    summary = metrics_markdown(spec_data, comparison)

    state = {
        "comparison": comparison,
        "spec": spec_data,
        "paths": {
            "step": str(step_path),
            "json": str(spec_path),
            "rebuild": str(rebuild_path),
            "recon_stl": str(recon_stl_path),
        },
    }
    return (
        summary,
        figure,
        str(step_path),
        str(spec_path),
        str(rebuild_path),
        str(recon_stl_path),
        state,
    )


def update_figure(
    state: dict[str, Any],
    show_original: bool,
    show_recon: bool,
    show_missing: bool,
    show_extra: bool,
) -> go.Figure:
    if not state:
        return make_figure(None, show_original, show_recon, show_missing, show_extra)
    return make_figure(
        state["comparison"],
        show_original,
        show_recon,
        show_missing,
        show_extra,
    )


with gr.Blocks(title="STL To Parametric STEP") as demo:
    gr.Markdown("## STL To Parametric STEP\nFit an STL, export a compact STEP reconstruction, and inspect voxel IoU plus missing/extra 3D regions.")

    fit_state = gr.State({})

    with gr.Row():
        input_file = gr.File(label="Input STL", file_types=[".stl"], type="filepath")
        with gr.Column():
            voxel_pitch = gr.Slider(0.2, 1.0, value=0.5, step=0.05, label="Fit Voxel Pitch (mm)")
            compare_pitch = gr.Slider(0.2, 1.0, value=0.5, step=0.05, label="Compare Voxel Pitch (mm)")
            max_steps = gr.Slider(4, 128, value=128, step=1, label="Max Steps")
            run_button = gr.Button("Fit And Compare", variant="primary")

    with gr.Row():
        show_original = gr.Checkbox(value=True, label="Show Original")
        show_recon = gr.Checkbox(value=True, label="Show Reconstruction")
        show_missing = gr.Checkbox(value=True, label="Show Missing")
        show_extra = gr.Checkbox(value=True, label="Show Extra")

    summary = gr.Markdown()
    figure = gr.Plot(label="3D Comparison")

    with gr.Row():
        step_file = gr.File(label="STEP Output")
        spec_file = gr.File(label="Fit JSON")
        rebuild_file = gr.File(label="Rebuild Script")
        recon_stl_file = gr.File(label="Reconstructed STL")

    run_button.click(
        fn=run_fit,
        inputs=[input_file, voxel_pitch, compare_pitch, max_steps],
        outputs=[summary, figure, step_file, spec_file, rebuild_file, recon_stl_file, fit_state],
    )

    for control in [show_original, show_recon, show_missing, show_extra]:
        control.change(
            fn=update_figure,
            inputs=[fit_state, show_original, show_recon, show_missing, show_extra],
            outputs=figure,
        )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0")
