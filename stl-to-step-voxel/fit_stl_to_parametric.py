#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import struct
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover
    ndimage = None

try:
    from skimage import measure
except ImportError:  # pragma: no cover
    measure = None

try:
    from shapely.affinity import translate
    from shapely.geometry import MultiPolygon, Point, Polygon, box
    from shapely.ops import unary_union
except ImportError:  # pragma: no cover
    translate = None
    MultiPolygon = None
    Point = None
    Polygon = None
    box = None
    unary_union = None

try:
    from stl import mesh as stl_mesh
except ImportError:  # pragma: no cover
    stl_mesh = None

try:
    import trimesh
except ImportError:  # pragma: no cover
    trimesh = None

try:
    from build123d import (
        BuildPart,
    BuildSketch,
    Circle,
    Face,
    Locations,
    Mode,
    Plane,
    Polygon as B3DPolygon,
    Rectangle,
    RectangleRounded,
    Wire,
    export_step,
    export_stl,
    extrude,
)
except ImportError:  # pragma: no cover
    BuildPart = None
    BuildSketch = None
    Circle = None
    Face = None
    Locations = None
    Mode = None
    Plane = None
    B3DPolygon = None
    Rectangle = None
    RectangleRounded = None
    Wire = None
    export_step = None
    export_stl = None
    extrude = None


def _require_runtime() -> None:
    missing: list[str] = []
    if ndimage is None:
        missing.append("scipy")
    if measure is None:
        missing.append("scikit-image")
    if Polygon is None:
        missing.append("shapely")
    if missing:
        joined = ", ".join(missing)
        raise SystemExit(
            f"Missing runtime dependencies: {joined}. "
            "Install them with `python -m pip install -r requirements.txt`."
        )


@dataclass
class Primitive2D:
    kind: str
    params: dict[str, Any]
    fit_error: float


@dataclass
class Feature:
    op: str
    z_start: float
    height: float
    primitive: Primitive2D
    centers: list[list[float]]
    area: float
    source: str


@dataclass
class ModelSpec:
    source_stl: str
    grid_pitch: float
    base_bottom: float
    base_top: float
    base_primitive: Primitive2D
    features: list[Feature] = field(default_factory=list)
    step_budget: int = 128
    step_count: int = 0
    notes: list[str] = field(default_factory=list)


@dataclass
class RasterData:
    x0: float
    y0: float
    pitch: float
    zmin: np.ndarray
    zmax: np.ndarray
    raw_occ: np.ndarray
    height_occ: np.ndarray
    footprint: np.ndarray


def primitive_center(primitive: Primitive2D, fallback_poly: Any) -> list[float]:
    return primitive.params.get(
        "center", [float(fallback_poly.centroid.x), float(fallback_poly.centroid.y)]
    )


def spec_to_dict(spec: ModelSpec | dict[str, Any]) -> dict[str, Any]:
    return asdict(spec) if isinstance(spec, ModelSpec) else spec


def quantize(value: float, step: float) -> float:
    if step <= 0:
        return float(value)
    return round(float(value) / step) * step


def quantized_mode(values: np.ndarray, step: float) -> float:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0
    q = np.round(vals / step) * step
    uniq, counts = np.unique(q, return_counts=True)
    return float(uniq[np.argmax(counts)])


def load_stl_triangles(path: Path) -> np.ndarray:
    if stl_mesh is not None:
        model = stl_mesh.Mesh.from_file(str(path))
        return np.asarray(model.vectors, dtype=np.float64)

    with path.open("rb") as handle:
        header = handle.read(80)
        try:
            tri_count = struct.unpack("<I", handle.read(4))[0]
        except struct.error as exc:  # pragma: no cover
            raise SystemExit(f"Could not parse STL header: {exc}") from exc
        data = handle.read()

    record = np.dtype(
        [
            ("normal", "<f4", 3),
            ("v1", "<f4", 3),
            ("v2", "<f4", 3),
            ("v3", "<f4", 3),
            ("attr", "<u2"),
        ]
    )
    arr = np.frombuffer(data, dtype=record, count=tri_count)
    return np.stack([arr["v1"], arr["v2"], arr["v3"]], axis=1).astype(np.float64)


def triangle_normals_and_areas(triangles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edge_a = triangles[:, 1] - triangles[:, 0]
    edge_b = triangles[:, 2] - triangles[:, 0]
    normals = np.cross(edge_a, edge_b)
    areas = np.linalg.norm(normals, axis=1) * 0.5
    good = areas > 1e-12
    normals[good] /= np.linalg.norm(normals[good], axis=1, keepdims=True)
    normals[~good] = np.array([0.0, 0.0, 1.0])
    return normals, areas


def ensure_thin_axis_is_z(
    triangles: np.ndarray, axis: str = "z"
) -> tuple[np.ndarray, list[int], list[str]]:
    axis_names = ["x", "y", "z"]
    verts = triangles.reshape(-1, 3)
    extents = verts.max(axis=0) - verts.min(axis=0)
    notes: list[str] = []

    if axis == "auto":
        thin_index = int(np.argmin(extents))
        if thin_index != 2:
            order = [i for i in range(3) if i != thin_index] + [thin_index]
            triangles = triangles[:, :, order]
            notes.append(
                f"Auto-reordered axes from {axis_names} to {[axis_names[i] for i in order]}."
            )
            return triangles, order, notes
        return triangles, [0, 1, 2], notes

    requested = axis_names.index(axis)
    if requested == 2:
        return triangles, [0, 1, 2], notes

    order = [i for i in range(3) if i != requested] + [requested]
    triangles = triangles[:, :, order]
    notes.append(
        f"Reordered axes from {axis_names} to {[axis_names[i] for i in order]}."
    )
    return triangles, order, notes


def rasterize_height_fields(
    triangles: np.ndarray, grid: int, pad: float
) -> RasterData:
    verts = triangles.reshape(-1, 3)
    mins = verts.min(axis=0)
    maxs = verts.max(axis=0)
    xy_min = mins[:2] - pad
    xy_max = maxs[:2] + pad
    pitch = float((xy_max - xy_min).max() / grid)

    nx = int(np.ceil((xy_max[0] - xy_min[0]) / pitch)) + 1
    ny = int(np.ceil((xy_max[1] - xy_min[1]) / pitch)) + 1

    zmin = np.full((ny, nx), np.inf, dtype=np.float64)
    zmax = np.full((ny, nx), -np.inf, dtype=np.float64)
    proj_occ = np.zeros((ny, nx), dtype=bool)

    for tri in triangles:
        pts_xy = tri[:, :2]
        den = (
            (pts_xy[1, 1] - pts_xy[2, 1]) * (pts_xy[0, 0] - pts_xy[2, 0])
            + (pts_xy[2, 0] - pts_xy[1, 0]) * (pts_xy[0, 1] - pts_xy[2, 1])
        )
        if abs(den) < 1e-12:
            continue

        bb0 = np.floor((pts_xy.min(axis=0) - xy_min) / pitch).astype(int)
        bb1 = np.ceil((pts_xy.max(axis=0) - xy_min) / pitch).astype(int)
        bb0 = np.maximum(bb0, 0)
        bb1 = np.minimum(bb1, [nx - 1, ny - 1])
        if bb1[0] < bb0[0] or bb1[1] < bb0[1]:
            continue

        xs = xy_min[0] + (np.arange(bb0[0], bb1[0] + 1) + 0.5) * pitch
        ys = xy_min[1] + (np.arange(bb0[1], bb1[1] + 1) + 0.5) * pitch
        xx, yy = np.meshgrid(xs, ys)

        w0 = (
            (pts_xy[1, 1] - pts_xy[2, 1]) * (xx - pts_xy[2, 0])
            + (pts_xy[2, 0] - pts_xy[1, 0]) * (yy - pts_xy[2, 1])
        ) / den
        w1 = (
            (pts_xy[2, 1] - pts_xy[0, 1]) * (xx - pts_xy[2, 0])
            + (pts_xy[0, 0] - pts_xy[2, 0]) * (yy - pts_xy[2, 1])
        ) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-6) & (w1 >= -1e-6) & (w2 >= -1e-6)
        if not inside.any():
            continue

        zz = w0 * tri[0, 2] + w1 * tri[1, 2] + w2 * tri[2, 2]
        sl = np.s_[bb0[1] : bb1[1] + 1, bb0[0] : bb1[0] + 1]
        proj_occ[sl] |= inside
        zmin[sl] = np.where(inside, np.minimum(zmin[sl], zz), zmin[sl])
        zmax[sl] = np.where(inside, np.maximum(zmax[sl], zz), zmax[sl])

    proj_occ = ndimage.binary_closing(proj_occ, iterations=1)

    labels, count = ndimage.label(proj_occ)
    if count:
        areas = ndimage.sum(proj_occ, labels, index=range(1, count + 1))
        largest = 1 + int(np.argmax(areas))
        proj_occ = labels == largest

    height_occ = proj_occ & np.isfinite(zmin) & np.isfinite(zmax) & ((zmax - zmin) > 1e-3)
    height_occ = ndimage.binary_closing(height_occ, iterations=2)
    footprint = ndimage.binary_fill_holes(proj_occ)
    return RasterData(
        x0=float(xy_min[0]),
        y0=float(xy_min[1]),
        pitch=pitch,
        zmin=zmin,
        zmax=zmax,
        raw_occ=proj_occ,
        height_occ=height_occ,
        footprint=footprint,
    )


def mask_to_polygon(mask: np.ndarray, x0: float, y0: float, pitch: float) -> Any:
    rows: list[Any] = []
    for row_index, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size == 0:
            continue
        y_min = y0 + row_index * pitch
        y_max = y_min + pitch
        for start, end in zip(starts, ends, strict=False):
            x_min = x0 + start * pitch
            x_max = x0 + end * pitch
            rows.append(box(x_min, y_min, x_max, y_max))

    if not rows:
        return Polygon()

    geom = unary_union(rows).buffer(0)
    if isinstance(geom, MultiPolygon):
        geom = max(geom.geoms, key=lambda g: g.area)
    return geom


def mask_to_geometry(mask: np.ndarray, x0: float, y0: float, pitch: float) -> Any:
    rows: list[Any] = []
    for row_index, row in enumerate(mask):
        padded = np.pad(row.astype(np.int8), (1, 1))
        changes = np.diff(padded)
        starts = np.flatnonzero(changes == 1)
        ends = np.flatnonzero(changes == -1)
        if starts.size == 0:
            continue
        y_min = y0 + row_index * pitch
        y_max = y_min + pitch
        for start, end in zip(starts, ends, strict=False):
            x_min = x0 + start * pitch
            x_max = x0 + end * pitch
            rows.append(box(x_min, y_min, x_max, y_max))

    if not rows:
        return Polygon()
    return unary_union(rows).buffer(0)


def fit_circle_coords(coords: np.ndarray) -> tuple[np.ndarray, float, float]:
    xy = coords[:, :2]
    a = np.column_stack([2.0 * xy[:, 0], 2.0 * xy[:, 1], np.ones(len(xy))])
    b = xy[:, 0] ** 2 + xy[:, 1] ** 2
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    cx, cy, c = solution
    radius = math.sqrt(max(c + cx * cx + cy * cy, 1e-12))
    distances = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    residual = float(np.sqrt(np.mean((distances - radius) ** 2)) / max(radius, 1e-9))
    return np.array([cx, cy]), float(radius), residual


def rounded_rectangle_polygon(bounds: tuple[float, float, float, float], radius: float) -> Any:
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    radius = min(radius, width * 0.5, height * 0.5)
    if radius <= 1e-6:
        return box(minx, miny, maxx, maxy)
    core = box(minx + radius, miny + radius, maxx - radius, maxy - radius)
    return core.buffer(radius, quad_segs=16, join_style=1)


def polygon_relative_points(poly: Any) -> tuple[list[list[float]], list[float]]:
    center = [float(poly.centroid.x), float(poly.centroid.y)]
    coords = np.asarray(poly.exterior.coords[:-1], dtype=float)
    rel = coords - np.asarray(center)
    return rel.tolist(), center


def profile_primitive_from_polygon(poly: Any, simplify_tol: float = 0.0) -> Primitive2D:
    if simplify_tol > 0:
        poly = poly.simplify(simplify_tol, preserve_topology=True)
    center = [float(poly.centroid.x), float(poly.centroid.y)]
    exterior = np.asarray(poly.exterior.coords[:-1], dtype=float) - np.asarray(center)
    holes: list[list[list[float]]] = []
    for interior in poly.interiors:
        hole = np.asarray(interior.coords[:-1], dtype=float) - np.asarray(center)
        holes.append(hole.tolist())
    return Primitive2D(
        "profile",
        {"center": center, "exterior": exterior.tolist(), "holes": holes},
        0.0,
    )


def fit_primitive(poly: Any, simplify_tol: float) -> Primitive2D:
    if poly.is_empty or poly.area <= 0:
        return Primitive2D("polygon", {"points": [], "center": [0.0, 0.0]}, 1.0)

    bounds = poly.bounds
    minx, miny, maxx, maxy = bounds
    width = maxx - minx
    height = maxy - miny
    center = [float((minx + maxx) * 0.5), float((miny + maxy) * 0.5)]

    if len(poly.interiors) == 1:
        outer_coords = np.asarray(poly.exterior.coords[:-1], dtype=float)
        inner_coords = np.asarray(poly.interiors[0].coords[:-1], dtype=float)
        outer_center, outer_radius, outer_error = fit_circle_coords(outer_coords)
        inner_center, inner_radius, inner_error = fit_circle_coords(inner_coords)
        concentric_error = float(np.linalg.norm(outer_center - inner_center))
        if (
            outer_error < 0.06
            and inner_error < 0.06
            and concentric_error < max(width, height) * 0.04
        ):
            return Primitive2D(
                "annulus",
                {
                    "center": outer_center.tolist(),
                    "outer_radius": outer_radius,
                    "inner_radius": inner_radius,
                },
                max(outer_error, inner_error),
            )

    exterior_coords = np.asarray(poly.exterior.coords[:-1], dtype=float)
    circle_center, circle_radius, circle_error = fit_circle_coords(exterior_coords)
    circularity = float(
        4.0 * math.pi * poly.area / max(poly.length * poly.length, 1e-9)
    )
    if circle_error < 0.05 and circularity > 0.92:
        return Primitive2D(
            "circle",
            {"center": circle_center.tolist(), "radius": circle_radius},
            circle_error,
        )

    rect_poly = box(minx, miny, maxx, maxy)
    rect_error = float(rect_poly.symmetric_difference(poly).area / max(poly.area, 1e-9))

    best_rr_error = rect_error
    best_radius = 0.0
    radius_limit = min(width, height) * 0.5
    if radius_limit > simplify_tol:
        for radius in np.linspace(0.0, radius_limit, 28):
            candidate = rounded_rectangle_polygon(bounds, float(radius))
            error = float(
                candidate.symmetric_difference(poly).area / max(poly.area, 1e-9)
            )
            if error < best_rr_error:
                best_rr_error = error
                best_radius = float(radius)

    if best_radius > simplify_tol and best_rr_error < min(rect_error * 0.92, 0.18):
        return Primitive2D(
            "rounded_rectangle",
            {
                "center": center,
                "width": float(width),
                "height": float(height),
                "radius": float(best_radius),
            },
            best_rr_error,
        )

    if rect_error < 0.12:
        return Primitive2D(
            "rectangle",
            {"center": center, "width": float(width), "height": float(height)},
            rect_error,
        )

    tri_poly = poly.simplify(simplify_tol, preserve_topology=True)
    if isinstance(tri_poly, Polygon):
        tri_coords = list(tri_poly.exterior.coords[:-1])
        if len(tri_coords) == 3:
            rel_points, tri_center = polygon_relative_points(tri_poly)
            tri_error = float(
                tri_poly.symmetric_difference(poly).area / max(poly.area, 1e-9)
            )
            return Primitive2D(
                "triangle",
                {"center": tri_center, "points": rel_points},
                tri_error,
            )

    simple_poly = poly.simplify(simplify_tol, preserve_topology=True)
    rel_points, poly_center = polygon_relative_points(simple_poly)
    poly_error = float(
        simple_poly.symmetric_difference(poly).area / max(poly.area, 1e-9)
    )
    return Primitive2D(
        "polygon",
        {"center": poly_center, "points": rel_points},
        poly_error,
    )


def iter_connected_masks(mask: np.ndarray) -> list[np.ndarray]:
    labels, count = ndimage.label(mask)
    parts: list[np.ndarray] = []
    for index in range(1, count + 1):
        component = labels == index
        if component.any():
            parts.append(component)
    return parts


def primitive_group_key(feature: Feature, tol: float = 0.1) -> Any:
    prim = feature.primitive
    base_key = (feature.op, quantize(feature.z_start, tol), quantize(feature.height, tol), prim.kind)
    signature = primitive_signature(prim, tol)
    if signature is None:
        return None
    return base_key + signature


def primitive_signature(primitive: Primitive2D, tol: float = 0.1) -> Any:
    prim = primitive
    params = prim.params
    if prim.kind == "circle":
        return (quantize(params["radius"], tol),)
    if prim.kind == "annulus":
        return (
            quantize(params["outer_radius"], tol),
            quantize(params["inner_radius"], tol),
        )
    if prim.kind == "rectangle":
        return (
            quantize(params["width"], tol),
            quantize(params["height"], tol),
        )
    if prim.kind == "rounded_rectangle":
        return (
            quantize(params["width"], tol),
            quantize(params["height"], tol),
            quantize(params["radius"], tol),
        )
    if prim.kind in {"triangle", "polygon"}:
        return tuple(
            (quantize(point[0], tol), quantize(point[1], tol))
            for point in params.get("points", [])
        )
    if prim.kind == "profile":
        exterior = tuple(
            (quantize(point[0], tol), quantize(point[1], tol))
            for point in params.get("exterior", [])
        )
        holes = tuple(
            tuple((quantize(point[0], tol), quantize(point[1], tol)) for point in hole)
            for hole in params.get("holes", [])
        )
        return (exterior, holes)
    return None


def regroup_features(features: list[Feature]) -> list[Feature]:
    grouped: dict[Any, Feature] = {}
    singles: list[Feature] = []
    for feature in features:
        key = primitive_group_key(feature)
        if key is None:
            singles.append(feature)
            continue
        if key not in grouped:
            grouped[key] = Feature(
                op=feature.op,
                z_start=feature.z_start,
                height=feature.height,
                primitive=feature.primitive,
                centers=list(feature.centers),
                area=feature.area,
                source=feature.source,
            )
        else:
            grouped[key].centers.extend(feature.centers)
            grouped[key].area += feature.area

    merged = list(grouped.values()) + singles
    merged.sort(key=lambda item: (-item.area, item.op, item.source))
    return merged


def merge_vertical_features(features: list[Feature], tol: float = 0.1) -> list[Feature]:
    def centers_key(item: Feature) -> tuple[tuple[float, float], ...]:
        return tuple(
            sorted((quantize(center[0], tol), quantize(center[1], tol)) for center in item.centers)
        )

    keyed = sorted(
        features,
        key=lambda item: (
            (item.op, repr(primitive_signature(item.primitive, tol))),
            centers_key(item),
            quantize(item.z_start, tol),
        ),
    )

    merged: list[Feature] = []
    for feature in keyed:
        if not merged:
            merged.append(feature)
            continue
        prev = merged[-1]
        same_shape = (
            prev.op == feature.op
            and primitive_signature(prev.primitive, tol)
            == primitive_signature(feature.primitive, tol)
        )
        same_centers = centers_key(prev) == centers_key(feature)
        contiguous = abs((prev.z_start + prev.height) - feature.z_start) <= tol
        if same_shape and same_centers and contiguous:
            prev.height += feature.height
            prev.area += feature.area
            prev.source = f"{prev.source}+{feature.source}"
        else:
            merged.append(feature)

    merged.sort(key=lambda item: (item.z_start, -item.area, item.op))
    return merged


def extract_features(
    raster: RasterData,
    level_step: float,
    feature_tol: float,
    simplify_tol: float,
) -> tuple[float, float, Primitive2D, list[Feature]]:
    thickness = np.where(raster.height_occ, raster.zmax - raster.zmin, np.nan)
    base_bottom = quantized_mode(raster.zmin[raster.height_occ], level_step)
    base_top = quantized_mode(raster.zmax[raster.height_occ], level_step)
    if base_top <= base_bottom + 1e-3:
        dominant_thickness = quantized_mode(thickness[raster.height_occ], level_step)
        base_top = base_bottom + dominant_thickness

    base_poly = mask_to_polygon(raster.footprint, raster.x0, raster.y0, raster.pitch)
    base_primitive = fit_primitive(base_poly, simplify_tol)

    features: list[Feature] = []
    hole_mask = raster.footprint & ~raster.raw_occ
    total_height = float(
        np.nanmax(raster.zmax[raster.height_occ]) - np.nanmin(raster.zmin[raster.height_occ])
    )

    for component in iter_connected_masks(hole_mask):
        poly = mask_to_polygon(component, raster.x0, raster.y0, raster.pitch)
        primitive = fit_primitive(poly, simplify_tol)
        center = primitive.params.get("center", [float(poly.centroid.x), float(poly.centroid.y)])
        features.append(
            Feature(
                op="cut",
                z_start=float(base_bottom),
                height=max(total_height, base_top - base_bottom),
                primitive=primitive,
                centers=[center],
                area=float(poly.area),
                source="through_hole",
            )
        )

    top_delta = np.where(raster.height_occ, raster.zmax - base_top, 0.0)
    top_add = raster.height_occ & (top_delta > feature_tol)
    top_cut = raster.height_occ & (top_delta < -feature_tol)

    bottom_delta = np.where(raster.height_occ, base_bottom - raster.zmin, 0.0)
    bottom_add = raster.height_occ & (bottom_delta > feature_tol)
    bottom_cut = raster.height_occ & ((raster.zmin - base_bottom) > feature_tol)

    def add_feature_components(
        mask: np.ndarray,
        op: str,
        source: str,
        level_map: np.ndarray,
        z_start_fn,
        height_fn,
    ) -> None:
        for component in iter_connected_masks(mask):
            poly = mask_to_polygon(component, raster.x0, raster.y0, raster.pitch)
            if poly.area < raster.pitch * raster.pitch * 4.0:
                continue
            primitive = fit_primitive(poly, simplify_tol)
            center = primitive.params.get("center", [float(poly.centroid.x), float(poly.centroid.y)])
            component_values = level_map[component]
            level = quantized_mode(component_values, level_step)
            height = float(max(height_fn(level), level_step))
            z_start = float(z_start_fn(level))
            features.append(
                Feature(
                    op=op,
                    z_start=z_start,
                    height=height,
                    primitive=primitive,
                    centers=[center],
                    area=float(poly.area),
                    source=source,
                )
            )

    add_feature_components(
        top_add,
        "add",
        "top_add",
        top_delta,
        lambda level: base_top,
        lambda level: max(level, level_step),
    )
    add_feature_components(
        top_cut,
        "cut",
        "top_cut",
        -top_delta,
        lambda level: base_top - level,
        lambda level: level,
    )
    add_feature_components(
        bottom_add,
        "add",
        "bottom_add",
        bottom_delta,
        lambda level: base_bottom - level,
        lambda level: level,
    )
    add_feature_components(
        bottom_cut,
        "cut",
        "bottom_cut",
        raster.zmin - base_bottom,
        lambda level: base_bottom,
        lambda level: level,
    )

    return float(base_bottom), float(base_top), base_primitive, regroup_features(features)


def solve_voxelized(input_path: Path, pitch: float, max_steps: int) -> ModelSpec | None:
    if trimesh is None:
        return None

    mesh = trimesh.load(str(input_path), force="mesh")
    if not mesh.is_watertight:
        return None

    voxels = mesh.voxelized(pitch).fill()
    matrix = np.asarray(voxels.matrix, dtype=bool)
    if matrix.ndim != 3 or not matrix.any():
        return None

    x0 = float(voxels.translation[0] - pitch * 0.5)
    y0 = float(voxels.translation[1] - pitch * 0.5)
    z0 = float(voxels.translation[2] - pitch * 0.5)
    simplify_tol = max(0.08, pitch * 0.35)
    bands: list[tuple[int, int, np.ndarray]] = []
    current = matrix[:, :, 0]
    start = 0
    for z_index in range(1, matrix.shape[2] + 1):
        changed = z_index == matrix.shape[2] or not np.array_equal(matrix[:, :, z_index], current)
        if changed:
            bands.append((start, z_index, current.T.copy()))
            if z_index < matrix.shape[2]:
                start = z_index
                current = matrix[:, :, z_index]

    features: list[Feature] = []
    for band_start, band_end, mask in bands:
        if not mask.any():
            continue
        height = float((band_end - band_start) * pitch)
        band_z = float(z0 + band_start * pitch)
        geom = mask_to_geometry(mask, x0, y0, pitch)
        polygons = list(geom.geoms) if isinstance(geom, MultiPolygon) else [geom]
        for poly in polygons:
            if poly.is_empty or poly.area < pitch * pitch * 2.0:
                continue
            primitive = profile_primitive_from_polygon(poly, simplify_tol=simplify_tol)
            features.append(
                Feature(
                    op="add",
                    z_start=band_z,
                    height=height,
                    primitive=primitive,
                    centers=[primitive_center(primitive, poly)],
                    area=float(poly.area * height),
                    source=f"voxel_band_{band_start}_{band_end}",
                )
            )

    features = regroup_features(features)
    features = merge_vertical_features(features)
    features = trim_to_budget(features, max_steps)
    if not features:
        return None

    lowest = min(features, key=lambda item: (item.z_start, -item.area))
    highest_z = max(item.z_start + item.height for item in features)
    spec = ModelSpec(
        source_stl=str(input_path),
        grid_pitch=float(pitch),
        base_bottom=float(z0),
        base_top=float(highest_z),
        base_primitive=lowest.primitive,
        features=features,
        step_budget=max_steps,
        step_count=len(features),
        notes=["Used watertight voxel slicing for the primary fit."],
    )
    return spec


def trim_to_budget(features: list[Feature], max_steps: int) -> list[Feature]:
    if 1 + len(features) <= max_steps:
        return features
    ordered = sorted(features, key=lambda item: (-item.area, item.primitive.fit_error))
    keep = max(max_steps - 1, 0)
    return ordered[:keep]


def solve(
    input_path: Path,
    grid: int,
    pad: float,
    thin_axis: str,
    max_steps: int,
    voxel_pitch: float = 0.5,
) -> ModelSpec:
    voxel_spec = solve_voxelized(input_path, pitch=voxel_pitch, max_steps=max_steps)
    if voxel_spec is not None:
        return voxel_spec

    triangles = load_stl_triangles(input_path)
    triangles, order, notes = ensure_thin_axis_is_z(triangles, thin_axis)
    normals, areas = triangle_normals_and_areas(triangles)
    del normals, areas  # reserved for future orientation refinement

    raster = rasterize_height_fields(triangles, grid=grid, pad=pad)
    simplify_tol = max(raster.pitch * 1.5, 0.12)
    level_step = 0.25
    feature_tol = max(0.3, raster.pitch * 1.8)

    base_bottom, base_top, base_primitive, features = extract_features(
        raster=raster,
        level_step=level_step,
        feature_tol=feature_tol,
        simplify_tol=simplify_tol,
    )
    features = trim_to_budget(features, max_steps)

    spec = ModelSpec(
        source_stl=str(input_path),
        grid_pitch=float(raster.pitch),
        base_bottom=base_bottom,
        base_top=base_top,
        base_primitive=base_primitive,
        features=features,
        step_budget=max_steps,
        step_count=1 + len(features),
        notes=notes,
    )
    if spec.step_count > max_steps:
        spec.notes.append("The feature budget was exceeded and features were trimmed.")
    return spec


def _primitive_to_build123d_code(primitive: Primitive2D, mode_expr: str) -> list[str]:
    params = primitive.params
    kind = primitive.kind
    if kind == "circle":
        return [f"Circle({params['radius']:.6f}, mode={mode_expr})"]
    if kind == "annulus":
        return [
            f"Circle({params['outer_radius']:.6f}, mode={mode_expr})",
            f"Circle({params['inner_radius']:.6f}, mode=Mode.SUBTRACT)",
        ]
    if kind == "rectangle":
        return [
            f"Rectangle({params['width']:.6f}, {params['height']:.6f}, mode={mode_expr})"
        ]
    if kind == "rounded_rectangle":
        return [
            "RectangleRounded("
            f"{params['width']:.6f}, {params['height']:.6f}, {params['radius']:.6f}, mode={mode_expr})"
        ]
    if kind in {"triangle", "polygon"}:
        pts = ", ".join(
            f"({point[0]:.6f}, {point[1]:.6f})" for point in params["points"]
        )
        return [f"Polygon({pts}, mode={mode_expr})"]
    if kind == "profile":
        outer = ", ".join(
            f"({point[0]:.6f}, {point[1]:.6f})" for point in params["exterior"]
        )
        lines = [f"Polygon({outer}, mode=Mode.ADD)"]
        for hole in params["holes"]:
            inner = ", ".join(
                f"({point[0]:.6f}, {point[1]:.6f})" for point in hole
            )
            lines.append(f"Polygon({inner}, mode=Mode.SUBTRACT)")
        return lines
    raise ValueError(f"Unsupported primitive kind: {kind}")


def _feature_plane_z(feature: Feature) -> float:
    return feature.z_start


def _draw_primitive_runtime(primitive: dict[str, Any], mode: Any = None) -> None:
    if BuildPart is None:
        raise RuntimeError("build123d is not installed in the active interpreter.")
    if mode is None:
        mode = Mode.ADD

    kind = primitive["kind"]
    params = primitive["params"]
    if kind == "circle":
        Circle(params["radius"], mode=mode)
    elif kind == "annulus":
        Circle(params["outer_radius"], mode=mode)
        Circle(params["inner_radius"], mode=Mode.SUBTRACT)
    elif kind == "rectangle":
        Rectangle(params["width"], params["height"], mode=mode)
    elif kind == "rounded_rectangle":
        RectangleRounded(params["width"], params["height"], params["radius"], mode=mode)
    elif kind in ("triangle", "polygon"):
        B3DPolygon(*params["points"], mode=mode)
    elif kind == "profile":
        B3DPolygon(*params["exterior"], mode=Mode.ADD)
        for hole in params["holes"]:
            B3DPolygon(*hole, mode=Mode.SUBTRACT)
    else:
        raise ValueError(f"Unsupported primitive: {kind}")


def _make_profile_face_runtime(
    primitive: dict[str, Any], center: list[float], plane_z: float
) -> Any:
    if Face is None or Wire is None:
        raise RuntimeError("build123d is not installed in the active interpreter.")

    params = primitive["params"]
    outer = Wire.make_polygon(
        [
            (center[0] + point[0], center[1] + point[1], plane_z)
            for point in params["exterior"]
        ],
        close=True,
    )
    holes = [
        Wire.make_polygon(
            [(center[0] + point[0], center[1] + point[1], plane_z) for point in hole],
            close=True,
        )
        for hole in params["holes"]
    ]
    return Face.make_surface(exterior=outer, interior_wires=holes)


def build_part_from_spec(spec: ModelSpec | dict[str, Any]) -> Any:
    spec_data = spec_to_dict(spec)
    if BuildPart is None:
        raise RuntimeError("build123d is not installed in the active interpreter.")

    with BuildPart() as part:
        for feature in spec_data["features"]:
            plane_z = feature["z_start"]
            mode = Mode.ADD if feature["op"] == "add" else Mode.SUBTRACT
            if feature["primitive"]["kind"] == "profile":
                for center in feature["centers"]:
                    face = _make_profile_face_runtime(feature["primitive"], center, plane_z)
                    extrude(face, amount=feature["height"], mode=mode)
            else:
                with BuildSketch(Plane(origin=(0, 0, plane_z), z_dir=(0, 0, 1))):
                    for center in feature["centers"]:
                        with Locations(tuple(center)):
                            _draw_primitive_runtime(feature["primitive"], Mode.ADD)
                extrude(amount=feature["height"], mode=mode)

    return part.part


def export_step_from_model_spec(spec: ModelSpec | dict[str, Any], path: Path) -> Path:
    part = build_part_from_spec(spec)
    export_step(part, path)
    return path


def export_stl_from_model_spec(
    spec: ModelSpec | dict[str, Any], path: Path, tolerance: float = 0.05
) -> Path:
    part = build_part_from_spec(spec)
    export_stl(part, path, tolerance=tolerance)
    return path


def write_rebuild_script(spec: ModelSpec, target: Path, spec_path: Path) -> None:
    script = f"""#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fit_stl_to_parametric import export_step_from_model_spec

def main() -> None:
    here = Path(__file__).resolve().parent
    with open(here / "{spec_path.name}", "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    out_path = here / "{target.with_suffix('.step').name}"
    export_step_from_model_spec(spec, out_path)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
"""
    target.write_text(script, encoding="utf-8")


def write_spec(spec: ModelSpec, path: Path) -> None:
    path.write_text(json.dumps(asdict(spec), indent=2), encoding="utf-8")


def export_step_from_spec(spec_path: Path, rebuild_script: Path) -> int:
    try:
        import build123d  # noqa: F401
    except ImportError:
        print(
            "Skipping STEP export because build123d is not installed in the active interpreter.",
            file=sys.stderr,
        )
        return 1

    import subprocess

    result = subprocess.run([sys.executable, str(rebuild_script)], check=False)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit a thin STL using a compact primitive history and emit a STEP rebuild."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input STL file")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("output"), help="Output folder"
    )
    parser.add_argument("--grid", type=int, default=320, help="Raster grid resolution")
    parser.add_argument(
        "--pad", type=float, default=0.6, help="XY padding around the sampled mesh in mm"
    )
    parser.add_argument(
        "--thin-axis",
        choices=["x", "y", "z", "auto"],
        default="z",
        help="Axis treated as the thin extrusion axis",
    )
    parser.add_argument(
        "--max-steps", type=int, default=128, help="Maximum emitted feature steps"
    )
    parser.add_argument(
        "--export-step",
        action="store_true",
        help="Run the generated rebuild script and export STEP",
    )
    parser.add_argument(
        "--voxel-pitch",
        type=float,
        default=0.5,
        help="Voxel pitch in mm for watertight solid fitting",
    )
    args = parser.parse_args()

    _require_runtime()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spec = solve(
        input_path=args.input,
        grid=args.grid,
        pad=args.pad,
        thin_axis=args.thin_axis,
        max_steps=args.max_steps,
        voxel_pitch=args.voxel_pitch,
    )

    stem = args.input.stem
    spec_path = args.output_dir / f"{stem}_fit.json"
    rebuild_path = args.output_dir / f"{stem}_rebuild.py"

    write_spec(spec, spec_path)
    write_rebuild_script(spec, rebuild_path, spec_path)

    print(json.dumps(asdict(spec), indent=2))
    print(f"Spec written to: {spec_path}")
    print(f"Rebuild script written to: {rebuild_path}")

    if args.export_step:
        return export_step_from_spec(spec_path, rebuild_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
