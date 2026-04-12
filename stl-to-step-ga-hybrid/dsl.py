#!/usr/bin/env python3
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


SUPPORTED_KINDS = ("rectangle", "circle", "rounded_rectangle")
SUPPORTED_AXES = ("x", "y", "z")


@dataclass
class SearchBounds:
    min_corner: np.ndarray
    max_corner: np.ndarray
    pitch: float

    def __post_init__(self) -> None:
        self.min_corner = np.asarray(self.min_corner, dtype=np.float32)
        self.max_corner = np.asarray(self.max_corner, dtype=np.float32)
        self.pitch = float(self.pitch)

    @property
    def extents(self) -> np.ndarray:
        return self.max_corner - self.min_corner

    @property
    def min_size(self) -> float:
        return max(self.pitch, 1e-3)

    def clip_xy(self, center_x: float, center_y: float) -> tuple[float, float]:
        cx = float(np.clip(center_x, self.min_corner[0], self.max_corner[0]))
        cy = float(np.clip(center_y, self.min_corner[1], self.max_corner[1]))
        return cx, cy

    def clip_xyz(self, center_x: float, center_y: float, center_z: float) -> tuple[float, float, float]:
        cx = float(np.clip(center_x, self.min_corner[0], self.max_corner[0]))
        cy = float(np.clip(center_y, self.min_corner[1], self.max_corner[1]))
        cz = float(np.clip(center_z, self.min_corner[2], self.max_corner[2]))
        return cx, cy, cz

    def clip_z(self, z_start: float, height: float) -> tuple[float, float]:
        min_height = self.min_size
        max_height = max(float(self.extents[2]), min_height)
        height = float(np.clip(height, min_height, max_height))
        z_min = float(self.min_corner[2])
        z_max = float(self.max_corner[2] - height)
        if z_max < z_min:
            return z_min, min_height
        z_start = float(np.clip(z_start, z_min, z_max))
        return z_start, height

    def max_radius(self) -> float:
        return float(max(min(self.extents[0], self.extents[1]) * 0.5, self.min_size))

    def max_radius_for_axis(self, axis: str) -> float:
        extents = self.extents
        if axis == "x":
            return float(max(min(extents[1], extents[2]) * 0.5, self.min_size))
        if axis == "y":
            return float(max(min(extents[0], extents[2]) * 0.5, self.min_size))
        return self.max_radius()

    def clip_center_length(self, axis: str, center: float, length: float) -> tuple[float, float]:
        axis_index = {"x": 0, "y": 1, "z": 2}[axis]
        min_length = self.min_size
        max_length = max(float(self.extents[axis_index]), min_length)
        length = float(np.clip(length, min_length, max_length))
        half = length * 0.5
        axis_min = float(self.min_corner[axis_index] + half)
        axis_max = float(self.max_corner[axis_index] - half)
        if axis_max < axis_min:
            center = float((self.min_corner[axis_index] + self.max_corner[axis_index]) * 0.5)
            return center, min_length
        center = float(np.clip(center, axis_min, axis_max))
        return center, length

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_corner": self.min_corner.tolist(),
            "max_corner": self.max_corner.tolist(),
            "pitch": self.pitch,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SearchBounds:
        return cls(
            min_corner=np.asarray(payload["min_corner"], dtype=np.float32),
            max_corner=np.asarray(payload["max_corner"], dtype=np.float32),
            pitch=float(payload["pitch"]),
        )


@dataclass
class PrimitiveGene:
    kind: str
    op: str
    center_x: float
    center_y: float
    z_start: float
    height: float
    size_x: float
    size_y: float
    aux: float = 0.0
    center_z: float | None = None
    axis: str = "z"
    angle_deg: float = 0.0

    def clone(self) -> PrimitiveGene:
        return PrimitiveGene(**self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        center_z = self.center_z
        if center_z is None:
            center_z = float(self.z_start + self.height * 0.5)
        return {
            "kind": self.kind,
            "op": self.op,
            "center_x": float(self.center_x),
            "center_y": float(self.center_y),
            "z_start": float(self.z_start),
            "height": float(self.height),
            "size_x": float(self.size_x),
            "size_y": float(self.size_y),
            "aux": float(self.aux),
            "center_z": float(center_z),
            "axis": self.axis,
            "angle_deg": float(self.angle_deg),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PrimitiveGene:
        height = float(payload["height"])
        z_start = float(payload["z_start"])
        return cls(
            kind=str(payload["kind"]),
            op=str(payload["op"]),
            center_x=float(payload["center_x"]),
            center_y=float(payload["center_y"]),
            z_start=z_start,
            height=height,
            size_x=float(payload["size_x"]),
            size_y=float(payload["size_y"]),
            aux=float(payload.get("aux", 0.0)),
            center_z=float(payload.get("center_z", z_start + height * 0.5)),
            axis=str(payload.get("axis", "z")),
            angle_deg=float(payload.get("angle_deg", 0.0)),
        )


@dataclass
class CandidateProgram:
    genes: list[PrimitiveGene] = field(default_factory=list)
    score: float | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    provenance: str = "random"

    def clone(self) -> CandidateProgram:
        return CandidateProgram(
            genes=[gene.clone() for gene in self.genes],
            score=self.score,
            metrics=dict(self.metrics),
            provenance=self.provenance,
        )

    def primitive_count(self) -> int:
        return len(self.genes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "genes": [gene.to_dict() for gene in self.genes],
            "score": None if self.score is None else float(self.score),
            "metrics": {key: float(value) for key, value in self.metrics.items()},
            "provenance": self.provenance,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CandidateProgram:
        return cls(
            genes=[PrimitiveGene.from_dict(item) for item in payload["genes"]],
            score=None if payload.get("score") is None else float(payload["score"]),
            metrics={key: float(value) for key, value in payload.get("metrics", {}).items()},
            provenance=str(payload.get("provenance", "random")),
        )


def primitive_area(gene: PrimitiveGene) -> float:
    if gene.kind == "rectangle":
        return float(gene.size_x * gene.size_y)
    if gene.kind == "circle":
        return float(np.pi * gene.size_x * gene.size_x)
    if gene.kind == "rounded_rectangle":
        radius = min(max(gene.aux, 0.0), gene.size_x * 0.5, gene.size_y * 0.5)
        core = max(gene.size_x - 2.0 * radius, 0.0) * max(gene.size_y - 2.0 * radius, 0.0)
        strips = 2.0 * radius * max(gene.size_x - 2.0 * radius, 0.0)
        strips += 2.0 * radius * max(gene.size_y - 2.0 * radius, 0.0)
        corners = np.pi * radius * radius
        return float(core + strips + corners)
    raise ValueError(f"Unsupported primitive kind: {gene.kind}")


def primitive_volume(gene: PrimitiveGene) -> float:
    return primitive_area(gene) * float(gene.height)


def gene_center_z(gene: PrimitiveGene) -> float:
    if gene.center_z is not None:
        return float(gene.center_z)
    return float(gene.z_start + gene.height * 0.5)


def gene_xy_rotation_rad(gene: PrimitiveGene) -> float:
    return math.radians(float(gene.angle_deg))


def gene_bounds(gene: PrimitiveGene) -> tuple[np.ndarray, np.ndarray]:
    center_z = gene_center_z(gene)
    if gene.kind == "circle":
        radius = float(gene.size_x)
        half_length = float(gene.height) * 0.5
        if gene.axis == "x":
            mins = np.array([gene.center_x - half_length, gene.center_y - radius, center_z - radius], dtype=np.float32)
            maxs = np.array([gene.center_x + half_length, gene.center_y + radius, center_z + radius], dtype=np.float32)
            return mins, maxs
        if gene.axis == "y":
            mins = np.array([gene.center_x - radius, gene.center_y - half_length, center_z - radius], dtype=np.float32)
            maxs = np.array([gene.center_x + radius, gene.center_y + half_length, center_z + radius], dtype=np.float32)
            return mins, maxs
        mins = np.array([gene.center_x - radius, gene.center_y - radius, center_z - half_length], dtype=np.float32)
        maxs = np.array([gene.center_x + radius, gene.center_y + radius, center_z + half_length], dtype=np.float32)
        return mins, maxs

    half_x = float(gene.size_x) * 0.5
    half_y = float(gene.size_y) * 0.5
    cos_a = abs(math.cos(gene_xy_rotation_rad(gene)))
    sin_a = abs(math.sin(gene_xy_rotation_rad(gene)))
    extent_x = cos_a * half_x + sin_a * half_y
    extent_y = sin_a * half_x + cos_a * half_y
    half_z = float(gene.height) * 0.5
    mins = np.array([gene.center_x - extent_x, gene.center_y - extent_y, center_z - half_z], dtype=np.float32)
    maxs = np.array([gene.center_x + extent_x, gene.center_y + extent_y, center_z + half_z], dtype=np.float32)
    return mins, maxs


def primitive_occupancy_mask(
    gene: PrimitiveGene,
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
) -> np.ndarray:
    x_grid = xs[:, None, None]
    y_grid = ys[None, :, None]
    z_grid = zs[None, None, :]
    center_z = gene_center_z(gene)

    if gene.kind == "circle":
        radius_sq = float(gene.size_x * gene.size_x)
        half_length = float(gene.height) * 0.5
        if gene.axis == "x":
            radial = (y_grid - gene.center_y) ** 2 + (z_grid - center_z) ** 2 <= radius_sq
            axial = np.abs(x_grid - gene.center_x) <= half_length
            return axial & radial
        if gene.axis == "y":
            radial = (x_grid - gene.center_x) ** 2 + (z_grid - center_z) ** 2 <= radius_sq
            axial = np.abs(y_grid - gene.center_y) <= half_length
            return radial & axial
        radial = (x_grid - gene.center_x) ** 2 + (y_grid - gene.center_y) ** 2 <= radius_sq
        axial = np.abs(z_grid - center_z) <= half_length
        return radial & axial

    theta = gene_xy_rotation_rad(gene)
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    x_rel = x_grid - gene.center_x
    y_rel = y_grid - gene.center_y
    local_x = cos_t * x_rel + sin_t * y_rel
    local_y = -sin_t * x_rel + cos_t * y_rel
    if gene.kind == "rectangle":
        mask_xy = (np.abs(local_x) <= gene.size_x * 0.5) & (np.abs(local_y) <= gene.size_y * 0.5)
    else:
        radius = min(max(gene.aux, 0.0), gene.size_x * 0.5, gene.size_y * 0.5)
        core_x = max(gene.size_x * 0.5 - radius, 0.0)
        core_y = max(gene.size_y * 0.5 - radius, 0.0)
        qx = np.maximum(np.abs(local_x) - core_x, 0.0)
        qy = np.maximum(np.abs(local_y) - core_y, 0.0)
        mask_xy = (qx * qx + qy * qy) <= (radius * radius)
    mask_z = np.abs(z_grid - center_z) <= (gene.height * 0.5)
    return mask_xy & mask_z


def clamp_gene(gene: PrimitiveGene, bounds: SearchBounds) -> PrimitiveGene:
    kind = gene.kind if gene.kind in SUPPORTED_KINDS else "rectangle"
    op = "cut" if gene.op == "cut" else "add"
    axis = gene.axis if gene.axis in SUPPORTED_AXES else "z"
    center_z = gene_center_z(gene)
    angle_deg = float(((float(gene.angle_deg) + 180.0) % 360.0) - 180.0)
    min_size = bounds.min_size
    if kind != "circle":
        axis = "z"
    if kind == "circle":
        cx, cy, center_z = bounds.clip_xyz(gene.center_x, gene.center_y, center_z)
        if axis == "x":
            cx, length = bounds.clip_center_length("x", cx, gene.height)
            radius = float(np.clip(gene.size_x, min_size, bounds.max_radius_for_axis("x")))
            cy = float(np.clip(cy, bounds.min_corner[1] + radius, bounds.max_corner[1] - radius))
            center_z = float(np.clip(center_z, bounds.min_corner[2] + radius, bounds.max_corner[2] - radius))
        elif axis == "y":
            cy, length = bounds.clip_center_length("y", cy, gene.height)
            radius = float(np.clip(gene.size_x, min_size, bounds.max_radius_for_axis("y")))
            cx = float(np.clip(cx, bounds.min_corner[0] + radius, bounds.max_corner[0] - radius))
            center_z = float(np.clip(center_z, bounds.min_corner[2] + radius, bounds.max_corner[2] - radius))
        else:
            center_z, length = bounds.clip_center_length("z", center_z, gene.height)
            radius = float(np.clip(gene.size_x, min_size, bounds.max_radius_for_axis("z")))
            cx = float(np.clip(cx, bounds.min_corner[0] + radius, bounds.max_corner[0] - radius))
            cy = float(np.clip(cy, bounds.min_corner[1] + radius, bounds.max_corner[1] - radius))
        return PrimitiveGene(
            kind=kind,
            op=op,
            center_x=cx,
            center_y=cy,
            z_start=float(center_z - length * 0.5),
            height=length,
            size_x=radius,
            size_y=radius,
            aux=0.0,
            center_z=center_z,
            axis=axis,
            angle_deg=0.0,
        )

    cx, cy, center_z = bounds.clip_xyz(gene.center_x, gene.center_y, center_z)
    center_z, height = bounds.clip_center_length("z", center_z, gene.height)
    max_width = max(float(bounds.extents[0]), min_size)
    max_height = max(float(bounds.extents[1]), min_size)
    width = float(np.clip(gene.size_x, min_size, max_width))
    depth = float(np.clip(gene.size_y, min_size, max_height))
    if kind == "rounded_rectangle":
        radius = float(np.clip(gene.aux, 0.0, min(width, depth) * 0.5))
        return PrimitiveGene(
            kind=kind,
            op=op,
            center_x=cx,
            center_y=cy,
            z_start=float(center_z - height * 0.5),
            height=height,
            size_x=width,
            size_y=depth,
            aux=radius,
            center_z=center_z,
            axis="z",
            angle_deg=angle_deg,
        )

    return PrimitiveGene(
        kind="rectangle",
        op=op,
        center_x=cx,
        center_y=cy,
        z_start=float(center_z - height * 0.5),
        height=height,
        size_x=width,
        size_y=depth,
        aux=0.0,
        center_z=center_z,
        axis="z",
        angle_deg=angle_deg,
    )


def normalize_program(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    max_primitives: int,
    rng: np.random.Generator | None = None,
) -> CandidateProgram:
    genes = [clamp_gene(gene, bounds) for gene in candidate.genes]
    genes = [gene for gene in genes if primitive_volume(gene) >= bounds.min_size**3]
    adds = sorted((gene for gene in genes if gene.op == "add"), key=primitive_volume, reverse=True)
    cuts = sorted((gene for gene in genes if gene.op == "cut"), key=primitive_volume, reverse=True)

    if not adds:
        if rng is None:
            rng = np.random.default_rng()
        adds = [random_gene(bounds, rng, allow_cut=False)]

    normalized = adds + cuts
    normalized = normalized[: max(1, max_primitives)]
    return CandidateProgram(
        genes=normalized,
        score=None,
        metrics={},
        provenance=candidate.provenance,
    )


def random_gene(
    bounds: SearchBounds,
    rng: np.random.Generator,
    allow_cut: bool = True,
) -> PrimitiveGene:
    kind = str(rng.choice(SUPPORTED_KINDS))
    op = "cut" if allow_cut and rng.random() < 0.45 else "add"
    center_x = float(rng.uniform(bounds.min_corner[0], bounds.max_corner[0]))
    center_y = float(rng.uniform(bounds.min_corner[1], bounds.max_corner[1]))
    center_z = float(rng.uniform(bounds.min_corner[2], bounds.max_corner[2]))
    z_span = max(float(bounds.extents[2]), bounds.min_size)
    height = float(rng.uniform(bounds.min_size, z_span))
    span_x = max(float(bounds.extents[0]), bounds.min_size)
    span_y = max(float(bounds.extents[1]), bounds.min_size)

    if kind == "circle":
        axis = str(rng.choice(np.asarray(("z", "z", "z", "x", "y"), dtype=object)))
        radius = float(rng.uniform(bounds.min_size, bounds.max_radius_for_axis(axis)))
        return clamp_gene(
            PrimitiveGene(
                kind=kind,
                op=op,
                center_x=center_x,
                center_y=center_y,
                z_start=float(center_z - height * 0.5),
                height=height,
                size_x=radius,
                size_y=radius,
                aux=0.0,
                center_z=center_z,
                axis=axis,
                angle_deg=0.0,
            ),
            bounds,
        )

    width = float(rng.uniform(bounds.min_size, span_x))
    depth = float(rng.uniform(bounds.min_size, span_y))
    corner = float(rng.uniform(0.0, min(width, depth) * 0.5))
    angle_deg = float(rng.uniform(-90.0, 90.0)) if rng.random() < 0.35 else 0.0
    return clamp_gene(
        PrimitiveGene(
            kind=kind,
            op=op,
            center_x=center_x,
            center_y=center_y,
            z_start=float(center_z - height * 0.5),
            height=height,
            size_x=width,
            size_y=depth,
            aux=corner,
            center_z=center_z,
            axis="z",
            angle_deg=angle_deg,
        ),
        bounds,
    )


def candidate_from_supported_features(
    spec: dict[str, Any],
    bounds: SearchBounds,
    max_primitives: int,
) -> CandidateProgram | None:
    genes: list[PrimitiveGene] = []
    for feature in spec.get("features", []):
        primitive = feature.get("primitive", {})
        kind = primitive.get("kind")
        if kind not in SUPPORTED_KINDS:
            continue
        params = primitive.get("params", {})
        centers = feature.get("centers", [])
        if not centers and "center_3d" in feature:
            center3d = feature["center_3d"]
            centers = [[float(center3d[0]), float(center3d[1])]]
        center_z = float(feature.get("center_3d", [0.0, 0.0, float(feature["z_start"]) + float(feature["height"]) * 0.5])[2])
        axis = str(feature.get("axis", primitive.get("axis", "z")))
        angle_deg = float(feature.get("rotation_deg", primitive.get("rotation_deg", 0.0)))
        for center in centers:
            if kind == "circle":
                gene = PrimitiveGene(
                    kind="circle",
                    op="cut" if feature.get("op") == "cut" else "add",
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    z_start=float(feature["z_start"]),
                    height=float(feature["height"]),
                    size_x=float(params["radius"]),
                    size_y=float(params["radius"]),
                    aux=0.0,
                    center_z=center_z,
                    axis=axis,
                    angle_deg=0.0,
                )
            elif kind == "rounded_rectangle":
                gene = PrimitiveGene(
                    kind="rounded_rectangle",
                    op="cut" if feature.get("op") == "cut" else "add",
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    z_start=float(feature["z_start"]),
                    height=float(feature["height"]),
                    size_x=float(params["width"]),
                    size_y=float(params["height"]),
                    aux=float(params["radius"]),
                    center_z=center_z,
                    axis="z",
                    angle_deg=angle_deg,
                )
            else:
                gene = PrimitiveGene(
                    kind="rectangle",
                    op="cut" if feature.get("op") == "cut" else "add",
                    center_x=float(center[0]),
                    center_y=float(center[1]),
                    z_start=float(feature["z_start"]),
                    height=float(feature["height"]),
                    size_x=float(params["width"]),
                    size_y=float(params["height"]),
                    aux=0.0,
                    center_z=center_z,
                    axis="z",
                    angle_deg=angle_deg,
                )
            genes.append(clamp_gene(gene, bounds))

    if not genes:
        return None

    return normalize_program(
        CandidateProgram(genes=genes, provenance="voxel-seed"),
        bounds=bounds,
        max_primitives=max_primitives,
    )


def seed_program_from_bounds(bounds: SearchBounds) -> CandidateProgram:
    center_z = float((bounds.min_corner[2] + bounds.max_corner[2]) * 0.5)
    base = PrimitiveGene(
        kind="rectangle",
        op="add",
        center_x=float((bounds.min_corner[0] + bounds.max_corner[0]) * 0.5),
        center_y=float((bounds.min_corner[1] + bounds.max_corner[1]) * 0.5),
        z_start=float(center_z - max(bounds.extents[2], bounds.min_size) * 0.5),
        height=float(max(bounds.extents[2], bounds.min_size)),
        size_x=float(max(bounds.extents[0], bounds.min_size)),
        size_y=float(max(bounds.extents[1], bounds.min_size)),
        aux=0.0,
        center_z=center_z,
        axis="z",
        angle_deg=0.0,
    )
    return CandidateProgram(genes=[base], provenance="bounds-seed")


def gene_to_feature(gene: PrimitiveGene) -> dict[str, Any]:
    if gene.kind == "circle":
        primitive = {
            "kind": "circle",
            "params": {"radius": float(gene.size_x)},
            "fit_error": 0.0,
        }
        primitive["axis"] = gene.axis
    elif gene.kind == "rounded_rectangle":
        width = float(gene.size_x)
        height = float(gene.size_y)
        safe_radius = min(
            float(gene.aux),
            max(min(width, height) * 0.5 - 1e-5, 0.0),
        )
        if safe_radius <= 1e-6:
            primitive = {
                "kind": "rectangle",
                "params": {
                    "width": width,
                    "height": height,
                },
                "fit_error": 0.0,
            }
        else:
            primitive = {
                "kind": "rounded_rectangle",
                "params": {
                    "width": width,
                    "height": height,
                    "radius": safe_radius,
                },
                "fit_error": 0.0,
            }
            primitive["rotation_deg"] = float(gene.angle_deg)
    else:
        primitive = {
            "kind": "rectangle",
            "params": {
                "width": float(gene.size_x),
                "height": float(gene.size_y),
            },
            "fit_error": 0.0,
        }
        primitive["rotation_deg"] = float(gene.angle_deg)

    mins, maxs = gene_bounds(gene)
    center_z = gene_center_z(gene)
    axis = gene.axis if gene.kind == "circle" else "z"

    return {
        "op": gene.op,
        "z_start": float(mins[2]),
        "height": float(gene.height),
        "primitive": primitive,
        "centers": [[float(gene.center_x), float(gene.center_y)]],
        "center_3d": [float(gene.center_x), float(gene.center_y), float(center_z)],
        "axis": axis,
        "rotation_deg": float(gene.angle_deg if gene.kind != "circle" else 0.0),
        "area": float(primitive_area(gene) * gene.height),
        "source": "ga_hybrid",
    }


def candidate_to_model_spec(
    candidate: CandidateProgram,
    input_path: str,
    grid_pitch: float,
    step_budget: int,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    features = [gene_to_feature(gene) for gene in candidate.genes]
    mins = [gene_bounds(gene)[0][2] for gene in candidate.genes]
    maxs = [gene_bounds(gene)[1][2] for gene in candidate.genes]
    base_bottom = min(mins)
    base_top = max(maxs)
    base_feature = next(feature for feature in features if feature["op"] == "add")
    spec_notes = list(notes or [])
    if candidate.metrics:
        spec_notes.append(
            "Best GA-hybrid metrics: "
            f"score={candidate.metrics.get('score', 0.0):.6f}, "
            f"iou={candidate.metrics.get('iou', 0.0):.6f}, "
            f"missing={candidate.metrics.get('missing_ratio', 0.0):.6f}, "
            f"extra={candidate.metrics.get('extra_ratio', 0.0):.6f}."
        )
    return {
        "source_stl": input_path,
        "grid_pitch": float(grid_pitch),
        "base_bottom": float(base_bottom),
        "base_top": float(base_top),
        "base_primitive": dict(base_feature["primitive"]),
        "features": features,
        "step_budget": int(step_budget),
        "step_count": len(features),
        "notes": spec_notes,
    }
