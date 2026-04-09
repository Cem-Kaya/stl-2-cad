#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


SUPPORTED_KINDS = ("rectangle", "circle", "rounded_rectangle")


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

    def clone(self) -> PrimitiveGene:
        return PrimitiveGene(**self.to_dict())

    def to_dict(self) -> dict[str, Any]:
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
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PrimitiveGene:
        return cls(
            kind=str(payload["kind"]),
            op=str(payload["op"]),
            center_x=float(payload["center_x"]),
            center_y=float(payload["center_y"]),
            z_start=float(payload["z_start"]),
            height=float(payload["height"]),
            size_x=float(payload["size_x"]),
            size_y=float(payload["size_y"]),
            aux=float(payload.get("aux", 0.0)),
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


def clamp_gene(gene: PrimitiveGene, bounds: SearchBounds) -> PrimitiveGene:
    kind = gene.kind if gene.kind in SUPPORTED_KINDS else "rectangle"
    op = "cut" if gene.op == "cut" else "add"
    cx, cy = bounds.clip_xy(gene.center_x, gene.center_y)
    z_start, height = bounds.clip_z(gene.z_start, gene.height)
    min_size = bounds.min_size
    max_width = max(float(bounds.extents[0]), min_size)
    max_height = max(float(bounds.extents[1]), min_size)
    if kind == "circle":
        radius = float(np.clip(gene.size_x, min_size, bounds.max_radius()))
        return PrimitiveGene(
            kind=kind,
            op=op,
            center_x=cx,
            center_y=cy,
            z_start=z_start,
            height=height,
            size_x=radius,
            size_y=radius,
            aux=0.0,
        )

    width = float(np.clip(gene.size_x, min_size, max_width))
    depth = float(np.clip(gene.size_y, min_size, max_height))
    if kind == "rounded_rectangle":
        radius = float(np.clip(gene.aux, 0.0, min(width, depth) * 0.5))
        return PrimitiveGene(
            kind=kind,
            op=op,
            center_x=cx,
            center_y=cy,
            z_start=z_start,
            height=height,
            size_x=width,
            size_y=depth,
            aux=radius,
        )

    return PrimitiveGene(
        kind="rectangle",
        op=op,
        center_x=cx,
        center_y=cy,
        z_start=z_start,
        height=height,
        size_x=width,
        size_y=depth,
        aux=0.0,
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
    z_span = max(float(bounds.extents[2]), bounds.min_size)
    z_start = float(rng.uniform(bounds.min_corner[2], bounds.max_corner[2]))
    height = float(rng.uniform(bounds.min_size, z_span))
    span_x = max(float(bounds.extents[0]), bounds.min_size)
    span_y = max(float(bounds.extents[1]), bounds.min_size)

    if kind == "circle":
        radius = float(rng.uniform(bounds.min_size, bounds.max_radius()))
        return clamp_gene(
            PrimitiveGene(
                kind=kind,
                op=op,
                center_x=center_x,
                center_y=center_y,
                z_start=z_start,
                height=height,
                size_x=radius,
                size_y=radius,
                aux=0.0,
            ),
            bounds,
        )

    width = float(rng.uniform(bounds.min_size, span_x))
    depth = float(rng.uniform(bounds.min_size, span_y))
    corner = float(rng.uniform(0.0, min(width, depth) * 0.5))
    return clamp_gene(
        PrimitiveGene(
            kind=kind,
            op=op,
            center_x=center_x,
            center_y=center_y,
            z_start=z_start,
            height=height,
            size_x=width,
            size_y=depth,
            aux=corner,
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
        for center in feature.get("centers", []):
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
    base = PrimitiveGene(
        kind="rectangle",
        op="add",
        center_x=float((bounds.min_corner[0] + bounds.max_corner[0]) * 0.5),
        center_y=float((bounds.min_corner[1] + bounds.max_corner[1]) * 0.5),
        z_start=float(bounds.min_corner[2]),
        height=float(max(bounds.extents[2], bounds.min_size)),
        size_x=float(max(bounds.extents[0], bounds.min_size)),
        size_y=float(max(bounds.extents[1], bounds.min_size)),
        aux=0.0,
    )
    return CandidateProgram(genes=[base], provenance="bounds-seed")


def gene_to_feature(gene: PrimitiveGene) -> dict[str, Any]:
    if gene.kind == "circle":
        primitive = {
            "kind": "circle",
            "params": {"radius": float(gene.size_x)},
            "fit_error": 0.0,
        }
    elif gene.kind == "rounded_rectangle":
        safe_radius = min(
            float(gene.aux),
            max(min(float(gene.size_x), float(gene.size_y)) * 0.5 - 1e-5, 0.0),
        )
        primitive = {
            "kind": "rounded_rectangle",
            "params": {
                "width": float(gene.size_x),
                "height": float(gene.size_y),
                "radius": safe_radius,
            },
            "fit_error": 0.0,
        }
    else:
        primitive = {
            "kind": "rectangle",
            "params": {
                "width": float(gene.size_x),
                "height": float(gene.size_y),
            },
            "fit_error": 0.0,
        }

    return {
        "op": gene.op,
        "z_start": float(gene.z_start),
        "height": float(gene.height),
        "primitive": primitive,
        "centers": [[float(gene.center_x), float(gene.center_y)]],
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
    base_bottom = min(feature["z_start"] for feature in features)
    base_top = max(feature["z_start"] + feature["height"] for feature in features)
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
