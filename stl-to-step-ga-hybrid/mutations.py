#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

from dsl import (
    CandidateProgram,
    PrimitiveGene,
    SearchBounds,
    SUPPORTED_KINDS,
    clamp_gene,
    gene_center_z,
    normalize_program,
    primitive_occupancy_mask,
    random_gene,
)


def _candidate_occupancy(candidate: CandidateProgram, target: object) -> np.ndarray:
    occupancy = np.zeros_like(target.occupancy, dtype=bool)

    for gene in candidate.genes:
        primitive = primitive_occupancy_mask(gene, target.xs, target.ys, target.zs)
        if gene.op == "add":
            occupancy |= primitive
        else:
            occupancy &= ~primitive

    return occupancy


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray | None, int]:
    if not mask.any():
        return None, 0

    structure = ndimage.generate_binary_structure(3, 1)
    labels, count = ndimage.label(mask, structure=structure)
    if count <= 1:
        return mask, int(mask.sum())

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    label_id = int(np.argmax(sizes))
    component = labels == label_id
    return component, int(sizes[label_id])


def _build_guided_gene(
    component_mask: np.ndarray,
    op: str,
    bounds: SearchBounds,
    target: object,
    rng: np.random.Generator,
) -> PrimitiveGene:
    coords = np.argwhere(component_mask)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)

    pitch = float(target.pitch)
    min_corner = np.array(
        [
            float(target.xs[mins[0]] - pitch * 0.5),
            float(target.ys[mins[1]] - pitch * 0.5),
            float(target.zs[mins[2]] - pitch * 0.5),
        ],
        dtype=float,
    )
    max_corner = np.array(
        [
            float(target.xs[maxs[0]] + pitch * 0.5),
            float(target.ys[maxs[1]] + pitch * 0.5),
            float(target.zs[maxs[2]] + pitch * 0.5),
        ],
        dtype=float,
    )
    span = np.maximum(max_corner - min_corner, bounds.min_size)
    center = 0.5 * (min_corner + max_corner)

    size_x = max(bounds.min_size, float(span[0] * rng.uniform(0.96, 1.18)))
    size_y = max(bounds.min_size, float(span[1] * rng.uniform(0.96, 1.18)))
    height = max(bounds.min_size, float(span[2] * rng.uniform(0.96, 1.18)))
    center_x = float(center[0] + rng.normal(0.0, max(span[0] * 0.04, pitch * 0.25)))
    center_y = float(center[1] + rng.normal(0.0, max(span[1] * 0.04, pitch * 0.25)))
    center_z = float(center[2] + rng.normal(0.0, max(span[2] * 0.04, pitch * 0.25)))

    aspect_ratio = max(size_x, size_y) / max(min(size_x, size_y), bounds.min_size)
    if aspect_ratio < 1.18 and rng.random() < 0.22:
        kind = "circle"
        axis = "z"
        longest_axis = int(np.argmax(span))
        if longest_axis == 0 and span[0] > max(span[1], span[2]) * 1.25:
            axis = "x"
            height = max(bounds.min_size, float(span[0] * rng.uniform(0.94, 1.12)))
        elif longest_axis == 1 and span[1] > max(span[0], span[2]) * 1.25:
            axis = "y"
            height = max(bounds.min_size, float(span[1] * rng.uniform(0.94, 1.12)))
        else:
            height = max(bounds.min_size, float(span[2] * rng.uniform(0.94, 1.12)))
        radius = max(bounds.min_size, 0.5 * max(size_x, size_y))
        gene = PrimitiveGene(
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
        )
    else:
        kind = "rounded_rectangle" if rng.random() < 0.35 else "rectangle"
        aux = 0.0
        if kind == "rounded_rectangle":
            aux = float(rng.uniform(0.0, min(size_x, size_y) * 0.28))
        xy_coords = coords[:, :2].astype(np.float64)
        angle_deg = 0.0
        if len(xy_coords) >= 3:
            centered = xy_coords - xy_coords.mean(axis=0, keepdims=True)
            cov = centered.T @ centered
            eigvals, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, int(np.argmax(eigvals))]
            angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))
        gene = PrimitiveGene(
            kind=kind,
            op=op,
            center_x=center_x,
            center_y=center_y,
            z_start=float(center_z - height * 0.5),
            height=height,
            size_x=size_x,
            size_y=size_y,
            aux=aux,
            center_z=center_z,
            axis="z",
            angle_deg=angle_deg,
        )
    return clamp_gene(gene, bounds)


def _select_replace_index(
    candidate: CandidateProgram,
    op: str,
    rng: np.random.Generator,
) -> int | None:
    matching = [index for index, gene in enumerate(candidate.genes) if gene.op == op]
    if matching:
        return int(rng.choice(matching))

    if op == "add":
        return int(rng.integers(0, len(candidate.genes)))

    add_count = sum(1 for gene in candidate.genes if gene.op == "add")
    convertible = [index for index, gene in enumerate(candidate.genes) if gene.op == "cut"]
    if add_count > 1:
        convertible.extend(index for index, gene in enumerate(candidate.genes) if gene.op == "add")
    if not convertible:
        return None
    return int(rng.choice(convertible))


def _apply_guided_mutation(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    rng: np.random.Generator,
    max_primitives: int,
    target: object,
) -> bool:
    occupancy = _candidate_occupancy(candidate, target)
    missing = target.occupancy & ~occupancy
    extra = occupancy & ~target.occupancy
    if not missing.any() and not extra.any():
        return False

    missing_component, missing_size = _largest_component(missing)
    extra_component, extra_size = _largest_component(extra)

    if missing_component is not None and extra_component is not None:
        missing_weight = max(0.72 * missing_size, 1e-9)
        extra_weight = max(0.58 * extra_size, 1e-9)
        if rng.random() < missing_weight / (missing_weight + extra_weight):
            component_mask = missing_component
            op = "add"
        else:
            component_mask = extra_component
            op = "cut"
    elif missing_component is not None:
        component_mask = missing_component
        op = "add"
    elif extra_component is not None:
        component_mask = extra_component
        op = "cut"
    else:
        return False

    guided_gene = _build_guided_gene(component_mask, op, bounds, target, rng)

    if len(candidate.genes) < max_primitives and (op == "add" or rng.random() < 0.7):
        candidate.genes.insert(int(rng.integers(0, len(candidate.genes) + 1)), guided_gene)
        return True

    replace_index = _select_replace_index(candidate, op, rng)
    if replace_index is None:
        return False
    candidate.genes[replace_index] = guided_gene
    return True


def crossover_programs(
    parent_a: CandidateProgram,
    parent_b: CandidateProgram,
    rng: np.random.Generator,
    bounds: SearchBounds,
    max_primitives: int,
) -> CandidateProgram:
    if not parent_a.genes:
        return normalize_program(parent_b.clone(), bounds, max_primitives, rng)
    if not parent_b.genes:
        return normalize_program(parent_a.clone(), bounds, max_primitives, rng)

    cut_a = int(rng.integers(0, len(parent_a.genes) + 1))
    cut_b = int(rng.integers(0, len(parent_b.genes) + 1))
    genes = [gene.clone() for gene in parent_a.genes[:cut_a]]
    genes.extend(gene.clone() for gene in parent_b.genes[cut_b:])
    child = CandidateProgram(genes=genes, provenance="crossover")
    return normalize_program(child, bounds, max_primitives, rng)


def mutate_program(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    rng: np.random.Generator,
    max_primitives: int,
    mutation_strength: float,
    structural_rate: float = 0.35,
    target: object | None = None,
    bias_rate: float = 0.15,
) -> CandidateProgram:
    child = candidate.clone()
    child.provenance = "mutated"
    if not child.genes:
        child.genes.append(random_gene(bounds, rng, allow_cut=False))

    guided = False
    if target is not None and bias_rate > 0.0 and rng.random() < bias_rate:
        guided = _apply_guided_mutation(child, bounds, rng, max_primitives, target)
        if guided:
            child.provenance = "guided_mutated"

    if not guided:
        action_roll = float(rng.random())
        if action_roll < structural_rate and len(child.genes) < max_primitives:
            insert_at = int(rng.integers(0, len(child.genes) + 1))
            child.genes.insert(insert_at, random_gene(bounds, rng, allow_cut=True))
        elif action_roll < structural_rate * 1.8 and len(child.genes) > 1:
            remove_at = int(rng.integers(0, len(child.genes)))
            del child.genes[remove_at]
        elif action_roll < structural_rate * 2.2 and len(child.genes) > 1:
            index_a = int(rng.integers(0, len(child.genes)))
            index_b = int(rng.integers(0, len(child.genes)))
            child.genes[index_a], child.genes[index_b] = child.genes[index_b], child.genes[index_a]
        else:
            mutate_gene(child.genes[int(rng.integers(0, len(child.genes)))], bounds, rng, mutation_strength)

    if not guided and rng.random() < 0.22:
        mutate_gene(child.genes[int(rng.integers(0, len(child.genes)))], bounds, rng, mutation_strength)

    return normalize_program(child, bounds, max_primitives, rng)


def mutate_gene(
    gene: PrimitiveGene,
    bounds: SearchBounds,
    rng: np.random.Generator,
    mutation_strength: float,
) -> PrimitiveGene:
    scale_xy = max(float(bounds.extents[:2].max()), bounds.min_size)
    scale_z = max(float(bounds.extents[2]), bounds.min_size)
    sigma_xy = max(scale_xy * mutation_strength * 0.22, bounds.min_size)
    sigma_z = max(scale_z * mutation_strength * 0.18, bounds.min_size)
    sigma_size = max(scale_xy * mutation_strength * 0.25, bounds.min_size)
    sigma_angle = max(6.0, 55.0 * mutation_strength)
    center_z = gene_center_z(gene)

    mode = int(rng.integers(0, 8))
    if mode == 0:
        gene.center_x += float(rng.normal(0.0, sigma_xy))
        gene.center_y += float(rng.normal(0.0, sigma_xy))
        if gene.kind == "circle" and gene.axis in {"x", "y"}:
            center_z += float(rng.normal(0.0, sigma_xy))
    elif mode == 1:
        center_z += float(rng.normal(0.0, sigma_z))
        gene.height *= float(math.exp(rng.normal(0.0, mutation_strength * 0.33)))
    elif mode == 2:
        if gene.kind == "circle":
            radius_scale = float(math.exp(rng.normal(0.0, mutation_strength * 0.38)))
            gene.size_x *= radius_scale
            gene.size_y = gene.size_x
        else:
            gene.size_x *= float(math.exp(rng.normal(0.0, mutation_strength * 0.38)))
            gene.size_y *= float(math.exp(rng.normal(0.0, mutation_strength * 0.38)))
    elif mode == 3:
        if gene.kind == "rounded_rectangle":
            gene.aux += float(rng.normal(0.0, sigma_size * 0.35))
        elif gene.kind != "circle":
            gene.angle_deg += float(rng.normal(0.0, sigma_angle))
    elif mode == 4:
        gene.op = "add" if gene.op == "cut" else "cut"
    elif mode == 5:
        if gene.kind == "circle":
            gene.axis = str(rng.choice(np.asarray(("x", "y", "z"), dtype=object)))
            if rng.random() < 0.45:
                gene.height *= float(math.exp(rng.normal(0.0, mutation_strength * 0.28)))
        else:
            gene.angle_deg += float(rng.normal(0.0, sigma_angle * 0.65))
            if rng.random() < 0.35:
                center_z += float(rng.normal(0.0, sigma_z * 0.8))
    elif mode == 6:
        center_z += float(rng.normal(0.0, sigma_z))
        if gene.kind != "circle":
            gene.angle_deg += float(rng.normal(0.0, sigma_angle * 0.45))
        if rng.random() < 0.4:
            gene.center_x += float(rng.normal(0.0, sigma_xy * 0.6))
            gene.center_y += float(rng.normal(0.0, sigma_xy * 0.6))
    else:
        gene.kind = str(rng.choice(SUPPORTED_KINDS))
        if gene.kind == "circle":
            gene.size_x = max(gene.size_x, bounds.min_size)
            gene.size_y = gene.size_x
            gene.aux = 0.0
            if gene.axis not in {"x", "y", "z"}:
                gene.axis = "z"
            gene.angle_deg = 0.0
        elif gene.kind == "rounded_rectangle":
            gene.aux = float(rng.uniform(0.0, min(gene.size_x, gene.size_y) * 0.5))
            gene.axis = "z"
        else:
            gene.aux = 0.0
            gene.axis = "z"

    gene.center_z = center_z
    gene.z_start = float(center_z - gene.height * 0.5)

    normalized = clamp_gene(gene, bounds)
    gene.kind = normalized.kind
    gene.op = normalized.op
    gene.center_x = normalized.center_x
    gene.center_y = normalized.center_y
    gene.z_start = normalized.z_start
    gene.height = normalized.height
    gene.size_x = normalized.size_x
    gene.size_y = normalized.size_y
    gene.aux = normalized.aux
    gene.center_z = normalized.center_z
    gene.axis = normalized.axis
    gene.angle_deg = normalized.angle_deg
    return gene


def perturb_continuous(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    rng: np.random.Generator,
    scale: float,
) -> CandidateProgram:
    child = candidate.clone()
    child.provenance = "refined"
    gene = child.genes[int(rng.integers(0, len(child.genes)))]
    mutate_gene(gene, bounds, rng, mutation_strength=max(scale, 0.02))
    return child
