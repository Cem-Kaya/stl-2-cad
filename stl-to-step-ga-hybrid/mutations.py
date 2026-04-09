#!/usr/bin/env python3
from __future__ import annotations

import math

import numpy as np

from dsl import (
    CandidateProgram,
    PrimitiveGene,
    SearchBounds,
    SUPPORTED_KINDS,
    clamp_gene,
    normalize_program,
    random_gene,
)


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
) -> CandidateProgram:
    child = candidate.clone()
    child.provenance = "mutated"
    if not child.genes:
        child.genes.append(random_gene(bounds, rng, allow_cut=False))

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

    if rng.random() < 0.22:
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

    mode = int(rng.integers(0, 6))
    if mode == 0:
        gene.center_x += float(rng.normal(0.0, sigma_xy))
        gene.center_y += float(rng.normal(0.0, sigma_xy))
    elif mode == 1:
        gene.z_start += float(rng.normal(0.0, sigma_z))
        gene.height *= float(math.exp(rng.normal(0.0, mutation_strength * 0.33)))
    elif mode == 2:
        gene.size_x *= float(math.exp(rng.normal(0.0, mutation_strength * 0.38)))
        gene.size_y *= float(math.exp(rng.normal(0.0, mutation_strength * 0.38)))
    elif mode == 3 and gene.kind == "rounded_rectangle":
        gene.aux += float(rng.normal(0.0, sigma_size * 0.35))
    elif mode == 4:
        gene.op = "add" if gene.op == "cut" else "cut"
    else:
        gene.kind = str(rng.choice(SUPPORTED_KINDS))
        if gene.kind == "circle":
            gene.size_x = max(gene.size_x, bounds.min_size)
            gene.size_y = gene.size_x
            gene.aux = 0.0
        elif gene.kind == "rounded_rectangle":
            gene.aux = float(rng.uniform(0.0, min(gene.size_x, gene.size_y) * 0.5))
        else:
            gene.aux = 0.0

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
