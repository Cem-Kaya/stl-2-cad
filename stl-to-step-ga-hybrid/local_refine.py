#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dsl import CandidateProgram, SearchBounds, clamp_gene, gene_center_z, normalize_program, primitive_volume
from mutations import mutate_gene, mutate_program, perturb_continuous
from scoring import ProgramScorer, ScoreWeights


def _candidate_signature(candidate: CandidateProgram) -> tuple[tuple[object, ...], ...]:
    rows: list[tuple[object, ...]] = []
    for gene in candidate.genes:
        rows.append(
            (
                gene.kind,
                gene.op,
                round(gene.center_x, 3),
                round(gene.center_y, 3),
                round(gene_center_z(gene), 3),
                round(gene.height, 3),
                round(gene.size_x, 3),
                round(gene.size_y, 3),
                round(gene.aux, 3),
                gene.axis,
                round(gene.angle_deg, 2),
            )
        )
    return tuple(rows)


def _attach_scores(
    candidates: list[CandidateProgram],
    scorer: ProgramScorer,
    weights: ScoreWeights,
    primitive_budget: int,
) -> list[CandidateProgram]:
    if not candidates:
        return []
    metrics = scorer.score_candidates(candidates, weights, primitive_budget)
    for candidate, metric in zip(candidates, metrics, strict=True):
        candidate.metrics = metric
        candidate.score = metric["score"]
    return candidates


def _unique_candidates(candidates: list[CandidateProgram]) -> list[CandidateProgram]:
    unique: list[CandidateProgram] = []
    seen: set[tuple[tuple[object, ...], ...]] = set()
    for candidate in candidates:
        signature = _candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        unique.append(candidate)
    return unique


def _pick_gene_indices(
    candidate: CandidateProgram,
    rng: np.random.Generator,
    count: int,
) -> list[int]:
    if not candidate.genes:
        return []
    if count >= len(candidate.genes):
        return list(range(len(candidate.genes)))
    weights = np.asarray(
        [max(primitive_volume(gene), 1e-6) for gene in candidate.genes],
        dtype=np.float64,
    )
    weights /= weights.sum()
    chosen = rng.choice(len(candidate.genes), size=count, replace=False, p=weights)
    return [int(item) for item in np.asarray(chosen, dtype=np.int64)]


def _coordinate_proposals(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    primitive_budget: int,
    rng: np.random.Generator,
    scale: float,
    budget: int,
) -> list[CandidateProgram]:
    if not candidate.genes or budget <= 0:
        return []

    scale = max(scale, 0.03)
    focus_count = min(len(candidate.genes), 1 + int(budget >= 8))
    focus_indices = _pick_gene_indices(candidate, rng, focus_count)
    span_xy = max(float(bounds.extents[:2].max()), bounds.min_size)
    span_z = max(float(bounds.extents[2]), bounds.min_size)
    proposals: list[CandidateProgram] = []
    for gene_index in focus_indices:
        gene = candidate.genes[gene_index]
        xy_step = max(span_xy * scale * 0.16, bounds.min_size * 0.5)
        z_step = max(span_z * scale * 0.18, bounds.min_size * 0.5)
        size_step = max(span_xy * scale * 0.14, bounds.min_size * 0.5)
        radius_step = max(min(float(gene.size_x), float(gene.size_y)) * scale * 0.22, bounds.min_size * 0.25)
        ops: list[tuple[str, float]] = [
            ("center_x", xy_step),
            ("center_y", xy_step),
            ("center_z", z_step),
            ("height", z_step),
        ]
        if gene.kind == "circle":
            ops.append(("radius", size_step))
        else:
            ops.extend([("size_x", size_step), ("size_y", size_step)])
            ops.append(("angle_deg", max(5.0, scale * 22.0)))
            if gene.kind == "rounded_rectangle":
                ops.append(("aux", radius_step))

        for field, delta in ops:
            for direction in (-1.0, 1.0):
                child = candidate.clone()
                child.provenance = "refine_coord"
                child_gene = child.genes[gene_index]
                offset = float(direction * delta)
                if field == "radius":
                    child_gene.size_x += offset
                    child_gene.size_y = child_gene.size_x
                elif field == "center_z":
                    child_gene.center_z = float(gene_center_z(child_gene) + offset)
                    child_gene.z_start = float(child_gene.center_z - child_gene.height * 0.5)
                else:
                    setattr(child_gene, field, float(getattr(child_gene, field) + offset))
                    if child_gene.kind == "circle" and field in {"size_x", "size_y"}:
                        child_gene.size_y = child_gene.size_x
                child.genes[gene_index] = clamp_gene(child_gene, bounds)
                proposals.append(normalize_program(child, bounds, primitive_budget, rng))

    if len(proposals) <= budget:
        return proposals
    chosen = rng.choice(len(proposals), size=budget, replace=False)
    return [proposals[int(index)] for index in np.asarray(chosen, dtype=np.int64)]


def _multi_gene_proposal(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    primitive_budget: int,
    rng: np.random.Generator,
    scale: float,
) -> CandidateProgram:
    child = candidate.clone()
    child.provenance = "refine_multi"
    if len(child.genes) < 2:
        mutate_gene(child.genes[0], bounds, rng, mutation_strength=max(scale * 0.85, 0.03))
        return normalize_program(child, bounds, primitive_budget, rng)

    touch_count = min(len(child.genes), int(rng.integers(2, min(4, len(child.genes)) + 1)))
    for gene_index in _pick_gene_indices(child, rng, touch_count):
        mutate_gene(child.genes[gene_index], bounds, rng, mutation_strength=max(scale * 0.85, 0.03))
    if len(child.genes) > 1 and rng.random() < 0.2:
        index_a, index_b = _pick_gene_indices(child, rng, 2)
        child.genes[index_a], child.genes[index_b] = child.genes[index_b], child.genes[index_a]
    return normalize_program(child, bounds, primitive_budget, rng)


def _guided_structural_proposal(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    primitive_budget: int,
    rng: np.random.Generator,
    scale: float,
    target: object | None,
) -> CandidateProgram:
    child = mutate_program(
        candidate,
        bounds=bounds,
        rng=rng,
        max_primitives=primitive_budget,
        mutation_strength=max(scale * 0.9, 0.04),
        structural_rate=min(0.5, 0.16 + scale * 0.75),
        target=target,
        bias_rate=1.0 if target is not None else 0.0,
    )
    child.provenance = "refine_guided"
    return child


def _build_step_proposals(
    candidate: CandidateProgram,
    bounds: SearchBounds,
    primitive_budget: int,
    rng: np.random.Generator,
    scale: float,
    budget: int,
    target: object | None,
    multi_gene_rate: float,
) -> list[CandidateProgram]:
    if budget <= 0:
        return []

    proposals: list[CandidateProgram] = []
    coord_budget = min(budget, max(2, int(np.ceil(budget * 0.45))))
    proposals.extend(
        _coordinate_proposals(
            candidate=candidate,
            bounds=bounds,
            primitive_budget=primitive_budget,
            rng=rng,
            scale=scale,
            budget=coord_budget,
        )
    )
    while len(proposals) < budget:
        roll = float(rng.random())
        if target is not None and roll < 0.24:
            proposals.append(
                _guided_structural_proposal(
                    candidate=candidate,
                    bounds=bounds,
                    primitive_budget=primitive_budget,
                    rng=rng,
                    scale=scale,
                    target=target,
                )
            )
        elif roll < 0.24 + multi_gene_rate:
            proposals.append(
                _multi_gene_proposal(
                    candidate=candidate,
                    bounds=bounds,
                    primitive_budget=primitive_budget,
                    rng=rng,
                    scale=scale,
                )
            )
        else:
            child = perturb_continuous(candidate, bounds, rng, scale)
            child.provenance = "refine_continuous"
            if rng.random() < multi_gene_rate * 0.6 and len(child.genes) > 1:
                for gene_index in _pick_gene_indices(child, rng, min(len(child.genes), 2)):
                    mutate_gene(child.genes[gene_index], bounds, rng, mutation_strength=max(scale * 0.6, 0.025))
                child = normalize_program(child, bounds, primitive_budget, rng)
                child.provenance = "refine_multi"
            proposals.append(child)
    return proposals[:budget]


def _sample_explorer(
    accepted: list[CandidateProgram],
    best_score: float,
    temperature: float,
    rng: np.random.Generator,
) -> CandidateProgram | None:
    if not accepted:
        return None
    temp = max(float(temperature), 1e-5)
    deltas = np.asarray([float(item.score or -1e9) - best_score for item in accepted], dtype=np.float64)
    weights = np.exp(np.clip(deltas / temp, -40.0, 0.0))
    weights_sum = float(weights.sum())
    if not np.isfinite(weights_sum) or weights_sum <= 0.0:
        return accepted[int(rng.integers(0, len(accepted)))].clone()
    weights /= weights_sum
    index = int(rng.choice(len(accepted), p=weights))
    return accepted[index].clone()


def refine_candidate(
    candidate: CandidateProgram,
    scorer: ProgramScorer,
    weights: ScoreWeights,
    bounds: SearchBounds,
    primitive_budget: int,
    rng: np.random.Generator,
    steps: int,
    proposals_per_step: int,
    initial_scale: float,
    target: object | None = None,
    beam_width: int = 3,
    acceptance_threshold: float = 0.0025,
    anneal_start: float = 0.0035,
    anneal_decay: float = 0.92,
    multi_gene_rate: float = 0.2,
) -> CandidateProgram:
    current = candidate.clone()
    if current.score is None:
        current = _attach_scores([current], scorer, weights, primitive_budget)[0]

    best = current.clone()
    beam = [current]
    scale = max(initial_scale, 0.02)
    beam_width = max(1, int(beam_width))
    temperature = max(float(anneal_start), float(acceptance_threshold))
    stagnation_steps = 0
    for _ in range(max(0, steps)):
        candidate_pool = [item.clone() for item in beam]
        active_beam = beam[:beam_width]
        base_budget = max(0, proposals_per_step) // max(1, len(active_beam))
        remainder = max(0, proposals_per_step) % max(1, len(active_beam))
        for beam_index, beam_candidate in enumerate(active_beam):
            budget = base_budget + (1 if beam_index < remainder else 0)
            if budget <= 0:
                continue
            candidate_pool.extend(
                _build_step_proposals(
                    candidate=beam_candidate,
                    bounds=bounds,
                    primitive_budget=primitive_budget,
                    rng=rng,
                    scale=scale,
                    budget=budget,
                    target=target,
                    multi_gene_rate=multi_gene_rate,
                )
            )

        ranked = _attach_scores(
            _unique_candidates(candidate_pool),
            scorer=scorer,
            weights=weights,
            primitive_budget=primitive_budget,
        )
        ranked.sort(key=lambda item: float(item.score or -1e9), reverse=True)
        if not ranked:
            break

        top_candidate = ranked[0]
        if float(top_candidate.score or -1e9) > float(best.score or -1e9):
            best = top_candidate.clone()
            scale = min(scale * 1.06, 0.65)
            stagnation_steps = 0
        else:
            scale = max(scale * 0.9, 0.02)
            stagnation_steps += 1

        next_beam: list[CandidateProgram] = [top_candidate.clone()]
        selected = {_candidate_signature(top_candidate)}
        threshold = float(acceptance_threshold) + temperature * (1.0 + 0.2 * min(stagnation_steps, 4))
        accepted = [
            item
            for item in ranked[1:]
            if float(item.score or -1e9) >= float(top_candidate.score or -1e9) - threshold
        ]
        explorer = _sample_explorer(accepted, float(top_candidate.score or -1e9), temperature, rng)
        if explorer is not None:
            explorer_signature = _candidate_signature(explorer)
            if explorer_signature not in selected and len(next_beam) < beam_width:
                selected.add(explorer_signature)
                next_beam.append(explorer)

        for item in ranked[1:]:
            signature = _candidate_signature(item)
            if signature in selected:
                continue
            selected.add(signature)
            next_beam.append(item.clone())
            if len(next_beam) >= beam_width:
                break

        beam = next_beam
        temperature = max(float(acceptance_threshold) * 0.5, temperature * float(anneal_decay))
    return best
