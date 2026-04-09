#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dsl import CandidateProgram, SearchBounds
from mutations import perturb_continuous
from scoring import ProgramScorer, ScoreWeights


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
) -> CandidateProgram:
    current = candidate.clone()
    if current.score is None:
        current.metrics = scorer.score_candidates([current], weights, primitive_budget)[0]
        current.score = current.metrics["score"]

    scale = max(initial_scale, 0.02)
    for _ in range(max(0, steps)):
        proposals = [
            perturb_continuous(current, bounds, rng, scale)
            for _ in range(max(1, proposals_per_step))
        ]
        metrics = scorer.score_candidates(proposals, weights, primitive_budget)
        best_index = max(range(len(proposals)), key=lambda idx: metrics[idx]["score"])
        best_metrics = metrics[best_index]
        if best_metrics["score"] > float(current.score):
            current = proposals[best_index]
            current.metrics = best_metrics
            current.score = best_metrics["score"]
            scale = min(scale * 1.08, 0.6)
        else:
            scale = max(scale * 0.86, 0.02)
    return current
