#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing
import os
import pickle
import resource
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import trimesh

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

try:  # pragma: no cover
    import psutil
except Exception:  # pragma: no cover
    psutil = None

from dsl import (
    CandidateProgram,
    SearchBounds,
    candidate_from_supported_features,
    candidate_to_model_spec,
    gene_center_z,
    normalize_program,
    primitive_occupancy_mask,
    random_gene,
    seed_program_from_bounds,
)
from local_refine import refine_candidate
from mutations import crossover_programs, mutate_program
from scoring import ProgramScorer, ScoreWeights, TargetGrid


HERE = Path(__file__).resolve().parent
VOXEL_DIR = HERE.parent / "stl-to-step-voxel"
if str(VOXEL_DIR) not in sys.path:
    sys.path.insert(0, str(VOXEL_DIR))

try:  # pragma: no cover
    from fit_stl_to_parametric import (
        export_step_from_model_spec,
        export_stl_from_model_spec,
        solve_voxelized,
        spec_to_dict,
    )
    from render_readme_assets import occupancy_to_surface_mesh, spec_to_voxel_occupancy
except ImportError:  # pragma: no cover
    export_step_from_model_spec = None
    export_stl_from_model_spec = None
    solve_voxelized = None
    spec_to_dict = None
    occupancy_to_surface_mesh = None
    spec_to_voxel_occupancy = None

CPU_WORKER_SCORER: ProgramScorer | None = None
WORKER_TARGET: TargetGrid | None = None
CHECKPOINT_VERSION = 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Hybrid GA + local refinement STL-to-STEP fitter."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input STL file.")
    parser.add_argument("--output-dir", type=Path, default=HERE / "run-output")
    parser.add_argument("--voxel-pitch", type=float, default=0.75)
    parser.add_argument("--population-size", "--population", dest="population_size", type=int, default=96)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--max-primitives", type=int, default=20)
    parser.add_argument(
        "--optimization-steps",
        "--local-steps",
        dest="optimization_steps",
        type=int,
        default=64,
    )
    parser.add_argument("--local-proposals", type=int, default=6)
    parser.add_argument("--elite-count", type=int, default=10)
    parser.add_argument("--refine-count", type=int, default=8)
    parser.add_argument("--refine-beam-width", type=int, default=3)
    parser.add_argument("--refine-accept-threshold", type=float, default=0.0025)
    parser.add_argument("--refine-anneal-start", type=float, default=0.0035)
    parser.add_argument("--refine-anneal-decay", type=float, default=0.92)
    parser.add_argument("--refine-multi-gene-rate", type=float, default=0.2)
    parser.add_argument("--mutation-strength", type=float, default=0.22)
    parser.add_argument("--structural-rate", type=float, default=0.35)
    parser.add_argument("--mutation-bias-rate", type=float, default=0.15)
    parser.add_argument("--score-batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"], default="auto")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 16) - 1))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no-step-export", action="store_true")
    parser.add_argument("--no-stl-export", action="store_true")
    parser.add_argument("--keep-history", type=int, default=32)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--frame-every", type=int, default=1)
    parser.add_argument("--camera-elev", type=float, default=35.2643897)
    parser.add_argument("--camera-azim", type=float, default=-45.0)
    parser.add_argument("--camera-motion", choices=["fixed", "orbit"], default="fixed")
    parser.add_argument("--camera-orbit-period", type=float, default=240.0)
    parser.add_argument("--frame-width", type=float, default=8.6)
    parser.add_argument("--frame-height", type=float, default=6.8)
    parser.add_argument("--frame-dpi", type=int, default=160)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--resume-from", type=Path)
    return parser


def chunked(items: list[Any], chunk_size: int) -> Iterable[list[Any]]:
    for start in range(0, len(items), chunk_size):
        yield items[start : start + chunk_size]


def candidate_signature(candidate: CandidateProgram) -> tuple[Any, ...]:
    rows: list[tuple[Any, ...]] = []
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


def write_rebuild_script(spec_path: Path, script_path: Path, step_path: Path) -> None:
    script = f"""#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fit_stl_to_parametric import export_step_from_model_spec


def main() -> None:
    here = Path(__file__).resolve().parent
    with open(here / "{spec_path.name}", "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    out_path = here / "{step_path.name}"
    export_step_from_model_spec(spec, out_path)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
"""
    script_path.write_text(script, encoding="utf-8")


def build_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "population_size": args.population_size,
        "generations": args.generations,
        "max_primitives": args.max_primitives,
        "optimization_steps": args.optimization_steps,
        "local_proposals": args.local_proposals,
        "elite_count": args.elite_count,
        "refine_count": args.refine_count,
        "refine_beam_width": args.refine_beam_width,
        "refine_accept_threshold": args.refine_accept_threshold,
        "refine_anneal_start": args.refine_anneal_start,
        "refine_anneal_decay": args.refine_anneal_decay,
        "refine_multi_gene_rate": args.refine_multi_gene_rate,
        "workers": args.workers,
        "voxel_pitch": args.voxel_pitch,
        "mutation_strength": args.mutation_strength,
        "structural_rate": args.structural_rate,
        "mutation_bias_rate": args.mutation_bias_rate,
        "score_batch_size": args.score_batch_size,
        "device": args.device,
        "seed": args.seed,
        "save_frames": args.save_frames,
        "frame_every": args.frame_every,
        "camera_elev": args.camera_elev,
        "camera_azim": args.camera_azim,
        "camera_motion": args.camera_motion,
        "camera_orbit_period": args.camera_orbit_period,
        "frame_width": args.frame_width,
        "frame_height": args.frame_height,
        "frame_dpi": args.frame_dpi,
        "checkpoint_every": args.checkpoint_every,
        "keep_history": args.keep_history,
        "no_step_export": args.no_step_export,
        "no_stl_export": args.no_stl_export,
    }


def checkpoint_paths(output_dir: Path, stem: str, generation: int) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    base = f"{stem}_ga_hybrid_checkpoint_gen_{generation:04d}"
    manifest_path = checkpoint_dir / f"{base}.json"
    data_path = checkpoint_dir / f"{base}.pkl.gz"
    return manifest_path, data_path


def save_checkpoint(
    *,
    output_dir: Path,
    input_path: Path,
    generation: int,
    population: list[CandidateProgram],
    history: list[dict[str, Any]],
    frame_records: list[dict[str, Any]],
    rng: np.random.Generator,
    args: argparse.Namespace,
    target: TargetGrid,
    best: CandidateProgram,
    backend_name: str,
    elapsed_seconds: float,
    cpu_total_seconds: float,
    resume_from: Path | None,
) -> dict[str, str]:
    manifest_path, data_path = checkpoint_paths(output_dir, input_path.stem, generation)
    payload = {
        "format": "ga_hybrid_checkpoint",
        "version": CHECKPOINT_VERSION,
        "input": str(input_path),
        "output_dir": str(output_dir),
        "generation": generation,
        "next_generation": generation + 1,
        "backend": backend_name,
        "config": build_config_payload(args),
        "target_grid": {
            "shape": list(target.shape),
            "voxel_count": target.target_voxels,
            "pitch": target.pitch,
        },
        "best_program": best.to_dict(),
        "population": [candidate.to_dict() for candidate in population],
        "history": history,
        "frame_records": frame_records,
        "rng_state": rng.bit_generator.state,
        "elapsed_offset_seconds": float(elapsed_seconds),
        "cpu_total_offset_seconds": float(cpu_total_seconds),
        "resume_from": None if resume_from is None else str(resume_from),
    }
    with gzip.open(data_path, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "format": payload["format"],
        "version": payload["version"],
        "input": payload["input"],
        "output_dir": payload["output_dir"],
        "generation": generation,
        "next_generation": generation + 1,
        "backend": backend_name,
        "config": payload["config"],
        "best_score": float(best.score or 0.0),
        "best_iou": float(best.metrics.get("iou", 0.0)),
        "best_missing_ratio": float(best.metrics.get("missing_ratio", 0.0)),
        "best_extra_ratio": float(best.metrics.get("extra_ratio", 0.0)),
        "best_primitives": best.primitive_count(),
        "population_size": len(population),
        "history_length": len(history),
        "frame_records_length": len(frame_records),
        "checkpoint_data_file": data_path.name,
        "checkpoint_data_path": str(data_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "manifest": str(manifest_path),
        "data": str(data_path),
    }


def load_resume_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    path = checkpoint_path.expanduser()
    if path.suffix == ".json":
        manifest = json.loads(path.read_text(encoding="utf-8"))
        data_file = manifest.get("checkpoint_data_file")
        data_ref = manifest.get("checkpoint_data_path")
        if data_file is not None:
            data_path = path.parent / str(data_file)
        elif data_ref is not None:
            data_path = Path(str(data_ref))
        else:
            raise ValueError(f"Checkpoint manifest is missing data path: {path}")
        path = data_path

    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)

    if payload.get("format") != "ga_hybrid_checkpoint":
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return payload


def assert_resume_compatible(args: argparse.Namespace, checkpoint: dict[str, Any]) -> None:
    source_path = Path(str(checkpoint["input"])).expanduser().resolve(strict=False)
    input_path = args.input.expanduser().resolve(strict=False)
    if source_path != input_path:
        raise ValueError(
            f"Checkpoint input mismatch: expected {source_path}, got {input_path}"
        )

    config = checkpoint.get("config", {})
    if abs(float(config.get("voxel_pitch", args.voxel_pitch)) - float(args.voxel_pitch)) > 1e-9:
        raise ValueError(
            f"Checkpoint voxel pitch mismatch: expected {config.get('voxel_pitch')}, got {args.voxel_pitch}"
        )
    if int(config.get("max_primitives", args.max_primitives)) != int(args.max_primitives):
        raise ValueError(
            f"Checkpoint max_primitives mismatch: expected {config.get('max_primitives')}, got {args.max_primitives}"
        )
    if int(config.get("population_size", args.population_size)) != int(args.population_size):
        raise ValueError(
            f"Checkpoint population_size mismatch: expected {config.get('population_size')}, got {args.population_size}"
        )


def _init_worker_context(target_payload: dict[str, Any], batch_size: int, init_cpu_scorer: bool) -> None:
    global CPU_WORKER_SCORER, WORKER_TARGET
    WORKER_TARGET = TargetGrid.from_payload(target_payload)
    CPU_WORKER_SCORER = None
    if init_cpu_scorer:
        CPU_WORKER_SCORER = ProgramScorer(WORKER_TARGET, requested_device="cpu", batch_size=batch_size)


def _score_chunk_worker(task: tuple[list[dict[str, Any]], dict[str, float], int]) -> list[dict[str, float]]:
    candidate_payloads, weights_payload, primitive_budget = task
    assert CPU_WORKER_SCORER is not None
    candidates = [CandidateProgram.from_dict(payload) for payload in candidate_payloads]
    weights = ScoreWeights.from_dict(weights_payload)
    return CPU_WORKER_SCORER.score_candidates(candidates, weights, primitive_budget)


def _refine_worker(
    task: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, float],
        int,
        int,
        int,
        float,
        int,
        float,
        float,
        float,
        float,
        int,
    ]
) -> dict[str, Any]:
    (
        candidate_payload,
        bounds_payload,
        weights_payload,
        primitive_budget,
        steps,
        proposals,
        scale,
        beam_width,
        acceptance_threshold,
        anneal_start,
        anneal_decay,
        multi_gene_rate,
        seed,
    ) = task
    assert CPU_WORKER_SCORER is not None
    candidate = CandidateProgram.from_dict(candidate_payload)
    bounds = SearchBounds.from_dict(bounds_payload)
    weights = ScoreWeights.from_dict(weights_payload)
    rng = np.random.default_rng(seed)
    refined = refine_candidate(
        candidate=candidate,
        scorer=CPU_WORKER_SCORER,
        weights=weights,
        bounds=bounds,
        primitive_budget=primitive_budget,
        rng=rng,
        steps=steps,
        proposals_per_step=proposals,
        initial_scale=scale,
        target=WORKER_TARGET,
        beam_width=beam_width,
        acceptance_threshold=acceptance_threshold,
        anneal_start=anneal_start,
        anneal_decay=anneal_decay,
        multi_gene_rate=multi_gene_rate,
    )
    return refined.to_dict()


def _breed_worker(
    task: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, Any],
        int,
        float,
        float,
        float,
        int,
    ]
) -> dict[str, Any]:
    parent_a_payload, parent_b_payload, bounds_payload, max_primitives, mutation_strength, structural_rate, mutation_bias_rate, seed = task
    bounds = SearchBounds.from_dict(bounds_payload)
    rng = np.random.default_rng(seed)
    parent_a = CandidateProgram.from_dict(parent_a_payload)
    parent_b = CandidateProgram.from_dict(parent_b_payload)

    if rng.random() < 0.82:
        child = crossover_programs(parent_a, parent_b, rng, bounds, max_primitives)
    else:
        child = parent_a.clone() if rng.random() < 0.5 else parent_b.clone()

    child = mutate_program(
        child,
        bounds=bounds,
        rng=rng,
        max_primitives=max_primitives,
        mutation_strength=mutation_strength,
        structural_rate=structural_rate,
        target=WORKER_TARGET,
        bias_rate=mutation_bias_rate,
    )
    return child.to_dict()


def score_population(
    population: list[CandidateProgram],
    scorer: ProgramScorer,
    weights: ScoreWeights,
    primitive_budget: int,
    pool: ProcessPoolExecutor | None,
    worker_count: int,
) -> list[CandidateProgram]:
    if not population:
        return []

    if pool is not None and scorer.backend == "numpy" and worker_count > 1 and len(population) >= worker_count * 2:
        payloads = [candidate.to_dict() for candidate in population]
        chunk_size = max(1, len(payloads) // worker_count)
        tasks = [
            (chunk, weights.to_dict(), primitive_budget)
            for chunk in chunked(payloads, chunk_size)
        ]
        metrics: list[dict[str, float]] = []
        for result in pool.map(_score_chunk_worker, tasks):
            metrics.extend(result)
    else:
        metrics = scorer.score_candidates(population, weights, primitive_budget)

    for candidate, metric in zip(population, metrics, strict=True):
        candidate.metrics = metric
        candidate.score = metric["score"]
    return population


def tournament_select(population: list[CandidateProgram], rng: np.random.Generator, size: int = 3) -> CandidateProgram:
    contenders = [population[int(rng.integers(0, len(population)))] for _ in range(size)]
    return max(contenders, key=lambda item: float(item.score or -1e9))


def build_initial_population(
    target: TargetGrid,
    input_path: Path,
    population_size: int,
    max_primitives: int,
    rng: np.random.Generator,
) -> list[CandidateProgram]:
    bounds = target.search_bounds
    population: list[CandidateProgram] = []

    base_seed = normalize_program(seed_program_from_bounds(bounds), bounds, max_primitives, rng)
    population.append(base_seed)

    if solve_voxelized is not None:
        try:
            voxel_spec = solve_voxelized(input_path, pitch=target.pitch, max_steps=max_primitives)
        except Exception:
            voxel_spec = None
        if voxel_spec is not None:
            spec_payload = spec_to_dict(voxel_spec) if spec_to_dict is not None else voxel_spec
            seed = candidate_from_supported_features(
                spec=spec_payload,
                bounds=bounds,
                max_primitives=max_primitives,
            )
            if seed is not None:
                population.append(seed)

    while len(population) < population_size:
        gene_count = int(rng.integers(1, max_primitives + 1))
        genes = [
            random_gene(bounds, rng, allow_cut=index > 0)
            for index in range(gene_count)
        ]
        population.append(
            normalize_program(
                CandidateProgram(genes=genes, provenance="random"),
                bounds=bounds,
                max_primitives=max_primitives,
                rng=rng,
            )
        )
    return population[:population_size]


def refine_elites(
    elites: list[CandidateProgram],
    scorer: ProgramScorer,
    weights: ScoreWeights,
    bounds: SearchBounds,
    target: TargetGrid,
    primitive_budget: int,
    steps: int,
    proposals: int,
    beam_width: int,
    acceptance_threshold: float,
    anneal_start: float,
    anneal_decay: float,
    multi_gene_rate: float,
    rng: np.random.Generator,
    pool: ProcessPoolExecutor | None,
) -> list[CandidateProgram]:
    if not elites or steps <= 0:
        return []

    scale = 0.18
    if pool is not None and scorer.backend == "numpy":
        tasks = [
            (
                elite.to_dict(),
                bounds.to_dict(),
                weights.to_dict(),
                primitive_budget,
                steps,
                proposals,
                scale,
                beam_width,
                acceptance_threshold,
                anneal_start,
                anneal_decay,
                multi_gene_rate,
                int(rng.integers(0, 2**31 - 1)),
            )
            for elite in elites
        ]
        return [CandidateProgram.from_dict(payload) for payload in pool.map(_refine_worker, tasks)]

    refined: list[CandidateProgram] = []
    for elite in elites:
        refined.append(
            refine_candidate(
                candidate=elite,
                scorer=scorer,
                weights=weights,
                bounds=bounds,
                primitive_budget=primitive_budget,
                rng=np.random.default_rng(int(rng.integers(0, 2**31 - 1))),
                steps=steps,
                proposals_per_step=proposals,
                initial_scale=scale,
                target=target,
                beam_width=beam_width,
                acceptance_threshold=acceptance_threshold,
                anneal_start=anneal_start,
                anneal_decay=anneal_decay,
                multi_gene_rate=multi_gene_rate,
            )
        )
    return refined


def breed_generation(
    population: list[CandidateProgram],
    offspring_count: int,
    bounds: SearchBounds,
    target: TargetGrid,
    max_primitives: int,
    mutation_strength: float,
    structural_rate: float,
    mutation_bias_rate: float,
    rng: np.random.Generator,
    pool: ProcessPoolExecutor | None,
) -> list[CandidateProgram]:
    if offspring_count <= 0:
        return []

    if pool is not None:
        tasks = []
        for _ in range(offspring_count):
            parent_a = tournament_select(population, rng, size=3)
            parent_b = tournament_select(population, rng, size=3)
            tasks.append(
                (
                    parent_a.to_dict(),
                    parent_b.to_dict(),
                    bounds.to_dict(),
                    max_primitives,
                    mutation_strength,
                    structural_rate,
                    mutation_bias_rate,
                    int(rng.integers(0, 2**31 - 1)),
                )
            )
        return [CandidateProgram.from_dict(payload) for payload in pool.map(_breed_worker, tasks)]

    offspring: list[CandidateProgram] = []
    for _ in range(offspring_count):
        parent_a = tournament_select(population, rng, size=3)
        parent_b = tournament_select(population, rng, size=3)
        if rng.random() < 0.82:
            child = crossover_programs(parent_a, parent_b, rng, bounds, max_primitives)
        else:
            child = parent_a.clone() if rng.random() < 0.5 else parent_b.clone()

        child = mutate_program(
            child,
            bounds=bounds,
            rng=rng,
            max_primitives=max_primitives,
            mutation_strength=mutation_strength,
            structural_rate=structural_rate,
            target=target,
            bias_rate=mutation_bias_rate,
        )
        offspring.append(child)
    return offspring


def select_next_population(candidates: list[CandidateProgram], population_size: int) -> list[CandidateProgram]:
    ordered = sorted(candidates, key=lambda item: float(item.score or -1e9), reverse=True)
    next_population: list[CandidateProgram] = []
    seen: set[tuple[Any, ...]] = set()
    for candidate in ordered:
        signature = candidate_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        next_population.append(candidate)
        if len(next_population) >= population_size:
            break
    if len(next_population) < population_size and ordered:
        refill_index = 0
        while len(next_population) < population_size:
            next_population.append(ordered[refill_index % len(ordered)].clone())
            refill_index += 1
    return next_population


def export_best_candidate(
    best: CandidateProgram,
    target: TargetGrid,
    output_dir: Path,
    input_path: Path,
    max_primitives: int,
    step_export: bool,
    stl_export: bool,
    backend_name: str,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem

    spec = candidate_to_model_spec(
        candidate=best,
        input_path=str(input_path),
        grid_pitch=target.pitch,
        step_budget=max_primitives,
        notes=[
            f"Hybrid GA search on backend {backend_name}.",
            "Semantic roadmap: box, rounded_box, cylinder, tube, slot, hole, patterns, and post-ops.",
        ],
    )
    spec_path = output_dir / f"{stem}_ga_hybrid_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2), encoding="utf-8")

    program_path = output_dir / f"{stem}_ga_hybrid_program.json"
    program_path.write_text(json.dumps(best.to_dict(), indent=2), encoding="utf-8")

    rebuild_path = output_dir / f"{stem}_ga_hybrid_rebuild.py"
    step_path = output_dir / f"{stem}_ga_hybrid.step"
    write_rebuild_script(spec_path=spec_path, script_path=rebuild_path, step_path=step_path)

    outputs = {
        "spec_json": str(spec_path),
        "program_json": str(program_path),
        "rebuild_script": str(rebuild_path),
    }

    if step_export and export_step_from_model_spec is not None:
        try:
            export_step_from_model_spec(spec, step_path)
            outputs["step"] = str(step_path)
        except Exception as exc:  # pragma: no cover
            outputs["step_error"] = str(exc)

    if stl_export and export_stl_from_model_spec is not None:
        stl_path = output_dir / f"{stem}_ga_hybrid.stl"
        try:
            export_stl_from_model_spec(spec, stl_path)
            outputs["stl"] = str(stl_path)
        except Exception as exc:  # pragma: no cover
            outputs["stl_error"] = str(exc)

    return outputs


def choose_pool_context(backend: str) -> multiprocessing.context.BaseContext:
    if backend == "numpy" and sys.platform != "win32":
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context("spawn")


def collect_resource_snapshot() -> dict[str, float]:
    if psutil is not None:
        process = psutil.Process(os.getpid())
        processes = [process]
        processes.extend(process.children(recursive=True))

        rss_tree = 0
        cpu_user = 0.0
        cpu_system = 0.0
        child_count = 0
        for proc in processes:
            try:
                mem = proc.memory_info()
                cpu = proc.cpu_times()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            rss_tree += int(mem.rss)
            cpu_user += float(getattr(cpu, "user", 0.0))
            cpu_system += float(getattr(cpu, "system", 0.0))
            if proc.pid != process.pid:
                child_count += 1

        try:
            self_rss = int(process.memory_info().rss)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            self_rss = 0

        return {
            "rss_self_mb": float(self_rss / (1024.0 * 1024.0)),
            "rss_tree_mb": float(rss_tree / (1024.0 * 1024.0)),
            "cpu_user_seconds_total": cpu_user,
            "cpu_system_seconds_total": cpu_system,
            "cpu_total_seconds_total": cpu_user + cpu_system,
            "child_processes": float(child_count),
        }

    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    max_rss_kb = float(self_usage.ru_maxrss + child_usage.ru_maxrss)
    cpu_user = float(self_usage.ru_utime + child_usage.ru_utime)
    cpu_system = float(self_usage.ru_stime + child_usage.ru_stime)
    return {
        "rss_self_mb": float(self_usage.ru_maxrss / 1024.0),
        "rss_tree_mb": float(max_rss_kb / 1024.0),
        "cpu_user_seconds_total": cpu_user,
        "cpu_system_seconds_total": cpu_system,
        "cpu_total_seconds_total": cpu_user + cpu_system,
        "child_processes": 0.0,
    }


def face_colors(mesh: trimesh.Trimesh, color: str) -> np.ndarray:
    rgb = np.asarray(to_rgb(color), dtype=float)
    normals = mesh.face_normals
    light_dir = np.array([0.58, -0.45, 0.68], dtype=float)
    light_dir /= np.linalg.norm(light_dir)
    diffuse = np.clip(normals @ light_dir, 0.0, 1.0)
    intensity = 0.34 + 0.66 * diffuse
    shaded = np.clip(rgb[None, :] * intensity[:, None], 0.0, 1.0)
    return np.concatenate([shaded, np.ones((len(shaded), 1), dtype=float)], axis=1)


def apply_camera(ax: Any, bounds: np.ndarray, elev: float, azim: float) -> None:
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
    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.dist = 8


def render_generation_frame(
    candidate: CandidateProgram,
    target: TargetGrid,
    output_path: Path,
    generation: int,
    camera_elev: float,
    camera_azim: float,
    camera_motion: str,
    camera_orbit_period: float,
    frame_width: float,
    frame_height: float,
    frame_dpi: int,
) -> str | None:
    if occupancy_to_surface_mesh is None:
        return "frame rendering helpers are unavailable"

    occupancy = np.zeros_like(target.occupancy, dtype=bool)
    for gene in candidate.genes:
        primitive = primitive_occupancy_mask(gene, target.xs, target.ys, target.zs)
        if gene.op == "add":
            occupancy |= primitive
        else:
            occupancy &= ~primitive
    if not occupancy.any():
        return "candidate occupancy is empty"

    min_corner = np.asarray(target.score_min_corner, dtype=float)
    max_corner = np.asarray(target.score_max_corner, dtype=float)
    mesh = occupancy_to_surface_mesh(occupancy, min_corner, target.pitch)
    if len(mesh.faces) == 0:
        return "candidate mesh is empty"

    extents = max_corner - min_corner
    pad = np.maximum(extents * 0.12, target.pitch * 2.0)
    bounds = np.array([min_corner - pad, max_corner + pad], dtype=float)

    triangles = mesh.vertices[mesh.faces]
    fig = plt.figure(figsize=(frame_width, frame_height), dpi=frame_dpi)
    ax = fig.add_subplot(111, projection="3d")
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors(mesh, "#ea580c"),
        edgecolors="none",
        linewidths=0.0,
        antialiased=True,
    )
    ax.add_collection3d(collection)
    orbit_period = max(float(camera_orbit_period), 1.0)
    if camera_motion == "orbit":
        azim = camera_azim + (360.0 * (float(generation) / orbit_period))
    else:
        azim = camera_azim
    apply_camera(ax, bounds, elev=camera_elev, azim=azim)
    ax.set_title(
        f"Generation {generation:03d} | IoU {candidate.metrics.get('iou', 0.0):.4f} | "
        f"Prims {candidate.primitive_count()}",
        fontsize=14,
        pad=14,
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return None


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    target = TargetGrid.from_stl(args.input, pitch=args.voxel_pitch)
    scorer = ProgramScorer(target, requested_device=args.device, batch_size=args.score_batch_size)
    weights = ScoreWeights()
    baseline_resources = collect_resource_snapshot()

    history: list[dict[str, Any]] = []
    frame_dir = args.output_dir / "frames"
    frame_records: list[dict[str, Any]] = []
    start_generation = 0
    elapsed_offset_seconds = 0.0
    cpu_total_offset_seconds = 0.0
    resume_checkpoint_path: Path | None = None
    latest_checkpoint: dict[str, str] | None = None
    pool: ProcessPoolExecutor | None = None
    if args.workers > 1:
        context = choose_pool_context(scorer.backend)
        if scorer.backend == "numpy":
            pool = ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
                initializer=_init_worker_context,
                initargs=(target.to_payload(), args.score_batch_size, True),
            )
        else:
            pool = ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
                initializer=_init_worker_context,
                initargs=(target.to_payload(), args.score_batch_size, False),
            )

    try:
        if args.resume_from is not None:
            resume_checkpoint_path = args.resume_from.expanduser().resolve(strict=False)
            checkpoint = load_resume_checkpoint(resume_checkpoint_path)
            assert_resume_compatible(args, checkpoint)
            rng.bit_generator.state = checkpoint["rng_state"]
            population = [CandidateProgram.from_dict(item) for item in checkpoint["population"]]
            history = list(checkpoint.get("history", []))[-max(1, args.keep_history) :]
            frame_records = list(checkpoint.get("frame_records", []))
            start_generation = int(checkpoint.get("next_generation", 0))
            elapsed_offset_seconds = float(checkpoint.get("elapsed_offset_seconds", 0.0))
            cpu_total_offset_seconds = float(checkpoint.get("cpu_total_offset_seconds", 0.0))
        else:
            population = build_initial_population(
                target=target,
                input_path=args.input,
                population_size=args.population_size,
                max_primitives=args.max_primitives,
                rng=rng,
            )
            population = score_population(
                population=population,
                scorer=scorer,
                weights=weights,
                primitive_budget=args.max_primitives,
                pool=pool,
                worker_count=args.workers,
            )

        start_time = time.time()
        prev_wall = start_time
        cpu_baseline_current = baseline_resources["cpu_total_seconds_total"]
        prev_cpu_total = cpu_total_offset_seconds
        for generation in range(start_generation, args.generations):
            population = sorted(population, key=lambda item: float(item.score or -1e9), reverse=True)
            elites = [candidate.clone() for candidate in population[: args.elite_count]]
            refined = refine_elites(
                elites=population[: args.refine_count],
                scorer=scorer,
                weights=weights,
                bounds=target.search_bounds,
                target=target,
                primitive_budget=args.max_primitives,
                steps=args.optimization_steps,
                proposals=args.local_proposals,
                beam_width=args.refine_beam_width,
                acceptance_threshold=args.refine_accept_threshold,
                anneal_start=args.refine_anneal_start,
                anneal_decay=args.refine_anneal_decay,
                multi_gene_rate=args.refine_multi_gene_rate,
                rng=rng,
                pool=pool,
            )

            if refined:
                refined = score_population(
                    population=refined,
                    scorer=scorer,
                    weights=weights,
                    primitive_budget=args.max_primitives,
                    pool=pool,
                    worker_count=args.workers,
                )

            offspring = breed_generation(
                population=population,
                offspring_count=max(args.population_size - len(elites) - len(refined), 0),
                bounds=target.search_bounds,
                target=target,
                max_primitives=args.max_primitives,
                mutation_strength=args.mutation_strength,
                structural_rate=args.structural_rate,
                mutation_bias_rate=args.mutation_bias_rate,
                rng=rng,
                pool=pool,
            )
            offspring = score_population(
                population=offspring,
                scorer=scorer,
                weights=weights,
                primitive_budget=args.max_primitives,
                pool=pool,
                worker_count=args.workers,
            )

            population = select_next_population(elites + refined + offspring + population[: args.elite_count], args.population_size)
            best = population[0]
            now = time.time()
            elapsed = elapsed_offset_seconds + (now - start_time)
            resources = collect_resource_snapshot()
            cpu_total = cpu_total_offset_seconds + (
                resources["cpu_total_seconds_total"] - cpu_baseline_current
            )
            record = {
                "generation": float(generation),
                "best_score": float(best.score or 0.0),
                "best_iou": float(best.metrics.get("iou", 0.0)),
                "best_missing_ratio": float(best.metrics.get("missing_ratio", 0.0)),
                "best_extra_ratio": float(best.metrics.get("extra_ratio", 0.0)),
                "elapsed_seconds": float(elapsed),
                "wall_seconds_step": float(now - prev_wall),
                "cpu_time_seconds_step": float(cpu_total - prev_cpu_total),
                "cpu_time_seconds_total": float(cpu_total),
                "rss_self_mb": float(resources["rss_self_mb"]),
                "rss_tree_mb": float(resources["rss_tree_mb"]),
                "child_processes": float(resources["child_processes"]),
            }
            history.append(record)
            history = history[-max(1, args.keep_history) :]
            if args.save_frames and generation % max(1, args.frame_every) == 0:
                frame_path = frame_dir / f"gen_{generation:04d}.png"
                if frame_path.exists():
                    frame_error = None
                else:
                    frame_error = render_generation_frame(
                        candidate=best,
                        target=target,
                        output_path=frame_path,
                        generation=generation,
                        camera_elev=args.camera_elev,
                        camera_azim=args.camera_azim,
                        camera_motion=args.camera_motion,
                        camera_orbit_period=args.camera_orbit_period,
                        frame_width=args.frame_width,
                        frame_height=args.frame_height,
                        frame_dpi=args.frame_dpi,
                    )
                frame_records.append(
                    {
                        "generation": generation,
                        "path": str(frame_path),
                        "error": frame_error,
                    }
                )
                if frame_error is None:
                    record["frame_path"] = str(frame_path)
                else:
                    record["frame_error"] = frame_error
            prev_wall = now
            prev_cpu_total = cpu_total
            checkpoint_saved = False
            if args.checkpoint_every > 0 and (
                ((generation + 1) % args.checkpoint_every == 0) or (generation + 1 == args.generations)
            ):
                latest_checkpoint = save_checkpoint(
                    output_dir=args.output_dir,
                    input_path=args.input,
                    generation=generation,
                    population=population,
                    history=history,
                    frame_records=frame_records,
                    rng=rng,
                    args=args,
                    target=target,
                    best=best,
                    backend_name=scorer.backend_name(),
                    elapsed_seconds=record["elapsed_seconds"],
                    cpu_total_seconds=record["cpu_time_seconds_total"],
                    resume_from=resume_checkpoint_path,
                )
                checkpoint_saved = True
            print(
                f"[gen {generation:03d}] "
                f"score={record['best_score']:.5f} "
                f"iou={record['best_iou']:.5f} "
                f"missing={record['best_missing_ratio']:.5f} "
                f"extra={record['best_extra_ratio']:.5f} "
                f"cpu_step={record['cpu_time_seconds_step']:.2f}s "
                f"rss_tree={record['rss_tree_mb']:.1f}MB "
                f"prims={best.primitive_count()} "
                f"backend={scorer.backend_name()}"
            )
            if checkpoint_saved and latest_checkpoint is not None:
                print(f"checkpoint: {latest_checkpoint['manifest']}")

        best = max(population, key=lambda item: float(item.score or -1e9))
        outputs = export_best_candidate(
            best=best,
            target=target,
            output_dir=args.output_dir,
            input_path=args.input,
            max_primitives=args.max_primitives,
            step_export=not args.no_step_export,
            stl_export=not args.no_stl_export,
            backend_name=scorer.backend_name(),
        )
        summary_path = args.output_dir / f"{args.input.stem}_ga_hybrid_summary.json"
        summary = {
            "input": str(args.input),
            "output_dir": str(args.output_dir),
            "backend": scorer.backend_name(),
            "config": build_config_payload(args),
            "resume_from": None if resume_checkpoint_path is None else str(resume_checkpoint_path),
            "start_generation": start_generation,
            "best_program": best.to_dict(),
            "target_grid": {
                "shape": list(target.shape),
                "voxel_count": target.target_voxels,
                "pitch": target.pitch,
            },
            "resource_baseline": baseline_resources,
            "history": history,
            "frame_records": frame_records,
            "latest_checkpoint": latest_checkpoint,
            "outputs": outputs,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        print("Best score:", f"{best.score or 0.0:.6f}")
        print("Best IoU:", f"{best.metrics.get('iou', 0.0):.6f}")
        print("Summary:", summary_path)
        for label, path in outputs.items():
            print(f"{label}: {path}")
    finally:
        if pool is not None:
            pool.shutdown(wait=True, cancel_futures=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
