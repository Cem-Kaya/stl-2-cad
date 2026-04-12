#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from dsl import CandidateProgram, SearchBounds, clamp_gene, gene_center_z, normalize_program
from scoring import ProgramScorer, ScoreWeights, TargetGrid


def load_checkpoint(checkpoint_path: Path) -> tuple[dict[str, Any], Path | None]:
    path = checkpoint_path.expanduser().resolve(strict=False)
    manifest_path: Path | None = None
    if path.suffix == ".json":
        manifest_path = path
        manifest = json.loads(path.read_text(encoding="utf-8"))
        data_file = manifest.get("checkpoint_data_file")
        data_ref = manifest.get("checkpoint_data_path")
        if data_file is not None:
            path = path.parent / str(data_file)
        elif data_ref is not None:
            path = Path(str(data_ref)).expanduser().resolve(strict=False)
        else:
            raise ValueError(f"Checkpoint manifest is missing data path: {checkpoint_path}")

    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("format") != "ga_hybrid_checkpoint":
        raise ValueError(f"Unsupported checkpoint format in {path}")
    return payload, manifest_path


def _checkpoint_output_paths(
    source_checkpoint: Path,
    output_dir: Path,
    suffix: str,
) -> tuple[Path, Path]:
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_name = source_checkpoint.stem
    if source_name.endswith(".pkl"):
        source_name = source_name[:-4]
    base = f"{source_name}{suffix}"
    return checkpoint_dir / f"{base}.json", checkpoint_dir / f"{base}.pkl.gz"


def _rotate_gene_variant(candidate: CandidateProgram, bounds: SearchBounds, rng: np.random.Generator) -> bool:
    options = [index for index, gene in enumerate(candidate.genes) if gene.kind != "circle"]
    if not options:
        return False
    gene_index = int(rng.choice(options))
    gene = candidate.genes[gene_index]
    target_angle = float(rng.choice(np.asarray([-45.0, -30.0, -22.5, -15.0, 15.0, 22.5, 30.0, 45.0])))
    gene.angle_deg = target_angle if abs(gene.angle_deg) < 1e-3 else float(gene.angle_deg + target_angle * 0.5)
    candidate.genes[gene_index] = clamp_gene(gene, bounds)
    return True


def _sideways_cylinder_variant(candidate: CandidateProgram, bounds: SearchBounds, rng: np.random.Generator) -> bool:
    options = [index for index, gene in enumerate(candidate.genes) if gene.kind == "circle"]
    if not options:
        return False
    gene_index = int(rng.choice(options))
    gene = candidate.genes[gene_index]
    if gene.axis == "z":
        gene.axis = str(rng.choice(np.asarray(("x", "y"), dtype=object)))
    else:
        gene.axis = str(rng.choice(np.asarray(("x", "y", "z"), dtype=object)))
    gene.center_z = gene_center_z(gene)
    candidate.genes[gene_index] = clamp_gene(gene, bounds)
    return True


def _hybrid_variant(candidate: CandidateProgram, bounds: SearchBounds, rng: np.random.Generator) -> bool:
    changed = False
    if rng.random() < 0.7:
        changed |= _rotate_gene_variant(candidate, bounds, rng)
    if rng.random() < 0.7:
        changed |= _sideways_cylinder_variant(candidate, bounds, rng)
    return changed


def patch_population(
    population: list[CandidateProgram],
    bounds: SearchBounds,
    max_primitives: int,
    rng: np.random.Generator,
    oriented_fraction: float,
    rotate_fraction: float,
    hybrid_fraction: float,
) -> list[CandidateProgram]:
    if not population:
        return population

    patched = [candidate.clone() for candidate in population]
    count = len(patched)
    rotate_count = min(count, max(0, int(round(count * max(rotate_fraction, 0.0)))))
    orient_count = min(count, max(0, int(round(count * max(oriented_fraction, 0.0)))))
    hybrid_count = min(count, max(0, int(round(count * max(hybrid_fraction, 0.0)))))

    patch_start = min(count - 1, max(0, int(count * 0.25)))
    indices = np.arange(patch_start, count, dtype=np.int64)
    if len(indices) == 0:
        indices = np.arange(count, dtype=np.int64)
    rng.shuffle(indices)
    cursor = 0

    for _ in range(rotate_count):
        idx = int(indices[cursor % count])
        cursor += 1
        child = patched[idx].clone()
        if _rotate_gene_variant(child, bounds, rng):
            patched[idx] = normalize_program(child, bounds, max_primitives, rng)

    for _ in range(orient_count):
        idx = int(indices[cursor % count])
        cursor += 1
        child = patched[idx].clone()
        if _sideways_cylinder_variant(child, bounds, rng):
            patched[idx] = normalize_program(child, bounds, max_primitives, rng)

    for _ in range(hybrid_count):
        idx = int(indices[cursor % count])
        cursor += 1
        child = patched[idx].clone()
        if _hybrid_variant(child, bounds, rng):
            patched[idx] = normalize_program(child, bounds, max_primitives, rng)

    return patched


def rescore_population(
    population: list[CandidateProgram],
    target: TargetGrid,
    device: str,
    batch_size: int,
    primitive_budget: int,
) -> tuple[list[CandidateProgram], CandidateProgram]:
    scorer = ProgramScorer(target, requested_device=device, batch_size=batch_size)
    weights = ScoreWeights()
    metrics = scorer.score_candidates(population, weights, primitive_budget)
    best: CandidateProgram | None = None
    for candidate, metric in zip(population, metrics, strict=True):
        candidate.metrics = metric
        candidate.score = float(metric["score"])
        if best is None or float(candidate.score or -1e9) > float(best.score or -1e9):
            best = candidate
    assert best is not None
    return population, best.clone()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Patch a GA checkpoint population for the extended DSL.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint manifest or data file.")
    parser.add_argument("--output-dir", type=Path, help="Directory that will receive the patched checkpoint.")
    parser.add_argument("--suffix", type=str, default="_dsl_patch", help="Suffix appended to the checkpoint file stem.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "gpu"], default="cpu")
    parser.add_argument("--score-batch-size", type=int, default=256)
    parser.add_argument("--oriented-fraction", type=float, default=0.12)
    parser.add_argument("--rotate-fraction", type=float, default=0.12)
    parser.add_argument("--hybrid-fraction", type=float, default=0.08)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload, manifest_path = load_checkpoint(args.checkpoint)
    rng = np.random.default_rng(args.seed)

    input_path = Path(str(payload["input"])).expanduser().resolve(strict=False)
    config = payload.get("config", {})
    voxel_pitch = float(config.get("voxel_pitch", 0.5))
    max_primitives = int(config.get("max_primitives", 128))
    output_dir = (
        args.output_dir.expanduser().resolve(strict=False)
        if args.output_dir is not None
        else Path(str(payload["output_dir"])).expanduser().resolve(strict=False)
    )

    target = TargetGrid.from_stl(input_path, pitch=voxel_pitch)
    bounds = target.search_bounds
    population = [CandidateProgram.from_dict(item) for item in payload["population"]]
    patched_population = patch_population(
        population=population,
        bounds=bounds,
        max_primitives=max_primitives,
        rng=rng,
        oriented_fraction=args.oriented_fraction,
        rotate_fraction=args.rotate_fraction,
        hybrid_fraction=args.hybrid_fraction,
    )
    patched_population, best = rescore_population(
        population=patched_population,
        target=target,
        device=args.device,
        batch_size=max(1, int(args.score_batch_size)),
        primitive_budget=max_primitives,
    )

    payload["population"] = [candidate.to_dict() for candidate in patched_population]
    payload["best_program"] = best.to_dict()
    payload["rng_state"] = rng.bit_generator.state
    payload["resume_from"] = str(args.checkpoint.expanduser().resolve(strict=False))
    payload["patch_info"] = {
        "seed": int(args.seed),
        "oriented_fraction": float(args.oriented_fraction),
        "rotate_fraction": float(args.rotate_fraction),
        "hybrid_fraction": float(args.hybrid_fraction),
        "dsl_extension": "axis_aligned_cylinders_and_xy_rotation",
    }

    source_ref = manifest_path if manifest_path is not None else args.checkpoint
    manifest_out, data_out = _checkpoint_output_paths(source_ref, output_dir, args.suffix)
    with gzip.open(data_out, "wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)

    manifest = {
        "format": payload["format"],
        "version": payload.get("version", 1),
        "input": payload["input"],
        "output_dir": str(output_dir),
        "generation": int(payload["generation"]),
        "next_generation": int(payload["next_generation"]),
        "backend": payload.get("backend", "patched"),
        "config": payload.get("config", {}),
        "best_score": float(best.score or 0.0),
        "best_iou": float(best.metrics.get("iou", 0.0)),
        "best_missing_ratio": float(best.metrics.get("missing_ratio", 0.0)),
        "best_extra_ratio": float(best.metrics.get("extra_ratio", 0.0)),
        "best_primitives": best.primitive_count(),
        "population_size": len(patched_population),
        "history_length": len(payload.get("history", [])),
        "frame_records_length": len(payload.get("frame_records", [])),
        "checkpoint_data_file": data_out.name,
        "checkpoint_data_path": str(data_out),
        "resume_from": payload["resume_from"],
        "patch_info": payload["patch_info"],
    }
    manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"patched_checkpoint={manifest_out}")
    print(f"best_score={float(best.score or 0.0):.8f}")
    print(f"best_iou={float(best.metrics.get('iou', 0.0)):.8f}")
    print(f"best_prims={best.primitive_count()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
