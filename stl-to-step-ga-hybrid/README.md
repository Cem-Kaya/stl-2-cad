# STL To STEP GA Hybrid

This folder now contains a working hybrid search fitter rather than a scaffold.

The current engine does four things:

1. Voxelize an input STL into a target occupancy grid.
2. Search over short CAD-like programs with a genetic algorithm.
3. Refine promising candidates with local stochastic parameter optimization.
4. Export the best candidate as a parametric spec plus optional STEP/STL rebuilds.

## What is implemented

- `ga_hybrid_search.py`: CLI entry point
- `dsl.py`: candidate program and primitive encoding
- `mutations.py`: crossover and mutation operators
- `local_refine.py`: local parameter refinement loop
- `scoring.py`: batched voxel scoring with CPU and optional CUDA backends

## Execution model

The search is designed to stay compute-heavy but still practical:

- batched scoring on a voxel grid
- optional CUDA scoring via `torch`
- multiprocessing-native offspring generation
- multiprocessing CPU scoring and refinement when you stay on `--device cpu`
- parametric population size, primitive budget, optimization steps, worker count, and score batch size

## Current primitive core

The fast kernel currently searches a compact base vocabulary:

- `box`
- `rounded_box`
- `cylinder`
- boolean `add`
- boolean `cut`

Internally those are lowered into the existing parametric rebuild path as rectangle, rounded-rectangle, and circle extrusions.

## Vocabulary roadmap

Your broader list makes sense, but it should be layered.

Fast direct solids and cuts:

- `box`
- `rounded_box`
- `cylinder`
- `tube`
- `slot`
- `hole`

Semantic feature macros lowered into the direct solids:

- `polygon_extrude`
- `counterbore_hole`
- `countersink_hole`
- `boss`
- `standoff`
- `rib`

Higher-cost post-ops and structure ops that should come after the base shape is stable:

- `cone`
- `sphere`
- `capsule`
- `shell`
- `fillet`
- `chamfer`
- `draft`
- `mirror`
- `linear_pattern`
- `circular_pattern`

That split keeps the scorer fast while still giving the DSL room to grow into a real CAD-program synthesis system.

## Setup

Base runtime:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r stl-to-step-ga-hybrid/requirements.txt
```

Optional CUDA backend:

```bash
. .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
```

If you are on a different CUDA stack, swap the wheel index for the matching PyTorch build.

## Run

CPU example:

```bash
. .venv/bin/activate
python stl-to-step-ga-hybrid/ga_hybrid_search.py \
  --input input/your-part.stl \
  --output-dir stl-to-step-ga-hybrid/run-output \
  --device cpu \
  --workers 16 \
  --population-size 128 \
  --max-primitives 24 \
  --optimization-steps 24
```

CUDA example:

```bash
. .venv/bin/activate
python stl-to-step-ga-hybrid/ga_hybrid_search.py \
  --input input/your-part.stl \
  --output-dir stl-to-step-ga-hybrid/run-output \
  --device cuda \
  --workers 16 \
  --population-size 256 \
  --max-primitives 32 \
  --optimization-steps 24 \
  --score-batch-size 64
```

Main tuning knobs:

- `--population-size`: number of candidates per generation
- `--max-primitives`: hard primitive budget for each candidate program
- `--optimization-steps`: local refinement steps per elite candidate
- `--workers`: CPU worker count
- `--score-batch-size`: batch size for the scorer, especially relevant on CUDA
- `--mutation-strength`: continuous mutation magnitude
- `--structural-rate`: add/remove/reorder mutation rate

Checkpointing and resume:

- `--checkpoint-every`: save resumable state every N generations, default `10`
- `--resume-from`: continue from a saved checkpoint manifest or `.pkl.gz` payload

Each checkpoint writes two files under `output-dir/checkpoints/`:

- `*_checkpoint_gen_XXXX.json`: small manifest with metrics and file pointers
- `*_checkpoint_gen_XXXX.pkl.gz`: full resumable population + RNG state

Resume example:

```bash
python stl-to-step-ga-hybrid/ga_hybrid_search.py \
  --input input/your-part.stl \
  --output-dir output/next-long-run \
  --resume-from output/next-long-run/checkpoints/your-part_ga_hybrid_checkpoint_gen_0099.json \
  --device cuda \
  --workers 28 \
  --population-size 1536 \
  --generations 256 \
  --max-primitives 96
```

## Outputs

Each run writes:

- `*_ga_hybrid_summary.json`
- `*_ga_hybrid_program.json`
- `*_ga_hybrid_spec.json`
- `*_ga_hybrid_rebuild.py`

If `build123d` is installed and STEP/STL export is enabled, the run also writes:

- `*_ga_hybrid.step`
- `*_ga_hybrid.stl`

## Verification

I verified three paths on `2026-04-09`:

- serial CPU smoke run
- multiprocessing CPU run with `--workers 2`
- CUDA run in WSL on an `RTX 4090 Laptop GPU`

The CUDA run reported `backend=torch:cuda` directly from the fitter. During a heavier run, `nvidia-smi` sampled about `19%` GPU utilization and about `3708 MiB / 16376 MiB` memory in use. WSL did not expose the Python process in `nvidia-smi pmon`, so process-level rows were not reliable even though the fitter itself confirmed `torch:cuda`.
