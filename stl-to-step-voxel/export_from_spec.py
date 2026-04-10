#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_stl_to_parametric import export_step_from_model_spec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a STEP file from a saved voxel fit spec."
    )
    parser.add_argument("--spec", required=True, type=Path, help="Input fit JSON")
    parser.add_argument("--out", required=True, type=Path, help="Output STEP path")
    parser.add_argument(
        "--batch-profiles",
        action="store_true",
        help="Batch same-layer profile faces into fewer exact extrusions",
    )
    parser.add_argument(
        "--no-batch-union-2d",
        action="store_true",
        help="Skip the 2D union step while batching profile faces",
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("start", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("spec", args.spec, flush=True)
    print("out", args.out, flush=True)
    export_step_from_model_spec(
        spec,
        args.out,
        batch_profiles=args.batch_profiles,
        batch_union_2d=not args.no_batch_union_2d,
    )
    print("done", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
