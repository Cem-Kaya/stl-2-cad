#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_stl_to_parametric import export_tessellated_step_from_model_spec


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a tessellated STEP file from a saved voxel fit spec."
    )
    parser.add_argument("--spec", required=True, type=Path, help="Input fit JSON")
    parser.add_argument("--out", required=True, type=Path, help="Output STEP path")
    parser.add_argument(
        "--mesh-source",
        choices=["voxel", "stl"],
        default="voxel",
        help="Use the voxelized source mesh or the raw STL mesh as the tessellated STEP source",
    )
    args = parser.parse_args()

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)

    print("start", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("spec", args.spec, flush=True)
    print("out", args.out, flush=True)
    print("mesh_source", args.mesh_source, flush=True)
    export_tessellated_step_from_model_spec(
        spec,
        args.out,
        mesh_source=args.mesh_source,
    )
    print("done", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print(args.out, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
