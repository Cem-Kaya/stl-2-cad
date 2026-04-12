#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

from fit_stl_to_parametric import export_step_from_model_spec


def main() -> None:
    here = Path(__file__).resolve().parent
    with open(here / "3d-enclosure-for-xiao-bus-servo-adapter-top_ga_hybrid_spec.json", "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    out_path = here / "3d-enclosure-for-xiao-bus-servo-adapter-top_ga_hybrid.step"
    export_step_from_model_spec(spec, out_path)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()
