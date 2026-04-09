# STL To STEP GA Hybrid

This folder is the scaffold for the next approach: a search-based fitter that treats STL-to-CAD as program synthesis.

The intended hybrid loop is:

1. Represent a CAD history as a short sequence of operations.
2. Maintain a population of candidate programs.
3. Mutate and recombine the operation lists.
4. Locally refine continuous parameters for promising candidates.
5. Score candidates on voxel IoU, missing regions, extra regions, and optional Chamfer distance.
6. Rebuild only the strongest candidates into exact CAD at the end.

## Why this approach

The voxel workflow is fast and useful, but it is still heuristic-heavy.

The GA hybrid approach is better suited when you want:

- a shorter editable CAD program
- explicit add and subtract operations
- automatic discovery of cylinders, rectangles, profiles, and repeated features
- a route toward combining structural search with local optimization

## Planned operation vocabulary

- boxes
- cylinders
- rounded rectangles
- profile extrusions
- boolean add
- boolean subtract
- optional chamfers and fillets as post-ops

## Planned scoring terms

- voxel IoU
- missing voxel penalty
- extra voxel penalty
- program length penalty
- invalid CAD penalty

## Suggested next files

- `requirements.txt`
- `ga_hybrid_search.py`
- `dsl.py`
- `mutations.py`
- `scoring.py`
- `local_refine.py`

This folder is intentionally clean right now so the GA work can start without carrying over the mistakes from the earlier quick-fit experiments.
