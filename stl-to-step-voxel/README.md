# STL To STEP Voxel

This folder contains the current working STL fitting workflow.

The idea is:

1. Load a watertight STL.
2. Convert it into voxel slices.
3. Merge similar `Z` bands into a compact reconstruction program.
4. Export a parametric rebuild script and preview meshes.
5. Inspect the fit in a Gradio UI with 3D toggles for missing and extra regions.

## What is here

- `fit_stl_to_parametric.py`: CLI fitter and exporters
- `app_gradio.py`: Gradio inspection UI
- `../input/`: local git-ignored STL inputs for testing
- `examples/`: current sample outputs from the voxel workflow
- `assets/`: preview images used in the README

## Preview

The image on the left is the original STL mesh. The image on the right is the voxel-fit reconstruction rendered as a full triangle surface mesh.

![Before vs After](./assets/before-after-comparison.png)

Individual views:

![Original STL](./assets/before-original-stl.png)

![Voxel Reconstruction](./assets/after-voxel-fit.png)

To regenerate these PNGs:

```bash
cd stl-to-step-voxel
. ../.venv/bin/activate
python render_readme_assets.py \
  --stl ../input/3d-enclosure-for-xiao-bus-servo-adapter-bottom.stl
```

## Setup

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run the CLI

```bash
cd stl-to-step-voxel
. ../.venv/bin/activate
python fit_stl_to_parametric.py \
  --input ../input/3d-enclosure-for-xiao-bus-servo-adapter-bottom.stl \
  --output-dir run-output \
  --voxel-pitch 0.5
```

## Run the Gradio UI

```bash
cd stl-to-step-voxel
. ../.venv/bin/activate
python app_gradio.py
```

The UI lets you:

- fit an STL
- export STEP, STL, JSON, and rebuild script
- inspect original vs reconstruction in 3D
- toggle original mesh, reconstructed mesh, missing regions, and extra regions
- read voxel IoU and voxel count summaries

## Current status

The best current output in this folder is the voxel preview reconstruction:

- [`examples/3d-enclosure-for-xiao-bus-servo-adapter-bottom_preview_voxel.glb`](./examples/3d-enclosure-for-xiao-bus-servo-adapter-bottom_preview_voxel.glb)

The voxel path is currently more trustworthy than the old simplified STEP export path. The next step is improving the exact CAD rebuild so the STEP result stays as close to the preview as possible.
