# Photo Reviver

`photo-reviver` is a clean Python project scaffold for an old-photo restoration pipeline.

The goal of this version is not to solve restoration with a model yet. Instead, it gives you the full project structure, the non-model parts of the pipeline, a simple CLI, saved stage outputs, and a clear place where a future Microsoft restoration repo can be connected.

## What This Project Already Does

This code already covers the parts that do **not** require a restoration model:

1. Save the original image into a run folder.
2. Read and validate the image.
3. Convert it to grayscale for analysis.
4. Check histogram / contrast information.
5. Estimate scratch severity with simple image-processing heuristics.
6. Detect whether a face is present with a lightweight OpenCV classical detector.
7. Decide whether a high-resolution path may be useful.
8. Preprocess the image with mild denoising and contrast correction.
9. Choose a restoration mode: `normal`, `scratch`, or `scratch+hr`.
10. Run a placeholder restoration stage.
11. Postprocess the output with optional enhancement, sharpening, and simple upscaling.
12. Create a before/after stage comparison image.
13. Compute simple metrics if you provide a clean reference image.
14. Save JSON reports for every stage.

## What This Project Does Not Do Yet

The actual learned restoration model is **not** included yet.

Right now the default restoration backend is `passthrough`, which means:

- the pipeline runs end to end,
- the restoration stage is structurally present,
- the preprocessed image is simply passed forward as a placeholder,
- you can later replace that stage with a Microsoft repo or another model.

There is now also an optional real backend for the Microsoft repo:

- backend name: `boptl`
- expected repo path: `external/bringing-old-photos-back-to-life`
- expected pretrained assets:
  - `Face_Detection/shape_predictor_68_face_landmarks.dat`
  - `Face_Enhancement/checkpoints/`
  - `Global/checkpoints/`

## Project Structure

```text
photo-reviver/
|-- artifacts/
|   `-- runs/
|-- configs/
|   |-- pipeline.boptl.json
|   `-- pipeline.example.json
|-- external/
|   `-- bringing-old-photos-back-to-life/
|-- src/
|   `-- photo_reviver/
|       |-- __init__.py
|       |-- analysis.py
|       |-- cli.py
|       |-- config.py
|       |-- decision.py
|       |-- evaluate.py
|       |-- io_utils.py
|       |-- pipeline.py
|       |-- postprocess.py
|       |-- preprocess.py
|       |-- restoration.py
|       `-- types.py
|-- tests/
|   |-- test_config.py
|   |-- test_decision.py
|   |-- test_evaluate.py
|   `-- test_restoration.py
|-- .gitignore
|-- pyproject.toml
|-- requirements-boptl.txt
|-- requirements.txt
|-- scripts/
|   `-- setup_boptl_assets.py
`-- README.md
```

## How The 7-Step Workflow Maps To The Code

### Step 1: User uploads image

Handled by:

- `photo_reviver.cli`
- `photo_reviver.pipeline`
- `photo_reviver.io_utils`

What happens:

- the input file path is received from the CLI,
- the file is copied into a timestamped run folder,
- the image is read,
- the format and dimensions are validated.

### Step 2: Analyze image

Handled by:

- `photo_reviver.analysis`

What happens:

- grayscale conversion,
- histogram calculation,
- histogram visualization saved as an image,
- low contrast detection,
- scratch severity estimation,
- face detection,
- decision helper for whether an HR path may be useful.

Important note:

- face detection uses OpenCV's built-in Haar cascade, which is a classical detector and much lighter than a deep restoration model.

### Step 3: Preprocess image

Handled by:

- `photo_reviver.preprocess`

What happens:

- mild denoise,
- CLAHE-based contrast correction,
- optional normalization,
- optional resizing,
- preprocessed output saved to disk.

### Step 4: Choose restoration mode

Handled by:

- `photo_reviver.decision`

What happens:

- if scratches are light, choose `normal`,
- if scratches are visible, choose `scratch`,
- if scratches are strong and the image also looks small, choose `scratch+hr`.

### Step 5: Run restoration engine

Handled by:

- `photo_reviver.restoration`

What happens now:

- default backend: `passthrough`
- this keeps the project runnable without a model
- it creates the correct stage output and report

What also works now:

- you can switch to the `boptl` backend
- the project will call the Microsoft `run.py` for you
- the config for that is in `configs/pipeline.boptl.json`

Optional advanced path:

- you can still use `external_command`
- that is useful if you later want to integrate a different external model

This means the project is already prepared for:

- `run.py` processing the image,
- model pipeline writing to an output path,
- optional model logs and intermediate files.

### Step 6: Postprocess output

Handled by:

- `photo_reviver.postprocess`

What happens:

- optional light enhancement,
- optional sharpening with unsharp masking,
- optional simple upscale with Lanczos,
- colorization is left as a future model-based step and is intentionally skipped.

### Step 7: Evaluate and present result

Handled by:

- `photo_reviver.evaluate`

What happens:

- a stage comparison image is generated,
- the project saves a visual summary,
- if you pass a clean reference image, simple metrics are computed.

Metrics included:

- MAE
- MSE
- PSNR

## Output Folder Structure

Every run creates a new folder inside `artifacts/runs/`.

Example:

```text
artifacts/runs/20260418_235959_old-photo/
|-- 01_input/
|-- 02_analysis/
|-- 03_preprocess/
|-- 04_decision/
|-- 05_restoration/
|-- 06_postprocess/
|-- 07_evaluation/
`-- run_summary.json
```

This makes it easy to inspect every stage separately.

## How Everything Is Connected

The flow is:

1. `cli.py` receives the arguments.
2. `config.py` loads default settings and optional JSON overrides.
3. `pipeline.py` orchestrates the whole run.
4. `io_utils.py` creates the run folders and handles image/file saving.
5. `analysis.py` inspects the image and produces analysis outputs.
6. `decision.py` converts analysis into a restoration mode.
7. `restoration.py` runs either the placeholder backend or an external command.
   It can also call the Microsoft BOPTL pipeline directly through the `boptl` backend.
8. `postprocess.py` refines the image after restoration.
9. `evaluate.py` creates the visual comparison and optional metrics.
10. `run_summary.json` ties all results together in one place.

So the project is split by responsibility:

- each stage has its own module,
- `pipeline.py` connects them,
- outputs are saved after every important step,
- later you can replace only the restoration part without rewriting the rest.

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

You can also install from `requirements.txt`, but `pip install -e .` is the best option because it installs the package and the CLI entry point.

## Extra Dependencies For The Microsoft Model

If you want to use the actual Microsoft backend, install its heavier dependencies too:

```powershell
python -m pip install -r requirements-boptl.txt
```

Notes:

- this backend is much heavier than the simple scaffold,
- CPU mode is supported through `gpu = -1`, but it can be very slow,
- the external repo is research code, so behavior can vary by platform and Python package versions.

## Basic Run

Use a full image path:

```powershell
photo-reviver --input "C:\full\path\to\old_photo.jpg"
```

Or run it directly as a module:

```powershell
python -m photo_reviver.cli --input "C:\full\path\to\old_photo.jpg"
```

## Run With A Config File

```powershell
photo-reviver --input "C:\full\path\to\old_photo.jpg" --config "configs/pipeline.example.json"
```

## Run With The Microsoft Backend

If the external repo and pretrained assets are already in place:

```powershell
photo-reviver --input ".\samples\old_photo_02.jpg" --config ".\configs\pipeline.boptl.json"
```

Or override the backend directly:

```powershell
photo-reviver --input ".\samples\old_photo_02.jpg" --backend boptl
```

The integrated repo path used by default is:

```text
external/bringing-old-photos-back-to-life
```

## Run With A Reference Image

If you have a synthetic-damage setup or a clean target image:

```powershell
photo-reviver --input "C:\full\path\to\damaged_photo.jpg" --reference "C:\full\path\to\clean_reference.jpg"
```

## Run With A Different Output Folder

```powershell
photo-reviver --input "C:\full\path\to\old_photo.jpg" --output-root "custom_runs"
```

## External Repo Layout

The integrated external repo now lives here:

```text
external/bringing-old-photos-back-to-life/
```

Its local runtime layout is intentionally simple:

- `Face_Detection/` contains the dlib landmark model
- `Face_Enhancement/checkpoints/` contains the face model weights
- `Global/checkpoints/` contains the global restoration weights
- unnecessary demo assets and installer leftovers can be cleaned away without touching the core inference path

If you need to recreate the model asset setup later:

```powershell
python .\scripts\setup_boptl_assets.py
```

## Testing

Run the lightweight tests with:

```powershell
python -m unittest discover -s tests
```

## Why This Structure Is Useful

This scaffold is useful because it separates the project into clear layers:

- input and file handling,
- image analysis,
- preprocessing,
- decision logic,
- restoration integration,
- postprocessing,
- evaluation and presentation.

That makes the project easier to:

- understand,
- debug,
- extend,
- replace piece by piece.

When you are ready, the next major step is to plug a real restoration model into `restoration.py` without changing the rest of the pipeline.
