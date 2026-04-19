# Photo Reviver

`photo-reviver` is a small Python project for restoring old photos with a clear staged pipeline.

This version is organized around three ideas:

- the Microsoft `Bringing-Old-Photos-Back-to-Life` repo handles the main restoration step,
- the app shows every stage so you can inspect what changed,
- final touches happen **after** the model output, with sliders for enhancement and sharpening, plus a button that runs **DeOldify** for colorization.

## What The Project Does

The pipeline is split into the same stages you described:

1. Save and validate the uploaded image.
2. Analyze the image:
   - grayscale conversion
   - histogram check
   - low-contrast detection
   - scratch estimation
   - face detection
   - HR-path suggestion
3. Preprocess the image.
4. Choose restoration mode:
   - `normal`
   - `scratch`
   - `scratch+hr`
5. Run the restoration engine.
6. Apply final touches on top of the model output:
   - enhancement
   - sharpening
   - optional DeOldify colorization
7. Save the final image and comparison outputs.

## Current App Logic

The app now works like this:

1. Upload a damaged image.
2. Press `Fix Photo`.
3. The restoration model runs first.
4. Open the `Final Touches` tab.
5. Adjust:
   - `Enhancement strength`
   - `Sharpening strength`
6. Press:
   - `Apply Final Touches` to update the restored image
   - `Run DeOldify Colorization` to colorize the current final-touch version

This means enhancement, sharpening, and colorization all happen **after** the model output, which matches the workflow you wanted.

## Project Structure

```text
photo-reviver/
|-- app.py
|-- artifacts/
|   `-- runs/
|-- configs/
|   |-- pipeline.boptl.json
|   `-- pipeline.example.json
|-- external/
|   |-- bringing-old-photos-back-to-life/
|   `-- deoldify/
|-- scripts/
|-- src/
|   `-- photo_reviver/
|       |-- __init__.py
|       |-- analysis.py
|       |-- app_utils.py
|       |-- cli.py
|       |-- colorization.py
|       |-- config.py
|       |-- decision.py
|       |-- evaluate.py
|       |-- io_utils.py
|       |-- pipeline.py
|       |-- postprocess.py
|       |-- preprocess.py
|       |-- restoration.py
|       |-- types.py
|       `-- web_app.py
|-- tests/
|-- pyproject.toml
|-- requirements-boptl.txt
|-- requirements.txt
`-- README.md
```

## Important Folders

### Microsoft restoration repo

Expected at:

```text
external/bringing-old-photos-back-to-life
```

Important files/folders:

- `run.py`
- `Face_Detection/shape_predictor_68_face_landmarks.dat`
- `Face_Enhancement/checkpoints/`
- `Global/checkpoints/`

### DeOldify repo

Expected at:

```text
external/deoldify
```

Important files/folders:

- `deoldify/visualize.py`
- `models/ColorizeArtistic_gen.pth`

The app is already wired to use that local folder directly.

## What Stays Out Of Git

These are local-only and should stay untracked:

- `external/bringing-old-photos-back-to-life/`
- `external/deoldify/`
- `artifacts/runs/`
- `.venv/` and `.venv-*`
- `result_images/`
- `samples/`

That keeps the repo focused on code, config, tests, and docs.

## Installation

### Basic app setup

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

### Full model setup

If you want to use the Microsoft model and DeOldify too:

```powershell
python -m pip install -r requirements-boptl.txt
```

That file includes the extra packages needed by the heavier model-based flow.

## Run The App

```powershell
streamlit run .\app.py
```

Then:

1. Upload an old photo.
2. Press `Fix Photo`.
3. Review the analysis and restoration tabs.
4. Go to `Final Touches`.
5. Adjust the sliders.
6. Press `Apply Final Touches` or `Run DeOldify Colorization`.

## Run From The CLI

### Simple run

```powershell
photo-reviver --input ".\samples\old_photo_03.png"
```

### Run with the Microsoft backend

```powershell
photo-reviver --input ".\samples\old_photo_03.png" --config ".\configs\pipeline.boptl.json"
```

### Run with a reference image

```powershell
photo-reviver --input "C:\full\path\to\damaged_photo.jpg" --reference "C:\full\path\to\clean_reference.jpg"
```

## Output Layout

Every run creates a folder inside `artifacts/runs/`.

Example:

```text
artifacts/runs/20260420_000000_old-photo/
|-- 01_input/
|-- 02_analysis/
|-- 03_preprocess/
|-- 04_decision/
|-- 05_restoration/
|-- 06_postprocess/
|-- 07_evaluation/
`-- run_summary.json
```

Useful outputs:

- `02_analysis/scratch_mask.png`
- `02_analysis/scratch_overlay.png`
- `05_restoration/restored_model_output.png`
- `06_postprocess/final_restored.png`
- `06_postprocess/colorized_output.png` when DeOldify runs
- `07_evaluation/stage_comparison.png`
- `run_summary.json`

## How The Code Is Connected

- `analysis.py` handles grayscale, histogram, scratch detection, and face detection.
- `preprocess.py` prepares the image before restoration.
- `decision.py` picks the restoration mode.
- `restoration.py` runs the Microsoft repo or another backend.
- `postprocess.py` applies enhancement, sharpening, and optional colorization.
- `colorization.py` is the DeOldify integration layer.
- `pipeline.py` connects the stages and also supports rerunning final touches after the model output.
- `web_app.py` is the Streamlit app.

## Scratch Detection

The scratch detector has been returned to the simpler version again.

It now uses a readable OpenCV heuristic instead of the more aggressive experimental path.  
That makes the analysis easier to understand and easier to tune later.

## Testing

Run the tests with:

```powershell
python -m unittest discover -s tests
```

## Notes

- DeOldify is integrated through your local `external/deoldify` folder.
- The app uses DeOldify only when you press the colorization button.
- Final touches are intentionally separate from the restoration run so you can try different values without rerunning the whole pipeline.
