# Photo Reviver

`photo-reviver` is a small Python app for restoring old photos.

It is built around:

- a simple Streamlit interface
- Microsoft `Bringing-Old-Photos-Back-to-Life` for the main restoration
- optional DeOldify colorization
- a stage-by-stage view so you can inspect what happened

## What This Repo Contains

This repo keeps the app code, config, tests, and UI.

This repo does **not** keep the heavy pretrained models inside Git.

That means on another PC you only need to:

1. copy or clone this repo
2. install Python packages
3. download the pretrained model folders into `external/`
4. run the app

## Folder Layout

Expected structure:

```text
photo-reviver/
|-- app.py
|-- configs/
|-- external/
|   |-- bringing-old-photos-back-to-life/
|   `-- deoldify/
|-- src/
|-- tests/
|-- requirements.txt
|-- requirements-boptl.txt
`-- README.md
```

Important external folders:

- `external/bringing-old-photos-back-to-life`
- `external/deoldify`

## Move To Another PC

The easiest way is:

1. Put this whole project folder on GitHub or copy it with a zip.
2. On the new PC, download or clone it.
3. Recreate the virtual environment.
4. Install the requirements.
5. Download the pretrained model repos and weights into `external/`.

You do **not** need to rewrite code or change paths if you keep the same folder names.

## Requirements

- Python 3.11 or newer
- Windows PowerShell
- internet access for installing packages and downloading pretrained models

## Quick Start

If you only want to open the app and test the basic interface:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
streamlit run .\app.py
```

This basic setup is enough for:

- the Streamlit app
- image upload
- preprocessing
- analysis
- passthrough mode

It is **not** enough for the Microsoft restoration model.

## Full Setup For Another PC

Use these steps if you want the real restoration model.

### 1. Clone or copy the repo

```powershell
git clone <your-repo-url>
cd photo-reviver
```

If you are not using Git, just copy the project folder and open PowerShell in that folder.

### 2. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3. Install Python packages

Install the app:

```powershell
python -m pip install -e .
```

Install the heavier model dependencies too:

```powershell
python -m pip install -r requirements-boptl.txt
```

### 4. Download the Microsoft restoration repo

Download the Microsoft `Bringing-Old-Photos-Back-to-Life` project and place it here:

```text
external/bringing-old-photos-back-to-life
```

After that, this file should exist:

```text
external/bringing-old-photos-back-to-life/run.py
```

You also need its pretrained assets. These paths must exist:

```text
external/bringing-old-photos-back-to-life/Face_Detection/shape_predictor_68_face_landmarks.dat
external/bringing-old-photos-back-to-life/Face_Enhancement/checkpoints/
external/bringing-old-photos-back-to-life/Global/checkpoints/
```

### 5. Optional: Download DeOldify

If you also want colorization, download DeOldify and place it here:

```text
external/deoldify
```

This file should exist:

```text
external/deoldify/deoldify/visualize.py
```

And this weight file should exist:

```text
external/deoldify/models/ColorizeArtistic_gen.pth
```

## Run The App

From the project root:

```powershell
streamlit run .\app.py
```

Then in the app:

1. upload a photo
2. choose the restoration engine in the sidebar
3. press `Fix Photo`
4. check the `Analysis` tab
5. check the `Restoration` tab
6. use the `Final Touches` tab if needed

## Which Backend To Choose

The app sidebar gives you two main choices:

- `Microsoft model (best quality, slower)`
- `Simple pipeline only (fast, no learned restoration)`

Use `Microsoft model` if the Microsoft repo and checkpoints are installed correctly.

If the model files are missing, the app will warn you in the sidebar.

## Scratch Detection In The App

The app can show two kinds of scratch detection:

- Microsoft pretrained scratch detector
- local OpenCV fallback detector

If the Microsoft backend is available, the app tries to use the Microsoft detector for the scratch preview in the `Analysis` tab.

If that detector cannot run, the app falls back to the local heuristic.

You can check the `Scratch detector` line in the `Analysis` tab to see which one was used.

## Run From The Command Line

Basic run:

```powershell
photo-reviver --input ".\samples\old_photo_03.png"
```

Run with the Microsoft backend config:

```powershell
photo-reviver --input ".\samples\old_photo_03.png" --config ".\configs\pipeline.boptl.json"
```

Run with a reference image:

```powershell
photo-reviver --input "C:\full\path\to\damaged_photo.jpg" --reference "C:\full\path\to\clean_reference.jpg"
```

## Output Files

Each run creates a folder inside:

```text
artifacts/runs/
```

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

Useful files:

- `02_analysis/scratch_mask.png`
- `02_analysis/scratch_overlay.png`
- `05_restoration/restored_model_output.png`
- `06_postprocess/final_restored.png`
- `06_postprocess/colorized_output.png`
- `07_evaluation/stage_comparison.png`

## Testing

Run tests with:

```powershell
python -m unittest discover -s tests
```

## Troubleshooting

### The Microsoft model does not appear in the app

Check that these exist:

- `external/bringing-old-photos-back-to-life/run.py`
- `external/bringing-old-photos-back-to-life/Face_Detection/shape_predictor_68_face_landmarks.dat`
- `external/bringing-old-photos-back-to-life/Face_Enhancement/checkpoints/`
- `external/bringing-old-photos-back-to-life/Global/checkpoints/`

### Colorization button is disabled

Check that these exist:

- `external/deoldify/deoldify/visualize.py`
- `external/deoldify/models/ColorizeArtistic_gen.pth`

### The app opens but restoration does not run

Make sure you installed both:

```powershell
python -m pip install -e .
python -m pip install -r requirements-boptl.txt
```

### I only want to move the code to another PC

You can move this repo without the `external/` model folders.

Later, on the new PC, just install requirements and download the pretrained model folders into:

- `external/bringing-old-photos-back-to-life`
- `external/deoldify`

## Notes

- final touches happen after the restoration output
- DeOldify is optional
- the Microsoft model is slower but gives the best restoration results in this project
