# Photo Reviver

Photo Reviver is a local Python app for restoring old or damaged photos. It provides a Streamlit interface, a command-line entry point, and a repeatable stage-based pipeline for analysis, preprocessing, restoration, final touches, and output review.

The app supports:

- image upload and preview in Streamlit
- scratch, contrast, and image-quality analysis
- preprocessing before restoration
- a simple passthrough backend for testing the pipeline
- optional Microsoft `Bringing-Old-Photos-Back-to-Life` restoration
- optional DeOldify colorization
- final enhancement and sharpening controls
- stage-by-stage output files and a run summary

Pretrained model repositories and weights are not stored in this repository. They must be downloaded separately into `external/`.

## Requirements

- Python 3.11 or newer
- Windows PowerShell for the commands below
- Internet access for installing packages and downloading model assets

The basic app only needs the package dependencies in `pyproject.toml` or `requirements.txt`. The Microsoft restoration backend and DeOldify require the heavier dependencies in `requirements-boptl.txt`.

## Installation

From the project root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

This installs the app, the Streamlit interface, and the simple local pipeline.

If you plan to use the Microsoft restoration backend or DeOldify, also install:

```powershell
python -m pip install -r requirements-boptl.txt
```

Some packages in the model dependency set, especially Torch and dlib, may need platform-specific wheels depending on the machine and GPU setup.

## Run the App

Start the Streamlit interface:

```powershell
streamlit run .\app.py
```

In the app:

1. Upload a photo.
2. Choose the restoration engine in the sidebar.
3. Select `Fix Photo`.
4. Review the analysis, preprocessing, restoration, and final result tabs.
5. Use final touches for enhancement, sharpening, or optional colorization.

If the Microsoft model assets are missing, the app will show which required paths were not found. The simple pipeline remains available for testing the interface and output flow.

## Optional Model Setup

Photo Reviver looks for external model repositories under `external/`.

### Microsoft Restoration

Download Microsoft's `Bringing-Old-Photos-Back-to-Life` project and place it here:

```text
external/bringing-old-photos-back-to-life
```

The following paths must exist:

```text
external/bringing-old-photos-back-to-life/run.py
external/bringing-old-photos-back-to-life/Face_Detection/shape_predictor_68_face_landmarks.dat
external/bringing-old-photos-back-to-life/Face_Enhancement/checkpoints/
external/bringing-old-photos-back-to-life/Global/checkpoints/
```

Use this config when running from the command line with the Microsoft backend:

```text
configs/pipeline.boptl.json
```

### DeOldify Colorization

DeOldify is optional and is used only for colorization.

Place DeOldify here:

```text
external/deoldify
```

The following paths must exist:

```text
external/deoldify/deoldify/visualize.py
external/deoldify/models/ColorizeArtistic_gen.pth
```

Use this config when running from the command line with Microsoft restoration and DeOldify enabled:

```text
configs/pipeline.boptl.deoldify.json
```

By default, colorization runs only when the original input image is grayscale.

## Command Line Usage

Run the basic pipeline:

```powershell
photo-reviver --input "C:\path\to\old_photo.jpg"
```

Run with the Microsoft backend:

```powershell
photo-reviver --input "C:\path\to\old_photo.jpg" --config ".\configs\pipeline.boptl.json"
```

Run with Microsoft restoration and DeOldify colorization:

```powershell
photo-reviver --input "C:\path\to\old_photo.jpg" --config ".\configs\pipeline.boptl.deoldify.json"
```

Run with a reference image for comparison metrics:

```powershell
photo-reviver --input "C:\path\to\old_photo.jpg" --reference "C:\path\to\reference.jpg"
```

Useful CLI options:

- `--config`: load a JSON config file
- `--output-root`: choose where run folders are written
- `--backend`: override the configured backend with `passthrough`, `boptl`, or `external_command`

## Output Files

Each run creates a timestamped folder under:

```text
artifacts/runs/
```

Example structure:

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

Common files:

- `02_analysis/scratch_mask.png`
- `02_analysis/scratch_overlay.png`
- `05_restoration/restored_model_output.png`
- `06_postprocess/final_restored.png`
- `06_postprocess/colorized_output.png`
- `07_evaluation/stage_comparison.png`
- `run_summary.json`

## Configuration

Pipeline behavior is controlled with JSON config files in `configs/`.

- `pipeline.example.json`: basic example using the local passthrough backend
- `pipeline.boptl.json`: Microsoft restoration backend
- `pipeline.boptl.deoldify.json`: Microsoft restoration with DeOldify colorization enabled

Config files can adjust output paths, analysis thresholds, preprocessing options, restoration backend settings, and postprocessing behavior.

## Testing

Run the test suite with:

```powershell
python -m unittest discover -s tests
```

## Troubleshooting

### The Microsoft model is unavailable

Check that the Microsoft repository, face landmark file, and checkpoint folders are present under `external/bringing-old-photos-back-to-life`. Also confirm that the model dependencies from `requirements-boptl.txt` are installed in the active virtual environment.

### Colorization is disabled

Check that DeOldify is present under `external/deoldify` and that `external/deoldify/models/ColorizeArtistic_gen.pth` exists. The default config colorizes only grayscale source images.

### Restoration fails from the command line

Start with the passthrough backend to confirm the pipeline works:

```powershell
photo-reviver --input "C:\path\to\old_photo.jpg" --backend passthrough
```

Then rerun with the model config after the external model files and dependencies are in place.

## Notes

- External model repositories and pretrained weights are intentionally excluded from Git.
- Generated runs are written to `artifacts/runs/` and are not intended to be committed.
- Review the licenses and usage terms of the external Microsoft and DeOldify projects before distributing model assets or restored outputs.
