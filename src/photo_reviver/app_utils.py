from __future__ import annotations

from pathlib import Path

from photo_reviver.colorization import colorization_assets_ready
from photo_reviver.types import (
    ImageAnalysis,
    ImageValidation,
    PostprocessResult,
    PreprocessResult,
    RestorationDecision,
    RestorationResult,
)


def yes_no(value: bool) -> str:
    return "Yes" if value else "No"


def backend_readiness(restoration_config: dict) -> tuple[bool, list[str]]:
    repo_root = Path(restoration_config.get("repo_root", "external/bringing-old-photos-back-to-life"))
    required_paths = [
        repo_root,
        repo_root / "run.py",
        repo_root / "Face_Detection" / "shape_predictor_68_face_landmarks.dat",
        repo_root / "Face_Enhancement" / "checkpoints",
        repo_root / "Global" / "checkpoints",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    return not missing, missing


def colorization_readiness(postprocess_config: dict) -> tuple[bool, list[str]]:
    return colorization_assets_ready(postprocess_config.get("colorization", {}))


def describe_validation(validation: ImageValidation) -> list[str]:
    return [
        f"Format: `{validation.image_format}`",
        f"Size: `{validation.width} x {validation.height}`",
        f"Channels: `{validation.channels}`",
        f"Grayscale already: `{yes_no(validation.is_grayscale)}`",
    ]


def describe_analysis(analysis: ImageAnalysis) -> list[str]:
    notes = [
        f"Low contrast detected: `{yes_no(analysis.low_contrast)}`",
        f"Scratch severity estimate: `{analysis.scratch_severity}`",
        f"Scratch ratio: `{analysis.scratch_ratio:.4f}`",
        f"Scratch detector: `{analysis.scratch_detection_method}`",
        f"Face detected: `{yes_no(analysis.face_detected)}`",
        f"Faces found: `{analysis.face_count}`",
        f"High-resolution path suggested: `{yes_no(analysis.needs_high_resolution_path)}`",
    ]
    notes.extend(analysis.notes)
    return notes


def describe_preprocess(preprocess: PreprocessResult) -> list[str]:
    lines = [
        f"Preprocess profile: `{preprocess.profile}`",
        f"Image size stayed at `{preprocess.processed_size[0]} x {preprocess.processed_size[1]}`"
        if preprocess.original_size == preprocess.processed_size
        else (
            f"Image was resized from `{preprocess.original_size[0]} x {preprocess.original_size[1]}` "
            f"to `{preprocess.processed_size[0]} x {preprocess.processed_size[1]}`"
        ),
    ]
    if preprocess.applied_steps:
        lines.append("Applied steps: " + ", ".join(preprocess.applied_steps))
    else:
        lines.append("No preprocessing adjustments were needed.")
    return lines


def describe_decision(decision: RestorationDecision) -> list[str]:
    lines = [f"Chosen restoration mode: `{decision.mode}`"]
    lines.extend(decision.reasons)
    return lines


def describe_restoration(restoration: RestorationResult) -> list[str]:
    lines = [f"Restoration backend: `{restoration.backend}`"]
    lines.extend(restoration.notes)
    if restoration.log_path:
        lines.append(f"Runtime log saved to `{restoration.log_path}`")
    return lines


def describe_postprocess(postprocess: PostprocessResult) -> list[str]:
    lines = []
    if postprocess.applied_steps:
        lines.append("Applied steps: " + ", ".join(postprocess.applied_steps))
    else:
        lines.append("No postprocessing adjustments were applied.")
    if postprocess.skipped_steps:
        lines.append("Skipped steps: " + ", ".join(postprocess.skipped_steps))
    lines.extend(postprocess.notes)
    if postprocess.colorized_path:
        lines.append("A DeOldify colorized image was generated.")
    lines.append(f"Final size: `{postprocess.final_size[0]} x {postprocess.final_size[1]}`")
    return lines
