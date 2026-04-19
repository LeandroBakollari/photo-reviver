from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RunPaths:
    run_root: Path
    input_dir: Path
    analysis_dir: Path
    preprocess_dir: Path
    decision_dir: Path
    restoration_dir: Path
    postprocess_dir: Path
    evaluation_dir: Path


@dataclass
class ImageValidation:
    source_path: Path
    copied_path: Path
    image_format: str
    width: int
    height: int
    channels: int
    is_grayscale: bool


@dataclass
class ImageAnalysis:
    grayscale_path: Path
    histogram_path: Path
    scratch_mask_path: Path
    scratch_overlay_path: Path
    brightness_mean: float
    brightness_std: float
    dynamic_range: int
    low_contrast: bool
    scratch_ratio: float
    scratch_severity: str
    scratch_detection_method: str
    face_detected: bool
    face_count: int
    face_detection_method: str
    needs_high_resolution_path: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class PreprocessResult:
    profile: str
    output_path: Path
    applied_steps: list[str]
    original_size: tuple[int, int]
    processed_size: tuple[int, int]


@dataclass
class RestorationDecision:
    mode: str
    reasons: list[str]


@dataclass
class RestorationResult:
    output_path: Path
    backend: str
    notes: list[str]
    command: list[str] | None = None
    log_path: Path | None = None


@dataclass
class PostprocessResult:
    output_path: Path
    applied_steps: list[str]
    skipped_steps: list[str]
    final_size: tuple[int, int]
    colorized_path: Path | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class QualityMetrics:
    reference_path: Path | None
    mae: float | None
    mse: float | None
    psnr: float | None
    notes: list[str] = field(default_factory=list)


@dataclass
class EvaluationResult:
    comparison_path: Path
    metrics: QualityMetrics
