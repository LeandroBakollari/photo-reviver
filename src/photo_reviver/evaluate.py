from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from photo_reviver.analysis import to_grayscale
from photo_reviver.io_utils import ensure_color, load_image, save_image, save_json
from photo_reviver.types import EvaluationResult, QualityMetrics


def fit_to_canvas(image: np.ndarray, width: int, height: int) -> np.ndarray:
    display = ensure_color(image)
    source_height, source_width = display.shape[:2]
    scale = min(width / source_width, height / source_height)
    scaled_width = max(1, int(source_width * scale))
    scaled_height = max(1, int(source_height * scale))

    resized = cv2.resize(display, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    x_offset = (width - scaled_width) // 2
    y_offset = (height - scaled_height) // 2
    canvas[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized
    return canvas


def build_panel(image: np.ndarray, label: str, width: int = 420, height: int = 300) -> np.ndarray:
    label_height = 42
    panel = np.full((height + label_height, width, 3), 255, dtype=np.uint8)
    panel[label_height:] = fit_to_canvas(image, width, height)
    cv2.putText(
        panel,
        label,
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (40, 40, 40),
        2,
        cv2.LINE_AA,
    )
    return panel


def create_comparison_grid(stage_images: list[tuple[str, np.ndarray]]) -> np.ndarray:
    panels = [build_panel(image, label) for label, image in stage_images]
    if len(panels) % 2 == 1:
        panels.append(np.full_like(panels[0], 255))

    rows = []
    for index in range(0, len(panels), 2):
        rows.append(np.hstack([panels[index], panels[index + 1]]))
    return np.vstack(rows)


def compute_reference_metrics(
    result_image: np.ndarray,
    reference_image: np.ndarray,
) -> QualityMetrics:
    notes: list[str] = []
    if reference_image.shape[:2] != result_image.shape[:2]:
        reference_image = cv2.resize(
            reference_image,
            (result_image.shape[1], result_image.shape[0]),
            interpolation=cv2.INTER_AREA,
        )
        notes.append("Reference image was resized to match the final output size.")

    result_gray = to_grayscale(result_image).astype(np.float32)
    reference_gray = to_grayscale(reference_image).astype(np.float32)
    difference = result_gray - reference_gray

    mse = float(np.mean(np.square(difference)))
    mae = float(np.mean(np.abs(difference)))
    psnr = None if mse == 0 else float(20 * math.log10(255.0 / math.sqrt(mse)))

    return QualityMetrics(
        reference_path=None,
        mae=mae,
        mse=mse,
        psnr=psnr,
        notes=notes,
    )


def evaluate_result(
    original_image: np.ndarray,
    preprocessed_image: np.ndarray,
    final_image: np.ndarray,
    stage_dir: Path,
    reference_path: Path | None = None,
) -> EvaluationResult:
    stage_images = [
        ("Original", original_image),
        ("Grayscale", to_grayscale(original_image)),
        ("Preprocessed", preprocessed_image),
        ("Final Output", final_image),
    ]

    comparison = create_comparison_grid(stage_images)
    comparison_path = save_image(stage_dir / "stage_comparison.png", comparison)

    if reference_path:
        reference_image = load_image(reference_path)
        metrics = compute_reference_metrics(final_image, reference_image)
        metrics.reference_path = reference_path.resolve()
    else:
        metrics = QualityMetrics(
            reference_path=None,
            mae=None,
            mse=None,
            psnr=None,
            notes=["No reference image was provided, so quality metrics were skipped."],
        )

    result = EvaluationResult(
        comparison_path=comparison_path.resolve(),
        metrics=metrics,
    )
    save_json(stage_dir / "evaluation.json", result)
    return result
