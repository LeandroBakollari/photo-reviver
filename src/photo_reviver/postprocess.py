from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from photo_reviver.io_utils import save_image, save_json
from photo_reviver.types import PostprocessResult


def apply_light_enhancement(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.convertScaleAbs(image, alpha=1.05, beta=4)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    merged = cv2.merge((lightness, green_red, blue_yellow))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def apply_unsharp_mask(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.2)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)


def upscale_image(image: np.ndarray, scale_factor: float) -> np.ndarray:
    if scale_factor <= 1.0:
        return image

    height, width = image.shape[:2]
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)


def postprocess_image(
    image: np.ndarray,
    postprocess_config: dict,
    stage_dir: Path,
) -> PostprocessResult:
    processed = image.copy()
    applied_steps: list[str] = []
    skipped_steps: list[str] = []

    if bool(postprocess_config["apply_enhancement"]):
        processed = apply_light_enhancement(processed)
        applied_steps.append("light enhancement")

    if bool(postprocess_config["apply_sharpening"]):
        processed = apply_unsharp_mask(processed)
        applied_steps.append("unsharp masking")

    scale_factor = float(postprocess_config["simple_upscale_factor"])
    if scale_factor > 1.0:
        processed = upscale_image(processed, scale_factor)
        applied_steps.append(f"simple Lanczos upscale x{scale_factor}")

    if bool(postprocess_config["attempt_colorization"]):
        skipped_steps.append("Colorization is reserved for a future model-based step.")

    output_path = save_image(stage_dir / "final_restored.png", processed)
    result = PostprocessResult(
        output_path=output_path.resolve(),
        applied_steps=applied_steps,
        skipped_steps=skipped_steps,
        final_size=(processed.shape[1], processed.shape[0]),
    )
    save_json(stage_dir / "postprocess.json", result)
    return result
