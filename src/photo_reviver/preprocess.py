from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from photo_reviver.analysis import to_grayscale
from photo_reviver.io_utils import save_image, save_json
from photo_reviver.types import PreprocessResult


def apply_clahe(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(image)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    merged = cv2.merge((lightness, green_red, blue_yellow))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def resize_to_longest_side(
    image: np.ndarray,
    longest_side: int | None,
) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = image.shape[:2]
    original_size = (width, height)

    if not longest_side or max(width, height) <= longest_side:
        return image, original_size

    scale = float(longest_side / max(width, height))
    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return resized, original_size


def preprocess_image(
    image: np.ndarray,
    preprocess_config: dict,
    stage_dir: Path,
) -> PreprocessResult:
    processed = image.copy()
    applied_steps: list[str] = []

    denoise_strength = int(preprocess_config["denoise_strength"])
    if denoise_strength > 0:
        if processed.ndim == 2:
            processed = cv2.fastNlMeansDenoising(processed, None, denoise_strength, 7, 21)
        else:
            processed = cv2.fastNlMeansDenoisingColored(
                processed,
                None,
                denoise_strength,
                denoise_strength,
                7,
                21,
            )
        applied_steps.append("mild denoise")

    if bool(preprocess_config["use_clahe"]):
        processed = apply_clahe(processed)
        applied_steps.append("contrast correction with CLAHE")

    if bool(preprocess_config["normalize_intensity"]):
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        applied_steps.append("intensity normalization")

    processed, original_size = resize_to_longest_side(
        processed,
        preprocess_config.get("resize_longest_side"),
    )
    processed_size = (processed.shape[1], processed.shape[0])
    if processed_size != original_size:
        applied_steps.append("resized for stable processing")

    if processed.ndim == 3 and processed.shape[2] == 3:
        preview_gray = to_grayscale(processed)
        save_image(stage_dir / "preprocessed_grayscale.png", preview_gray)

    output_path = save_image(stage_dir / "preprocessed.png", processed)
    result = PreprocessResult(
        output_path=output_path.resolve(),
        applied_steps=applied_steps,
        original_size=original_size,
        processed_size=processed_size,
    )
    save_json(stage_dir / "preprocess.json", result)
    return result
