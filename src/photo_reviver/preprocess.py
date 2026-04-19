from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from photo_reviver.analysis import to_grayscale
from photo_reviver.io_utils import save_image, save_json
from photo_reviver.types import ImageAnalysis, PreprocessResult


MODEL_SAFE_BACKENDS = {"boptl"}


def resolve_preprocess_profile(preprocess_config: dict, backend: str) -> str:
    profile = preprocess_config.get("profile", "auto")
    if profile == "auto":
        return "model_safe" if backend in MODEL_SAFE_BACKENDS else "classic"
    return str(profile)


def blend_images(original: np.ndarray, adjusted: np.ndarray, strength: float) -> np.ndarray:
    clamped_strength = max(0.0, min(1.0, float(strength)))
    if clamped_strength <= 0.0:
        return original.copy()
    if clamped_strength >= 1.0:
        return adjusted
    return cv2.addWeighted(original, 1.0 - clamped_strength, adjusted, clamped_strength, 0.0)


def apply_clahe(
    image: np.ndarray,
    clip_limit: float,
    strength: float,
) -> np.ndarray:
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
        enhanced = clahe.apply(image)
        return blend_images(image, enhanced, strength)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(16, 16))
    enhanced_lightness = clahe.apply(lightness)
    merged = cv2.merge((enhanced_lightness, green_red, blue_yellow))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return blend_images(image, enhanced, strength)


def apply_light_denoise(
    image: np.ndarray,
    denoise_strength: int,
    blend: float,
) -> np.ndarray:
    if image.ndim == 2:
        denoised = cv2.fastNlMeansDenoising(image, None, denoise_strength, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoisingColored(
            image,
            None,
            denoise_strength,
            denoise_strength,
            7,
            21,
        )
    return blend_images(image, denoised, blend)


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
    analysis: ImageAnalysis,
    backend: str,
    preprocess_config: dict,
    stage_dir: Path,
) -> PreprocessResult:
    processed = image.copy()
    applied_steps: list[str] = []
    profile = resolve_preprocess_profile(preprocess_config, backend)

    denoise_strength = int(preprocess_config["denoise_strength"])
    resize_longest_side = preprocess_config.get("resize_longest_side")
    use_clahe = bool(preprocess_config["use_clahe"])

    if profile == "model_safe":
        denoise_strength = int(preprocess_config.get("model_safe_denoise_strength", 0))
        resize_longest_side = preprocess_config.get("model_safe_resize_longest_side")
        use_clahe = bool(preprocess_config.get("model_safe_use_clahe", False))
        applied_steps.append("model-safe preprocessing kept the image close to the original")

    if denoise_strength > 0:
        processed = apply_light_denoise(
            processed,
            denoise_strength=denoise_strength,
            blend=float(preprocess_config.get("denoise_blend", 0.3)),
        )
        applied_steps.append("mild denoise")

    if use_clahe and analysis.low_contrast:
        processed = apply_clahe(
            processed,
            clip_limit=float(preprocess_config.get("clahe_clip_limit", 1.4)),
            strength=float(preprocess_config.get("clahe_strength", 0.3)),
        )
        applied_steps.append("gentle contrast correction")

    if bool(preprocess_config["normalize_intensity"]):
        processed = cv2.normalize(processed, None, 0, 255, cv2.NORM_MINMAX)
        applied_steps.append("intensity normalization")

    processed, original_size = resize_to_longest_side(
        processed,
        resize_longest_side,
    )
    processed_size = (processed.shape[1], processed.shape[0])
    if processed_size != original_size:
        applied_steps.append("resized for stable processing")

    if processed.ndim == 3 and processed.shape[2] == 3:
        preview_gray = to_grayscale(processed)
        save_image(stage_dir / "preprocessed_grayscale.png", preview_gray)

    output_path = save_image(stage_dir / "preprocessed.png", processed)
    result = PreprocessResult(
        profile=profile,
        output_path=output_path.resolve(),
        applied_steps=applied_steps,
        original_size=original_size,
        processed_size=processed_size,
    )
    save_json(stage_dir / "preprocess.json", result)
    return result
