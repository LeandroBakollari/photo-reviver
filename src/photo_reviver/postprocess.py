from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from photo_reviver.colorization import apply_deoldify_colorization
from photo_reviver.io_utils import save_image, save_json
from photo_reviver.types import PostprocessResult


def blend_images(original: np.ndarray, adjusted: np.ndarray, strength: float) -> np.ndarray:
    clamped = max(0.0, min(1.0, float(strength)))
    if clamped <= 0.0:
        return original.copy()
    if clamped >= 1.0:
        return adjusted
    return cv2.addWeighted(original, 1.0 - clamped, adjusted, clamped, 0.0)


def apply_light_enhancement(
    image: np.ndarray,
    strength: float,
    clip_limit: float,
) -> np.ndarray:
    if image.ndim == 2:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(12, 12))
        enhanced = clahe.apply(image)
        return blend_images(image, enhanced, strength)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(12, 12))
    enhanced_lightness = clahe.apply(lightness)
    merged = cv2.merge((enhanced_lightness, green_red, blue_yellow))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return blend_images(image, enhanced, strength)


def apply_unsharp_mask(image: np.ndarray, strength: float, sigma: float) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return blend_images(image, sharpened, strength)


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
    original_is_grayscale: bool = False,
) -> PostprocessResult:
    processed = image.copy()
    applied_steps: list[str] = []
    skipped_steps: list[str] = []
    notes: list[str] = []
    colorized_path: Path | None = None

    enhancement_strength = float(postprocess_config.get("enhancement_strength", 0.0))
    use_enhancement = bool(
        postprocess_config.get("apply_enhancement", enhancement_strength > 0.0)
    )
    if use_enhancement and enhancement_strength > 0.0:
        processed = apply_light_enhancement(
            processed,
            strength=enhancement_strength,
            clip_limit=float(postprocess_config.get("enhancement_clip_limit", 1.25)),
        )
        applied_steps.append(f"gentle enhancement ({enhancement_strength:.2f})")
    else:
        skipped_steps.append("Enhancement was disabled or set to zero strength.")

    sharpening_strength = float(postprocess_config.get("sharpening_strength", 0.0))
    use_sharpening = bool(
        postprocess_config.get("apply_sharpening", sharpening_strength > 0.0)
    )
    if use_sharpening and sharpening_strength > 0.0:
        processed = apply_unsharp_mask(
            processed,
            strength=sharpening_strength,
            sigma=float(postprocess_config.get("sharpen_sigma", 1.0)),
        )
        applied_steps.append(f"soft sharpening ({sharpening_strength:.2f})")
    else:
        skipped_steps.append("Sharpening was disabled or set to zero strength.")

    scale_factor = float(postprocess_config["simple_upscale_factor"])
    if scale_factor > 1.0:
        processed = upscale_image(processed, scale_factor)
        applied_steps.append(f"simple Lanczos upscale x{scale_factor}")
    else:
        skipped_steps.append("Upscaling was not requested.")

    colorization_config = postprocess_config.get("colorization", {})
    colorization_enabled = bool(
        colorization_config.get("enabled", False)
        or postprocess_config.get("attempt_colorization", False)
    )
    only_if_input_grayscale = bool(
        colorization_config.get("only_if_input_grayscale", True)
    )
    if colorization_enabled and only_if_input_grayscale and not original_is_grayscale:
        skipped_steps.append(
            "Colorization was skipped because the original input image was not grayscale."
        )
        notes.append(
            "DeOldify was enabled, but it only runs when the original input is grayscale."
        )
    elif colorization_enabled:
        colorization_result = apply_deoldify_colorization(
            processed,
            colorization_config,
            stage_dir,
        )
        processed = colorization_result.output_image
        notes.extend(colorization_result.notes)
        if colorization_result.applied:
            colorized_path = save_image(stage_dir / "colorized_output.png", processed).resolve()
            applied_steps.append("DeOldify colorization")
        else:
            skipped_steps.append("Colorization was requested but not applied.")
    else:
        skipped_steps.append("Colorization was disabled.")

    output_path = save_image(stage_dir / "final_restored.png", processed)
    result = PostprocessResult(
        output_path=output_path.resolve(),
        applied_steps=applied_steps,
        skipped_steps=skipped_steps,
        final_size=(processed.shape[1], processed.shape[0]),
        colorized_path=colorized_path,
        notes=notes,
    )
    save_json(stage_dir / "postprocess.json", result)
    return result
