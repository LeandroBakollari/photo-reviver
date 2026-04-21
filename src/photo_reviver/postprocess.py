from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from photo_reviver.analysis import to_grayscale
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


def clamp_int(value: float, minimum: int, maximum: int) -> int:
    return int(max(minimum, min(maximum, round(value))))


def measure_enhancement_inputs(image: np.ndarray) -> dict[str, float]:
    color_image = image.copy()
    if color_image.ndim == 2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2BGR)

    gray_image = to_grayscale(color_image)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    lab_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2LAB)

    brightness = float(np.mean(gray_image))
    contrast = float(np.std(gray_image))
    saturation = float(np.mean(hsv_image[:, :, 1]))
    shadows = float(np.mean(gray_image < 24))
    highlights = float(np.mean(gray_image > 232))

    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    sharpness = float(laplacian.var())

    median = cv2.medianBlur(gray_image, 3)
    noise = float(np.std(gray_image.astype(np.float32) - median.astype(np.float32)))

    blue_mean, green_mean, red_mean = [
        float(np.mean(channel)) for channel in cv2.split(color_image)
    ]
    warmth_cast = red_mean - blue_mean
    tint_cast = green_mean - ((red_mean + blue_mean) / 2.0)
    lab_a = float(np.mean(lab_image[:, :, 1])) - 128.0
    lab_b = float(np.mean(lab_image[:, :, 2])) - 128.0

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "shadows": shadows,
        "highlights": highlights,
        "sharpness": sharpness,
        "noise": noise,
        "warmth_cast": warmth_cast,
        "tint_cast": tint_cast,
        "lab_a": lab_a,
        "lab_b": lab_b,
    }


def build_recommended_enhancement_settings(image: np.ndarray) -> dict[str, int]:
    metrics = measure_enhancement_inputs(image)

    brightness_delta = 128.0 - metrics["brightness"]
    brightness = clamp_int(brightness_delta * 0.22, -22, 24)
    gamma = clamp_int(brightness_delta * 0.17, -18, 22)

    if metrics["highlights"] > 0.08:
        brightness = min(brightness, -6)
        gamma = min(gamma, -6)
    if metrics["shadows"] > 0.18:
        gamma = max(gamma, 10)

    contrast_target = 58.0
    contrast = clamp_int((contrast_target - metrics["contrast"]) * 0.52 + 8, -6, 32)
    clarity = clamp_int((50.0 - metrics["contrast"]) * 0.28 + 10, 4, 28)

    saturation_target = 72.0
    saturation = clamp_int((saturation_target - metrics["saturation"]) * 0.28 + 4, -14, 28)
    vibrance = clamp_int((82.0 - metrics["saturation"]) * 0.35 + 6, -8, 34)

    warmth = clamp_int(-(metrics["warmth_cast"] * 0.16) - (metrics["lab_b"] * 0.18), -18, 18)
    tint = clamp_int(-(metrics["tint_cast"] * 0.22) - (metrics["lab_a"] * 0.12), -16, 16)

    denoise = clamp_int((metrics["noise"] - 3.5) * 4.0, 0, 32)
    if metrics["sharpness"] < 80:
        sharpness = clamp_int(24 - denoise * 0.25, 8, 28)
    elif metrics["sharpness"] < 220:
        sharpness = clamp_int(18 - denoise * 0.2, 6, 22)
    else:
        sharpness = clamp_int(10 - denoise * 0.1, 4, 14)

    if denoise > 20:
        clarity = max(4, clarity - 5)

    return {
        "brightness": brightness,
        "contrast": contrast,
        "gamma": gamma,
        "saturation": saturation,
        "vibrance": vibrance,
        "warmth": warmth,
        "tint": tint,
        "clarity": clarity,
        "denoise": denoise,
        "sharpness": sharpness,
    }


def describe_enhancement_recommendation(image: np.ndarray) -> str:
    metrics = measure_enhancement_inputs(image)
    notes: list[str] = []

    if metrics["brightness"] < 105:
        notes.append("dark exposure")
    elif metrics["brightness"] > 165:
        notes.append("bright exposure")
    else:
        notes.append("balanced exposure")

    if metrics["contrast"] < 42:
        notes.append("low contrast")
    elif metrics["contrast"] > 75:
        notes.append("strong contrast")

    if metrics["saturation"] < 45:
        notes.append("muted color")
    elif metrics["saturation"] > 95:
        notes.append("strong color")

    if metrics["noise"] > 8:
        notes.append("visible fine noise")

    if metrics["sharpness"] < 100:
        notes.append("soft details")

    if abs(metrics["warmth_cast"]) > 16 or abs(metrics["tint_cast"]) > 12:
        notes.append("color cast")

    return "Recommended from image analysis: " + ", ".join(notes) + "."


def apply_gamma_adjustment(image: np.ndarray, gamma_slider: float) -> np.ndarray:
    if gamma_slider == 0:
        return image
    gamma = 1.0 - (float(gamma_slider) / 150.0)
    gamma = max(0.25, min(2.5, gamma))
    table = np.array(
        [((index / 255.0) ** gamma) * 255 for index in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, table)


def apply_color_temperature(image: np.ndarray, warmth: float, tint: float) -> np.ndarray:
    adjusted = image.astype(np.float32)
    adjusted[:, :, 2] += float(warmth) * 0.65
    adjusted[:, :, 0] -= float(warmth) * 0.55

    adjusted[:, :, 1] -= float(tint) * 0.45
    adjusted[:, :, 2] += float(tint) * 0.28
    adjusted[:, :, 0] += float(tint) * 0.18
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_saturation_controls(
    image: np.ndarray,
    saturation: float,
    vibrance: float,
) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)

    if saturation:
        hsv_image[:, :, 1] *= 1.0 + (float(saturation) / 100.0)

    if vibrance:
        current_saturation = hsv_image[:, :, 1]
        low_saturation_weight = 1.0 - (current_saturation / 255.0)
        boost = float(vibrance) * 1.2 * low_saturation_weight
        hsv_image[:, :, 1] += boost

    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)


def apply_clarity(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, green_red, blue_yellow = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=1.0 + float(strength) / 35.0, tileGridSize=(8, 8))
    enhanced_lightness = clahe.apply(lightness)
    merged = cv2.merge((enhanced_lightness, green_red, blue_yellow))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return blend_images(image, enhanced, min(0.65, float(strength) / 100.0))


def apply_denoise_control(image: np.ndarray, strength: float) -> np.ndarray:
    if strength <= 0:
        return image
    denoise_strength = max(1, int(float(strength) / 5.0))
    denoised = cv2.fastNlMeansDenoisingColored(
        image,
        None,
        denoise_strength,
        denoise_strength,
        7,
        21,
    )
    return blend_images(image, denoised, min(0.75, float(strength) / 100.0))


def apply_enhancement_controls(image: np.ndarray, settings: dict) -> np.ndarray:
    processed = image.copy()
    if processed.ndim == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)

    processed = apply_denoise_control(
        processed,
        float(settings.get("denoise", 0)),
    )

    contrast = float(settings.get("contrast", 0))
    brightness = float(settings.get("brightness", 0))
    alpha = 1.0 + (contrast / 100.0)
    processed = np.clip(processed.astype(np.float32) * alpha + brightness, 0, 255).astype(np.uint8)

    processed = apply_gamma_adjustment(processed, float(settings.get("gamma", 0)))
    processed = apply_color_temperature(
        processed,
        warmth=float(settings.get("warmth", 0)),
        tint=float(settings.get("tint", 0)),
    )
    processed = apply_saturation_controls(
        processed,
        saturation=float(settings.get("saturation", 0)),
        vibrance=float(settings.get("vibrance", 0)),
    )
    processed = apply_clarity(processed, float(settings.get("clarity", 0)))

    sharpness = float(settings.get("sharpness", 0))
    if sharpness > 0:
        processed = apply_unsharp_mask(
            processed,
            strength=min(1.2, sharpness / 70.0),
            sigma=1.0,
        )

    return processed


def save_enhancement_result(
    image: np.ndarray,
    settings: dict,
    stage_dir: Path,
    output_name: str = "final_restored.png",
) -> Path:
    stage_dir.mkdir(parents=True, exist_ok=True)
    enhanced = apply_enhancement_controls(image, settings)
    output_path = save_image(stage_dir / output_name, enhanced).resolve()
    save_json(
        stage_dir / "enhancement_controls.json",
        {
            "output_path": output_path,
            "settings": settings,
        },
    )
    return output_path


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
