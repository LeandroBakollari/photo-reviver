from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import shutil
import sys

import cv2
import numpy as np

from photo_reviver.analysis import to_grayscale
from photo_reviver.io_utils import ensure_color, load_image, save_image, save_json


@dataclass
class ColorizationResult:
    output_image: np.ndarray
    applied: bool
    notes: list[str]


@dataclass
class StagedColorizationResult:
    output_image: np.ndarray
    pre_model_image: np.ndarray
    base_image: np.ndarray
    used_deoldify: bool
    applied: bool
    notes: list[str]


PALETTE_PRESETS: dict[str, dict] = {
    "classic": {
        "name": "Classic warm",
        "description": "Warm skin tones and slightly aged highlights.",
        "control_points": [(35, 28, 22), (118, 92, 72), (224, 214, 194)],
        "saturation": 0.88,
        "contrast": 1.03,
    },
    "cool": {
        "name": "Cool restored",
        "description": "Colder blues and cleaner neutral highlights.",
        "control_points": [(38, 34, 34), (132, 111, 92), (236, 226, 208)],
        "saturation": 0.78,
        "contrast": 1.01,
    },
    "vivid": {
        "name": "More colorful",
        "description": "Stronger color separation for lively portraits.",
        "control_points": [(28, 26, 36), (96, 122, 132), (236, 226, 196)],
        "saturation": 1.18,
        "contrast": 1.06,
    },
    "sepia": {
        "name": "Sepia archive",
        "description": "Brown archival tone for antique photos.",
        "control_points": [(28, 22, 16), (100, 82, 54), (215, 200, 160)],
        "saturation": 0.72,
        "contrast": 1.04,
    },
    "pastel": {
        "name": "Soft pastel",
        "description": "Gentler colors with lifted highlights.",
        "control_points": [(55, 48, 58), (145, 128, 110), (242, 229, 214)],
        "saturation": 0.62,
        "contrast": 0.98,
    },
    "film": {
        "name": "Natural film",
        "description": "Balanced colors with a subtle film print feel.",
        "control_points": [(30, 32, 30), (112, 104, 88), (230, 222, 198)],
        "saturation": 0.95,
        "contrast": 1.02,
    },
}


def available_palette_presets() -> list[dict[str, str]]:
    return [
        {
            "key": key,
            "name": preset["name"],
            "description": preset["description"],
        }
        for key, preset in PALETTE_PRESETS.items()
    ]


def recommend_palette_key(image: np.ndarray) -> str:
    gray_image = to_grayscale(image)
    brightness = float(np.mean(gray_image))
    contrast = float(np.std(gray_image))

    color_image = ensure_color(image)
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
    saturation = float(np.mean(hsv_image[:, :, 1]))

    if contrast < 34:
        return "vivid"
    if brightness < 88:
        return "classic"
    if saturation > 70:
        return "film"
    if brightness > 178:
        return "pastel"
    return "classic"


def _apply_palette_saturation(image: np.ndarray, multiplier: float) -> np.ndarray:
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1] * multiplier, 0, 255)
    return cv2.cvtColor(hsv_image.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _apply_palette_contrast(image: np.ndarray, multiplier: float) -> np.ndarray:
    adjusted = (image.astype(np.float32) - 128.0) * multiplier + 128.0
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_palette_colorization(
    image: np.ndarray,
    palette_key: str,
    intensity: float = 0.85,
) -> ColorizationResult:
    if palette_key not in PALETTE_PRESETS:
        raise ValueError(f"Unsupported color palette: {palette_key}")

    preset = PALETTE_PRESETS[palette_key]
    gray_image = to_grayscale(image)
    control_values = np.array([0, 128, 255], dtype=np.float32)
    control_colors = np.array(preset["control_points"], dtype=np.float32)

    channels = [
        np.interp(gray_image, control_values, control_colors[:, channel_index])
        for channel_index in range(3)
    ]
    colorized = np.dstack(channels).astype(np.uint8)
    colorized = _apply_palette_saturation(colorized, float(preset["saturation"]))
    colorized = _apply_palette_contrast(colorized, float(preset["contrast"]))

    base_image = ensure_color(image)
    clamped_intensity = max(0.0, min(1.0, float(intensity)))
    blended = cv2.addWeighted(
        base_image,
        1.0 - clamped_intensity,
        colorized,
        clamped_intensity,
        0.0,
    )

    return ColorizationResult(
        output_image=blended,
        applied=True,
        notes=[
            f"Palette colorization used '{preset['name']}' at {clamped_intensity:.2f} intensity.",
        ],
    )


def save_palette_colorization(
    image: np.ndarray,
    palette_key: str,
    intensity: float,
    stage_dir: Path,
    output_name: str = "colorized_output.png",
) -> ColorizationResult:
    stage_dir.mkdir(parents=True, exist_ok=True)
    result = apply_palette_colorization(image, palette_key, intensity)
    output_path = save_image(stage_dir / output_name, result.output_image).resolve()
    save_json(
        stage_dir / "palette_colorization.json",
        {
            "output_path": output_path,
            "palette_key": palette_key,
            "palette_name": PALETTE_PRESETS[palette_key]["name"],
            "intensity": intensity,
            "notes": result.notes,
        },
    )
    return result


def apply_deoldify_palette_colorization(
    image: np.ndarray,
    colorization_config: dict,
    palette_key: str,
    intensity: float,
    stage_dir: Path,
) -> ColorizationResult:
    deoldify_result = apply_deoldify_colorization(
        image=image,
        colorization_config=colorization_config,
        stage_dir=stage_dir,
    )
    base_image = deoldify_result.output_image if deoldify_result.applied else image
    palette_result = apply_palette_colorization(
        base_image,
        palette_key=palette_key,
        intensity=intensity,
    )

    notes = [*deoldify_result.notes]
    if deoldify_result.applied:
        notes.append("The selected palette was applied as a color grade after DeOldify.")
    else:
        notes.append("The app fell back to fast palette colorization.")
    notes.extend(palette_result.notes)

    return ColorizationResult(
        output_image=palette_result.output_image,
        applied=palette_result.applied,
        notes=notes,
    )


def apply_staged_colorization(
    image: np.ndarray,
    colorization_config: dict,
    stage_dir: Path,
    use_deoldify: bool,
    before_palette_key: str | None = None,
    before_intensity: float = 0.35,
    after_palette_key: str | None = None,
    after_intensity: float = 0.85,
) -> StagedColorizationResult:
    processed = ensure_color(image)
    notes: list[str] = []
    applied = False
    used_deoldify = False

    if before_palette_key:
        before_result = apply_palette_colorization(
            processed,
            palette_key=before_palette_key,
            intensity=before_intensity,
        )
        processed = before_result.output_image
        notes.append("A palette was applied before DeOldify.")
        notes.extend(before_result.notes)
        applied = True

    pre_model_image = processed.copy()

    if use_deoldify:
        deoldify_result = apply_deoldify_colorization(
            image=processed,
            colorization_config=colorization_config,
            stage_dir=stage_dir,
        )
        notes.extend(deoldify_result.notes)
        if deoldify_result.applied:
            processed = deoldify_result.output_image
            used_deoldify = True
            applied = True
        else:
            notes.append("DeOldify was unavailable, so the app kept the pre-model image.")

    base_image = processed.copy()

    if after_palette_key:
        after_result = apply_palette_colorization(
            processed,
            palette_key=after_palette_key,
            intensity=after_intensity,
        )
        processed = after_result.output_image
        notes.append("A palette was applied after DeOldify.")
        notes.extend(after_result.notes)
        applied = True

    if not applied:
        notes.append("No colorization step was applied.")

    return StagedColorizationResult(
        output_image=processed,
        pre_model_image=pre_model_image,
        base_image=base_image,
        used_deoldify=used_deoldify,
        applied=applied,
        notes=notes,
    )


def colorization_assets_ready(colorization_config: dict) -> tuple[bool, list[str]]:
    repo_root = Path(colorization_config.get("repo_root", "external/deoldify"))
    weights_name = str(colorization_config.get("weights_name", "ColorizeArtistic_gen"))

    required_paths = [
        repo_root,
        repo_root / "deoldify" / "visualize.py",
        repo_root / "models" / f"{weights_name}.pth",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]

    dependency_names = {
        "torch": "torch",
        "torchvision": "torchvision",
        "fastprogress": "fastprogress",
        "ffmpeg": "ffmpeg-python",
        "yt_dlp": "yt-dlp",
        "IPython": "IPython",
    }
    for module_name, package_name in dependency_names.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append(f"python package: {package_name}")

    return not missing, missing


def ensure_deoldify_on_path(repo_root: Path) -> None:
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


@contextmanager
def patched_torch_load():
    import torch

    original_load = torch.load

    def patched_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = patched_load
    try:
        yield
    finally:
        torch.load = original_load


def resolve_deoldify_device(device_name: str, device_id_enum):
    normalized = str(device_name).strip().lower()
    if normalized == "cpu":
        return device_id_enum.CPU
    if normalized.startswith("gpu"):
        gpu_index = normalized.replace("gpu", "", 1)
        return getattr(device_id_enum, f"GPU{gpu_index}")
    raise ValueError(f"Unsupported DeOldify device value: {device_name}")


@lru_cache(maxsize=4)
def load_deoldify_colorizer(
    repo_root_str: str,
    artistic: bool,
    weights_name: str,
    render_factor: int,
    device_name: str,
):
    repo_root = Path(repo_root_str).resolve()
    ensure_deoldify_on_path(repo_root)

    with patched_torch_load():
        from deoldify import device
        from deoldify.device_id import DeviceId
        from deoldify.visualize import (
            get_artistic_image_colorizer,
            get_stable_image_colorizer,
        )

        device.set(resolve_deoldify_device(device_name, DeviceId))
        if artistic:
            return get_artistic_image_colorizer(
                root_folder=repo_root,
                weights_name=weights_name,
                render_factor=render_factor,
            )
        return get_stable_image_colorizer(
            root_folder=repo_root,
            weights_name=weights_name,
            render_factor=render_factor,
        )


def apply_deoldify_colorization(
    image: np.ndarray,
    colorization_config: dict,
    stage_dir: Path,
) -> ColorizationResult:
    ready, missing = colorization_assets_ready(colorization_config)
    if not ready:
        return ColorizationResult(
            output_image=image,
            applied=False,
            notes=[
                "DeOldify colorization was requested but the repo or dependencies were missing.",
                *missing,
            ],
        )

    repo_root = Path(colorization_config.get("repo_root", "external/deoldify")).resolve()
    artistic = bool(colorization_config.get("artistic", True))
    weights_name = str(colorization_config.get("weights_name", "ColorizeArtistic_gen"))
    render_factor = int(colorization_config.get("render_factor", 25))
    device_name = str(colorization_config.get("device", "cpu"))
    post_process = bool(colorization_config.get("post_process", True))
    watermarked = bool(colorization_config.get("watermarked", False))

    temp_input_path = stage_dir / "_deoldify_input.png"
    temp_results_dir = stage_dir / "_deoldify_output"
    temp_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        save_image(temp_input_path, ensure_color(image))
        colorizer = load_deoldify_colorizer(
            str(repo_root),
            artistic=artistic,
            weights_name=weights_name,
            render_factor=render_factor,
            device_name=device_name,
        )
        result_path = colorizer.plot_transformed_image(
            path=str(temp_input_path),
            results_dir=temp_results_dir,
            render_factor=render_factor,
            compare=False,
            post_process=post_process,
            watermarked=watermarked,
        )
        output_image = load_image(Path(result_path))
    except Exception as error:
        return ColorizationResult(
            output_image=image,
            applied=False,
            notes=[f"DeOldify colorization failed: {error}"],
        )
    finally:
        temp_input_path.unlink(missing_ok=True)
        shutil.rmtree(temp_results_dir, ignore_errors=True)

    return ColorizationResult(
        output_image=output_image,
        applied=True,
        notes=[
            f"DeOldify colorization was applied with render factor {render_factor}.",
        ],
    )
