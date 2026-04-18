from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from photo_reviver.types import ImageValidation, RunPaths

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "-", value.lower()).strip("-")
    return cleaned or "image"


def build_run_paths(output_root: Path, image_name: str) -> RunPaths:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = output_root / f"{timestamp}_{slugify(image_name)}"

    paths = RunPaths(
        run_root=run_root,
        input_dir=run_root / "01_input",
        analysis_dir=run_root / "02_analysis",
        preprocess_dir=run_root / "03_preprocess",
        decision_dir=run_root / "04_decision",
        restoration_dir=run_root / "05_restoration",
        postprocess_dir=run_root / "06_postprocess",
        evaluation_dir=run_root / "07_evaluation",
    )

    for directory in (
        paths.run_root,
        paths.input_dir,
        paths.analysis_dir,
        paths.preprocess_dir,
        paths.decision_dir,
        paths.restoration_dir,
        paths.postprocess_dir,
        paths.evaluation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    return paths


def copy_input_file(source_path: Path, destination_dir: Path) -> Path:
    destination_dir.mkdir(parents=True, exist_ok=True)
    target_path = destination_dir / source_path.name
    shutil.copy2(source_path, target_path)
    return target_path


def load_image(path: Path) -> np.ndarray:
    raw_bytes = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw_bytes, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    return image


def save_image(path: Path, image: np.ndarray) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix or ".png"
    success, encoded = cv2.imencode(suffix, image)
    if not success:
        raise ValueError(f"Could not encode image for saving: {path}")

    encoded.tofile(str(path))
    return path


def image_is_grayscale(image: np.ndarray) -> bool:
    if image.ndim == 2:
        return True
    if image.ndim == 3 and image.shape[2] == 1:
        return True
    if image.ndim == 3 and image.shape[2] == 3:
        blue, green, red = cv2.split(image)
        return bool(np.array_equal(blue, green) and np.array_equal(green, red))
    return False


def validate_image(
    source_path: Path,
    copied_path: Path,
    image: np.ndarray,
    min_width: int,
    min_height: int,
) -> ImageValidation:
    extension = copied_path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image format: {extension}. "
            f"Use one of: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    height, width = image.shape[:2]
    if width < min_width or height < min_height:
        raise ValueError(
            f"Image is too small ({width}x{height}). "
            f"Minimum size is {min_width}x{min_height}."
        )

    channels = 1 if image.ndim == 2 else image.shape[2]
    return ImageValidation(
        source_path=source_path.resolve(),
        copied_path=copied_path.resolve(),
        image_format=extension.lstrip("."),
        width=width,
        height=height,
        channels=channels,
        is_grayscale=image_is_grayscale(image),
    )


def ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def json_ready(value: Any) -> Any:
    if is_dataclass(value):
        return json_ready(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def save_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(json_ready(payload), file, indent=2)
    return path
