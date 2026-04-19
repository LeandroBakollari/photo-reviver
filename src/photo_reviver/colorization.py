from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
import importlib.util
from pathlib import Path
import shutil
import sys

import numpy as np

from photo_reviver.io_utils import ensure_color, load_image, save_image


@dataclass
class ColorizationResult:
    output_image: np.ndarray
    applied: bool
    notes: list[str]


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
