from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "paths": {
        "output_root": "artifacts/runs",
    },
    "analysis": {
        "min_width": 64,
        "min_height": 64,
        "low_contrast_std_threshold": 38.0,
        "low_contrast_dynamic_range_threshold": 90,
        "small_image_longest_side_threshold": 1200,
        "scratch_ratio_thresholds": {
            "medium": 0.015,
            "high": 0.04,
        },
    },
    "preprocess": {
        "profile": "auto",
        "denoise_strength": 4,
        "denoise_blend": 0.3,
        "use_clahe": True,
        "clahe_clip_limit": 1.4,
        "clahe_strength": 0.3,
        "resize_longest_side": 1600,
        "normalize_intensity": False,
        "model_safe_denoise_strength": 0,
        "model_safe_use_clahe": False,
        "model_safe_resize_longest_side": 768,
    },
    "restoration": {
        "backend": "passthrough",
        "repo_root": "external/bringing-old-photos-back-to-life",
        "python_executable": "python",
        "gpu": "-1",
        "checkpoint_name": "Setting_9_epoch_100",
        "external_command": [
            "python",
            "external/microsoft-repo/run.py",
            "--input",
            "{input_path}",
            "--output",
            "{output_path}",
            "--mode",
            "{mode}",
        ],
    },
    "postprocess": {
        "apply_enhancement": False,
        "enhancement_strength": 0.12,
        "enhancement_clip_limit": 1.2,
        "apply_sharpening": False,
        "sharpening_strength": 0.08,
        "sharpen_sigma": 1.0,
        "simple_upscale_factor": 1.0,
        "attempt_colorization": False,
        "colorization": {
            "enabled": False,
            "only_if_input_grayscale": True,
            "repo_root": "external/deoldify",
            "weights_name": "ColorizeArtistic_gen",
            "artistic": True,
            "render_factor": 25,
            "post_process": True,
            "watermarked": False,
            "device": "cpu",
        },
    },
}


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | None = None) -> dict[str, Any]:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if not config_path:
        return config

    with Path(config_path).open("r", encoding="utf-8") as file:
        user_config = json.load(file)

    if not isinstance(user_config, dict):
        raise ValueError("Config file must contain a JSON object.")

    return deep_merge(config, user_config)


def apply_cli_overrides(
    config: dict[str, Any],
    output_root: str | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    if output_root:
        updated["paths"]["output_root"] = output_root
    if backend:
        updated["restoration"]["backend"] = backend
    return updated
