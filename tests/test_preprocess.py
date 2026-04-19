import tempfile
import unittest
from pathlib import Path

import numpy as np

from photo_reviver.preprocess import preprocess_image, resolve_preprocess_profile
from photo_reviver.types import ImageAnalysis


def make_analysis(low_contrast: bool = False) -> ImageAnalysis:
    return ImageAnalysis(
        grayscale_path=Path("grayscale.png"),
        histogram_path=Path("histogram.png"),
        scratch_mask_path=Path("mask.png"),
        scratch_overlay_path=Path("overlay.png"),
        brightness_mean=100.0,
        brightness_std=20.0,
        dynamic_range=80,
        low_contrast=low_contrast,
        scratch_ratio=0.01,
        scratch_severity="low",
        scratch_detection_method="test",
        face_detected=False,
        face_count=0,
        face_detection_method="test",
        needs_high_resolution_path=False,
        notes=[],
    )


class PreprocessTests(unittest.TestCase):
    def test_auto_profile_uses_model_safe_for_boptl(self) -> None:
        profile = resolve_preprocess_profile({"profile": "auto"}, "boptl")
        self.assertEqual(profile, "model_safe")

    def test_model_safe_profile_preserves_image_when_steps_are_disabled(self) -> None:
        image = np.full((24, 24, 3), 120, dtype=np.uint8)
        config = {
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
            "model_safe_resize_longest_side": None,
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            result = preprocess_image(
                image=image,
                analysis=make_analysis(low_contrast=True),
                backend="boptl",
                preprocess_config=config,
                stage_dir=Path(temp_dir),
            )

        self.assertEqual(result.profile, "model_safe")
        self.assertEqual(result.processed_size, (24, 24))
        self.assertIn("model-safe preprocessing kept the image close to the original", result.applied_steps)


if __name__ == "__main__":
    unittest.main()
