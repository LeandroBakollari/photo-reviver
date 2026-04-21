import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photo_reviver.postprocess import (
    apply_enhancement_controls,
    build_recommended_enhancement_settings,
    describe_enhancement_recommendation,
    measure_enhancement_inputs,
    postprocess_image,
)


class PostprocessTests(unittest.TestCase):
    def test_colorization_is_skipped_for_non_grayscale_input_when_required(self) -> None:
        image = np.full((10, 10, 3), 120, dtype=np.uint8)
        config = {
            "apply_enhancement": False,
            "apply_sharpening": False,
            "simple_upscale_factor": 1.0,
            "attempt_colorization": True,
            "colorization": {
                "enabled": True,
                "only_if_input_grayscale": True,
            },
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch("photo_reviver.postprocess.apply_deoldify_colorization") as mock_colorize:
                result = postprocess_image(
                    image=image,
                    postprocess_config=config,
                    stage_dir=Path(temp_dir),
                    original_is_grayscale=False,
                )

        mock_colorize.assert_not_called()
        self.assertIsNone(result.colorized_path)
        self.assertTrue(any("not grayscale" in step for step in result.skipped_steps))
        self.assertTrue(any("only runs when the original input is grayscale" in note for note in result.notes))

    def test_recommended_enhancement_settings_include_slider_keys(self) -> None:
        image = np.full((18, 18, 3), 80, dtype=np.uint8)

        settings = build_recommended_enhancement_settings(image)

        self.assertIn("brightness", settings)
        self.assertIn("contrast", settings)
        self.assertIn("sharpness", settings)

    def test_recommended_enhancement_settings_react_to_image_inputs(self) -> None:
        dark_muted = np.full((32, 32, 3), 45, dtype=np.uint8)
        bright_color = np.zeros((32, 32, 3), dtype=np.uint8)
        bright_color[:, :, 1] = 210
        bright_color[:, :, 2] = 240

        dark_settings = build_recommended_enhancement_settings(dark_muted)
        bright_settings = build_recommended_enhancement_settings(bright_color)

        self.assertGreater(dark_settings["brightness"], bright_settings["brightness"])
        self.assertGreater(dark_settings["vibrance"], bright_settings["vibrance"])

    def test_enhancement_recommendation_describes_measured_image(self) -> None:
        image = np.full((32, 32, 3), 45, dtype=np.uint8)

        metrics = measure_enhancement_inputs(image)
        summary = describe_enhancement_recommendation(image)

        self.assertLess(metrics["brightness"], 105)
        self.assertIn("dark exposure", summary)

    def test_enhancement_controls_preserve_image_shape(self) -> None:
        image = np.full((18, 18, 3), 120, dtype=np.uint8)
        settings = {
            "brightness": 8,
            "contrast": 10,
            "gamma": 5,
            "saturation": 12,
            "vibrance": 10,
            "warmth": 4,
            "tint": 0,
            "clarity": 8,
            "denoise": 5,
            "sharpness": 8,
        }

        result = apply_enhancement_controls(image, settings)

        self.assertEqual(result.shape, image.shape)
        self.assertEqual(result.dtype, image.dtype)


if __name__ == "__main__":
    unittest.main()
