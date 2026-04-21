import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photo_reviver.postprocess import postprocess_image


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


if __name__ == "__main__":
    unittest.main()
