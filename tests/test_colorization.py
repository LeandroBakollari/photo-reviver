import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photo_reviver.colorization import (
    apply_deoldify_colorization,
    colorization_assets_ready,
)
from photo_reviver.io_utils import save_image


class FakeColorizer:
    def plot_transformed_image(
        self,
        path: str,
        results_dir: Path,
        render_factor: int,
        compare: bool,
        post_process: bool,
        watermarked: bool,
    ) -> Path:
        del render_factor, compare, post_process, watermarked
        result_path = Path(results_dir) / Path(path).name
        save_image(result_path, np.full((12, 12, 3), 180, dtype=np.uint8))
        return result_path


class ColorizationTests(unittest.TestCase):
    def test_asset_readiness_checks_repo_and_weights(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready, missing = colorization_assets_ready(
                {
                    "repo_root": str(root / "missing-deoldify"),
                    "weights_name": "ColorizeArtistic_gen",
                }
            )

        self.assertFalse(ready)
        self.assertTrue(missing)

    def test_colorization_returns_note_when_assets_are_missing(self) -> None:
        image = np.full((16, 16, 3), 120, dtype=np.uint8)
        result = apply_deoldify_colorization(
            image=image,
            colorization_config={
                "repo_root": "missing",
                "weights_name": "ColorizeArtistic_gen",
            },
            stage_dir=Path(tempfile.gettempdir()),
        )

        self.assertFalse(result.applied)
        self.assertEqual(result.output_image.shape, image.shape)
        self.assertTrue(result.notes)

    def test_colorization_uses_fake_deoldify_runner(self) -> None:
        image = np.full((12, 12, 3), 90, dtype=np.uint8)
        config = {
            "repo_root": "external/deoldify",
            "weights_name": "ColorizeArtistic_gen",
            "artistic": True,
            "render_factor": 25,
            "post_process": True,
            "watermarked": False,
            "device": "cpu",
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "photo_reviver.colorization.colorization_assets_ready",
                return_value=(True, []),
            ):
                with patch(
                    "photo_reviver.colorization.load_deoldify_colorizer",
                    return_value=FakeColorizer(),
                ):
                    result = apply_deoldify_colorization(
                        image=image,
                        colorization_config=config,
                        stage_dir=Path(temp_dir),
                    )

        self.assertTrue(result.applied)
        self.assertEqual(result.output_image.shape, (12, 12, 3))
        self.assertTrue(any("DeOldify" in note for note in result.notes))


if __name__ == "__main__":
    unittest.main()
