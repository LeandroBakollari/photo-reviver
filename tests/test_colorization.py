import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photo_reviver.colorization import (
    ColorizationResult,
    apply_deoldify_colorization,
    apply_deoldify_palette_colorization,
    apply_palette_colorization,
    apply_staged_colorization,
    available_palette_presets,
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

    def test_palette_colorization_provides_six_options(self) -> None:
        palettes = available_palette_presets()

        self.assertEqual(len(palettes), 6)
        self.assertTrue(all(item["key"] for item in palettes))

    def test_palette_colorization_creates_color_output(self) -> None:
        gray = np.tile(np.arange(16, dtype=np.uint8), (16, 1)) * 16

        result = apply_palette_colorization(gray, "vivid", intensity=1.0)

        self.assertTrue(result.applied)
        self.assertEqual(result.output_image.shape, (16, 16, 3))
        self.assertFalse(
            np.array_equal(result.output_image[:, :, 0], result.output_image[:, :, 2])
        )

    def test_deoldify_palette_colorization_grades_deoldify_output(self) -> None:
        image = np.full((12, 12, 3), 90, dtype=np.uint8)
        deoldify_output = np.full((12, 12, 3), 150, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "photo_reviver.colorization.apply_deoldify_colorization",
                return_value=ColorizationResult(
                    output_image=deoldify_output,
                    applied=True,
                    notes=["DeOldify colorization was applied."],
                ),
            ):
                result = apply_deoldify_palette_colorization(
                    image=image,
                    colorization_config={},
                    palette_key="cool",
                    intensity=0.75,
                    stage_dir=Path(temp_dir),
                )

        self.assertTrue(result.applied)
        self.assertEqual(result.output_image.shape, image.shape)
        self.assertTrue(any("after DeOldify" in note for note in result.notes))

    def test_staged_colorization_can_apply_palettes_before_and_after_model(self) -> None:
        image = np.full((12, 12, 3), 90, dtype=np.uint8)
        deoldify_output = np.full((12, 12, 3), 150, dtype=np.uint8)

        with tempfile.TemporaryDirectory() as temp_dir:
            with patch(
                "photo_reviver.colorization.apply_deoldify_colorization",
                return_value=ColorizationResult(
                    output_image=deoldify_output,
                    applied=True,
                    notes=["DeOldify colorization was applied."],
                ),
            ):
                result = apply_staged_colorization(
                    image=image,
                    colorization_config={},
                    stage_dir=Path(temp_dir),
                    use_deoldify=True,
                    before_palette_key="classic",
                    before_intensity=0.25,
                    after_palette_key="vivid",
                    after_intensity=0.75,
                )

        self.assertTrue(result.applied)
        self.assertTrue(result.used_deoldify)
        self.assertEqual(result.base_image.shape, image.shape)
        self.assertEqual(result.output_image.shape, image.shape)
        self.assertTrue(any("before DeOldify" in note for note in result.notes))
        self.assertTrue(any("after DeOldify" in note for note in result.notes))

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
