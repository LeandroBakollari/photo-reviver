import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from photo_reviver.analysis import analyze_image
from photo_reviver.types import ImageValidation


def make_validation() -> ImageValidation:
    return ImageValidation(
        source_path=Path("source.png"),
        copied_path=Path("copied.png"),
        image_format="png",
        width=128,
        height=128,
        channels=3,
        is_grayscale=False,
    )


def make_analysis_config() -> dict:
    return {
        "low_contrast_std_threshold": 38.0,
        "low_contrast_dynamic_range_threshold": 90,
        "small_image_longest_side_threshold": 1200,
        "scratch_ratio_thresholds": {
            "medium": 0.015,
            "high": 0.04,
        },
    }


class AnalysisTests(unittest.TestCase):
    def test_uses_microsoft_detector_when_boptl_backend_is_ready(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        validation = make_validation()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_dir = Path(temp_dir)
            mask = np.zeros((128, 128), dtype=np.uint8)
            mask[:, :10] = 255
            with patch(
                "photo_reviver.analysis.estimate_scratch_severity_with_boptl",
                return_value=(0.08, "high", mask, "Microsoft U-Net scratch detector"),
            ) as microsoft_detector, patch(
                "photo_reviver.analysis.estimate_scratch_severity"
            ) as heuristic_detector:
                analysis = analyze_image(
                    image=image,
                    validation=validation,
                    analysis_config=make_analysis_config(),
                    stage_dir=stage_dir,
                    restoration_config={
                        "backend": "boptl",
                        "repo_root": temp_dir,
                        "python_executable": "python",
                        "gpu": "-1",
                    },
                )

        microsoft_detector.assert_called_once()
        heuristic_detector.assert_not_called()
        self.assertEqual(analysis.scratch_detection_method, "Microsoft U-Net scratch detector")
        self.assertEqual(analysis.scratch_severity, "high")

    def test_falls_back_to_heuristic_when_microsoft_detector_fails(self) -> None:
        image = np.zeros((128, 128, 3), dtype=np.uint8)
        validation = make_validation()
        with tempfile.TemporaryDirectory() as temp_dir:
            stage_dir = Path(temp_dir)
            mask = np.zeros((128, 128), dtype=np.uint8)
            with patch(
                "photo_reviver.analysis.estimate_scratch_severity_with_boptl",
                side_effect=RuntimeError("detector failed"),
            ), patch(
                "photo_reviver.analysis.estimate_scratch_severity",
                return_value=(0.0, "low", mask, "blackhat threshold heuristic"),
            ) as heuristic_detector:
                analysis = analyze_image(
                    image=image,
                    validation=validation,
                    analysis_config=make_analysis_config(),
                    stage_dir=stage_dir,
                    restoration_config={
                        "backend": "boptl",
                        "repo_root": temp_dir,
                        "python_executable": "python",
                        "gpu": "-1",
                    },
                )

        heuristic_detector.assert_called_once()
        self.assertEqual(analysis.scratch_detection_method, "blackhat threshold heuristic")
        self.assertTrue(
            any("fell back" in note.lower() for note in analysis.notes),
            analysis.notes,
        )


if __name__ == "__main__":
    unittest.main()
