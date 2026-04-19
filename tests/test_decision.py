import unittest
from pathlib import Path

from photo_reviver.decision import choose_restoration_mode
from photo_reviver.types import ImageAnalysis


def make_analysis(
    scratch_severity: str,
    needs_hr: bool,
    low_contrast: bool = False,
) -> ImageAnalysis:
    return ImageAnalysis(
        grayscale_path=Path("grayscale.png"),
        histogram_path=Path("histogram.png"),
        scratch_mask_path=Path("mask.png"),
        scratch_overlay_path=Path("overlay.png"),
        brightness_mean=100.0,
        brightness_std=20.0,
        dynamic_range=80,
        low_contrast=low_contrast,
        scratch_ratio=0.05,
        scratch_severity=scratch_severity,
        scratch_detection_method="test",
        face_detected=False,
        face_count=0,
        face_detection_method="test",
        needs_high_resolution_path=needs_hr,
        notes=[],
    )


class DecisionTests(unittest.TestCase):
    def test_selects_scratch_hr_for_hard_case(self) -> None:
        decision = choose_restoration_mode(make_analysis("high", True))
        self.assertEqual(decision.mode, "scratch+hr")

    def test_selects_scratch_when_scratches_are_medium(self) -> None:
        decision = choose_restoration_mode(make_analysis("medium", False))
        self.assertEqual(decision.mode, "scratch")

    def test_selects_normal_when_scratches_are_light(self) -> None:
        decision = choose_restoration_mode(make_analysis("low", False, low_contrast=True))
        self.assertEqual(decision.mode, "normal")
        self.assertTrue(any("contrast" in reason.lower() for reason in decision.reasons))


if __name__ == "__main__":
    unittest.main()
