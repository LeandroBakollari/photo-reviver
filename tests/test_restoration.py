import unittest
from pathlib import Path

from photo_reviver.restoration import build_boptl_command


class RestorationTests(unittest.TestCase):
    def test_boptl_command_for_normal_mode(self) -> None:
        command = build_boptl_command(
            python_executable="python",
            checkpoint_name="Setting_9_epoch_100",
            input_dir=Path("in"),
            output_dir=Path("out"),
            gpu="-1",
            mode="normal",
        )

        self.assertNotIn("--with_scratch", command)
        self.assertNotIn("--HR", command)

    def test_boptl_command_for_scratch_hr_mode(self) -> None:
        command = build_boptl_command(
            python_executable="python",
            checkpoint_name="Setting_9_epoch_100",
            input_dir=Path("in"),
            output_dir=Path("out"),
            gpu="-1",
            mode="scratch+hr",
        )

        self.assertIn("--with_scratch", command)
        self.assertIn("--HR", command)


if __name__ == "__main__":
    unittest.main()
