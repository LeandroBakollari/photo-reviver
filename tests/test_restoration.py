import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from subprocess import CalledProcessError
from unittest.mock import patch

from photo_reviver.restoration import BoptlRestorationRunner, build_boptl_command


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

    def test_boptl_runner_uses_temp_paths_without_spaces_on_windows(self) -> None:
        with TemporaryDirectory(prefix="repo_root_") as repo_dir, TemporaryDirectory(
            prefix="stage dir "
        ) as temp_dir:
            input_path = Path(temp_dir) / "input.png"
            input_path.write_bytes(b"fake-image")
            stage_dir = Path(temp_dir) / "nested stage dir"

            captured_command: list[str] = []

            def fake_run(command, **kwargs):
                captured_command[:] = command
                raise CalledProcessError(
                    returncode=1,
                    cmd=command,
                    output="stdout",
                    stderr="stderr",
                )

            runner = BoptlRestorationRunner(repo_root=repo_dir)
            with patch("photo_reviver.restoration.subprocess.run", side_effect=fake_run):
                with self.assertRaises(RuntimeError) as ctx:
                    runner.run(input_path=input_path, mode="scratch", stage_dir=stage_dir)

            self.assertTrue(captured_command)
            self.assertTrue(stage_dir.exists())
            input_dir = Path(captured_command[captured_command.index("--input_folder") + 1])
            output_dir = Path(captured_command[captured_command.index("--output_folder") + 1])

            self.assertNotIn(" ", str(input_dir))
            self.assertNotIn(" ", str(output_dir))
            self.assertIn("boptl.log", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
