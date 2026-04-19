import tempfile
import unittest
from pathlib import Path

from photo_reviver.app_utils import backend_readiness, colorization_readiness, yes_no


class AppUtilsTests(unittest.TestCase):
    def test_yes_no(self) -> None:
        self.assertEqual(yes_no(True), "Yes")
        self.assertEqual(yes_no(False), "No")

    def test_backend_readiness_reports_missing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ready, missing = backend_readiness({"repo_root": str(Path(temp_dir) / "missing-repo")})

        self.assertFalse(ready)
        self.assertTrue(missing)

    def test_colorization_readiness_reports_missing_assets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            ready, missing = colorization_readiness(
                {
                    "colorization": {
                        "repo_root": str(root / "missing-deoldify"),
                        "weights_name": "ColorizeArtistic_gen",
                    }
                }
            )

        self.assertFalse(ready)
        self.assertTrue(missing)


if __name__ == "__main__":
    unittest.main()
