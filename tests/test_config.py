import unittest

from photo_reviver.config import deep_merge


class ConfigTests(unittest.TestCase):
    def test_deep_merge_keeps_defaults(self) -> None:
        base = {"analysis": {"min_width": 64, "min_height": 64}, "restoration": {"backend": "passthrough"}}
        overrides = {"analysis": {"min_width": 128}}

        merged = deep_merge(base, overrides)

        self.assertEqual(merged["analysis"]["min_width"], 128)
        self.assertEqual(merged["analysis"]["min_height"], 64)
        self.assertEqual(merged["restoration"]["backend"], "passthrough")


if __name__ == "__main__":
    unittest.main()
