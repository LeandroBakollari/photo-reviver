import unittest

import numpy as np

from photo_reviver.evaluate import compute_reference_metrics


class EvaluateTests(unittest.TestCase):
    def test_identical_images_have_zero_error(self) -> None:
        image = np.full((32, 32), 128, dtype=np.uint8)
        metrics = compute_reference_metrics(image, image.copy())

        self.assertEqual(metrics.mae, 0.0)
        self.assertEqual(metrics.mse, 0.0)
        self.assertIsNone(metrics.psnr)


if __name__ == "__main__":
    unittest.main()
