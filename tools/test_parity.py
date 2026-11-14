import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import soundfile as sf
from parity import compare_files


class SampleRateTest(unittest.TestCase):
    def test_identical_samples_require_matching_rates(self):
        audio = np.full((32, 2), 0.125, dtype=np.float32)
        for reference_rate, estimate_rate in [(44100, 44100), (44100, 48000), (48000, 44100), (48000, 48000)]:
            with (
                self.subTest(reference_rate=reference_rate, estimate_rate=estimate_rate),
                TemporaryDirectory() as directory,
            ):
                reference = Path(directory) / "reference.wav"
                estimate = Path(directory) / "estimate.wav"
                sf.write(reference, audio, reference_rate, subtype="FLOAT")
                sf.write(estimate, audio, estimate_rate, subtype="FLOAT")

                passed, summary = compare_files(reference, estimate)
                expected = reference_rate == estimate_rate
                self.assertEqual(passed, expected, summary)
                if not expected:
                    self.assertIn("sample rate", summary)


if __name__ == "__main__":
    unittest.main()
