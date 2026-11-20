"""Check submission validation and paired statistical decisions."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from competition.runner import read_tokens
from competition.judge import paired_statistics, validate_submission
from train import file_hash


class SubmissionTests(unittest.TestCase):
    def test_stream_and_windows_have_identical_targets(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tokens.npy"
            np.save(p, np.arange(513, dtype=np.uint16))
            windows = read_tokens(p)
            self.assertEqual(windows.shape, (2, 257))
            self.assertEqual(windows[0, -1], windows[1, 0])
            self.assertEqual(windows[:, 1:].flatten().tolist(), list(range(1, 513)))
            np.save(p, windows.numpy().astype(np.uint16))
            self.assertTrue(np.array_equal(read_tokens(p), windows))

    def test_reject_malformed_artifacts_and_changed_payload(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tokens.npy"
            for values in [np.zeros((2, 256), np.uint16), np.zeros(513, np.float32),
                           np.full(513, 4096, np.uint16), np.zeros(0, np.uint16)]:
                np.save(p, values)
                with self.assertRaises(ValueError):
                    read_tokens(p)
            np.save(p, np.zeros(513, np.uint16))
            spec = Path(d) / "submission.json"
            spec.write_text(json.dumps({"name": "test", "track": "data-only", "artifact": str(p),
                                       "artifact_sha256": file_hash(p), "source": "test", "license": "test",
                                       "construction": {}}))
            validate_submission(spec)
            np.save(p, np.ones(513, np.uint16))
            with self.assertRaises(ValueError):
                validate_submission(spec)

    def test_paired_bound_detects_clear_gain_and_uncertainty(self):
        gain = paired_statistics([-.02, -.021, -.019, -.02, -.022, -.018, -.02, -.02])
        self.assertLess(gain["upper_95_one_sided"], 0)
        uncertain = paired_statistics([-.1, .1] * 4)
        self.assertGreater(uncertain["upper_95_one_sided"], .01)


if __name__ == "__main__":
    unittest.main()
