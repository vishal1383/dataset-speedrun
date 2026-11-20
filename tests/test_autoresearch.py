"""Check compressed-artifact storage and decoder behavior."""

import io
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research/autoresearch"))
from engine import slice_artifact, factors, GradientDecoder


class CompressionInvariants(unittest.TestCase):
    def test_slicing_materializes_storage_and_preserves_endpoints(self):
        frame = {"weight": {"left": torch.ones(10, 8), "right": torch.ones(8, 10)}}
        source = {"knots": [1, 5, 10, 20], "frames": [frame] * 4,
                  "source_seed": 17, "initial_state_sha256": "example"}
        compressed = slice_artifact(source, rank=2, stride=2)
        self.assertEqual(compressed["knots"], [1, 10, 20])
        tensor = compressed["frames"][0]["weight"]["left"]
        self.assertEqual(tensor.untyped_storage().nbytes(), tensor.numel() * tensor.element_size())
        tensor.zero_()
        self.assertEqual(frame["weight"]["left"].sum().item(), 80)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA decoder check")
    def test_decoder_uses_known_factors_and_time_interpolation(self):
        model = torch.nn.Linear(3, 2, bias=False).cuda()
        a = {"weight": {"left": torch.ones(2, 1), "right": torch.ones(1, 3)}}
        b = {"weight": {"left": torch.ones(2, 1) * 3, "right": torch.ones(1, 3)}}
        decoder = GradientDecoder({"knots": [1, 3], "frames": [a, b]})
        decoder.apply(model, 2)
        self.assertTrue(torch.equal(model.weight.grad, torch.full_like(model.weight, 2)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA factorization check")
    def test_low_rank_reconstruction_and_rng_isolation(self):
        torch.manual_seed(88)
        matrix = torch.randn(32, 2, device="cuda") @ torch.randn(2, 24, device="cuda")
        state = torch.cuda.get_rng_state().clone()
        encoded = factors(matrix, 2)
        self.assertTrue(torch.equal(state, torch.cuda.get_rng_state()))
        decoded = encoded["left"].float().cuda() @ encoded["right"].float().cuda()
        self.assertLess(((matrix - decoded).norm() / matrix.norm()).item(), 0.002)


if __name__ == "__main__":
    unittest.main()
