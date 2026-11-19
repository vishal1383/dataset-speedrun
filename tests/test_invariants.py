"""Check the invariants that make dataset comparisons valid."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from prepare import select_recipes
from train import GPT, load_sequences, make_optimizers, state_hash


class Invariants(unittest.TestCase):
    def test_recipe_membership_and_reproducibility(self):
        docs = [list(range(2, 10 + i)) for i in range(100)]
        a, b = select_recipes(docs), select_recipes(docs)
        for name, ids in a.items():
            self.assertEqual(len(ids), 100 if name == "original" else 25)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(np.array_equal(ids, b[name]))
            self.assertTrue(np.all((ids >= 0) & (ids < 100)))

    def test_eval_targets_count_once(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tokens.npy"
            np.save(path, np.arange(18, dtype=np.uint16))
            sequences = load_sequences(path, length=4)
            self.assertEqual(sequences[:, 1:].flatten().tolist(), list(range(1, 17)))
            self.assertEqual(sequences[:, 0].tolist(), [0, 4, 8, 12])
            with self.assertRaises(ValueError):
                load_sequences(path, length=4, count=5)

    def test_initialization_optimizer_partition_and_causality(self):
        config = dict(vocab_size=32, width=16, heads=2, layers=2, sequence_length=8,
                      muon_lr=0.02, adamw_lr=0.001, weight_decay=0.01)
        torch.manual_seed(17)
        model = GPT(config)
        torch.manual_seed(17)
        other = GPT(config)
        self.assertEqual(state_hash(model), state_hash(other))
        optimizers = make_optimizers(model, config, "cpu")
        ids = [id(p) for o in optimizers for g in o.param_groups for p in g["params"]]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters()})
        logits = []
        hook = model.head.register_forward_hook(lambda m, x, y: logits.append(y.detach()))
        x = torch.randint(0, 32, (2, 8))
        y = x.clone()
        model(x, y)
        x[:, -1] = (x[:, -1] + 1) % 32
        loss = model(x, y)
        hook.remove()
        self.assertTrue(torch.equal(logits[0][:, :-1], logits[1][:, :-1]))
        initial = state_hash(model)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        self.assertNotEqual(state_hash(model), initial)


if __name__ == "__main__":
    unittest.main()
