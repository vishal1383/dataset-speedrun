"""Construct a lexical coverage map from training windows only."""

import sys
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from competition.runner import read_tokens, write


def main():
    start = time.perf_counter()
    torch.set_num_threads(8)
    tokens = read_tokens(ROOT / "data/tinystories-v0/original.npy").cuda()
    # Fixed random projection of TF-IDF token counts keeps the map small.
    counts = torch.zeros(len(tokens), 4096, device="cuda")
    counts.scatter_add_(1, tokens[:, 1:], torch.ones_like(tokens[:, 1:], dtype=torch.float))
    idf = torch.log((len(tokens) + 1) / ((counts > 0).sum(0) + 1))
    counts = counts.sqrt() * idf
    torch.manual_seed(4912)
    features = counts @ torch.randn(4096, 128, device="cuda")
    features = torch.nn.functional.normalize(features, dim=1)
    centers = features[torch.randperm(len(features), device="cuda")[:128]].clone()
    for _ in range(30):
        labels = (features @ centers.T).argmax(1)
        sums = torch.zeros_like(centers).index_add_(0, labels, features)
        nonempty = sums.norm(dim=1) > 0
        centers[nonempty] = torch.nn.functional.normalize(sums[nonempty], dim=1)
    labels = (features @ centers.T).argmax(1).cpu().numpy()
    np.save(ROOT / "runs/round2/clusters.npy", labels)
    write(ROOT / "runs/round2/coverage-cost.json", {"seconds": time.perf_counter() - start,
          "method": "128 spherical clusters over a fixed 128-dimensional projection of sqrt-TF-IDF",
          "seed": 4912, "training_only": True})


if __name__ == "__main__":
    main()
