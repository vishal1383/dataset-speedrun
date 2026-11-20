"""Build a reproducible random-subset starter submission."""

import argparse
from pathlib import Path
import time

import numpy as np

from competition.runner import ROOT, read_tokens, write
from train import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fraction", type=float, default=.75)
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--output", type=Path, default=Path("submissions/random75.json"))
    args = p.parse_args()
    if not 0 < args.fraction <= 1:
        raise ValueError("Fraction must be in (0,1].")
    start = time.perf_counter()
    original = ROOT / "data/tinystories-v0/original.npy"
    full = read_tokens(original).numpy().astype(np.uint16)
    ids = np.sort(np.random.default_rng(args.seed).choice(len(full), max(1, int(len(full) * args.fraction)), replace=False))
    artifact = ROOT / "runs/submissions" / (args.output.stem + ".npy")
    if artifact.exists() or args.output.exists():
        raise FileExistsError("Choose a new output name.")
    artifact.parent.mkdir(parents=True, exist_ok=True)
    np.save(artifact, full[ids])
    write(args.output, {"name": args.output.stem, "track": "data-only",
                       "artifact": str(artifact.relative_to(ROOT)), "artifact_sha256": file_hash(artifact),
                       "source": "TinyStories fixed 20,000-document training pool",
                       "license": "CDLA-Sharing-1.0", "construction": {"method": "uniform random windows",
                       "seed": args.seed, "fraction": args.fraction, "seconds": time.perf_counter() - start,
                       "teacher_compute": "none", "original_sha256": file_hash(original)}})
    print(args.output)


if __name__ == "__main__":
    main()
