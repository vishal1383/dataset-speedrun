"""Rebuild the exact finalist from its published training-window indices."""

import json
import argparse
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import file_hash, load_sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = json.loads((ROOT / "submissions/round2-influence75.json").read_text())
    ids = np.array(json.loads((ROOT / "research/round2/results/selected-window-ids.json").read_text()))
    full = load_sequences(ROOT / "data/tinystories-v0/original.npy", 256).numpy().astype(np.uint16)
    artifact = args.output or ROOT / spec["artifact"]
    artifact.parent.mkdir(parents=True, exist_ok=True)
    if not artifact.exists():
        np.save(artifact, full[ids])
    if file_hash(artifact) != spec["artifact_sha256"]:
        raise ValueError("Reconstructed artifact does not match the frozen submission.")
    print(f"Verified {artifact.stat().st_size} bytes: {spec['artifact_sha256']}")


if __name__ == "__main__":
    main()
