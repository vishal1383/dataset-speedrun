"""Remove strongly repetitive training documents without consulting validation data."""

import argparse
import json
from pathlib import Path
import time

import numpy as np

try:
    from .prepare import ROOT, inspect_shard
except ImportError:
    from prepare import ROOT, inspect_shard


class ShardWriter:
    def __init__(self, directory, capacity=100_000_000):
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=False)
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=np.uint16)
        self.used = 0
        self.files = {}

    def append(self, tokens):
        while len(tokens):
            count = min(len(tokens), self.capacity - self.used)
            self.buffer[self.used:self.used + count] = tokens[:count]
            self.used += count
            tokens = tokens[count:]
            if self.used == self.capacity:
                self.flush()

    def flush(self):
        if not self.used:
            return
        path = self.directory / f"fineweb_train_{len(self.files)+1:06d}.bin"
        header = np.zeros(256, dtype=np.int32)
        header[:3] = [20240520, 1, self.used]
        with path.open("wb") as f:
            header.tofile(f)
            self.buffer[:self.used].tofile(f)
        self.files[path.name] = inspect_shard(path)
        self.used = 0


def repetitive(tokens):
    """Flag obvious repetition; preserve normal short documents and document boundaries."""
    body = tokens[1:] if len(tokens) and tokens[0] == 50256 else tokens
    if len(body) < 128:
        return False
    adjacent = float(np.count_nonzero(body[1:] == body[:-1])) / (len(body) - 1)
    # Test unique-token fraction on a fixed-size prefix to avoid a document-length bias.
    prefix = body[:128]
    unique = len(np.unique(prefix)) / len(prefix)
    return adjacent > 0.20 or unique < 0.10


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("runs/fineweb-v1/repetition-filter"))
    args = p.parse_args()
    start = time.perf_counter()
    manifest = json.loads((ROOT / "data/fineweb10B/manifest.json").read_text())
    names = [f"fineweb_train_{i:06d}.bin" for i in range(1, 21)]
    if any(n not in manifest["files"] for n in names):
        raise ValueError("Download and verify all 20 pinned training shards first.")
    writer = ShardWriter(args.output)
    carry = np.empty(0, np.uint16)
    kept_docs = removed_docs = kept_tokens = removed_tokens = 0

    def emit(doc):
        nonlocal kept_docs, removed_docs, kept_tokens, removed_tokens
        if repetitive(doc):
            removed_docs += 1
            removed_tokens += len(doc)
        else:
            writer.append(doc)
            kept_docs += 1
            kept_tokens += len(doc)

    for name in names:
        path = ROOT / "data/fineweb10B" / name
        if inspect_shard(path) != manifest["files"][name]:
            raise ValueError(f"Source shard changed: {name}")
        values = np.memmap(path, mode="r", dtype=np.uint16, offset=1024)
        starts = np.flatnonzero(values == 50256)
        if not len(starts):
            carry = np.concatenate((carry, values))
            continue
        prefix = np.concatenate((carry, values[:starts[0]]))
        if len(prefix):
            emit(prefix)
        for lo, hi in zip(starts[:-1], starts[1:]):
            emit(values[lo:hi])
        carry = np.array(values[starts[-1]:], copy=True)
        print(json.dumps({"source": name, "kept_tokens": kept_tokens, "removed_tokens": removed_tokens}), flush=True)
    if len(carry):
        emit(carry)
    writer.flush()
    assert kept_tokens + removed_tokens == sum(manifest["files"][n]["tokens"] for n in names)
    # The unchanged upstream loader drops a short tail at each shard boundary.
    available_updates = sum((v["tokens"] - 2) // 524288 for v in writer.files.values())
    result = {"name": "repetition-filter", "construction": "Train-only document filtering; no teacher or validation access.",
              "source_dataset": manifest["dataset"], "source_revision": manifest["revision"],
              "source_files": {n: manifest["files"][n] for n in names}, "files": writer.files,
              "kept_documents": kept_docs, "removed_documents": removed_docs,
              "kept_tokens": kept_tokens, "removed_tokens": removed_tokens,
              "available_updates_without_repetition": available_updates,
              "construction_seconds": time.perf_counter() - start,
              "thresholds": {"min_document_tokens": 128, "adjacent_repeat_fraction_gt": 0.20, "prefix128_unique_fraction_lt": 0.10},
              "status": "complete" if available_updates >= 3250 else "insufficient_tokens"}
    (args.output / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    if available_updates < 3250:
        raise ValueError("Filtered corpus cannot supply the complete fixed schedule.")


if __name__ == "__main__":
    main()
