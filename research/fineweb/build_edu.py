"""Prepare the educational-data variant with exact validation-document exclusion."""

import hashlib
import json
from pathlib import Path
import time

import numpy as np

try:
    from .build_candidate import ShardWriter
    from .prepare import ROOT, inspect_shard
except ImportError:
    from build_candidate import ShardWriter
    from prepare import ROOT, inspect_shard


def documents(paths):
    carry = np.empty(0, dtype=np.uint16)
    for path in paths:
        values = np.memmap(path, mode="r", dtype=np.uint16, offset=1024)
        starts = np.flatnonzero(values == 50256)
        if not len(starts):
            carry = np.concatenate((carry, values))
            continue
        prefix = np.concatenate((carry, values[:starts[0]]))
        if len(prefix):
            yield prefix
        for lo, hi in zip(starts[:-1], starts[1:]):
            yield values[lo:hi]
        carry = np.array(values[starts[-1]:], copy=True)
    if len(carry):
        yield carry


def doc_hash(doc):
    return hashlib.sha256(memoryview(np.ascontiguousarray(doc))).digest()


def main():
    started = time.perf_counter()
    source = ROOT / "data/finewebedu10B"
    manifest = json.loads((source / "manifest.json").read_text())
    names = [f"finewebedu_train_{i:06d}.bin" for i in range(1, 21)]
    for name in names:
        if inspect_shard(source / name) != manifest["files"][name]:
            raise ValueError(f"Source shard changed: {name}")
    validation = ROOT / "data/fineweb10B/fineweb_val_000000.bin"
    validation_record = inspect_shard(validation)
    excluded = {doc_hash(doc) for doc in documents([validation])}
    out = ROOT / "runs/fineweb-v1/finewebedu-decontaminated"
    writer = ShardWriter(out)
    kept = dropped = kept_tokens = dropped_tokens = 0
    for doc in documents([source / name for name in names]):
        if doc_hash(doc) in excluded:
            dropped += 1
            dropped_tokens += len(doc)
        else:
            writer.append(doc)
            kept += 1
            kept_tokens += len(doc)
        if (kept + dropped) % 100000 == 0:
            print(json.dumps({"documents": kept + dropped, "excluded_documents": dropped}), flush=True)
    writer.flush()
    assert kept_tokens + dropped_tokens == 2_000_000_000
    available = sum((r["tokens"] - 2) // 524288 for r in writer.files.values())
    record = {"dataset": manifest["dataset"], "revision": manifest["revision"], "source_files": manifest["files"],
              "files": writer.files, "validation_exclusion": validation_record,
              "kept_documents": kept, "removed_documents": dropped, "kept_tokens": kept_tokens,
              "removed_tokens": dropped_tokens, "available_updates_without_repetition": available,
              "construction_seconds": time.perf_counter() - started,
              "method": "Public FineWeb-Edu selection, preserving source order. Remove exact token-document matches against the full canonical validation shard.",
              "upstream_selection_cost": "Reused public educational classifier selection; construction compute is not reconstructed or claimed free.",
              "track": "open-curation-data-only", "near_duplicate_exclusion": False,
              "validation_loss_used": False, "status": "complete" if available >= 3250 else "insufficient_tokens"}
    (out / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    if available < 3250:
        raise ValueError("Candidate cannot supply the frozen schedule.")
    print(json.dumps({k: v for k, v in record.items() if k not in ["source_files", "files"]}), flush=True)


if __name__ == "__main__":
    main()
