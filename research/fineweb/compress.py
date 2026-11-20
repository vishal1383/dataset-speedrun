"""Build a training-only text coreset and a sparse conditional target codebook."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

try:
    from .prepare import ROOT, inspect_shard, sha256
    from .build_candidate import ShardWriter
except ImportError:
    from prepare import ROOT, inspect_shard, sha256
    from build_candidate import ShardWriter

BLOCK = 8192
FEATURES = 128
VOCAB = 50304
SEED = 20261120


def select_coreset(features, repetition, seed=SEED, group_size=64):
    """Remove 25% while minimizing local feature-mean error and repetition."""
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(features))
    selected = []
    errors = []
    for start in range(0, len(permutation), group_size * 256):
        batch_ids = permutation[start:start + group_size * 256]
        n = len(batch_ids) // group_size
        ids = batch_ids[:n * group_size].reshape(n, group_size)
        if n:
            values = features[ids]
            centered = values - values.mean(1, keepdims=True)
            residual = np.zeros((n, features.shape[-1]), dtype=np.float32)
            removed = np.zeros((n, group_size), dtype=bool)
            rows = np.arange(n)
            for _ in range(group_size // 4):
                # The residual is the sum of removed, centered feature vectors.
                costs = ((centered + residual[:, None, :]) ** 2).sum(-1)
                costs -= 0.25 * repetition[ids]
                costs[removed] = np.inf
                choice = costs.argmin(1)
                residual += centered[rows, choice]
                removed[rows, choice] = True
            selected.extend(ids[~removed].tolist())
            errors.extend(np.linalg.norm(residual / (group_size * 0.75), axis=1).tolist())
        tail = batch_ids[n * group_size:]
        selected.extend(tail[:int(len(tail) * 0.75)].tolist())
    return np.array(selected, dtype=np.int64), float(np.mean(errors))


def make_codebook(events, max_contexts=262144, minimum=16, topk=4):
    """Fit empirical categorical targets, then retain frequent contexts and top-k mass."""
    unique, counts = np.unique(events, return_counts=True)
    context = unique >> np.uint64(16)
    starts = np.r_[0, np.flatnonzero(context[1:] != context[:-1]) + 1]
    ends = np.r_[starts[1:], len(unique)]
    totals = np.add.reduceat(counts, starts)
    eligible = np.flatnonzero(totals >= minimum)
    ranked = eligible[np.argsort(-totals[eligible], kind="stable")[:max_contexts]]
    ranked.sort()
    keys = context[starts[ranked]].astype(np.int64)
    target_ids = np.zeros((len(ranked), topk), dtype=np.int64)
    probabilities = np.zeros((len(ranked), topk), dtype=np.float32)
    support = totals[ranked].astype(np.int32)
    for row, group in enumerate(ranked):
        lo, hi = starts[group], ends[group]
        chosen = np.argsort(-counts[lo:hi], kind="stable")[:topk] + lo
        target_ids[row, :len(chosen)] = (unique[chosen] & np.uint64(65535)).astype(np.int64)
        probabilities[row, :len(chosen)] = counts[chosen] / totals[group]
    return keys, target_ids, probabilities, support


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/fineweb-compression-v1/artifact")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    begin = time.perf_counter()
    source = ROOT / "data/fineweb10B"
    manifest = json.loads((source / "manifest.json").read_text())
    paths = [source / f"fineweb_train_{i:06d}.bin" for i in range(1, 21)]
    for path in paths:
        if inspect_shard(path) != manifest["files"][path.name]:
            raise ValueError(f"Training source changed: {path}")
    unigram = np.zeros(VOCAB, dtype=np.int64)
    offsets, hashes, repetition, raw_features, events = [], [], [], [], []
    rng = np.random.default_rng(SEED)
    for shard, path in enumerate(paths):
        tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=1024)
        unigram += np.bincount(tokens, minlength=VOCAB)
        count = len(tokens) // BLOCK
        blocks = tokens[:count * BLOCK].reshape(count, BLOCK)
        sample = blocks[:, ::16].astype(np.int64)
        hashed = ((sample * 2654435761) ^ (sample >> 3)) % FEATURES
        bins = hashed + np.arange(count)[:, None] * FEATURES
        raw_features.append(np.bincount(bins.ravel(), minlength=count * FEATURES).reshape(count, FEATURES).astype(np.float32))
        repetition.append((blocks[:, 1:] == blocks[:, :-1]).mean(1).astype(np.float32))
        offsets.extend((shard, int(i * BLOCK)) for i in range(count))
        hashes.extend(hashlib.blake2b(memoryview(b), digest_size=8).digest() for b in blocks)
        # Sample across every source shard. Never read validation tokens or losses.
        position = np.arange(2 + int(rng.integers(32)), len(tokens), 32)
        a = tokens[position - 2].astype(np.uint64)
        b = tokens[position - 1].astype(np.uint64)
        target = tokens[position].astype(np.uint64)
        valid = (a != 50256) & (b != 50256)
        events.append((((a * VOCAB + b) << np.uint64(16)) | target)[valid])
        print(json.dumps({"stage": "scan", "shard": shard + 1, "seconds": time.perf_counter() - begin}), flush=True)
    raw = np.concatenate(raw_features)
    idf = np.log((len(raw) + 1) / ((raw > 0).sum(0) + 1)) + 1
    features = np.sqrt(raw) * idf
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-12)
    repetition = np.concatenate(repetition)
    # Exact duplicate blocks need only one representative in the stored text pool.
    _, first = np.unique(np.array(hashes, dtype="S8"), return_index=True)
    first.sort()
    local, mean_error = select_coreset(features[first], repetition[first])
    selected = first[local]
    difficulty = np.zeros(len(offsets), dtype=np.float32)
    logp = -np.log((unigram + 1) / (unigram.sum() + VOCAB))
    cursor = 0
    for path in paths:
        tokens = np.memmap(path, dtype=np.uint16, mode="r", offset=1024)
        n = len(tokens) // BLOCK
        difficulty[cursor:cursor+n] = logp[tokens[:n*BLOCK].reshape(n, BLOCK)[:, ::16]].mean(1)
        cursor += n
    # A noisy easy-to-hard order retains diversity throughout the first pass.
    normalized = (difficulty[selected] - difficulty[selected].mean()) / max(float(difficulty[selected].std()), 1e-8)
    order = np.argsort(normalized + rng.normal(0, 2, len(selected)), kind="stable")
    selected = selected[order]
    offsets = np.array(offsets, dtype=np.int64)
    np.save(out / "selected-blocks.npy", offsets[selected])
    writer = ShardWriter(out / "text")
    arrays = [np.memmap(p, dtype=np.uint16, mode="r", offset=1024) for p in paths]
    for shard, offset in offsets[selected]:
        writer.append(arrays[shard][offset:offset+BLOCK])
    writer.flush()
    # Decode replay order into symlinks; no source corpus is required by the student.
    stream = out / "stream"
    stream.mkdir()
    payload = sorted((out / "text").glob("*.bin"))
    replay = list(range(len(payload))) + rng.permutation(len(payload)).tolist()
    available = 0
    links = []
    for index in replay:
        path = payload[index]
        name = f"fineweb_train_{len(links)+1:06d}.bin"
        (stream / name).symlink_to(Path("../text") / path.name)
        links.append({"file": name, "payload": path.name})
        available += (writer.files[path.name]["tokens"] - 2) // 524288
        if available >= 3250:
            break
    if available < 3250:
        raise ValueError("Compressed replay schedule is too short.")
    print(json.dumps({"stage": "fit_latent_codebook", "sampled_events": sum(map(len, events))}), flush=True)
    keys, ids, probs, counts = make_codebook(np.concatenate(events))
    np.savez(out / "latent.npz", keys=keys.astype(np.uint32), ids=ids.astype(np.uint16),
             probabilities=probs.astype(np.float16), support=counts)
    (out / "replay.json").write_text(json.dumps(links, indent=2) + "\n")
    metadata = {"method": "75% feature-mean coreset after exact-block deduplication; noisy easy-to-hard curriculum; sparse conditional target codebook",
                "seed": SEED, "block_tokens": BLOCK, "source_tokens": int(unigram.sum()),
                "source_files": {p.name: manifest["files"][p.name] for p in paths},
                "selected_blocks": len(selected), "unique_source_blocks": len(first),
                "text_tokens": len(selected) * BLOCK, "text_payload_bytes": sum(p.stat().st_size for p in payload),
                "latent_bytes": (out / "latent.npz").stat().st_size, "latent_contexts": len(keys),
                "latent_sample_stride": 32, "latent_topk": 4, "latent_minimum_support": 16,
                "mean_group_feature_error": mean_error, "validation_access": False,
                "feature_gradient_claim": "Lexical feature matching is a proxy, not a neural-gradient estimate.",
                "latent_fit": "Empirical categorical maximum likelihood on sampled training trigrams; top-four retained probabilities, remaining mass stays on gold.",
                "construction_seconds": time.perf_counter() - begin,
                "files": {"text/" + p.name: writer.files[p.name] for p in payload},
                "replay_logical_bytes": sum((stream / x["file"]).stat().st_size for x in links),
                "available_updates": available, "status": "complete"}
    for name in ["latent.npz", "selected-blocks.npy", "replay.json"]:
        p = out / name
        metadata["files"][name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    metadata["text_artifact_bytes"] = metadata["text_payload_bytes"] + metadata["files"]["replay.json"]["bytes"]
    metadata["latent_arm_artifact_bytes"] = metadata["text_artifact_bytes"] + metadata["latent_bytes"]
    (out / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({k: v for k, v in metadata.items() if k not in ["files", "source_files"]}), flush=True)


if __name__ == "__main__":
    main()
