"""Build fixed data recipes using training documents only."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import requests
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
BASE = f"https://huggingface.co/datasets/roneneldan/TinyStories/resolve/{REVISION}"
RECIPES = ("original", "random_quarter", "diverse_quarter", "stratified_quarter")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def doc_hash(text):
    return hashlib.sha256(" ".join(text.split()).encode()).hexdigest()


def source_documents(filename, count, cache):
    path = cache / f"{filename}.{count}.json"
    if path.exists():
        docs = json.loads(path.read_text())
        if len(docs) != count:
            raise ValueError(f"Invalid source cache: {path}")
        return docs
    docs, lines = [], []
    with requests.get(f"{BASE}/{filename}", stream=True, timeout=(30, 120)) as r:
        r.raise_for_status()
        r.encoding = "utf-8"
        for line in r.iter_lines(decode_unicode=True):
            if line.strip() == "<|endoftext|>":
                text = "\n".join(lines).strip()
                lines = []
                if text:
                    docs.append(text)
                if len(docs) == count:
                    break
            else:
                lines.append(line)
    if len(docs) != count:
        raise ValueError(f"Expected {count} documents, got {len(docs)}")
    path.write_text(json.dumps(docs, ensure_ascii=False))
    return docs


def select_recipes(encoded, seed=2026):
    n = len(encoded)
    k = n // 4
    rng = np.random.default_rng(seed)
    random_ids = np.sort(rng.choice(n, k, replace=False))
    # Normalize distinct-token count by length to reduce the length preference.
    scores = np.array([len(set(x)) / np.sqrt(len(x)) for x in encoded])
    diverse_ids = np.sort(np.argsort(-scores, kind="stable")[:k])
    lengths = np.array([len(x) for x in encoded])
    bins = np.array_split(np.argsort(lengths, kind="stable"), 10)
    chosen = []
    for i, bucket in enumerate(bins):
        quota = k // 10 + int(i < k % 10)
        chosen.extend(rng.choice(bucket, quota, replace=False).tolist())
    return dict(original=np.arange(n), random_quarter=random_ids,
                diverse_quarter=diverse_ids, stratified_quarter=np.sort(chosen))


def write_stream(path, encoded, ids):
    tokens = np.concatenate([np.array(encoded[int(i)] + [0], dtype=np.uint16) for i in ids])
    np.save(path, tokens)
    return {"sha256": sha256(path), "tokens": int(tokens.size),
            "documents": len(ids), "bytes": path.stat().st_size}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/tinystories-v0"))
    parser.add_argument("--train-docs", type=int, default=20000)
    parser.add_argument("--eval-docs", type=int, default=2048)
    args = parser.parse_args()
    start = time.perf_counter()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        for name, info in manifest["files"].items():
            if sha256(out / name) != info["sha256"]:
                raise ValueError(f"Artifact hash mismatch: {name}")
        print("Existing artifacts verified.")
        return
    cache = out / "source"
    cache.mkdir(exist_ok=True)
    raw_train = source_documents("TinyStories-train.txt", args.train_docs, cache)
    raw_eval = source_documents("TinyStories-valid.txt", args.eval_docs, cache)
    # Exact normalized-document decontamination is the only use of eval text here.
    eval_unique, seen = [], set()
    for text in raw_eval:
        h = doc_hash(text)
        if h not in seen:
            eval_unique.append(text)
            seen.add(h)
    train, removed = [], 0
    for text in raw_train:
        h = doc_hash(text)
        if h in seen:
            removed += 1
        else:
            train.append(text)
            seen.add(h)
    half = len(eval_unique) // 2
    dev, test = eval_unique[:half], eval_unique[half:]
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=4096, special_tokens=["<eos>", "<unk>"],
                                 initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tokenizer.train_from_iterator(train, trainer=trainer)
    tokenizer.save(str(out / "tokenizer.json"))
    assert tokenizer.token_to_id("<eos>") == 0
    encoded = [x.ids for x in tokenizer.encode_batch(train)]
    selection_start = time.perf_counter()
    selections = select_recipes(encoded)
    selection_seconds = time.perf_counter() - selection_start
    files = {"tokenizer.json": {"sha256": sha256(out / "tokenizer.json")}}
    for name, ids in selections.items():
        files[f"{name}.npy"] = write_stream(out / f"{name}.npy", encoded, ids)
        (out / f"{name}.ids.json").write_text(json.dumps(ids.tolist()) + "\n")
        files[f"{name}.ids.json"] = {"sha256": sha256(out / f"{name}.ids.json")}
    for name, docs in [("dev", dev), ("test", test)]:
        values = [x.ids for x in tokenizer.encode_batch(docs)]
        files[f"{name}.npy"] = write_stream(out / f"{name}.npy", values, list(range(len(values))))
    manifest = {"source": "roneneldan/TinyStories", "revision": REVISION,
                "source_files": ["TinyStories-train.txt", "TinyStories-valid.txt"],
                "source_license": "CDLA-Sharing-1.0", "train_documents_requested": args.train_docs,
                "eval_documents_requested": args.eval_docs, "train_documents": len(train),
                "removed_train_documents": removed, "dev_documents": len(dev), "test_documents": len(test),
                "vocab_size": tokenizer.get_vocab_size(), "selection_seed": 2026,
                "selection_seconds": selection_seconds, "preparation_seconds": time.perf_counter() - start,
                "source_prefix_hashes": {p.name: sha256(p) for p in sorted(cache.glob("*.json"))},
                "files": files}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
