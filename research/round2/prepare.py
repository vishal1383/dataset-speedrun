"""Prepare broader development data and a separate reserved audit."""

import json
from pathlib import Path
import sys
import numpy as np
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from prepare import source_documents, doc_hash
from train import file_hash, load_sequences


def main():
    data = ROOT / "data/tinystories-v0"
    out = ROOT / "data/round2"
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        for name in ["dev", "audit"]:
            assert file_hash(out / f"{name}.npy") == manifest[name]["sha256"]
        return
    docs = source_documents("TinyStories-valid.txt", 8192, data / "source")
    training = json.loads((data / "source/TinyStories-train.txt.20000.json").read_text())
    excluded = {doc_hash(x) for x in training}
    tokenizer = Tokenizer.from_file(str(data / "tokenizer.json"))
    manifest = {"tokenizer_sha256": file_hash(data / "tokenizer.json"),
                "source_sha256": file_hash(data / "source/TinyStories-valid.txt.8192.json")}
    rng = np.random.default_rng(19381)
    for name, first, last, n in [("dev", 0, 4096, 1024), ("audit", 4096, 8192, 2048)]:
        texts = []
        for text in docs[first:last]:
            h = doc_hash(text)
            if h not in excluded:
                texts.append(text)
                excluded.add(h)
        tokens = np.concatenate([np.array(x.ids + [0], np.uint16) for x in tokenizer.encode_batch(texts)])
        stream = out / f"{name}-stream.npy"
        np.save(stream, tokens)
        windows = load_sequences(stream, 256).numpy().astype(np.uint16)
        ids = np.sort(rng.choice(len(windows), n, replace=False))
        np.save(out / f"{name}.npy", windows[ids])
        manifest[name] = {"documents": len(texts), "range": [first, last], "windows": n,
                          "tokens": n * 256, "sha256": file_hash(out / f"{name}.npy"),
                          "window_ids": ids.tolist()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared disjoint development and reserved audit arrays.")


if __name__ == "__main__":
    main()
