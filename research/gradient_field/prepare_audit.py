"""Reserve fresh audit documents without evaluating a model on them."""

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
    source = ROOT / "data/tinystories-v0/source"
    out = ROOT / "data/gradient-field-v1"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        assert file_hash(out / "audit.npy") == manifest["sha256"]
        return
    docs = source_documents("TinyStories-valid.txt", 12288, source)
    training = json.loads((source / "TinyStories-train.txt.20000.json").read_text())
    seen = {doc_hash(x) for x in training + docs[:8192]}
    kept = []
    for text in docs[8192:12288]:
        h = doc_hash(text)
        if h not in seen:
            kept.append(text)
            seen.add(h)
    tokenizer = Tokenizer.from_file(str(ROOT / "data/tinystories-v0/tokenizer.json"))
    tokens = np.concatenate([np.array(x.ids + [0], np.uint16) for x in tokenizer.encode_batch(kept)])
    stream = out / "audit-stream.npy"
    np.save(stream, tokens)
    windows = load_sequences(stream, 256).numpy().astype(np.uint16)
    rng = np.random.default_rng(771928)
    ids = np.sort(rng.choice(len(windows), 2048, replace=False))
    np.save(out / "audit.npy", windows[ids])
    manifest = {"source_document_range": [8192, 12288], "documents_after_exact_deduplication": len(kept),
                "windows": 2048, "target_tokens": 524288, "sha256": file_hash(out / "audit.npy"),
                "tokenizer_sha256": file_hash(ROOT / "data/tinystories-v0/tokenizer.json"),
                "source_sha256": file_hash(source / "TinyStories-valid.txt.12288.json"), "window_ids": ids.tolist(),
                "audit_used_to_choose_candidate": False}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "window_ids"}))


if __name__ == "__main__":
    main()
