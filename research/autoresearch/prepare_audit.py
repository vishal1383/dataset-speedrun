"""Reserve new audit documents without using their loss for search."""

import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from engine import ROOT, DATA, PROTOCOL, file_hash
from prepare import source_documents, doc_hash


def main():
    out = ROOT / "data/autoresearch-v1"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        manifest = json.loads((out / "manifest.json").read_text())
        assert file_hash(out / "audit.npy") == manifest["audit_sha256"]
        print("Existing audit artifact verified.")
        return
    first, last = PROTOCOL["audit_source_document_range"]
    source = source_documents("TinyStories-valid.txt", last, DATA / "source")
    train = json.loads((DATA / "source/TinyStories-train.txt.20000.json").read_text())
    excluded = {doc_hash(text) for text in train + source[:first]}
    docs = []
    for text in source[first:last]:
        digest = doc_hash(text)
        if digest not in excluded:
            docs.append(text)
            excluded.add(digest)
    tokenizer = Tokenizer.from_file(str(DATA / "tokenizer.json"))
    tokens = np.concatenate([np.array(x.ids + [0], dtype=np.uint16) for x in tokenizer.encode_batch(docs)])
    np.save(out / "audit.npy", tokens)
    manifest = {"source_documents": [first, last], "documents_after_exact_dedup": len(docs),
                "tokens": len(tokens), "audit_sha256": file_hash(out / "audit.npy"),
                "tokenizer_sha256": file_hash(DATA / "tokenizer.json"),
                "evaluation_tokens": PROTOCOL["audit_sequences"] * 256,
                "selection_access": "No audit loss is exposed to the search code."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
