"""Download the pinned upstream GPT-2 token shards without retokenization."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def inspect_shard(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
        raise ValueError(f"Invalid upstream token shard: {path}")
    count = int(header[2])
    if path.stat().st_size != 1024 + 2 * count:
        raise ValueError(f"Truncated shard: {path}")
    values = np.memmap(path, mode="r", dtype=np.uint16, offset=1024, shape=(count,))
    if int(values.max()) > 50256:
        raise ValueError(f"Token outside GPT-2 vocabulary: {path}")
    return {"bytes": path.stat().st_size, "tokens": count, "sha256": sha256(path)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shards", type=int, default=20)
    p.add_argument("--dataset", choices=["fineweb", "finewebedu"], default="fineweb")
    args = p.parse_args()
    pin = json.loads((Path(__file__).parent / "pin.json").read_text())
    educational = args.dataset == "finewebedu"
    dataset = pin["candidate_dataset"] if educational else pin["dataset"]
    revision = pin["candidate_dataset_revision"] if educational else pin["dataset_revision"]
    out = ROOT / ("data/finewebedu10B" if educational else "data/fineweb10B")
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {"dataset": dataset, "revision": revision, "files": {}}
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old["revision"] != revision:
            raise ValueError("Existing dataset revision differs from the pinned revision.")
        manifest = old
    names = ([] if educational else ["fineweb_val_000000.bin"]) + [f"{args.dataset}_train_{i:06d}.bin" for i in range(1, args.shards + 1)]
    for name in names:
        path = Path(hf_hub_download(repo_id=dataset, repo_type="dataset", filename=name,
                                    revision=revision, local_dir=out))
        record = inspect_shard(path)
        if name in manifest["files"] and record != manifest["files"][name]:
            raise ValueError(f"Existing manifest differs from downloaded shard: {name}")
        manifest["files"][name] = record
        temp = manifest_path.with_suffix(".tmp")
        temp.write_text(json.dumps(manifest, indent=2) + "\n")
        temp.replace(manifest_path)
        print(json.dumps({"file": name, **record}), flush=True)


if __name__ == "__main__":
    main()
