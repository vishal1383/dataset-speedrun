"""Confirm frozen candidates on fresh seeds and the reserved audit split."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from engine import ROOT, PROTOCOL, run, file_hash, setup, losses, load_sequences
from distill import train as train_distill
from search import write


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=[0, 1], required=True)
    parser.add_argument("--root", type=Path, default=ROOT / "runs/autoresearch-v1")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root / "frozen-finalists.json").read_text())
    assert manifest["audit_used_for_selection"] is False
    for candidate in manifest["candidates"].values():
        assert file_hash(ROOT / candidate["artifact"]) == candidate["artifact_sha256"]
    completed = []
    # Assign each recipe to both hosts across seeds.
    tasks = [(kind, seed) for i, seed in enumerate(manifest["seeds"])
             for j, kind in enumerate(["original", "selected_data", "distilled_targets", "gradient_latents"])
             if (i + j) % 2 == args.worker]
    for kind, seed in tasks:
        name = f"{kind}-seed{seed}"
        write(root / f"confirmation-{args.worker}-status.json", {"status": "running", "active": name, "completed": completed})
        output = root / "confirmation" / name
        if kind == "original":
            reserved = root / "reserved-controls" / f"seed{seed}"
            if (reserved / "checkpoint-400.pt").exists():
                output.mkdir(parents=True, exist_ok=False)
                result = json.loads((reserved / "result.json").read_text())
                assert result["audit_accessed"] is False
                model, _ = setup(seed)
                model.load_state_dict(torch.load(reserved / "checkpoint-400.pt", weights_only=True))
                audit_path = ROOT / "data/autoresearch-v1/audit.npy"
                audit_data = load_sequences(audit_path, 256, PROTOCOL["audit_sequences"]).cuda()
                result.update(audit_accessed=True, audit_loss=losses(model, audit_data).mean().item(),
                              audit_sha256=file_hash(audit_path), reused_control_checkpoint=True)
                del model, audit_data
            else:
                result = run(seed, output, audit=True)
        else:
            candidate = manifest["candidates"][kind]
            path = ROOT / candidate["artifact"]
            if kind == "selected_data":
                seq = torch.from_numpy(np.load(path).astype(np.int64))
                result = run(seed, output, sequences=seq, audit=True)
            elif kind == "gradient_latents":
                artifact = torch.load(path, weights_only=True)
                result = run(seed, output, gradient_artifact=artifact, audit=True)
            else:
                with np.load(path) as f:
                    payload = {k: f[k] for k in f.files}
                result = train_distill(output, seed, payload, candidate["alpha"], audit=True)
            result.update(artifact_bytes=candidate["artifact_bytes"], artifact_sha256=candidate["artifact_sha256"])
        result.update(candidate=kind, physical_host=f"spark-{'a5a1' if args.worker == 0 else '664c'}",
                      frozen_manifest_sha256=file_hash(root / "frozen-finalists.json"))
        write(output / "result.json", result)
        completed.append(name)
    write(root / f"confirmation-{args.worker}-status.json", {"status": "complete", "completed": completed})


if __name__ == "__main__":
    main()
