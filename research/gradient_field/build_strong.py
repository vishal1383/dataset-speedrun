"""Build the adaptive follow-up from the existing 1,600-update training-only teacher."""

import json
from pathlib import Path
import time

import torch

from experiment import ROOT, decode, encode_tensor, file_hash, losses, read_tokens, setup, write


def main():
    root = ROOT / "runs/gradient-field-strong-v1"
    out = root / "artifacts"
    out.mkdir(parents=True, exist_ok=True)
    teacher_path = ROOT / "runs/autoresearch-v1/strong-teacher/teacher.pt"
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
    record = json.loads((teacher_path.parent / "result.json").read_text())
    write(root / "amendment.json", {
        "trigger": "The 400-update teacher has dev loss 3.0452; the seed-17 full-data reference has loss 3.0339.",
        "decision": "Test a stronger training-only teacher to separate endpoint quality from gradient-field convergence.",
        "teacher_updates": 1600, "teacher_seed": 271, "teacher_train_seconds": record["train_seconds"],
        "kinds": ["q4", "q8"], "potentials": ["quadratic", "huber"], "development_seeds": [17, 42],
        "teacher_cost_is_not_student_cost": True, "audit_accessed_for_selection": False,
        "scope": "Checkpoint coding through a state-dependent loss. Not a token-data leaderboard entry."
    })
    dev = read_tokens(ROOT / "data/round2/dev.npy").cuda()
    for kind in ["q4", "q8"]:
        path = out / f"{kind}.pt"
        if path.exists():
            continue
        start = time.perf_counter()
        parameters = {n: encode_tensor(p, int(kind[1:])) for n, p in teacher.items()}
        artifact = {"kind": kind, "parameters": parameters, "teacher_sha256": file_hash(teacher_path),
                    "teacher_updates": 1600, "teacher_train_seconds": record["train_seconds"],
                    "builder_sha256": file_hash(__file__), "codec_sha256": file_hash(Path(__file__).with_name("experiment.py")),
                    "config_sha256": file_hash(ROOT / "configs/tinystories-v0.json")}
        torch.save(artifact, path)
        construction = time.perf_counter() - start
        model, _ = setup(17)
        torch.cuda.synchronize()
        start = time.perf_counter()
        decoded = decode(torch.load(path, weights_only=True, map_location="cpu"))
        model.load_state_dict(decoded)
        torch.cuda.synchronize()
        direct_load_seconds = time.perf_counter() - start
        result = {"kind": kind, "artifact_bytes": path.stat().st_size, "artifact_sha256": file_hash(path),
                  "teacher_sha256": file_hash(teacher_path), "teacher_updates": 1600,
                  "teacher_train_seconds": record["train_seconds"], "construction_seconds": construction,
                  "direct_load_seconds": direct_load_seconds, "direct_load_dev_loss": losses(model, dev).mean().item()}
        write(path.with_suffix(".json"), result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
