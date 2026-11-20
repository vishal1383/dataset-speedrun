"""Run frozen paired confirmation trials for the experimental gradient field."""

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from scipy.stats import t

from experiment import ROOT, decode, file_hash, losses, read_tokens, run, setup, write


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", type=int, choices=[0, 1], default=0)
    args = p.parse_args()
    freeze_path = ROOT / "research/gradient_field/results/frozen.json"
    frozen = json.loads(freeze_path.read_text())
    for path, digest in frozen["input_sha256"].items():
        if file_hash(ROOT / path) != digest:
            raise ValueError(f"Frozen input changed: {path}")
    output = ROOT / "runs/gradient-field-confirmation-v1"
    output.mkdir(parents=True, exist_ok=True)
    artifact = ROOT / frozen["artifact"]
    audit = ROOT / frozen["evaluation"]
    seeds = frozen["worker_seeds"][str(args.worker)]
    pairs = []
    for seed in seeds:
        # Alternate order to limit a systematic thermal or first-run timing advantage.
        order = ["full", "field"] if seed % 2 else ["field", "full"]
        completed = {}
        for name in order:
            completed[name] = run(output / f"{name}-{seed}", seed,
                                  artifact if name == "field" else None, frozen["field"], audit,
                                  frozen["updates"] if name == "field" else 400)
        full, field = completed["full"], completed["field"]
        assert full["initial_state_sha256"] == field["initial_state_sha256"]
        pairs.append({"seed": seed, "full_loss": full["final_loss"], "field_loss": field["final_loss"],
                      "delta": field["final_loss"] - full["final_loss"],
                      "full_train_seconds": full["train_seconds"], "field_train_seconds": field["train_seconds"],
                      "field_decode_seconds": field["decode_seconds"],
                      "speedup_training_loop": full["train_seconds"] / field["train_seconds"]})
    model, _ = setup(0)
    data = read_tokens(audit).cuda()
    timings = []
    for _ in range(5):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        target = decode(torch.load(artifact, map_location="cpu", weights_only=True))
        model.load_state_dict(target)
        torch.cuda.synchronize()
        timings.append(time.perf_counter() - begin)
    direct = losses(model, data).mean().item()
    delta = np.array([p["delta"] for p in pairs])
    upper = float(delta.mean() + t.ppf(0.95, len(delta) - 1) * delta.std(ddof=1) / np.sqrt(len(delta)))
    summary = {"pairs": pairs, "seeds": seeds, "worker": args.worker,
               "mean_full_loss": float(np.mean([p["full_loss"] for p in pairs])),
               "mean_field_loss": float(np.mean([p["field_loss"] for p in pairs])),
               "mean_paired_delta": float(delta.mean()), "one_sided_95_upper_delta": upper,
               "mean_full_train_seconds": float(np.mean([p["full_train_seconds"] for p in pairs])),
               "mean_field_train_seconds": float(np.mean([p["field_train_seconds"] for p in pairs])),
               "direct_load_loss": direct, "direct_load_seconds_samples": timings,
               "direct_load_seconds_median": float(np.median(timings)),
               "artifact_bytes": artifact.stat().st_size, "artifact_sha256": file_hash(artifact),
               "evaluation_sha256": file_hash(audit), "frozen_manifest_sha256": file_hash(freeze_path),
               "status": "complete", "scope": "Experimental checkpoint potential. Not the official dataset-only track."}
    write(output / f"worker-{args.worker}.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
