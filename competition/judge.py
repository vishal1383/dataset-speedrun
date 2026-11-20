"""Judge a frozen data submission against matched full-data controls."""

import argparse
import json
from pathlib import Path
import statistics

from competition.runner import ROOT, train, write, read_tokens
from train import file_hash


def validate_submission(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    for key in ("name", "track", "artifact", "artifact_sha256", "source", "license", "construction"):
        if key not in spec:
            raise ValueError(f"Missing submission field: {key}")
    if spec["track"] != "data-only":
        raise ValueError("This judge accepts the data-only track. Training signals need a separate judge.")
    artifact = (ROOT / spec["artifact"]).resolve()
    if file_hash(artifact) != spec["artifact_sha256"]:
        raise ValueError("Submission artifact hash mismatch.")
    read_tokens(artifact)
    return spec, artifact


def paired_statistics(deltas):
    import scipy.stats
    n = len(deltas)
    if n < 2:
        raise ValueError("At least two matched seeds are required.")
    mean = statistics.mean(deltas)
    se = statistics.stdev(deltas) / n**0.5
    return {"mean": mean, "standard_error": se,
            "upper_95_one_sided": mean + float(scipy.stats.t.ppf(.95, n - 1)) * se,
            "lower_95_one_sided": mean - float(scipy.stats.t.ppf(.95, n - 1)) * se}


def summarize(output):
    output = Path(output)
    frozen = json.loads((output / "frozen.json").read_text())
    rows, differences = [], []
    for seed in frozen["seeds"]:
        baseline = json.loads((output / f"original-s{seed}/result.json").read_text())
        candidate = json.loads((output / f"candidate-s{seed}/result.json").read_text())
        for row, artifact_hash in [(baseline, frozen["baseline_sha256"]),
                                   (candidate, frozen["artifact_sha256"])]:
            if (row["artifact_sha256"] != artifact_hash or
                    row["evaluation_sha256"] != frozen["evaluation_sha256"] or
                    row["seed"] != seed or row["updates"] != 400 or row["status"] != "complete"):
                raise ValueError("Result does not match the frozen submission.")
        for key in ("initial_state_sha256", "source_hashes", "evaluation_windows", "host", "torch", "cuda", "gpu"):
            if baseline[key] != candidate[key]:
                raise ValueError(f"Paired control mismatch: {key}")
        delta = candidate["loss"] - baseline["loss"]
        differences.append(delta)
        rows.append({"seed": seed, "baseline_loss": baseline["loss"],
                     "candidate_loss": candidate["loss"], "delta": delta,
                     "host": baseline["host"]})
    stats = paired_statistics(differences)
    smaller = frozen["artifact_bytes"] < frozen["baseline_bytes"]
    result = {"name": frozen["name"], "track": "data-only", "updates": 400,
              "seeds": frozen["seeds"], "pairs": rows, "paired_delta": stats,
              "baseline_loss": statistics.mean(r["baseline_loss"] for r in rows),
              "candidate_loss": statistics.mean(r["candidate_loss"] for r in rows),
              "artifact_bytes": frozen["artifact_bytes"], "baseline_bytes": frozen["baseline_bytes"],
              "compression_ratio": frozen["baseline_bytes"] / frozen["artifact_bytes"],
              "strict_quality_pass": stats["mean"] <= 0,
              "evidence_pass": len(rows) >= 8 and stats["upper_95_one_sided"] <= .01,
              "record": smaller and len(rows) >= 8 and stats["mean"] <= 0 and stats["upper_95_one_sided"] <= .01,
              "frozen_sha256": file_hash(output / "frozen.json"),
              "evaluation_sha256": frozen["evaluation_sha256"],
              "interpretation": "Mean loss must not increase; the one-sided confidence bound must exclude harm above 0.01 nats. This is not proof of exactly equal population loss."}
    write(output / "summary.json", result)
    print(json.dumps(result, indent=2))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--submission", type=Path)
    p.add_argument("--evaluation", type=Path)
    p.add_argument("--seeds", default="1001,1002,1003,1004,1005,1006,1007,1008")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--worker", type=int, default=0)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--freeze-only", action="store_true")
    p.add_argument("--summarize", action="store_true")
    args = p.parse_args()
    if args.summarize:
        summarize(args.output)
        return
    if not 0 <= args.worker < args.workers:
        raise ValueError("Invalid worker assignment.")
    spec, artifact = validate_submission(args.submission)
    baseline = ROOT / "data/tinystories-v0/original.npy"
    seeds = [int(s) for s in args.seeds.split(",")]
    if len(seeds) != len(set(seeds)) or len(seeds) < 8:
        raise ValueError("Use at least eight distinct confirmation seeds.")
    frozen = {"name": spec["name"], "artifact_sha256": file_hash(artifact),
              "artifact_bytes": artifact.stat().st_size, "baseline_sha256": file_hash(baseline),
              "baseline_bytes": baseline.stat().st_size, "evaluation_sha256": file_hash(args.evaluation),
              "submission_sha256": file_hash(args.submission), "seeds": seeds,
              "protocol_sha256": file_hash(ROOT / "research/round2/protocol.json"),
              "judge_sha256": file_hash(__file__)}
    args.output.mkdir(parents=True, exist_ok=True)
    frozen_path = args.output / "frozen.json"
    if frozen_path.exists():
        if json.loads(frozen_path.read_text()) != frozen:
            raise ValueError("Existing cohort has different frozen inputs.")
    else:
        write(frozen_path, frozen)
    if args.freeze_only:
        print(json.dumps(frozen, indent=2))
        return
    for seed in seeds[args.worker::args.workers]:
        # Keep each candidate/control pair on the same GPU and software stack.
        for name, path in [("original", baseline), ("candidate", artifact)]:
            out = args.output / f"{name}-s{seed}"
            if not (out / "result.json").exists():
                train(path, seed, args.evaluation, out)
    if args.workers == 1:
        summarize(args.output)


if __name__ == "__main__":
    main()
