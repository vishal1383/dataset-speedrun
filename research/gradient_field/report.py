"""Summarize all completed development experiments without hiding failures."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("runs/gradient-field-v1"))
    p.add_argument("--output", type=Path, default=Path("research/gradient_field/results"))
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = [json.loads(p.read_text()) | {"run": str(p.parent)} for p in sorted((args.root / "search").glob("*/result.json"))]
    baselines = {r["seed"]: r for r in runs if r["mode"] == "full-data"}
    for r in runs:
        control = baselines[r["seed"]]
        r["matched_control_initialization"] = r["initial_state_sha256"] == control["initial_state_sha256"]
        r["control_final_loss"] = control["final_loss"]
        r["loss_gap_at_400"] = r["final_loss"] - control["final_loss"]
        crossing = next((v for v in r["curve"] if v["loss"] <= control["final_loss"] + 0.01), None)
        r["first_observed_quality_crossing"] = crossing
        r["observed_student_speedup"] = control["train_seconds"] / crossing["train_seconds"] if crossing and crossing["train_seconds"] else None
    artifacts = [json.loads(p.read_text()) for p in sorted((args.root / "artifacts").glob("*.json"))]
    summary = {"runs": runs, "artifacts": artifacts, "completed_runs": len(runs),
               "summed_training_seconds": sum(r["train_seconds"] for r in runs),
               "interpretation": "Development only. An endpoint potential encodes a checkpoint. Direct loading is a required control.",
               "quality_rule": "First measured loss <= same-seed full-data final loss + 0.01 nats. No interpolation between evaluations."}
    (args.output / "development.json").write_text(json.dumps(summary, indent=2) + "\n")
    for r in runs:
        print(r["run"], f'loss={r["final_loss"]:.5f}', f'gap={r["loss_gap_at_400"]:+.5f}',
              f'seconds={r["train_seconds"]:.3f}', f'crossing={r["first_observed_quality_crossing"]}')


if __name__ == "__main__":
    main()
