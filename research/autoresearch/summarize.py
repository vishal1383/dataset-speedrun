"""Verify and export all searches and fresh-audit confirmation results."""

import csv
import json
from pathlib import Path
import shutil
import statistics

from engine import ROOT, file_hash
from search import write


def main():
    root = ROOT / "runs/autoresearch-v1"
    out = Path(__file__).parent / "results"
    out.mkdir(exist_ok=True)
    frozen = json.loads((root / "frozen-finalists.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((root / "confirmation").glob("*/result.json"))]
    assert len(rows) == 12
    expected = {(kind, seed) for kind in ["original", *frozen["candidates"]] for seed in frozen["seeds"]}
    assert {(r["candidate"], r["seed"]) for r in rows} == expected
    assert len({r["audit_sha256"] for r in rows}) == 1
    assert {r["frozen_manifest_sha256"] for r in rows} == {file_hash(root / "frozen-finalists.json")}
    for seed in frozen["seeds"]:
        assert len({r["initial_state_sha256"] for r in rows if r["seed"] == seed}) == 1
    for r in rows:
        assert r["updates"] == 400 and r["audit_accessed"] is True
        assert r["curve"][-1]["step"] == 400
        assert r["first_crossing"] == next((x["step"] for x in r["curve"] if x["dev_loss"] <= 3.5), None)
    for mode in ["data", "gradient", "distill"]:
        history = json.loads((root / f"{mode}-history.json").read_text())
        assert all(r["audit_accessed"] is False for r in history)
        shutil.copy2(root / f"{mode}-history.json", out / f"{mode}-history.json")
        shutil.copy2(root / f"{mode}-finalist.json", out / f"{mode}-finalist.json")
    shutil.copy2(root / "frozen-finalists.json", out / "frozen-finalists.json")
    shutil.copy2(ROOT / "data/autoresearch-v1/manifest.json", out / "audit-manifest.json")
    for name in ["score-cost.json", "distill-plan.json", "muon-diagnostic.json"]:
        if (root / name).exists():
            shutil.copy2(root / name, out / name)
    for name in ["teacher", "strong-teacher", "gradient-source"]:
        shutil.copy2(root / name / "result.json", out / f"{name}.json")
    write(out / "confirmation.json", rows)
    base = {r["seed"]: r for r in rows if r["candidate"] == "original"}
    summary = []
    for kind in ["original", *frozen["candidates"]]:
        group = sorted([r for r in rows if r["candidate"] == kind], key=lambda r: r["seed"])
        values = [r["audit_loss"] for r in group]
        delta = [r["audit_loss"] - base[r["seed"]]["audit_loss"] for r in group]
        size = frozen["reference_artifact_bytes"] if kind == "original" else frozen["candidates"][kind]["artifact_bytes"]
        summary.append({"candidate": kind, "artifact_bytes": size,
                        "compression_ratio": frozen["reference_artifact_bytes"] / size,
                        "dev_loss_mean": statistics.mean(r["final_dev_loss"] for r in group),
                        "audit_loss_mean": statistics.mean(values), "audit_loss_sample_sd": statistics.stdev(values),
                        "paired_audit_delta_mean": statistics.mean(delta), "paired_audit_deltas": delta,
                        "threshold_steps": [r["first_crossing"] for r in group],
                        "train_seconds_mean": statistics.mean(r["train_seconds"] for r in group)})
    write(out / "summary.json", summary)
    columns = ["candidate", "seed", "final_dev_loss", "audit_loss", "first_crossing", "train_seconds"]
    with (out / "confirmation.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in columns})
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = {"original": "Full pool", "selected_data": "Teacher-selected pool",
              "distilled_targets": "Sparse teacher targets", "gradient_latents": "Gradient latents"}
    for row in summary:
        ax = axes[1] if row["candidate"] == "gradient_latents" else axes[0]
        ax.errorbar(row["artifact_bytes"] / 1e6, row["audit_loss_mean"], yerr=row["audit_loss_sample_sd"],
                    fmt="o", capsize=4, label=labels[row["candidate"]])
    for ax in axes:
        ax.axhline(summary[0]["audit_loss_mean"], color="gray", linestyle="--", linewidth=1)
        ax.set(xlabel="Serialized training artifact (MB)", ylabel="Fresh audit loss (nats/token)")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_title("Data and state-dependent targets")
    axes[1].set_title("Open-loop gradient replay")
    fig.suptitle("Fixed model and optimizer · 400 updates · 3 fresh seeds")
    fig.tight_layout()
    fig.savefig(out / "compression_frontier.png", dpi=180)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
