"""Export complete confirmation evidence and a scientific result figure."""

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from competition.judge import summarize
from competition.runner import write


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    root = ROOT / "runs/round2/confirmation"
    out = ROOT / "research/round2/results"
    summary = summarize(root)
    write(out / "confirmation-summary.json", summary)
    runs = [json.loads(p.read_text()) for p in sorted(root.glob("*/result.json"))]
    write(out / "confirmation-runs.json", runs)
    with (out / "paired-results.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary["pairs"][0]))
        writer.writeheader()
        writer.writerows(summary["pairs"])
    history = json.loads((out / "search-history.json").read_text())
    write(out / "compute.json", {"search_runs": len(history), "confirmation_runs": len(runs),
          "search_train_seconds": sum(r["train_seconds"] for r in history),
          "confirmation_train_seconds": sum(r["train_seconds"] for r in runs),
          "search_run_seconds": sum(r["elapsed_seconds"] for r in history),
          "confirmation_run_seconds": sum(r["elapsed_seconds"] for r in runs),
          "influence_scoring": json.loads((ROOT / "runs/round2/influence-cost.json").read_text()),
          "exclusions": "Image startup, downloads, source development, transfers, previous campaign, and other subset construction costs are excluded from training-loop totals."})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    colors = ["#64748b", "#087f8c"]
    sizes = [summary["baseline_bytes"] / 1e6, summary["artifact_bytes"] / 1e6]
    axes[0].barh(["Full data", "Selected data"], sizes, color=colors)
    for y, value in enumerate(sizes):
        axes[0].text(value + .12, y, f"{value:.2f} MB", va="center")
    axes[0].set_xlim(0, 11)
    axes[0].set_xlabel("Uncompressed training artifact")
    axes[0].set_title("Fixed model, optimizer, and 400 updates")
    deltas = [r["delta"] for r in summary["pairs"]]
    axes[1].scatter(range(1, len(deltas) + 1), deltas, color=colors[1], s=40)
    axes[1].axhline(0, color=colors[0], linewidth=1)
    axes[1].axhline(summary["paired_delta"]["mean"], color=colors[1], linestyle="--", label="Mean difference")
    axes[1].set_xlabel("Fresh training seed (paired control)")
    axes[1].set_ylabel("Selected-data loss − full-data loss (nats/token)")
    axes[1].set_title("Fresh audit: negative values favor selection")
    axes[1].legend(frameon=False)
    fig.savefig(out / "record.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
