"""Export the completed pilot and a figure for the experimental track."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = ROOT / "research/gradient_field/results"
    source = ROOT / "runs/gradient-field-confirmation-v1"
    workers = [json.loads((source / f"worker-{i}.json").read_text()) for i in [0, 1]]
    for key in ["artifact_sha256", "evaluation_sha256", "frozen_manifest_sha256"]:
        assert len({w[key] for w in workers}) == 1, f"Worker mismatch: {key}"
    pairs = sorted([p for w in workers for p in w["pairs"]], key=lambda p: p["seed"])
    assert [p["seed"] for p in pairs] == list(range(1009, 1017))
    deltas = np.array([p["delta"] for p in pairs])
    timings = [s for w in workers for s in w["direct_load_seconds_samples"]]
    summary = {k: v for k, v in workers[0].items() if k not in ["worker", "pairs", "seeds"]}
    summary.update(pairs=pairs, seeds=[p["seed"] for p in pairs], workers=[0, 1],
                   mean_full_loss=float(np.mean([p["full_loss"] for p in pairs])),
                   mean_field_loss=float(np.mean([p["field_loss"] for p in pairs])),
                   mean_paired_delta=float(deltas.mean()),
                   one_sided_95_upper_delta=float(deltas.mean() + t.ppf(0.95, len(deltas)-1) * deltas.std(ddof=1)/np.sqrt(len(deltas))),
                   mean_full_train_seconds=float(np.mean([p["full_train_seconds"] for p in pairs])),
                   mean_field_train_seconds=float(np.mean([p["field_train_seconds"] for p in pairs])),
                   direct_load_seconds_samples=timings, direct_load_seconds_median=float(np.median(timings)))
    (out / "confirmation-workers.json").write_text(json.dumps(workers, indent=2) + "\n")
    (out / "confirmation-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    runs = [json.loads(p.read_text()) for p in sorted(source.glob("*/result.json"))]
    (out / "confirmation-runs.json").write_text(json.dumps(runs, indent=2) + "\n")
    initial = json.loads((out / "development.json").read_text())
    strong = json.loads((out / "strong/development.json").read_text())
    construction = json.loads((out / "strong-amendment.json").read_text())
    compute = {
        "development_runs": len(initial["runs"]) + len(strong["runs"]),
        "development_train_seconds": initial["summed_training_seconds"] + strong["summed_training_seconds"],
        "confirmation_runs": len(runs), "confirmation_train_seconds": sum(r["train_seconds"] for r in runs),
        "reused_400_update_teacher_train_seconds": 13.320215069688857,
        "reused_1600_update_teacher_train_seconds": construction["teacher_train_seconds"],
        "artifact_encoding_seconds": sum(a["construction_seconds"] for a in initial["artifacts"] + strong["artifacts"]),
        "scope": "Training-loop and encoding times. Process startup, evaluation, preparation, and prior campaigns are additional costs."
    }
    (out / "compute.json").write_text(json.dumps(compute, indent=2) + "\n")
    release = {"q4.pt": {"bytes": summary["artifact_bytes"], "sha256": summary["artifact_sha256"],
                         "source": "runs/gradient-field-strong-v1/artifacts/q4.pt", "kind": "4-bit quantized teacher endpoint"}}
    (out / "release-artifacts.json").write_text(json.dumps(release, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for kind, label, color in [(None, "Full data", "#444444"), ("q4", "4-bit target potential", "#167a67")]:
        r = next(r for r in strong["runs"] if r["kind"] == kind and (kind is None or r["field"] == "quadratic"))
        axes[0].plot([p["step"] for p in r["curve"]], [p["loss"] for p in r["curve"]],
                     marker="o", markersize=3, label=label, color=color)
    direct = next(a for a in strong["artifacts"] if a["kind"] == "q4")["direct_load_dev_loss"]
    axes[0].axhline(direct, linestyle=":", color="#a76712", label="Directly loaded target")
    axes[0].set(xlabel="Student optimizer updates", ylabel="Development loss (nats/token)", ylim=(2.2, 8.6),
                title="Same architecture and optimizer; changed objective")
    axes[0].legend(fontsize=8)
    pairs = summary["pairs"]
    for pair in pairs:
        axes[1].plot([0, 1], [pair["full_loss"], pair["field_loss"]], marker="o", alpha=0.65, color="#167a67")
    axes[1].scatter([2], [summary["direct_load_loss"]], color="#a76712", marker="D", s=55)
    axes[1].set_xticks([0, 1, 2], ["Full data\n400 updates", "Target potential\n200 updates", "Direct load\n0 updates"])
    axes[1].set(ylabel="Fresh-audit loss (nats/token)", title="Eight fresh initialization seeds")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", alpha=0.15)
    fig.suptitle("A 3.57 MB gradient generator that encodes a trained checkpoint", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "gradient-field.png", dpi=180)
    print(json.dumps({"summary": summary, "compute": compute}, indent=2))


if __name__ == "__main__":
    main()
