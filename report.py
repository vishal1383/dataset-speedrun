"""Validate complete runs and generate a public result table and loss plot."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--config", type=Path, default=Path("configs/tinystories-v0.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    rows = [json.loads(p.read_text()) for p in sorted(args.results.glob("*/result.json"))
            if not p.parent.name.startswith("smoke")]
    expected = {(r, s) for r in config["recipes"] for s in config["seeds"]}
    assert len(rows) == len(expected), "Missing or duplicate runs."
    assert {(r["recipe"], r["seed"]) for r in rows} == expected
    assert len({r["train_source_sha256"] for r in rows}) == 1
    assert len({r["data_manifest_sha256"] for r in rows}) == 1
    config_hash = hashlib.sha256(args.config.read_bytes()).hexdigest()
    for r in rows:
        assert r["status"] == "complete" and r["config"] == config
        assert r["config_sha256"] == config_hash
        assert math.isfinite(r["test_loss"]) and math.isfinite(r["final_dev_loss"])
        assert r["curve"][-1]["step"] == config["updates"]
        assert r["curve"][-1]["tokens_seen"] == config["updates"] * config["batch_size"] * config["sequence_length"]
        crossing = next((x for x in r["curve"] if x["dev_loss"] <= config["target_loss"]), None)
        assert crossing == r["first_crossing"]
    for seed in config["seeds"]:
        assert len({r["initial_state_sha256"] for r in rows if r["seed"] == seed}) == 1
    fields = ["recipe", "seed", "final_dev_loss", "test_loss", "threshold_step", "train_seconds", "host"]
    with (args.results / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fields[:4]} | {
                "threshold_step": r["first_crossing"]["step"] if r["first_crossing"] else "not_reached",
                "train_seconds": r["train_seconds"], "host": r["hardware"]["host"]})
    lines = ["| Recipe | Dev loss, mean ± sample SD | Audit test loss, mean ± sample SD | Target steps by seed | Train seconds, mean |",
             "|---|---:|---:|---|---:|"]
    for recipe in config["recipes"]:
        group = sorted([r for r in rows if r["recipe"] == recipe], key=lambda r: r["seed"])
        dev = [r["final_dev_loss"] for r in group]
        test = [r["test_loss"] for r in group]
        steps = ", ".join(f"{r['seed']}: {r['first_crossing']['step'] if r['first_crossing'] else 'not reached'}" for r in group)
        lines.append(f"| {recipe} | {statistics.mean(dev):.4f} ± {statistics.stdev(dev):.4f} | "
                     f"{statistics.mean(test):.4f} ± {statistics.stdev(test):.4f} | {steps} | "
                     f"{statistics.mean(r['train_seconds'] for r in group):.1f} |")
    (args.results / "TABLE.md").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for recipe in config["recipes"]:
        group = [r for r in rows if r["recipe"] == recipe]
        steps = [x["step"] for x in group[0]["curve"]]
        losses = [[r["curve"][i]["dev_loss"] for r in group] for i in range(len(steps))]
        line, = ax.plot(steps, [statistics.mean(v) for v in losses], label=recipe, marker=".")
        ax.fill_between(steps, [min(v) for v in losses], [max(v) for v in losses], color=line.get_color(), alpha=0.12)
    ax.axhline(config["target_loss"], color="black", linestyle="--", linewidth=1, label="Target")
    ax.set(xlabel="Optimizer updates", ylabel="Development cross-entropy (nats/token)",
           title="Dataset Speedrun · TinyStories v0 · two seeds")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.results / "loss_curves.png", dpi=180)
    plt.close(fig)
    print("All eight runs passed the comparison checks.")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
