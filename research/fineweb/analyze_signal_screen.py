"""Compare development screens without claiming a full-budget training record."""

import argparse
import json
from pathlib import Path


def compare(summary):
    rows = []
    for row in summary["results"]:
        if row["status"] != "complete":
            rows.append({"arm": row["arm"], "data_order": row["data_order"], "status": row["status"]})
            continue
        reference = next(item for item in summary["results"]
                         if item["arm"] == "fresh" and item["data_order"] == row["data_order"])
        total = row["total_seconds_excluding_teacher_training"]
        # Use observed points; do not interpolate a learning curve into a claimed speedup.
        earlier = [point for point in reference["curve"] if point["train_seconds"] <= total]
        later = [point for point in reference["curve"] if point["train_seconds"] >= total]
        rows.append({
            "arm": row["arm"], "data_order": row["data_order"], "status": "complete",
            "final_loss": row["final_loss"], "delta_at_fixed_updates": row["final_loss"] - reference["final_loss"],
            "student_seconds": row["train_seconds"], "construction_plus_student_seconds": total,
            "teacher_training_seconds_additional": row["construction"].get("teacher_training_seconds_excluded_from_construction", 0),
            "backward_batches": row["backward_calls"], "unique_source_tokens": row["unique_source_tokens"],
            "payload_bytes": row["construction"].get("artifact_bytes", row["cached_gradient_bytes"]),
            "baseline_point_at_or_before_candidate_cost": earlier[-1] if earlier else None,
            "baseline_point_at_or_after_candidate_cost": later[0] if later else None,
            "interpretation": "Matched-update development comparison; observed baseline points bracket the cost comparison."
        })
    return {"record_eligible": False, "initial_dev_loss": summary["initial_dev_loss"], "rows": rows,
            "limitations": summary["limitations"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text())
    value = compare(summary)
    args.output.write_text(json.dumps(value, indent=2) + "\n")
    for row in value["rows"]:
        print(json.dumps(row))
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), layout="constrained")
        for row in summary["results"]:
            if row["status"] != "complete" or row["data_order"] != 0:
                continue
            curve = row["curve"]
            cost = row["construction"].get("construction_seconds", 0)
            axes[0].plot([point["step"] for point in curve], [point["loss"] for point in curve],
                         marker="o", markersize=3, label=row["arm"])
            axes[1].plot([point["train_seconds"] + cost for point in curve], [point["loss"] for point in curve],
                         marker="o", markersize=3, label=row["arm"])
        axes[0].set_xlabel("Optimizer updates")
        axes[1].set_xlabel("Student + target construction time (seconds)")
        for axis in axes:
            axis.set_ylabel("Development loss (nats)")
            axis.grid(alpha=0.2)
        axes[1].legend(fontsize=8)
        figure.suptitle("162M model: gradient and target screen\nReduced batch; one shared initialization; not a full-budget record", fontsize=12)
        figure.supxlabel("Teacher pretraining (11.92 hours), shared warmup, and evaluation are excluded from the time axis.", fontsize=9)
        figure.savefig(args.plot, dpi=180)
        plt.close(figure)


if __name__ == "__main__":
    main()
