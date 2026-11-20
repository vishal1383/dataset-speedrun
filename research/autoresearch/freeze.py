"""Freeze all promotion decisions before confirmation accesses audit loss."""

import json
from pathlib import Path

from engine import ROOT, DATA, PROTOCOL, file_hash
from search import write


def main():
    root = ROOT / "runs/autoresearch-v1"
    destination = root / "frozen-finalists.json"
    if destination.exists():
        raise FileExistsError("Finalists are already frozen.")
    for name in ["data", "gradient", "distill"]:
        assert json.loads((root / f"{name}-status.json").read_text())["status"] == "complete"
    history = json.loads((root / "data-history.json").read_text())
    baseline = sum(json.loads((ROOT / f"results/original-seed{s}/result.json").read_text())["final_dev_loss"]
                   for s in [17, 42]) / 2
    groups = {}
    for row in history:
        groups.setdefault(row["name"], []).append(row)
    eligible = [group for group in groups.values() if {r["seed"] for r in group} == {17, 42}
                and sum(r["final_dev_loss"] for r in group) / 2 <= baseline]
    if eligible:
        chosen = min(eligible, key=lambda group: group[0]["artifact_bytes"])[0]
        reason = "Smallest two-seed artifact with mean development loss no worse than the registered full-pool baseline."
    else:
        finalist = json.loads((root / "data-finalist.json").read_text())
        chosen = next(r for r in history if r["name"] == finalist["name"])
        reason = "No candidate passed the development quality constraint; selected lowest mean development loss."
    candidates = {"selected_data": {k: chosen[k] for k in ["name", "artifact", "artifact_bytes", "fraction", "method"]}}
    for key, source in [("gradient_latents", "gradient"), ("distilled_targets", "distill")]:
        candidates[key] = json.loads((root / f"{source}-finalist.json").read_text())
    for value in candidates.values():
        value["artifact_sha256"] = file_hash(ROOT / value["artifact"])
    frozen = {"seeds": PROTOCOL["confirmation_seeds"], "audit_used_for_selection": False,
              "data_selection_reason": reason, "reference_development_loss": baseline,
              "reference_artifact_bytes": (DATA / "original.npy").stat().st_size,
              "candidates": candidates, "audit_manifest_sha256": file_hash(ROOT / "data/autoresearch-v1/manifest.json")}
    write(destination, frozen)
    print(json.dumps(frozen, indent=2))


if __name__ == "__main__":
    main()
