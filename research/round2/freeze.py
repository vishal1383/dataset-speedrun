"""Select a finalist from development results and freeze its artifact."""

import collections
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/round2"))
from competition.runner import write
from train import file_hash
from research.round2.search import select


def main():
    root = ROOT / "runs/round2"
    groups = collections.defaultdict(dict)
    history = []
    for p in sorted((root / "search").glob("*/result.json")):
        row = json.loads(p.read_text())
        groups[row["name"]][row["seed"]] = row
        history.append(row)
    eligible = []
    for name, rows in groups.items():
        if name == "original" or not all(seed in rows for seed in [17, 42]):
            continue
        if all(rows[s]["loss"] <= groups["original"][s]["loss"] for s in [17, 42]):
            eligible.append((rows[17]["artifact_bytes"], sum(r["loss"] for r in rows.values()) / len(rows), name))
    if not eligible:
        raise RuntimeError("No finalist passes the development selection rule.")
    _, _, name = min(eligible)
    row = groups[name][17]
    artifact = ROOT / row["artifact"]
    scores = np.load(root / "scores.npz")
    ids = select(scores, row["method"], row["fraction"])
    results = ROOT / "research/round2/results"
    write(results / "selected-window-ids.json", ids.tolist())
    write(results / "search-history.json", history)
    write(results / "selection.json", {"finalist": name, "development_seeds": [17, 42],
          "artifact_sha256": file_hash(artifact), "selection_ids_sha256": file_hash(results / "selected-window-ids.json"),
          "scores_sha256": file_hash(root / "scores.npz"), "audit_accessed": False,
          "candidate_dev": {str(s): groups[name][s]["loss"] for s in [17, 42]},
          "control_dev": {str(s): groups["original"][s]["loss"] for s in [17, 42]},
          "protocol_sha256": file_hash(ROOT / "research/round2/protocol.json")})
    write(ROOT / "submissions/round2-influence75.json", {"name": name, "track": "data-only",
          "artifact": row["artifact"], "artifact_sha256": file_hash(artifact),
          "source": "TinyStories fixed 20,000-document training pool; development gradients from exposed validation documents [0,4096)",
          "license": "CDLA-Sharing-1.0", "construction": {
              "method": "Select top 75% by mean standardized training/development gradient alignment at teacher steps 50 and 400",
              "teacher": "Same architecture, seed 271, 400 updates; checkpoints at 50 and 400, reused from first campaign",
              "teacher_training_seconds": json.loads((ROOT / "runs/autoresearch-v1/teacher/result.json").read_text())["train_seconds"],
              "scoring": json.loads((root / "influence-cost.json").read_text()),
              "search_runs": len(history), "search_train_seconds": sum(r["train_seconds"] for r in history),
              "previous_campaign": "44 earlier search trials and their teacher construction also informed this campaign",
              "audit_access": False}})
    print(name)


if __name__ == "__main__":
    main()
