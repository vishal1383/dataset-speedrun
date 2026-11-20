"""Execute one frozen data-only comparison sequentially on the local Spark."""

# Superseded on 2026-11-20: the user requested text and latent compression.
# The running controller has its original code in memory. This source change
# makes its frozen-source check reject the pending FineWeb-Edu run after the
# baseline finishes. compression_queue.py owns subsequent experiments.

import json
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .launch import ROOT, HERE, digest
except ImportError:
    from launch import ROOT, HERE, digest


def verify_pair(records, runs):
    full, candidate = records
    for record, run in zip(records, runs):
        if record["status"] != "complete" or record["smoke"] or record["exit_code"] != 0:
            raise ValueError("Both benchmark runs must finish successfully without smoke mode.")
        if not record["curve"] or record["curve"][-1]["step"] != run["updates"]:
            raise ValueError("Final validation does not match the frozen update budget.")
        if record["seed"] != run["seed"] or record["updates"] != run["updates"]:
            raise ValueError("Run differs from its frozen configuration.")
    for key in ["initial_state_sha256", "validation_files", "validation_tokens", "seed", "updates",
                "schedule_updates", "global_batch_tokens", "microbatch_sequences", "compile_model", "upstream"]:
        if full[key] != candidate[key]:
            raise ValueError(f"Paired learner configurations differ: {key}")
    if not full["initial_state_sha256"] or full["validation_tokens"] != 10485760:
        raise ValueError("Missing initialization hash or noncanonical validation budget.")


def write(path, obj):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(obj, indent=2) + "\n")
    temporary.replace(path)


def main():
    plan_path = HERE / "plan.json"
    plan = json.loads(plan_path.read_text())
    directory = ROOT / plan["output"]
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / "queue.lock"
    fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    status = {"status": "running", "pid": os.getpid(), "plan_sha256": digest(plan_path),
              "started_unix": time.time(), "completed": [], "active": None}
    status_path = directory / "queue-status.json"
    write(status_path, status)
    try:
        for run in plan["runs"]:
            if digest(ROOT / plan["validation"]["path"]) != plan["validation"]["sha256"]:
                raise ValueError("Frozen validation shard changed.")
            for name, expected in plan["source_sha256"].items():
                if digest(ROOT / name) != expected:
                    raise ValueError(f"Frozen source changed: {name}")
            manifest_path = ROOT / run["manifest"]
            if digest(manifest_path) != run["manifest_sha256"]:
                raise ValueError("Frozen training manifest changed.")
            manifest = json.loads(manifest_path.read_text())
            files = sorted(ROOT.glob(run["train_pattern"]))
            for path in files:
                if digest(path) != manifest["files"][path.name]["sha256"]:
                    raise ValueError(f"Frozen training shard changed: {path}")
            expected_names = {n for n in manifest["files"] if n.startswith("fineweb_train_")}
            if {p.name for p in files} != expected_names:
                raise ValueError("Training shard membership differs from the frozen manifest.")
            output = directory / run["name"]
            if output.exists():
                raise ValueError(f"Run directory already exists; refusing duplicate execution: {output}")
            command = [sys.executable, str(HERE / "launch.py"), "--output", str(output),
                       "--train-pattern", run["train_pattern"], "--seed", str(run["seed"]),
                       "--updates", str(run["updates"]), "--microbatch", "64"]
            status.update(active=run["name"], command=command)
            write(status_path, status)
            print(json.dumps({"starting": run["name"], "command": command}), flush=True)
            result = subprocess.run(command, cwd=ROOT)
            if result.returncode:
                raise RuntimeError(f"Run {run['name']} failed with exit code {result.returncode}.")
            status["completed"].append(run["name"])
            write(status_path, status)
        records = [json.loads((directory / r["name"] / "result.json").read_text()) for r in plan["runs"]]
        full, candidate = records
        verify_pair(records, plan["runs"])
        a, b = full["curve"][-1], candidate["curve"][-1]
        summary = {"seed": plan["runs"][0]["seed"], "updates": 3250, "target_loss": 3.28,
                   "reference": a, "candidate": b, "paired_final_loss_delta": b["loss"] - a["loss"],
                   "candidate_below_target": b["loss"] <= 3.28,
                   "interpretation": "One paired pilot, not a confirmed record. No per-run stopping or automatic hyperparameter changes."}
        write(directory / "comparison.json", summary)
        status.update(status="complete", active=None, finished_unix=time.time())
        write(status_path, status)
    except BaseException as error:
        status.update(status="failed", error=repr(error), finished_unix=time.time())
        write(status_path, status)
        raise


if __name__ == "__main__":
    main()
