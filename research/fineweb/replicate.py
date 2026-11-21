"""Replicate a completed pilot improvement with a new paired initialization."""

import json
import math
import os
import subprocess
import sys
import time

try:
    from .launch import ROOT, HERE, digest
    from .run_pair import verify_pair, write
    from .compression_queue import compare, verify_artifact
except ImportError:
    from launch import ROOT, HERE, digest
    from run_pair import verify_pair, write
    from compression_queue import compare, verify_artifact


def choose_winner(summary):
    """Select by final pilot loss. Require an improvement; prefer text on ties."""
    reference = summary["baseline_loss"]
    if not math.isfinite(reference):
        raise ValueError("Nonfinite baseline loss.")
    rows = summary["arms"]
    if len(rows) != 2 or {r["arm"] for r in rows} != {"text", "latent"}:
        raise ValueError("Both completed pilot arms are required.")
    for row in rows:
        if not math.isfinite(row["final_loss"]):
            raise ValueError("Nonfinite pilot loss.")
    winner = min(rows, key=lambda r: (r["final_loss"], r["arm"] != "text"))
    return winner["arm"] if winner["final_loss"] < reference else None


def main():
    plan_path = HERE / "replication-plan.json"
    plan = json.loads(plan_path.read_text())
    pilot_plan_path = HERE / "compression-plan.json"
    pilot_plan = json.loads(pilot_plan_path.read_text())
    directory = ROOT / plan["output"]
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / "queue.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    status = {"status": "starting", "pid": os.getpid(), "started_unix": time.time(),
              "plan_sha256": digest(plan_path), "active": None, "completed": []}

    def state(stage, **fields):
        status.update(status=stage, updated_unix=time.time(), **fields)
        write(directory / "queue-status.json", status)
        print(json.dumps(status), flush=True)

    def verify():
        if digest(pilot_plan_path) != plan["pilot_plan_sha256"]:
            raise ValueError("Pilot plan changed.")
        for name, expected in plan["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Frozen replication source changed: {name}")
        for name, expected in pilot_plan["baseline_training_files"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Baseline data changed: {name}")
        if digest(ROOT / pilot_plan["validation"]["path"]) != pilot_plan["validation"]["sha256"]:
            raise ValueError("Validation data changed.")
        verify_artifact(ROOT / pilot_plan["artifact"], pilot_plan["artifact_manifest_sha256"])

    try:
        verify()
        pilot = ROOT / pilot_plan["output"]
        state("waiting_for_pilot", active=str(pilot.relative_to(ROOT)))
        while True:
            queue = json.loads((pilot / "queue-status.json").read_text())
            if queue["status"] == "complete":
                break
            if queue["status"] == "failed":
                raise RuntimeError(f"Pilot failed: {queue.get('error')}")
            active = queue.get("active")
            progress = pilot / str(active) / "progress.json"
            if active and progress.exists() and time.time() - progress.stat().st_mtime > 1800:
                raise RuntimeError("Pilot progress has been stale for 30 minutes.")
            time.sleep(30)
        baseline17 = json.loads((ROOT / pilot_plan["baseline"] / "result.json").read_text())
        text17 = json.loads((pilot / "text-seed17/result.json").read_text())
        latent17 = json.loads((pilot / "latent-seed17/result.json").read_text())
        summary = compare(baseline17, text17, latent17)
        winner = choose_winner(summary)
        write(directory / "pilot-comparison.json", summary)
        if winner is None:
            state("no_pilot_improvement", active=None,
                  next_action="Inspect text replay and latent-minus-text curves before changing the recipe. No unsupported winner was submitted.")
            return
        selection = {"arm": winner, "rule": plan["selection_rule"], "pilot_comparison": summary,
                     "artifact_manifest_sha256": pilot_plan["artifact_manifest_sha256"],
                     "selected_before_replication": True, "selected_unix": time.time()}
        write(directory / "selection.json", selection)
        state("selected", active=None, selected_arm=winner)
        records = []
        for arm in ["baseline", winner]:
            verify()
            output = directory / f"{arm}-seed{plan['seed']}"
            pattern = "data/fineweb10B/fineweb_train_*.bin" if arm == "baseline" else pilot_plan["artifact"] + "/stream/fineweb_train_*.bin"
            command = [sys.executable, str(HERE / "launch.py"), "--output", str(output),
                       "--train-pattern", pattern, "--seed", str(plan["seed"]),
                       "--updates", str(plan["updates"]), "--microbatch", "64"]
            if arm == "latent":
                command += ["--latent-codebook", str(ROOT / pilot_plan["artifact"] / "latent.npz")]
            state("running", active=output.name, command=command)
            subprocess.run(command, cwd=ROOT, check=True)
            record = json.loads((output / "result.json").read_text())
            if record["status"] != "complete" or record["smoke"] or record["curve"][-1]["step"] != plan["updates"]:
                raise ValueError("Replication did not finish its fixed budget.")
            if not all(math.isfinite(r["loss"]) for r in record["curve"]):
                raise ValueError("Nonfinite replication loss.")
            records.append(record)
            status["completed"].append(output.name)
        verify_pair(records, [{"seed": plan["seed"], "updates": plan["updates"]}] * 2)
        if records[0]["initial_state_sha256"] == baseline17["initial_state_sha256"]:
            raise ValueError("Replication reused the pilot initialization.")
        reference, candidate = (r["curve"][-1]["loss"] for r in records)
        reference_curve = {r["step"]: r["loss"] for r in records[0]["curve"]}
        result = {"arm": winner, "seed": plan["seed"], "updates": plan["updates"],
                  "baseline_loss": reference, "candidate_loss": candidate, "delta": candidate - reference,
                  "improvement_repeated": candidate < reference, "candidate_below_3_28": candidate <= 3.28,
                  "curve": [{"step": r["step"], "loss": r["loss"], "delta": r["loss"] - reference_curve[r["step"]]}
                            for r in records[1]["curve"]],
                  "plan_sha256": digest(plan_path), "selection_sha256": digest(directory / "selection.json"),
                  "interpretation": "One fresh paired seed after pilot selection. Same public validation set. This does not establish a multi-seed record or fresh-data generalization."}
        write(directory / "comparison.json", result)
        state("complete", active=None, finished_unix=time.time(), improvement_repeated=candidate < reference)
    except BaseException as error:
        state("failed", error=repr(error), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
