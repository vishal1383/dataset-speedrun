"""Finish the existing baseline, then run fixed text and latent compression arms."""

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

try:
    from .launch import ROOT, HERE, digest
    from .run_pair import verify_pair, write
except ImportError:
    from launch import ROOT, HERE, digest
    from run_pair import verify_pair, write


def verify_artifact(directory, expected_manifest):
    if digest(directory / "manifest.json") != expected_manifest:
        raise ValueError("Frozen compression manifest changed.")
    manifest = json.loads((directory / "manifest.json").read_text())
    for name, record in manifest["files"].items():
        if digest(directory / name) != record["sha256"]:
            raise ValueError(f"Compressed artifact changed: {name}")
    replay = json.loads((directory / "replay.json").read_text())
    if {p.name for p in (directory / "stream").glob("*.bin")} != {r["file"] for r in replay}:
        raise ValueError("Replay stream membership changed.")
    for row in replay:
        if (directory / "stream" / row["file"]).resolve() != (directory / "text" / row["payload"]).resolve():
            raise ValueError("Replay link no longer resolves to the frozen text artifact.")
    return manifest


def compare(baseline, text, latent):
    rows = []
    for name, result in [("text", text), ("latent", latent)]:
        runs = [{"seed": 17, "updates": 3250}] * 2
        verify_pair([baseline, result], runs)
        if not all(math.isfinite(r["loss"]) for r in result["curve"]):
            raise ValueError("Nonfinite validation loss.")
        reference_curve = {r["step"]: r for r in baseline["curve"]}
        deltas = [{"step": r["step"], "loss": r["loss"],
                   "delta_from_baseline": r["loss"] - reference_curve[r["step"]]["loss"]}
                  for r in result["curve"] if r["step"] in reference_curve]
        final = result["curve"][-1]["loss"]
        rows.append({"arm": name, "final_loss": final, "final_delta": deltas[-1]["delta_from_baseline"],
                     "below_3_28": final <= 3.28, "beats_baseline": final < baseline["curve"][-1]["loss"],
                     "curve": deltas})
    return {"baseline_loss": baseline["curve"][-1]["loss"], "arms": rows,
            "latent_minus_text": latent["curve"][-1]["loss"] - text["curve"][-1]["loss"],
            "interpretation": "One paired seed. Fixed updates and validation; latent training objective differs. No confirmed record or end-to-end speed claim."}


def main():
    plan_path = HERE / "compression-plan.json"
    plan = json.loads(plan_path.read_text())
    directory = ROOT / plan["output"]
    directory.mkdir(parents=True, exist_ok=True)
    fd = os.open(directory / "queue.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    os.write(fd, f"{os.getpid()}\n".encode())
    os.close(fd)
    status_path = directory / "queue-status.json"
    status = {"status": "starting", "pid": os.getpid(), "plan_sha256": digest(plan_path),
              "started_unix": time.time(), "completed": [], "active": None}

    def state(stage, **fields):
        status.update(status=stage, updated_unix=time.time(), **fields)
        write(status_path, status)
        print(json.dumps(status), flush=True)

    def verify_sources():
        for name, expected in plan["source_sha256"].items():
            if digest(ROOT / name) != expected:
                raise ValueError(f"Frozen compression source changed: {name}")
        if digest(ROOT / plan["validation"]["path"]) != plan["validation"]["sha256"]:
            raise ValueError("Frozen validation changed.")

    try:
        verify_sources()
        artifact = ROOT / plan["artifact"]
        metadata = verify_artifact(artifact, plan["artifact_manifest_sha256"])
        state("waiting_for_baseline", active=plan["baseline"])
        baseline_path = ROOT / plan["baseline"]
        while True:
            try:
                baseline = json.loads((baseline_path / "result.json").read_text())
            except json.JSONDecodeError:
                time.sleep(1)
                continue
            if baseline["status"] == "complete":
                break
            if baseline["status"] != "running":
                raise RuntimeError("The baseline failed. No compression run was submitted.")
            if time.time() - (baseline_path / "progress.json").stat().st_mtime > 1800:
                raise RuntimeError("Baseline progress has been stale for 30 minutes.")
            time.sleep(30)
        # The original controller rejects its pending Edu run after the explicit source amendment.
        prior_status = baseline_path.parent / "queue-status.json"
        for _ in range(120):
            old = json.loads(prior_status.read_text())
            if old["status"] == "failed" and "Frozen source changed" in old.get("error", ""):
                break
            time.sleep(1)
        else:
            raise RuntimeError("Original queue did not acknowledge supersession; refusing GPU overlap.")
        if baseline["launcher_sha256"] != plan["baseline_launcher_sha256"]:
            raise ValueError("Baseline launcher differs from the retained reference.")
        for name, expected in plan["baseline_training_files"].items():
            if baseline["training_files"][name]["sha256"] != expected:
                raise ValueError("Baseline training input differs from the retained reference.")
        status["completed"].append("baseline")
        results = {}
        for smoke in [True, False]:
            for arm in ["text", "latent"]:
                verify_sources()
                verify_artifact(artifact, plan["artifact_manifest_sha256"])
                name = f"{arm}-smoke" if smoke else f"{arm}-seed17"
                output = directory / name
                command = [sys.executable, str(HERE / "launch.py"), "--output", str(output),
                           "--train-pattern", str((artifact / "stream/fineweb_train_*.bin").relative_to(ROOT)),
                           "--seed", "17", "--updates", "2" if smoke else "3250", "--microbatch", "64"]
                if smoke:
                    command += ["--smoke", "--validation-tokens", "65536"]
                if arm == "latent":
                    command += ["--latent-codebook", str(artifact / "latent.npz")]
                state("running", active=name, command=command)
                subprocess.run(command, cwd=ROOT, check=True)
                result = json.loads((output / "result.json").read_text())
                if result["initial_state_sha256"] != baseline["initial_state_sha256"]:
                    raise ValueError("Student initialization differs from the paired baseline.")
                if result["status"] != "complete" or not result["curve"]:
                    raise ValueError("Missing completed run record.")
                if smoke:
                    values = [r["loss"] for r in result["curve"]]
                    if not all(math.isfinite(v) for v in values) or values[-1] >= values[0]:
                        raise ValueError("GPU smoke did not produce finite, decreasing validation loss.")
                else:
                    verify_pair([baseline, result], [{"seed": 17, "updates": 3250}] * 2)
                    results[arm] = result
                status["completed"].append(name)
                state("between_runs", active=None)
        summary = compare(baseline, results["text"], results["latent"])
        summary["artifacts"] = {k: metadata[k] for k in ["text_artifact_bytes", "latent_arm_artifact_bytes", "construction_seconds"]}
        summary["plan_sha256"] = digest(plan_path)
        write(directory / "comparison.json", summary)
        state("complete", active=None, finished_unix=time.time())
    except BaseException as error:
        state("failed", error=repr(error), finished_unix=time.time())
        raise


if __name__ == "__main__":
    main()
