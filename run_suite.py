"""Run balanced experiment queues on two independent GPUs."""

import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", type=int, choices=[0, 1], required=True)
    parser.add_argument("--config", default="configs/tinystories-v0.json")
    parser.add_argument("--data", default="data/tinystories-v0")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    if len(config["seeds"]) != 2:
        raise ValueError("This launch plan requires two seeds.")
    Path("logs").mkdir(exist_ok=True)
    lock = Path(f"logs/worker-{args.worker}.lock").open("w")
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = Path(f"logs/worker-{args.worker}.json")
    completed = []

    def status(state, active=None):
        status_path.write_text(json.dumps({"status": state, "active": active,
                                          "completed": completed, "updated_unix": time.time()}, indent=2))

    for i, recipe in enumerate(config["recipes"]):
        seed = config["seeds"][(args.worker + i) % 2]
        name = f"{recipe}-seed{seed}"
        output = Path("results") / name
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite or duplicate {output}")
        command = [sys.executable, "-u", "train.py", "--config", args.config, "--data", args.data,
                   "--recipe", recipe, "--seed", str(seed), "--output", str(output)]
        status("running", name)
        with (Path("logs") / f"{name}.log").open("x") as log:
            proc = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode:
            status("failed", name)
            raise SystemExit(proc.returncode)
        completed.append(name)
        status("between_runs")
    status("complete")


if __name__ == "__main__":
    main()
