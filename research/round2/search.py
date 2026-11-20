"""Search token subsets using development data only."""

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/autoresearch"))
from competition.runner import train, write
from engine import setup, losses
from train import load_sequences, file_hash

RUN = ROOT / "runs/round2"
DATA = ROOT / "data/tinystories-v0"


def select(scores, method, fraction):
    weak, strong = scores["weak"], scores["strong"]
    n = len(weak)
    k = int(n * fraction)
    rng = np.random.default_rng(5819)
    if method == "random":
        ids = rng.choice(n, k, replace=False)
    elif method in ("easy400", "easy1600", "learnable", "middle"):
        score = {"easy400": weak, "easy1600": strong,
                 "learnable": 2 * strong - weak,
                 "middle": np.abs(weak - np.median(weak))}[method]
        ids = np.argsort(score, kind="stable")[:k]
    elif method in ("stratified", "coverage_easy", "coverage_learnable"):
        # Preserve source position and teacher-difficulty coverage.
        buckets = []
        for section in np.array_split(np.arange(n), 16):
            buckets.extend(np.array_split(section[np.argsort(strong[section])], 10))
        quotas = np.array([len(b) * fraction for b in buckets])
        sizes = np.floor(quotas).astype(int)
        for i in np.argsort(-(quotas - sizes))[:k - sizes.sum()]:
            sizes[i] += 1
        parts = []
        for bucket, count in zip(buckets, sizes):
            if method == "stratified":
                parts.append(rng.choice(bucket, count, replace=False))
            else:
                score = weak[bucket] if method == "coverage_easy" else 2 * strong[bucket] - weak[bucket]
                parts.append(bucket[np.argsort(score)[:count]])
        ids = np.concatenate(parts)
    else:
        raise ValueError(method)
    return np.sort(ids)


def prepare_scores():
    RUN.mkdir(exist_ok=True, parents=True)
    path = RUN / "scores.npz"
    if path.exists():
        return
    start = time.perf_counter()
    sequences = load_sequences(DATA / "original.npy", 256).cuda()
    model, _ = setup(271)
    values = {}
    for name, checkpoint in [("weak", ROOT / "runs/autoresearch-v1/teacher/checkpoint-400.pt"),
                             ("strong", ROOT / "runs/autoresearch-v1/strong-teacher/teacher.pt")]:
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        values[name] = losses(model, sequences).cpu().numpy()
    np.savez(path, **values)
    write(RUN / "score-cost.json", {"seconds": time.perf_counter() - start,
                                    "source_teachers": [400, 1600], "additional_teacher_training": False})


def worker(worker_id, plan):
    RUN.mkdir(exist_ok=True, parents=True)
    plan = json.loads(Path(plan).read_text())
    scores = np.load(RUN / "scores.npz")
    full = load_sequences(DATA / "original.npy", 256).numpy().astype(np.uint16)
    history = []
    for spec in plan[worker_id::2]:
        name, seed = spec["name"], spec["seed"]
        out = RUN / "search" / f"{name}-s{seed}"
        if (out / "result.json").exists():
            continue
        artifact = DATA / "original.npy"
        if name != "original":
            artifact = RUN / "artifacts" / f"{name}.npy"
            artifact.parent.mkdir(exist_ok=True, parents=True)
            ids = select(scores, spec["method"], spec["fraction"])
            if not artifact.exists():
                np.save(artifact, full[ids])
        write(RUN / f"worker{worker_id}-status.json", {"active": name, "seed": seed, "state": "running"})
        row = train(artifact, seed, ROOT / "data/round2/dev.npy", out)
        row.update(spec)
        row["artifact"] = str(artifact.relative_to(ROOT))
        write(out / "result.json", row)
        history.append(row)
        gc.collect()
        torch.cuda.empty_cache()
    write(RUN / f"worker{worker_id}-status.json", {"state": "complete", "completed_this_invocation": len(history)})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", action="store_true")
    p.add_argument("--worker", type=int, choices=[0, 1])
    p.add_argument("--plan", type=Path)
    args = p.parse_args()
    if args.scores:
        prepare_scores()
    else:
        worker(args.worker, args.plan)


if __name__ == "__main__":
    main()
