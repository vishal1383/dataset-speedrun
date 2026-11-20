"""Run bounded adaptive searches. Never read audit data during search."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch

from engine import (CONFIG, PROTOCOL, DATA, ROOT, setup, load_sequences, losses, run,
                    slice_artifact, file_hash)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def score_teacher(root):
    start = time.perf_counter()
    sequences = load_sequences(DATA / "original.npy", CONFIG["sequence_length"]).cuda()
    model, _ = setup(PROTOCOL["teacher_seed"])
    scores = {}
    for step in PROTOCOL["teacher_checkpoints"]:
        model.load_state_dict(torch.load(root / "teacher" / f"checkpoint-{step}.pt", weights_only=True))
        scores[f"loss_{step}"] = losses(model, sequences).cpu().numpy()
    np.savez(root / "scores.npz", **scores)
    write(root / "score-cost.json", {"seconds": time.perf_counter() - start, "sequences": len(sequences)})
    del sequences, model
    gc.collect()
    torch.cuda.empty_cache()
    return scores


def choose(scores, method, fraction):
    n = len(scores["loss_400"])
    k = max(1, int(n * fraction))
    rng = np.random.default_rng(20261119)
    if method == "random":
        return np.sort(rng.choice(n, k, replace=False))
    if method == "easy":
        ordering = np.argsort(scores["loss_400"], kind="stable")
    elif method == "progress":
        ordering = np.argsort(-(scores["loss_50"] - scores["loss_400"]), kind="stable")
    elif method == "middle":
        ordering = np.argsort(np.abs(scores["loss_400"] - np.median(scores["loss_400"])), kind="stable")
    elif method == "stratified":
        bins = np.array_split(np.argsort(scores["loss_400"], kind="stable"), 20)
        values = [rng.choice(bucket, k // 20 + (i < k % 20), replace=False) for i, bucket in enumerate(bins)]
        return np.sort(np.concatenate(values))
    else:
        raise ValueError(method)
    return np.sort(ordering[:k])


def data_search(root):
    if not (root / "teacher/result.json").exists():
        run(PROTOCOL["teacher_seed"], root / "teacher", checkpoint=True)
    scores = score_teacher(root)
    full = load_sequences(DATA / "original.npy", CONFIG["sequence_length"])
    history = []
    candidates = {}

    def trial(method, fraction, seed=17, phase="initial"):
        name = f"{method}-f{fraction:.5f}"
        ids = choose(scores, method, fraction)
        selected = full[torch.from_numpy(ids)]
        artifact = root / "artifacts" / f"{name}.npy"
        artifact.parent.mkdir(exist_ok=True)
        if not artifact.exists():
            np.save(artifact, selected.numpy().astype(np.uint16))
        write(root / "data-status.json", {"status": "running", "active": f"{name}-seed{seed}", "completed": len(history)})
        row = run(seed, root / "data-search" / f"{name}-seed{seed}", sequences=selected)
        row.update(name=name, method=method, fraction=fraction, phase=phase,
                   artifact_bytes=artifact.stat().st_size, artifact_sha256=file_hash(artifact),
                   artifact=str(artifact.relative_to(ROOT)), selected_sequences=len(ids))
        history.append(row)
        candidates[name] = {"method": method, "fraction": fraction, "artifact": row["artifact"]}
        write(root / "data-history.json", history)
        gc.collect()
        torch.cuda.empty_cache()
        return row

    for fraction in [0.125, 0.25, 0.5]:
        for method in ["random", "easy", "progress", "middle", "stratified"]:
            trial(method, fraction)
    best = min(history, key=lambda r: r["final_dev_loss"])
    # Refine the most promising region, using observed development loss.
    for fraction in [best["fraction"] / np.sqrt(2), min(0.75, best["fraction"] * np.sqrt(2))]:
        trial(best["method"], float(fraction), phase="adaptive_fraction")
    top = sorted(history, key=lambda r: r["final_dev_loss"])[:3]
    for candidate in top:
        trial(candidate["method"], candidate["fraction"], seed=42, phase="adaptive_seed_check")
    def mean_dev(candidate):
        return np.mean([r["final_dev_loss"] for r in history if r["name"] == candidate["name"]])
    winner = min(top, key=mean_dev)
    selection = {**candidates[winner["name"]], "name": winner["name"],
                 "mean_selection_dev_loss": float(mean_dev(winner)), "selection_seeds": [17, 42],
                 "audit_accessed": False, "history_sha256": file_hash(root / "data-history.json")}
    write(root / "data-finalist.json", selection)
    write(root / "data-status.json", {"status": "complete", "completed": len(history), "finalist": selection})


def gradient_search(root):
    if not (root / "gradient-source/result.json").exists():
        run(PROTOCOL["search_seed"], root / "gradient-source", capture=True)
    source = torch.load(root / "gradient-source/gradient-source.pt", weights_only=True)
    history = []

    def trial(rank, stride, phase="initial"):
        name = f"gradient-r{rank}-s{stride}"
        artifact = slice_artifact(source, rank, stride)
        path = root / "artifacts" / f"{name}.pt"
        path.parent.mkdir(exist_ok=True)
        torch.save(artifact, path)
        write(root / "gradient-status.json", {"status": "running", "active": name, "completed": len(history)})
        row = run(PROTOCOL["search_seed"], root / "gradient-search" / name, gradient_artifact=artifact)
        row.update(name=name, rank=rank, stride=stride, knots=len(artifact["knots"]), phase=phase,
                   artifact_bytes=path.stat().st_size, artifact_sha256=file_hash(path),
                   artifact=str(path.relative_to(ROOT)), initialization="source_seed")
        history.append(row)
        write(root / "gradient-history.json", history)
        gc.collect()
        torch.cuda.empty_cache()
        return row

    for stride in [1, 2, 4]:
        for rank in [1, 2, 4]:
            trial(rank, stride)
    best = min(history, key=lambda r: r["final_dev_loss"])
    # If the rank boundary wins, increase rank; otherwise explore temporal density.
    for stride in sorted({best["stride"], 1, 2, 4}):
        trial(8, stride, phase="adaptive_rank")
    # The budget counts the serialized payload, including tensor metadata.
    budget = (DATA / "original.npy").stat().st_size
    eligible = [r for r in history if r["artifact_bytes"] < budget]
    winner = min(eligible or history, key=lambda r: r["final_dev_loss"])
    selection = {k: winner[k] for k in ["name", "rank", "stride", "artifact", "artifact_bytes", "final_dev_loss"]}
    selection.update(audit_accessed=False, fits_data_byte_budget=winner["artifact_bytes"] < budget,
                     history_sha256=file_hash(root / "gradient-history.json"))
    write(root / "gradient-finalist.json", selection)
    write(root / "gradient-status.json", {"status": "complete", "completed": len(history), "finalist": selection})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["data", "gradients"])
    parser.add_argument("--root", type=Path, default=ROOT / "runs/autoresearch-v1")
    args = parser.parse_args()
    args.root = args.root.resolve()
    args.root.mkdir(parents=True, exist_ok=True)
    (data_search if args.mode == "data" else gradient_search)(args.root)


if __name__ == "__main__":
    main()
