"""Adaptive follow-up: sparse teacher targets generate state-dependent gradients."""

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from engine import ROOT, DATA, CONFIG, PROTOCOL, setup, autocast, logits, losses, load_sequences, lr_ratio, file_hash, state_hash
from search import write


def train(output, seed=17, payload=None, alpha=0, teacher=False, audit=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    model, optimizers = setup(seed)
    initial = state_hash(model)
    c = dict(CONFIG)
    if teacher:
        c["updates"] = 1600
    seq = load_sequences(DATA / "original.npy", 256).cuda() if payload is None else torch.from_numpy(payload["tokens"].astype(np.int64)).cuda()
    ids = torch.from_numpy(payload["teacher_ids"].astype(np.int64)).cuda() if payload is not None else None
    probs = torch.from_numpy(payload["teacher_probs"].astype(np.float32)).cuda() / 255 if payload is not None else None
    if probs is not None:
        probs /= probs.sum(-1, keepdim=True).clamp_min(1)
    dev = load_sequences(DATA / "dev.npy", 256, CONFIG["dev_sequences"]).cuda()
    rng = torch.Generator().manual_seed(seed + 100000)
    elapsed = 0.0
    curve = []
    for step in range(c["updates"] + 1):
        if step % (400 if teacher else 50) == 0:
            row = {"step": step, "dev_loss": losses(model, dev).mean().item(), "train_seconds": elapsed}
            curve.append(row)
            print(json.dumps(row), flush=True)
        if step == c["updates"]:
            break
        torch.cuda.synchronize()
        begin = time.perf_counter()
        selection = torch.randint(len(seq), (32,), generator=rng).cuda()
        batch = seq[selection]
        for optimizer, base in zip(optimizers, [c["muon_lr"], c["adamw_lr"]]):
            for group in optimizer.param_groups:
                group["lr"] = base * lr_ratio(step + 1, c)
            optimizer.zero_grad(set_to_none=True)
        with autocast():
            if payload is None or alpha == 0:
                loss = model(batch[:, :-1], batch[:, 1:])
            else:
                logp = F.log_softmax(logits(model, batch[:, :-1]), dim=-1)
                gold = -logp.gather(-1, batch[:, 1:, None]).squeeze(-1)
                mass = probs[selection].sum(-1)
                teacher_terms = -(logp.gather(-1, ids[selection]) * probs[selection]).sum(-1)
                # Put unrecorded teacher probability mass on the original target.
                loss = ((1 - alpha * mass) * gold + alpha * teacher_terms).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"], error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        torch.cuda.synchronize()
        elapsed += time.perf_counter() - begin
    if teacher:
        torch.save(model.state_dict(), output / "teacher.pt")
    result = {"seed": seed, "alpha": alpha, "mode": "teacher" if teacher else "sparse_target_distillation",
              "updates": c["updates"], "initial_state_sha256": initial, "curve": curve,
              "final_dev_loss": curve[-1]["dev_loss"], "train_seconds": elapsed,
              "first_crossing": next((r["step"] for r in curve if r["dev_loss"] <= 3.5), None),
              "audit_accessed": audit, "source_sha256": file_hash(__file__),
              "training_config": c, "teacher_target_format": "top2 uint16 IDs + uint8 probabilities; residual mass on gold"}
    if audit:
        audit_path = ROOT / "data/autoresearch-v1/audit.npy"
        data = load_sequences(audit_path, 256, PROTOCOL["audit_sequences"]).cuda()
        result.update(audit_loss=losses(model, data).mean().item(), audit_sha256=file_hash(audit_path))
    result["total_seconds"] = time.perf_counter() - start
    write(output / "result.json", result)
    return result


@torch.no_grad()
def materialize(root, fraction):
    start = time.perf_counter()
    full = load_sequences(DATA / "original.npy", 256)
    rng = np.random.default_rng(20261119)
    selection = np.sort(rng.choice(len(full), int(len(full) * fraction), replace=False))
    selected = full[torch.from_numpy(selection)].cuda()
    model, _ = setup(271)
    model.load_state_dict(torch.load(root / "strong-teacher/teacher.pt", weights_only=True))
    all_ids, all_probs = [], []
    for batch in selected.split(32):
        with autocast():
            p = logits(model, batch[:, :-1]).softmax(-1)
            values, ids = p.topk(2, dim=-1)
        all_ids.append(ids.cpu().numpy().astype(np.uint16))
        all_probs.append((values * 255).round().cpu().numpy().astype(np.uint8))
    path = root / "artifacts" / f"distill-f{fraction:.5f}.npz"
    path.parent.mkdir(exist_ok=True)
    np.savez(path, tokens=selected.cpu().numpy().astype(np.uint16),
             teacher_ids=np.concatenate(all_ids), teacher_probs=np.concatenate(all_probs))
    write(path.with_suffix(".json"), {"artifact_bytes": path.stat().st_size, "sha256": file_hash(path),
                                     "construction_seconds": time.perf_counter() - start,
                                     "fraction": fraction, "sequences": len(selection)})
    return path


def search(root):
    root.mkdir(parents=True, exist_ok=True)
    # This follow-up was selected after the open-loop gradient sweep failed.
    write(root / "distill-plan.json", {"trigger": "Low-rank open-loop gradients produced dev loss above 10.",
        "teacher_updates": 1600, "teacher_seed": 271, "fractions": [0.0625, 0.125],
        "initial_alphas": [0, 0.25, 0.5, 0.75], "adaptive": "Refine alpha around the lowest development loss.",
        "objective_changed": True, "model_and_optimizer_fixed": True, "audit_accessed": False})
    write(root / "distill-status.json", {"status": "running", "active": "strong-teacher"})
    train(root / "strong-teacher", seed=271, teacher=True)
    history = []
    paths = {}

    def trial(fraction, alpha, seed=17, phase="initial"):
        path = paths[fraction]
        name = f"distill-f{fraction:.5f}-a{alpha:.3f}"
        write(root / "distill-status.json", {"status": "running", "active": f"{name}-seed{seed}", "completed": len(history)})
        with np.load(path) as f:
            payload = {k: f[k] for k in f.files}
        row = train(root / "distill-search" / f"{name}-seed{seed}", seed, payload, alpha)
        row.update(name=name, fraction=fraction, phase=phase, artifact=str(path.relative_to(ROOT)),
                   artifact_bytes=path.stat().st_size, artifact_sha256=file_hash(path))
        history.append(row)
        write(root / "distill-history.json", history)
        gc.collect()
        torch.cuda.empty_cache()
        return row

    for fraction in [0.0625, 0.125]:
        paths[fraction] = materialize(root, fraction)
        for alpha in [0, 0.25, 0.5, 0.75]:
            trial(fraction, alpha)
    best = min(history, key=lambda r: r["final_dev_loss"])
    for alpha in sorted({max(0.0625, best["alpha"] - 0.125), min(1.0, best["alpha"] + 0.125)}):
        trial(best["fraction"], alpha, phase="adaptive_alpha")
    top = sorted(history, key=lambda r: r["final_dev_loss"])[:2]
    for row in top:
        trial(row["fraction"], row["alpha"], seed=42, phase="adaptive_seed_check")
    winner = min(top, key=lambda r: np.mean([x["final_dev_loss"] for x in history if x["name"] == r["name"]]))
    selection = {k: winner[k] for k in ["name", "fraction", "alpha", "artifact", "artifact_bytes"]}
    selection.update(audit_accessed=False, history_sha256=file_hash(root / "distill-history.json"))
    write(root / "distill-finalist.json", selection)
    write(root / "distill-status.json", {"status": "complete", "completed": len(history), "finalist": selection})


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "runs/autoresearch-v1")
    args = parser.parse_args()
    search(args.root.resolve())
