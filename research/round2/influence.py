"""Estimate training-window alignment with development gradients."""

import json
import sys
from pathlib import Path
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/autoresearch"))
from engine import setup, autocast, losses
from train import load_sequences
from competition.runner import read_tokens, write


def main():
    start = time.perf_counter()
    root = ROOT / "runs/round2"
    scores = dict(np.load(root / "scores.npz"))
    full = load_sequences(ROOT / "data/tinystories-v0/original.npy", 256).cuda()
    dev = read_tokens(ROOT / "data/round2/dev.npy").cuda()
    effects = []
    for step in [50, 400]:
        model, _ = setup(271)
        model.load_state_dict(torch.load(ROOT / f"runs/autoresearch-v1/teacher/checkpoint-{step}.pt", weights_only=True))
        model.zero_grad(set_to_none=True)
        for batch in dev.split(32):
            with autocast():
                loss = model(batch[:, :-1], batch[:, 1:]) / (len(dev) / 32)
            loss.backward()
        # Symmetric finite differences estimate per-window gradient alignment.
        # This scoring teacher changes; the submitted learner does not.
        with torch.no_grad():
            norm = torch.sqrt(sum(p.grad.square().sum() for p in model.parameters()))
            direction = {n: p.grad.clone() / norm for n, p in model.named_parameters()}
            original = {n: p.clone() for n, p in model.named_parameters()}
            estimates = []
            for sign in [-1, 1]:
                for n, p in model.named_parameters():
                    p.copy_(original[n] + sign * .05 * direction[n])
                estimates.append(losses(model, full).cpu().numpy())
        effect = (estimates[1] - estimates[0]) / .1
        effects.append(effect / max(float(effect.std()), 1e-8))
    scores["influence"] = np.mean(effects, axis=0)
    np.savez(root / "scores.npz", **scores)
    write(root / "influence-cost.json", {"seconds": time.perf_counter() - start,
        "method": "Mean standardized symmetric finite-difference gradient alignment at teacher steps 50 and 400",
        "development_access": True, "audit_access": False})


if __name__ == "__main__":
    main()
