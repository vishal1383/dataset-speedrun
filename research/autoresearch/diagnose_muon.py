"""Measure low-rank distortion before and after Muon's update transform."""

import json

import torch

from engine import ROOT, DATA, CONFIG, setup, load_sequences, autocast, factors
from search import write


def main():
    model, _ = setup(17)
    seq = load_sequences(DATA / "original.npy", 256).cuda()
    rng = torch.Generator().manual_seed(100017)
    batch = seq[torch.randint(len(seq), (32,), generator=rng).cuda()]
    with autocast():
        loss = model(batch[:, :-1], batch[:, 1:])
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    rows = []
    for name, p in model.named_parameters():
        if not name.startswith("blocks.0.") or p.ndim != 2:
            continue
        g = p.grad.detach().clone()
        for rank in [1, 4, 8]:
            latent = factors(g, rank)
            approx = latent["left"].float().cuda() @ latent["right"].float().cuda()
            updates = []
            for grad in [g, approx]:
                # Zero parameters isolate the gradient-driven update from weight decay.
                q = torch.nn.Parameter(torch.zeros_like(p))
                optimizer = torch.optim.Muon([q], lr=0.02, weight_decay=0, adjust_lr_fn="original")
                q.grad = grad.clone()
                optimizer.step()
                updates.append(q.detach().clone())
            rows.append({"parameter": name, "rank": rank,
                         "raw_relative_error": ((g - approx).norm() / g.norm()).item(),
                         "muon_update_relative_error": ((updates[0] - updates[1]).norm() / updates[0].norm()).item()})
    path = ROOT / "runs/autoresearch-v1/muon-diagnostic.json"
    write(path, {"scope": "First optimizer update, seed 17, first transformer block; diagnostic only.", "rows": rows})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
