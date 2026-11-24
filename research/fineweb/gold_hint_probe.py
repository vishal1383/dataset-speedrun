"""Short matched probe of answer-conditioned distillation on FineWeb."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F
from signal_screen import ROOT, definitions, mapped_tokens, sparse_loss, write


def main(out, teacher_kind):
    out.mkdir(parents=True, exist_ok=False)
    dist.init_process_group("nccl", device_id=torch.device("cuda:0"))
    ns = definitions()
    net = ns["GPT"](50304, 12, 768).cuda()
    teacher = ns["GPT"](50304, 12, 768).cuda().eval()
    start = torch.load(
        ROOT / "runs/signal-screen20261123-v5/shared.pt",
        map_location="cpu",
        weights_only=True,
    )
    if teacher_kind == "frozen_student":
        teacher.load_state_dict(start["model"])
    else:
        teacher.load_state_dict(
            torch.load(
                ROOT / "runs/fineweb-v1/paired17/fineweb-seed17/latest.pt",
                map_location="cpu",
                weights_only=True,
            )["model"]
        )
    width, examples, steps = 16, 256, 16
    source = mapped_tokens(ROOT / "data/fineweb10B/fineweb_train_000001.bin")
    dev = mapped_tokens(ROOT / "data/fineweb10B/fineweb_val_000000.bin")
    methods = ["hard", "ordinary", "answer", "shuffled"]
    labels = {k: [] for k in methods[1:]}
    stats = {k: [] for k in labels}

    def read(data, offset, rows):
        a = torch.from_numpy(
            np.array(data[offset : offset + rows * width + 1], dtype=np.int64)
        ).cuda()
        return a[:-1].reshape(rows, width), a[1:].reshape(rows, width)

    def logits(model, x, positions=None):
        h = model.norm1(model.embed(x))
        for block in model.blocks:
            h = block(h)
        if positions is not None:
            h = h[torch.arange(len(h), device=h.device), positions]
        z = model.proj(model.norm2(h)).float()
        return 15 * z * (z.square() + 225).rsqrt()

    begun = time.monotonic()
    with torch.no_grad():
        for step in range(steps):
            x, y = read(source, 20_000_000 + step * examples * width, examples)
            outputs = {k: [] for k in labels}
            # One row per prediction. Read only the final prefix position of each prompt.
            for left in range(0, examples, 16):
                seq = x[left : left + 16].repeat_interleave(width, 0)
                ends = torch.arange(width, device="cuda").repeat(16)
                gold = y[left : left + 16].flatten()
                logits0 = logits(teacher, seq, ends)
                col = torch.arange(2 * width + 2, device="cuda")[None, :]
                n = ends[:, None] + 1
                idx = (
                    torch.where(col < n, col, col - n - 2)
                    .clamp(0, width - 1)
                    .expand(len(seq), -1)
                )
                base = seq.gather(1, idx)
                base = torch.where(col == n + 1, 50256, base)
                variants = [("ordinary", logits0)]
                for key, hint in [
                    ("answer", gold),
                    ("shuffled", gold.view(16, width).roll(1, 0).flatten()),
                ]:
                    prompt = torch.where(col == n, hint[:, None], base)
                    variants.append((key, logits(teacher, prompt, 2 * ends + 3)))
                for key, z in variants:
                    p = z.softmax(-1)
                    v, i = p.topk(8)
                    pgold = p.gather(1, gold[:, None]).mean().item()
                    stats[key].append(
                        [pgold, (p.argmax(-1) == gold).float().mean().item()]
                    )
                    outputs[key].append((v.half().cpu(), i.int().cpu()))
            for key, parts in outputs.items():
                labels[key].append(
                    tuple(torch.cat(t).reshape(examples, width, 8) for t in zip(*parts))
                )
            if (step + 1) % 4 == 0:
                print("Label batches", step + 1, flush=True)
    target_seconds = time.monotonic() - begun
    torch.cuda.empty_cache()

    @torch.no_grad()
    def evaluate():
        x, y = read(dev, 2_000_000, 4096)
        return (
            sum(net(a, b).item() for a, b in zip(x.split(128), y.split(128)))
            / y.numel()
        )

    rows = {}
    for key in methods:
        net.load_state_dict(start["model"])
        adam = torch.optim.AdamW(
            [
                dict(params=[net.embed.weight], lr=0.7),
                dict(params=[net.proj.weight], lr=0.004),
                dict(params=[p for p in net.parameters() if p.ndim < 2], lr=0.015),
            ],
            betas=(0.8, 0.95),
            eps=1e-10,
            weight_decay=0.001,
            fused=True,
        )
        mu = ns["Muon"](
            [p for p in net.blocks.parameters() if p.ndim >= 2],
            lr=0.025,
            weight_decay=0.05,
        )
        for opt, state in zip([adam, mu], start["optimizers"]):
            opt.load_state_dict(state)
        curve = [evaluate()]
        for step in range(steps):
            x, y = read(source, 20_000_000 + step * examples * width, examples)
            for b in range(0, examples, 64):
                xx, yy = x[b : b + 64], y[b : b + 64]
                z = logits(net, xx)
                if key == "hard":
                    loss = F.cross_entropy(
                        z.flatten(0, 1), yy.flatten(), reduction="sum"
                    )
                else:
                    v, i = labels[key][step]
                    loss = sparse_loss(
                        z, yy, i[b : b + 64].cuda(), v[b : b + 64].float().cuda(), 0.5
                    )
                (loss * (524288 / (width * examples))).backward()
            adam.step()
            mu.step()
            net.zero_grad(set_to_none=True)
            if step + 1 in (8, 16):
                curve.append(evaluate())
        rows[key] = curve
        result = dict(
            teacher_kind=teacher_kind,
            width=width,
            updates=steps,
            examples_per_step=examples,
            dev_tokens=65536,
            dev_offset=2000000,
            train_offset=20000000,
            loss_steps=[0, 8, 16],
            curves=rows,
            teacher_statistics={
                k: dict(
                    mean_gold_probability=float(np.mean(a, 0)[0]),
                    gold_top1=float(np.mean(a, 0)[1]),
                )
                for k, a in stats.items()
            },
            joint_target_construction_seconds=target_seconds,
        )
        write(out / "result.json", result)
        print(key, curve, flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--teacher", choices=["mature", "frozen_student"], default="mature"
    )
    args = parser.parse_args()
    main(args.output, args.teacher)
