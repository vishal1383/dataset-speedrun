"""Evaluate compressed parameter targets as state-dependent gradient generators."""

import argparse
import json
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/autoresearch"))
from engine import CONFIG, autocast, losses, setup
from train import file_hash, lr_ratio, state_hash
from competition.runner import read_tokens, write

STEPS = [0, 10, 20, 50, 100, 150, 200, 300, 400]


def encode_tensor(value, bits, block=128):
    """Use symmetric block quantization and pack two four-bit values per byte."""
    flat = value.float().flatten().cpu()
    padded = F.pad(flat, (0, (-flat.numel()) % block)).reshape(-1, block)
    limit = 2 ** (bits - 1) - 1
    scale = (padded.abs().amax(1) / limit).clamp_min(1e-7).half()
    q = (padded / scale.float()[:, None]).round().clamp(-limit, limit).to(torch.int16)
    if bits == 4:
        unsigned = (q.flatten() + limit).to(torch.uint8)
        packed = unsigned[::2] | (unsigned[1::2] << 4)
    else:
        packed = q.to(torch.int8).flatten()
    return {"values": packed, "scale": scale, "shape": list(value.shape), "bits": bits, "block": block}


def decode_tensor(code):
    bits, block = code["bits"], code["block"]
    q = code["values"]
    if bits == 4:
        q = torch.stack((q & 15, q >> 4), dim=1).flatten().float() - 7
    else:
        q = q.float()
    flat = (q.reshape(-1, block) * code["scale"].float()[:, None]).flatten()
    return flat[:math.prod(code["shape"])].reshape(code["shape"])


def decode(artifact):
    kind = artifact["kind"]
    if kind.startswith("q"):
        return {n: decode_tensor(v).cuda() for n, v in artifact["parameters"].items()}
    if kind == "fp16":
        return {n: v.cuda().float() for n, v in artifact["parameters"].items()}
    initial, _ = setup(artifact["base_seed"])
    if state_hash(initial) != artifact["base_state_sha256"]:
        raise ValueError("Low-rank decoder initialization changed.")
    values = {}
    for n, p in initial.named_parameters():
        v = artifact["parameters"][n]
        delta = v["dense"].cuda().float() if "dense" in v else v["left"].cuda().float() @ v["right"].cuda().float()
        values[n] = p.detach() + delta
    return values


def build(output, kinds):
    output.mkdir(parents=True, exist_ok=True)
    teacher_path = ROOT / "runs/autoresearch-v1/teacher/checkpoint-400.pt"
    teacher = torch.load(teacher_path, map_location="cpu", weights_only=True)
    initial, _ = setup(271)
    base_hash = state_hash(initial)
    base = {n: p.detach().cpu() for n, p in initial.named_parameters()}
    dev = read_tokens(ROOT / "data/round2/dev.npy").cuda()
    for kind in kinds:
        path = output / f"{kind}.pt"
        if path.exists():
            continue
        start = time.perf_counter()
        if kind.startswith("q"):
            parameters = {n: encode_tensor(p, int(kind[1:])) for n, p in teacher.items()}
        elif kind == "fp16":
            parameters = {n: p.half() for n, p in teacher.items()}
        else:
            rank = int(kind[4:])
            parameters = {}
            for n, p in teacher.items():
                delta = (p - base[n]).cuda()
                if delta.ndim < 2:
                    parameters[n] = {"dense": delta.half().cpu()}
                else:
                    torch.manual_seed(7181)
                    u, s, v = torch.svd_lowrank(delta, q=min(rank + 8, min(delta.shape)), niter=3)
                    parameters[n] = {"left": (u[:, :rank] * s[:rank]).half().cpu(), "right": v[:, :rank].T.half().cpu()}
        torch.cuda.synchronize()
        construction = time.perf_counter() - start
        artifact = {"kind": kind, "parameters": parameters, "base_seed": 271,
                    "base_state_sha256": base_hash, "teacher_sha256": file_hash(teacher_path),
                    "teacher_updates": 400, "config_sha256": file_hash(ROOT / "configs/tinystories-v0.json")}
        torch.save(artifact, path)
        decoded = decode(artifact)
        model, _ = setup(17)
        model.load_state_dict(decoded)
        record = {"kind": kind, "artifact_bytes": path.stat().st_size, "artifact_sha256": file_hash(path),
                  "teacher_sha256": file_hash(teacher_path), "construction_seconds": construction,
                  "direct_load_dev_loss": losses(model, dev).mean().item(),
                  "reconstruction_relative_error": float(torch.sqrt(sum((decoded[n].cpu() - p).square().sum() for n, p in teacher.items()) /
                                                                         sum(p.square().sum() for p in teacher.values())))}
        write(path.with_suffix(".json"), record)
        print(json.dumps(record), flush=True)


def potential(model, target, kind):
    terms = []
    for name, p in model.named_parameters():
        if kind == "quadratic":
            terms.append((p - target[name]).square().sum() * 0.5)
        else:
            terms.append(F.huber_loss(p, target[name], reduction="sum", delta=0.01))
    return torch.stack(terms).sum()


def run(output, seed, artifact_path=None, field="quadratic", evaluation=None, updates=400):
    if (output / "result.json").exists():
        return json.loads((output / "result.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    wall_start = time.perf_counter()
    evaluation = evaluation or ROOT / "data/round2/dev.npy"
    samples = read_tokens(ROOT / "data/tinystories-v0/original.npy").cuda() if artifact_path is None else None
    eval_data = read_tokens(evaluation).cuda()
    target, artifact = None, None
    # Decode before initializing the student because a low-rank decoder uses its own seed.
    torch.cuda.synchronize()
    decode_start = time.perf_counter()
    if artifact_path:
        artifact = torch.load(artifact_path, map_location="cpu", weights_only=True)
        if artifact["config_sha256"] != file_hash(ROOT / "configs/tinystories-v0.json"):
            raise ValueError("Artifact learner configuration mismatch.")
        target = decode(artifact)
    torch.cuda.synchronize()
    decode_seconds = time.perf_counter() - decode_start
    model, optimizers = setup(seed)
    initial = state_hash(model)
    rng = torch.Generator().manual_seed(seed + 100000)
    seconds, curve = 0.0, []
    for step in range(updates + 1):
        if step in STEPS or step == updates:
            row = {"step": step, "loss": losses(model, eval_data).mean().item(), "train_seconds": seconds}
            curve.append(row)
            print(json.dumps({"run": str(output), **row}), flush=True)
        if step == updates:
            break
        torch.cuda.synchronize()
        start = time.perf_counter()
        for opt, base in zip(optimizers, [CONFIG["muon_lr"], CONFIG["adamw_lr"]]):
            for group in opt.param_groups:
                group["lr"] = base * lr_ratio(step + 1, CONFIG)
            opt.zero_grad(set_to_none=True)
        if target is None:
            ids = torch.randint(len(samples), (CONFIG["batch_size"],), generator=rng).cuda()
            batch = samples[ids]
            with autocast():
                loss = model(batch[:, :-1], batch[:, 1:])
        else:
            loss = potential(model, target, field)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
        for opt in optimizers:
            opt.step()
        torch.cuda.synchronize()
        seconds += time.perf_counter() - start
    result = {"mode": "full-data" if target is None else "compressed-target-potential",
              "seed": seed, "kind": artifact["kind"] if artifact else None, "field": field if artifact else None,
              "updates": updates, "curve": curve, "final_loss": curve[-1]["loss"], "train_seconds": seconds,
              "decode_seconds": decode_seconds, "wall_seconds": time.perf_counter() - wall_start,
              "initial_state_sha256": initial, "evaluation_sha256": file_hash(evaluation),
              "artifact_sha256": file_hash(artifact_path) if artifact_path else file_hash(ROOT / "data/tinystories-v0/original.npy"),
              "artifact_bytes": artifact_path.stat().st_size if artifact_path else (ROOT / "data/tinystories-v0/original.npy").stat().st_size,
              "source_sha256": file_hash(__file__), "trainer_sha256": file_hash(ROOT / "train.py"),
              "config_sha256": file_hash(ROOT / "configs/tinystories-v0.json"),
              "host": platform.node(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
              "gpu": torch.cuda.get_device_name(), "training_objective_changed": target is not None,
              "training_token_exposures": updates * 8192 if target is None else 0,
              "training_data_read_by_candidate": target is None, "status": "complete"}
    write(output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["build", "search", "run"])
    p.add_argument("--root", type=Path, default=Path("runs/gradient-field-v1"))
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--kinds", nargs="+", default=["q4", "q8", "fp16", "rank8", "rank32", "rank64"])
    p.add_argument("--field", choices=["quadratic", "huber"], default="quadratic")
    p.add_argument("--artifact", type=Path)
    p.add_argument("--evaluation", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--updates", type=int, default=400)
    args = p.parse_args()
    if args.mode == "build":
        build(args.root / "artifacts", args.kinds)
    elif args.mode == "run":
        run(args.output, args.seed, args.artifact, args.field, args.evaluation, args.updates)
    else:
        run(args.root / f"search/full-{args.seed}", args.seed)
        for kind in args.kinds:
            for field in ["quadratic", "huber"]:
                run(args.root / f"search/{kind}-{field}-{args.seed}", args.seed, args.root / f"artifacts/{kind}.pt", field)


if __name__ == "__main__":
    main()
