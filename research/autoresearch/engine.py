"""Fixed-learner training and compressed-gradient experiments."""

import contextlib
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from train import GPT, load_sequences, make_optimizers, lr_ratio, state_hash, file_hash

CONFIG = json.loads((ROOT / "configs/tinystories-v0.json").read_text())
PROTOCOL = json.loads((Path(__file__).parent / "protocol.json").read_text())
DATA = ROOT / "data/tinystories-v0"


def setup(seed):
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT(CONFIG).cuda()
    return model, make_optimizers(model, CONFIG, "cuda")


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16)


def logits(model, x):
    h = model.token(x) + model.position(torch.arange(x.shape[1], device=x.device))
    for block in model.blocks:
        h = block(h)
    return model.head(model.norm(h)).float()


@torch.no_grad()
def losses(model, sequences, batch_size=32):
    model.eval()
    values = []
    for batch in sequences.split(batch_size):
        with autocast():
            out = logits(model, batch[:, :-1])
            loss = F.cross_entropy(out.flatten(0, 1), batch[:, 1:].flatten(), reduction="none")
        values.append(loss.view(len(batch), -1).mean(1))
    model.train()
    return torch.cat(values)


def factors(matrix, rank):
    # The random SVD must not change the model's training RNG state.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        torch.manual_seed(12345)
        u, s, v = torch.svd_lowrank(matrix.float(), q=min(rank + 8, min(matrix.shape)), niter=3)
    return {"left": (u[:, :rank] * s[:rank]).half().cpu(), "right": v[:, :rank].T.half().cpu()}


def compress_gradients(model, rank):
    return {name: factors(p.grad, rank) if p.ndim == 2 else {"dense": p.grad.half().cpu()}
            for name, p in model.named_parameters() if p.grad is not None}


def slice_artifact(source, rank, stride):
    knots = source["knots"][::stride]
    if knots[-1] != source["knots"][-1]:
        knots = knots + [source["knots"][-1]]
    frames = []
    for step in knots:
        frame = source["frames"][source["knots"].index(step)]
        frames.append({name: {"dense": value["dense"].clone()} if "dense" in value else
                       {"left": value["left"][:, :rank].clone(), "right": value["right"][:rank].clone()}
                       for name, value in frame.items()})
    return {"knots": knots, "frames": frames, "rank": rank, "source_seed": source["source_seed"],
            "initial_state_sha256": source["initial_state_sha256"], "kind": "low_rank_gradient_trajectory"}


class GradientDecoder:
    def __init__(self, artifact, scale=1.0):
        self.knots = artifact["knots"]
        self.frames = [{name: {k: v.cuda().float() for k, v in value.items()}
                        for name, value in frame.items()} for frame in artifact["frames"]]
        self.scale = scale
        self.cache = {}

    def frame(self, index):
        if index not in self.cache:
            self.cache[index] = {name: value["dense"] if "dense" in value else value["left"] @ value["right"]
                                 for name, value in self.frames[index].items()}
        return self.cache[index]

    def apply(self, model, step):
        hi = min(int(np.searchsorted(self.knots, step)), len(self.knots) - 1)
        lo = max(0, hi - 1)
        mix = 0 if hi == lo else (step - self.knots[lo]) / (self.knots[hi] - self.knots[lo])
        a, b = self.frame(lo), self.frame(hi)
        for name, p in model.named_parameters():
            p.grad = torch.lerp(a[name], b[name], mix).mul(self.scale)
        self.cache = {i: value for i, value in self.cache.items() if i in (lo, hi)}


def run(seed, output, sequences=None, gradient_artifact=None, capture=False, checkpoint=False,
        gradient_scale=1.0, audit=False, updates=None):
    """Run the fixed update rule. Audit access is disabled during search."""
    start = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    model, optimizers = setup(seed)
    initial_hash = state_hash(model)
    if sequences is None and gradient_artifact is None:
        sequences = load_sequences(DATA / "original.npy", CONFIG["sequence_length"])
    if sequences is not None:
        sequences = sequences.cuda()
    dev = load_sequences(DATA / "dev.npy", CONFIG["sequence_length"], CONFIG["dev_sequences"]).cuda()
    decoder = GradientDecoder(gradient_artifact, gradient_scale) if gradient_artifact is not None else None
    rng = torch.Generator().manual_seed(seed + 100000)
    frames, knots, curve = [], [], []
    train_seconds, capture_seconds = 0.0, 0.0
    total_updates = updates or CONFIG["updates"]
    log = (output / "metrics.jsonl").open("x")

    def evaluate(step):
        value = losses(model, dev).mean().item()
        row = {"step": step, "dev_loss": value, "train_seconds": train_seconds,
               "elapsed_seconds": time.perf_counter() - start}
        curve.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    evaluate(0)
    for step in range(1, total_updates + 1):
        torch.cuda.synchronize()
        step_start = time.perf_counter()
        for optimizer, base in zip(optimizers, [CONFIG["muon_lr"], CONFIG["adamw_lr"]]):
            for group in optimizer.param_groups:
                group["lr"] = base * lr_ratio(step, CONFIG)
            optimizer.zero_grad(set_to_none=True)
        if decoder is None:
            indices = torch.randint(len(sequences), (CONFIG["batch_size"],), generator=rng).cuda()
            batch = sequences[indices]
            with autocast():
                loss = model(batch[:, :-1], batch[:, 1:])
            loss.backward()
        else:
            decoder.apply(model, step)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
        torch.cuda.synchronize()
        before_capture = time.perf_counter()
        if capture and step in PROTOCOL["gradient_knots"]:
            frames.append(compress_gradients(model, PROTOCOL["gradient_max_rank"]))
            knots.append(step)
        torch.cuda.synchronize()
        captured = time.perf_counter() - before_capture
        capture_seconds += captured
        for optimizer in optimizers:
            optimizer.step()
        torch.cuda.synchronize()
        train_seconds += time.perf_counter() - step_start - captured
        if checkpoint and step in PROTOCOL["teacher_checkpoints"]:
            torch.save(model.state_dict(), output / f"checkpoint-{step}.pt")
        if step % CONFIG["eval_every"] == 0 or step == total_updates:
            evaluate(step)
    result = {"seed": seed, "updates": total_updates, "initial_state_sha256": initial_hash,
              "config_sha256": file_hash(ROOT / "configs/tinystories-v0.json"),
              "engine_sha256": file_hash(__file__), "trainer_sha256": file_hash(ROOT / "train.py"),
              "protocol_sha256": file_hash(Path(__file__).parent / "protocol.json"),
              "mode": "gradient_replay" if decoder else "data", "gradient_scale": gradient_scale,
              "curve": curve, "final_dev_loss": curve[-1]["dev_loss"],
              "first_crossing": next((x["step"] for x in curve if x["dev_loss"] <= CONFIG["target_loss"]), None),
              "train_seconds": train_seconds, "capture_seconds": capture_seconds,
              "audit_accessed": audit, "train_token_exposures": 0 if decoder else total_updates * CONFIG["batch_size"] * CONFIG["sequence_length"],
              "software": {"torch": str(torch.__version__), "cuda": torch.version.cuda}}
    if audit:
        audit_path = ROOT / "data/autoresearch-v1/audit.npy"
        result["audit_sha256"] = file_hash(audit_path)
        audit_data = load_sequences(audit_path, CONFIG["sequence_length"], PROTOCOL["audit_sequences"]).cuda()
        result["audit_loss"] = losses(model, audit_data).mean().item()
    if capture:
        torch.save({"knots": knots, "frames": frames, "source_seed": seed,
                    "initial_state_sha256": initial_hash}, output / "gradient-source.pt")
    result["total_seconds"] = time.perf_counter() - start
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    log.close()
    return result
