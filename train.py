"""Train a fixed GPT recipe and record loss at a fixed update budget."""

import argparse
import contextlib
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(width, bias=False)
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width, bias=False)
        self.norm2 = nn.LayerNorm(width, bias=False)
        self.up = nn.Linear(width, 4 * width, bias=False)
        self.down = nn.Linear(4 * width, width, bias=False)

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(self.norm1(x)).reshape(b, t, 3, self.heads, c // self.heads).unbind(2)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                          v.transpose(1, 2), is_causal=True)
        x = x + self.proj(y.transpose(1, 2).contiguous().view(b, t, c))
        return x + self.down(F.gelu(self.up(self.norm2(x)), approximate="tanh"))


class GPT(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.token = nn.Embedding(c["vocab_size"], c["width"])
        self.position = nn.Embedding(c["sequence_length"], c["width"])
        self.blocks = nn.ModuleList([Block(c["width"], c["heads"]) for _ in range(c["layers"])])
        self.norm = nn.LayerNorm(c["width"], bias=False)
        self.head = nn.Linear(c["width"], c["vocab_size"], bias=False)
        self.apply(self.initialize)
        for block in self.blocks:
            nn.init.normal_(block.proj.weight, std=0.02 / math.sqrt(2 * c["layers"]))
            nn.init.normal_(block.down.weight, std=0.02 / math.sqrt(2 * c["layers"]))

    @staticmethod
    def initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    def forward(self, x, y):
        h = self.token(x) + self.position(torch.arange(x.shape[1], device=x.device))
        for block in self.blocks:
            h = block(h)
        logits = self.head(self.norm(h))
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), y.reshape(-1))


def make_optimizers(model, c, device):
    hidden = [p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2]
    ids = {id(p) for p in hidden}
    other = [p for p in model.parameters() if id(p) not in ids]
    return [torch.optim.Muon(hidden, lr=c["muon_lr"], weight_decay=c["weight_decay"],
                             momentum=0.95, nesterov=True, ns_steps=5, adjust_lr_fn="original"),
            torch.optim.AdamW(other, lr=c["adamw_lr"], betas=(0.9, 0.95), eps=1e-8,
                              weight_decay=c["weight_decay"], fused=(device == "cuda"))]


def load_sequences(path, length, count=None):
    tokens = np.load(path)
    n = (len(tokens) - 1) // length
    if count is not None:
        if n < count:
            raise ValueError(f"Insufficient evaluation tokens in {path}")
        n = count
    # Adjacent contexts share their boundary token; each target appears once.
    starts = np.arange(n)[:, None] * length
    return torch.from_numpy(tokens[starts + np.arange(length + 1)].astype(np.int64))


def state_hash(model):
    h = hashlib.sha256()
    for name, p in model.state_dict().items():
        h.update(name.encode())
        h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def lr_ratio(step, c):
    if step <= c["warmup_updates"]:
        return step / c["warmup_updates"]
    progress = (step - c["warmup_updates"]) / (c["updates"] - c["warmup_updates"])
    return c["minimum_lr_ratio"] + (1 - c["minimum_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * progress))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/tinystories-v0.json"))
    parser.add_argument("--data", type=Path, default=Path("data/tinystories-v0"))
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    start = time.perf_counter()
    c = json.loads(args.config.read_text())
    if args.recipe not in c["recipes"]:
        raise ValueError("Recipe is outside this track.")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.data / "manifest.json").read_text())
    for name in [f"{args.recipe}.npy", "dev.npy", "test.npy", "tokenizer.json"]:
        if file_hash(args.data / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Data hash mismatch: {name}")
    if manifest["vocab_size"] != c["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match config.")
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    model = GPT(c).to(args.device)
    initial_hash = state_hash(model)
    optimizers = make_optimizers(model, c, args.device)
    train_model = torch.compile(model) if c["compile"] else model
    train = load_sequences(args.data / f"{args.recipe}.npy", c["sequence_length"]).to(args.device)
    dev = load_sequences(args.data / "dev.npy", c["sequence_length"], c["dev_sequences"]).to(args.device)
    test = load_sequences(args.data / "test.npy", c["sequence_length"], c["test_sequences"]).to(args.device)
    generator = torch.Generator(device="cpu").manual_seed(args.seed + 100000)

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    def autocast():
        return torch.autocast("cuda", dtype=torch.bfloat16) if args.device == "cuda" else contextlib.nullcontext()

    @torch.no_grad()
    def evaluate(data):
        model.eval()
        total = 0.0
        for batch in data.split(c["batch_size"]):
            with autocast():
                loss = model(batch[:, :-1], batch[:, 1:])
            total += loss.item() * len(batch)
        model.train()
        return total / len(data)

    curve = []
    train_seconds = 0.0
    first_crossing = None
    sync()
    setup_seconds = time.perf_counter() - start
    training_start = time.perf_counter()
    log = (args.output / "metrics.jsonl").open("x")

    def record(step, loss=None):
        nonlocal first_crossing
        dev_loss = evaluate(dev)
        row = {"step": step, "tokens_seen": step * c["batch_size"] * c["sequence_length"],
               "dev_loss": dev_loss, "train_loss": loss, "train_seconds": train_seconds,
               "elapsed_seconds": time.perf_counter() - training_start}
        if dev_loss <= c["target_loss"] and first_crossing is None:
            first_crossing = dict(row)
        curve.append(row)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)

    record(0)
    for step in range(1, c["updates"] + 1):
        sync()
        step_start = time.perf_counter()
        indices = torch.randint(len(train), (c["batch_size"],), generator=generator).to(args.device)
        batch = train[indices]
        for optimizer, base_lr in zip(optimizers, [c["muon_lr"], c["adamw_lr"]]):
            for group in optimizer.param_groups:
                group["lr"] = base_lr * lr_ratio(step, c)
            optimizer.zero_grad(set_to_none=True)
        with autocast():
            loss = train_model(batch[:, :-1], batch[:, 1:])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), c["gradient_clip"], error_if_nonfinite=True)
        for optimizer in optimizers:
            optimizer.step()
        sync()
        train_seconds += time.perf_counter() - step_start
        if step % c["eval_every"] == 0 or step == c["updates"]:
            record(step, loss.item())
    # The audit test is evaluated once, after the fixed update budget.
    test_loss = evaluate(test)
    sync()
    experiment_seconds = time.perf_counter() - start
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    result = {"status": "complete", "track": c["track"], "recipe": args.recipe, "seed": args.seed,
              "config": c, "config_sha256": file_hash(args.config), "train_source_sha256": file_hash(__file__),
              "git_revision": revision, "data_manifest_sha256": file_hash(args.data / "manifest.json"),
              "train_artifact": manifest["files"][f"{args.recipe}.npy"],
              "initial_state_sha256": initial_hash, "parameters": sum(p.numel() for p in model.parameters()),
              "train_sequences": len(train), "final_dev_loss": curve[-1]["dev_loss"],
              "test_loss": test_loss, "first_crossing": first_crossing, "train_seconds": train_seconds,
              "setup_seconds": setup_seconds, "experiment_seconds": experiment_seconds,
              "curve": curve, "hardware": {"host": platform.node(), "machine": platform.machine(),
              "device": torch.cuda.get_device_name() if args.device == "cuda" else "cpu"},
              "software": {"torch": str(torch.__version__), "cuda": torch.version.cuda,
                           "numpy": np.__version__, "python": platform.python_version()},
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated() if args.device == "cuda" else 0}
    (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    log.close()
    print(json.dumps({"status": "complete", "recipe": args.recipe, "seed": args.seed,
                      "final_dev_loss": result["final_dev_loss"], "test_loss": test_loss,
                      "train_seconds": train_seconds}), flush=True)


if __name__ == "__main__":
    main()
