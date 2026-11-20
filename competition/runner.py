"""Train submitted token arrays with the registered, unchanged learner."""

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research/autoresearch"))
from engine import setup, autocast, losses, CONFIG
from train import file_hash, load_sequences, lr_ratio, state_hash

CONFIG_HASH = "5e0aac9d0607ca972a39abfc85cfd3e9bdfa7fb15639e3cc0793774112749d98"
TRAINER_HASH = "a89bfbb7edfd297bba1cbc69dde249ddd6a9bdfe13dc1e26f32dc3c575958746"


def read_tokens(path):
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.uint16 or values.ndim not in (1, 2):
        raise ValueError("Submit a uint16 .npy token stream or [N, 257] window array.")
    if values.size == 0 or int(values.max()) >= CONFIG["vocab_size"]:
        raise ValueError("Empty artifact or token outside the fixed vocabulary.")
    if values.ndim == 1:
        if len(values) < 257:
            raise ValueError("A token stream needs at least 257 tokens.")
        return load_sequences(path, 256)
    if values.shape[1] != 257:
        raise ValueError("Every window must contain 257 tokens.")
    return torch.from_numpy(values.astype(np.int64))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(obj, indent=2) + "\n")
    temp.replace(path)


def train(artifact, seed, evaluation, output, save_model=False):
    artifact, evaluation, output = map(Path, (artifact, evaluation, output))
    if file_hash(ROOT / "configs/tinystories-v0.json") != CONFIG_HASH:
        raise ValueError("Registered learner configuration changed.")
    if file_hash(ROOT / "train.py") != TRAINER_HASH:
        raise ValueError("Registered trainer changed.")
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    model, optimizers = setup(seed)
    initial_hash = state_hash(model)
    samples = read_tokens(artifact).cuda()
    eval_data = read_tokens(evaluation).cuda()
    rng = torch.Generator().manual_seed(seed + 100000)
    seconds = 0.0
    curve = []
    for step in range(1, 401):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for opt, base in zip(optimizers, [CONFIG["muon_lr"], CONFIG["adamw_lr"]]):
            for group in opt.param_groups:
                group["lr"] = base * lr_ratio(step, CONFIG)
            opt.zero_grad(set_to_none=True)
        ids = torch.randint(len(samples), (32,), generator=rng).cuda()
        batch = samples[ids]
        with autocast():
            loss = model(batch[:, :-1], batch[:, 1:])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["gradient_clip"], error_if_nonfinite=True)
        for opt in optimizers:
            opt.step()
        torch.cuda.synchronize()
        seconds += time.perf_counter() - start
        if step % 100 == 0:
            curve.append({"step": step, "loss": losses(model, eval_data).mean().item()})
    per_window = losses(model, eval_data).cpu().numpy()
    np.save(output / "evaluation-losses.npy", per_window)
    if save_model:
        torch.save(model.state_dict(), output / "model.pt")
    sources = {name: file_hash(ROOT / name) for name in
               ["train.py", "competition/runner.py", "research/autoresearch/engine.py",
                "configs/tinystories-v0.json"]}
    for name in sources:
        target = output / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
    result = {"seed": seed, "updates": 400, "target_tokens": 3276800,
              "initial_state_sha256": initial_hash, "artifact_sha256": file_hash(artifact),
              "artifact_bytes": artifact.stat().st_size, "artifact_tokens": int(np.load(artifact).size),
              "evaluation_sha256": file_hash(evaluation), "evaluation_windows": len(eval_data),
              "loss": float(per_window.mean()), "curve": curve, "train_seconds": seconds,
              "elapsed_seconds": time.perf_counter() - started, "source_hashes": sources,
              "host": platform.node(), "torch": str(torch.__version__),
              "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
              "mode": "data-only", "status": "complete"}
    write(output / "result.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--artifact", type=Path, required=True)
    p.add_argument("--evaluation", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--save-model", action="store_true")
    args = p.parse_args()
    train(args.artifact, args.seed, args.evaluation, args.output, args.save_model)


if __name__ == "__main__":
    main()
