"""Screen gradient refresh and teacher targets on the pinned full-size learner.

This reduced-batch development screen is not a speedrun result.
"""

import argparse
import ast
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ARMS = ["fresh", "replay4", "cache2", "cache4", "extrapolate2", "teacher_top8"]


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def definitions():
    source = HERE / "upstream/train_gpt_simple.py"
    pin = json.loads((HERE / "pin.json").read_text())
    assert sha(source) == pin["source_sha256"]
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(
        node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))]
    namespace = {}
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace


def sparse_loss(logits, targets, ids, probabilities, strength):
    """Keep omitted teacher mass on the observed target."""
    hard = F.cross_entropy(logits.flatten(0, 1), targets.flatten(), reduction="sum")
    gold = logits.gather(-1, targets.unsqueeze(-1))
    other = logits.gather(-1, ids.long())
    return hard + strength * (probabilities * (gold - other)).sum()


def predicted_gradient(current, previous=None, fraction=0.0):
    """Return independent storage because the upstream Muon mutates gradients."""
    if previous is None or fraction == 0:
        return current.clone()
    return current + fraction * (current - previous)


def cpu_copy(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def mapped_tokens(path):
    header = np.fromfile(path, dtype=np.int32, count=256)
    assert header[0] == 20240520 and header[1] == 1
    return np.memmap(path, dtype=np.uint16, mode="r", offset=1024, shape=(int(header[2]),))


def batch(tokens, offset, sequences):
    values = torch.from_numpy(np.array(tokens[offset:offset + sequences * 1024 + 1], dtype=np.int64))
    assert len(values) == sequences * 1024 + 1
    return values[:-1].reshape(sequences, 1024).cuda(), values[1:].reshape(sequences, 1024).cuda()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--dev-tokens", type=int, default=262144)
    parser.add_argument("--arms", nargs="+", choices=ARMS, default=ARMS)
    parser.add_argument("--eager", action="store_true")
    args = parser.parse_args()
    assert len(set(args.arms)) == len(args.arms), "Duplicate arms would overwrite results."
    assert args.batch > 0 and 512 % args.batch == 0
    assert args.microbatch > 0 and args.batch % args.microbatch == 0
    assert args.dev_tokens % (args.batch * 1024) == 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    status = {"status": "initializing", "host": socket.gethostname(), "pid": os.getpid(), "started_unix": time.time()}
    write(output / "status.json", status)

    def report(stage, **fields):
        status.update(status=stage, updated_unix=time.time(), **fields)
        write(output / "status.json", status)
        print(json.dumps(status), flush=True)

    torch.cuda.set_device(0)
    dist.init_process_group("nccl", device_id=torch.device("cuda", 0))
    assert dist.get_world_size() == 1, "This screen has a documented service-memory exception to two-host DDP."
    ns = definitions()
    GPT, Muon = ns["GPT"], ns["Muon"]

    class ScreenGPT(GPT):
        def logits(self, inputs):
            x = self.norm1(self.embed(inputs))
            for block in self.blocks:
                x = block(x)
            z = self.proj(self.norm2(x)).float()
            return 15 * z * (z.square() + 15**2).rsqrt()

        def forward(self, inputs, targets, ids, probabilities, strength):
            return sparse_loss(self.logits(inputs), targets, ids, probabilities, strength)

    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    model = ScreenGPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()
    # Upstream seeds the explicit initialization after constructing the module.
    torch.manual_seed(17)
    torch.cuda.manual_seed_all(17)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith("weight"):
                if "proj" in name:
                    parameter.zero_()
                elif "embed" in name:
                    parameter.normal_()
                else:
                    parameter.normal_(std=0.33**0.5 / parameter.size(-1)**0.5)
            elif name.endswith("bias"):
                parameter.zero_()
            elif name.endswith("gains"):
                parameter.normal_(mean=1, std=0)
            else:
                raise ValueError(name)
    initial_digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        initial_digest.update(name.encode())
        initial_digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    baseline_record = json.loads((ROOT / "runs/fineweb-v1/paired17/fineweb-seed17/result.json").read_text())
    assert initial_digest.hexdigest() == baseline_record["initial_state_sha256"]

    def optimizers():
        adam = torch.optim.AdamW([
            dict(params=[model.embed.weight], lr=0.7),
            dict(params=[model.proj.weight], lr=0.004),
            dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
            betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
        muon = Muon([p for p in model.blocks.parameters() if p.ndim >= 2], lr=0.025, weight_decay=0.05)
        return [adam, muon]

    opts = optimizers()
    forward = model.forward if args.eager else torch.compile(model.forward, dynamic=False)
    train_path = ROOT / "data/fineweb10B/fineweb_train_000001.bin"
    validation_path = next((ROOT / "data/fineweb10B").glob("fineweb_val_*.bin"))
    teacher_path = ROOT / "runs/fineweb-v1/paired17/fineweb-seed17/latest.pt"
    config = dict(vars(args), output=str(output), arms=args.arms)
    config.update(
        track="development_signal_screen", record_eligible=False, seed=17,
        architecture=json.loads((HERE / "pin.json").read_text())["architecture"],
        source_sha256=sha(Path(__file__)), upstream_sha256=sha(HERE / "upstream/train_gpt_simple.py"),
        train_sha256=sha(train_path), validation_sha256=sha(validation_path),
        teacher_checkpoint_sha256=sha(teacher_path),
        teacher_training_seconds=baseline_record["elapsed_seconds"], initial_state_sha256=initial_digest.hexdigest(),
        loss_scale=512 / args.batch, schedule="constant original plateau LR; all steps below cooldown",
        changes="Reduced statistical batch; sum loss rescaled to original batch magnitude. Cached gradients change the training signal.",
        selection_split=f"first {args.dev_tokens} canonical validation tokens; development only",
        confirmation="fresh data order; same shared initialization, not a fresh model seed",
        ddp_exception="Second Spark retains protected 100GiB service; 14GiB available versus prior 35GiB training allocation",
        torch_version=torch.__version__, gpu=torch.cuda.get_device_name())
    write(output / "plan.json", config)
    (output / "source.py").write_bytes(Path(__file__).read_bytes())
    training, validation = mapped_tokens(train_path), mapped_tokens(validation_path)
    count = args.batch * 1024
    dummy_ids = torch.zeros((args.microbatch, 1024, 8), device="cuda", dtype=torch.int32)
    dummy_probs = torch.zeros_like(dummy_ids, dtype=torch.float32)
    strength = torch.zeros((), device="cuda")

    def backward(offset, teacher=None):
        total = 0.0
        for part in range(0, args.batch, args.microbatch):
            x, y = batch(training, offset + part * 1024, args.microbatch)
            if teacher is None:
                ids, probs = dummy_ids, dummy_probs
                strength.zero_()
            else:
                ids, probs = (value[part:part + args.microbatch].cuda().to(dtype)
                              for value, dtype in zip(teacher, [torch.int32, torch.float32]))
                strength.fill_(0.5)
            loss = forward(x, y, ids, probs, strength)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("Nonfinite training loss")
            (loss * (512 / args.batch)).backward()
            total += loss.detach().item()
        return total / count

    def update():
        for opt in opts:
            opt.step()
        model.zero_grad(set_to_none=True)

    @torch.no_grad()
    def evaluate():
        model.eval()
        strength.zero_()
        total = 0.0
        for offset in range(0, args.dev_tokens, args.microbatch * 1024):
            x, y = batch(validation, offset, args.microbatch)
            total += forward(x, y, dummy_ids, dummy_probs, strength).item()
        model.train()
        return total / args.dev_tokens

    report("warmup", step=0, steps=args.warmup)
    start = time.perf_counter()
    for step in range(args.warmup):
        backward(step * count)
        update()
        if (step + 1) % 8 == 0:
            report("warmup", step=step + 1, seconds=time.perf_counter() - start)
    torch.cuda.synchronize()
    shared = {"model": cpu_copy(model.state_dict()), "optimizers": cpu_copy([opt.state_dict() for opt in opts]),
              "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}
    torch.save(shared, output / "shared.pt")
    initial_loss = evaluate()
    shared_hash = sha(output / "shared.pt")
    report("warmup_complete", dev_loss=initial_loss, checkpoint_sha256=shared_hash)

    def restore():
        nonlocal opts
        model.load_state_dict(shared["model"])
        opts = optimizers()
        for opt, saved in zip(opts, shared["optimizers"]):
            opt.load_state_dict(copy.deepcopy(saved))
        torch.set_rng_state(shared["cpu_rng"])
        torch.cuda.set_rng_state(shared["cuda_rng"])
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    # Verify zero-strength equivalence and complete branch restoration on the real learner.
    restore()
    x, y = batch(training, args.warmup * count, min(8, args.microbatch))
    reference = GPT.forward(model, x, y)
    actual = model(x, y, dummy_ids[:len(x)], dummy_probs[:len(x)], strength.zero_())
    torch.testing.assert_close(actual, reference)
    selected = [model.embed.weight, model.blocks[0].attn.q.weight, model.proj.weight]
    reference_grad = torch.autograd.grad(reference, selected)
    actual_grad = torch.autograd.grad(actual, selected)
    relative_errors = []
    for actual_tensor, reference_tensor in zip(actual_grad, reference_grad):
        # BF16 embedding accumulation can vary with atomic reduction order.
        relative = (actual_tensor.float() - reference_tensor.float()).norm() / reference_tensor.float().norm().clamp_min(1e-20)
        relative_errors.append(relative.item())
        assert relative < 0.005, "Zero-strength gradient differs beyond BF16 precision tolerance."
    del reference, actual, reference_grad, actual_grad, x, y
    backward(args.warmup * count)
    update()
    restore()

    def equal_tree(left, right):
        if torch.is_tensor(left):
            assert torch.equal(left.cpu(), right.cpu()), "Branch restoration changed a tensor."
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal_tree(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal_tree(a, b)
        else:
            assert left == right

    equal_tree(shared["model"], model.state_dict())
    equal_tree(shared["optimizers"], [opt.state_dict() for opt in opts])
    write(output / "verification.json", {"zero_strength_matches_upstream_loss_and_selected_gradients": True,
                                       "selected_gradient_relative_errors": relative_errors,
                                       "gradient_relative_tolerance": 0.005,
                                       "restored_model_and_all_optimizer_states_exactly_equal": True})

    def build_teacher(offset, order):
        report("teacher_construction", data_order=order)
        torch.cuda.synchronize()
        start = time.perf_counter()
        teacher = ScreenGPT(vocab_size=50304, num_layers=12, model_dim=768).cuda().eval()
        checkpoint = torch.load(teacher_path, map_location="cpu", weights_only=False)
        assert checkpoint["step"] == 3250
        teacher.load_state_dict(checkpoint["model"])
        del checkpoint
        labels = []
        with torch.no_grad():
            for step in range(args.steps):
                ids_parts, probs_parts = [], []
                for part in range(0, args.batch, 8):
                    x, _ = batch(training, offset + step * count + part * 1024, min(8, args.batch - part))
                    logits = teacher.logits(x)
                    top, ids = logits.topk(8, dim=-1)
                    probabilities = (top - logits.logsumexp(-1, keepdim=True)).exp()
                    ids_parts.append(ids.int().cpu())
                    probs_parts.append(probabilities.half().cpu())
                labels.append((torch.cat(ids_parts), torch.cat(probs_parts)))
                if (step + 1) % 8 == 0:
                    report("teacher_construction", data_order=order, step=step + 1)
        torch.cuda.synchronize()
        path = output / f"teacher-order{order}.pt"
        torch.save(labels, path)
        del teacher
        torch.cuda.empty_cache()
        seconds = time.perf_counter() - start
        return labels, {"construction_seconds": seconds, "artifact_bytes": path.stat().st_size,
                        "teacher_training_seconds_excluded_from_construction": config["teacher_training_seconds"]}

    def run_arm(arm, offset, order, labels=None, construction=None):
        restore()
        torch.cuda.reset_peak_memory_stats()
        report("branch", arm=arm, data_order=order, step=0)
        row = {"arm": arm, "data_order": order, "shared_checkpoint_sha256": shared_hash,
               "curve": [{"step": 0, "loss": initial_loss, "train_seconds": 0.0}],
               "construction": construction or {}, "source_start_token": offset,
               "backward_calls": 0, "cached_gradient_bytes": 0}
        interval = 4 if arm in {"cache4", "replay4"} else 2 if arm in {"cache2", "extrapolate2"} else 1
        cache = previous = None
        elapsed = 0.0
        try:
            for step in range(args.steps):
                torch.cuda.synchronize()
                start = time.perf_counter()
                refresh = step % interval == 0
                if arm in {"fresh", "teacher_top8", "replay4"} or refresh:
                    source_step = step - step % interval if arm == "replay4" else step
                    backward(offset + source_step * count, labels[step] if arm == "teacher_top8" else None)
                    row["backward_calls"] += 1
                    if arm in {"cache2", "cache4", "extrapolate2"}:
                        previous = cache if arm == "extrapolate2" else None
                        cache = [p.grad.detach().clone() for p in model.parameters()]
                        row["cached_gradient_bytes"] = sum(g.numel() * g.element_size() for g in cache) * (2 if previous is not None else 1)
                else:
                    fraction = (step % interval) / interval if arm == "extrapolate2" else 0.0
                    for index, parameter in enumerate(model.parameters()):
                        parameter.grad = predicted_gradient(cache[index], None if previous is None else previous[index], fraction)
                update()
                torch.cuda.synchronize()
                elapsed += time.perf_counter() - start
                if (step + 1) % 16 == 0 or step + 1 == args.steps:
                    loss = evaluate()
                    if not math.isfinite(loss):
                        raise FloatingPointError("Nonfinite development loss")
                    row["curve"].append({"step": step + 1, "loss": loss, "train_seconds": elapsed})
                    write(output / f"{arm}-order{order}.json", row)
                    report("branch", arm=arm, data_order=order, step=step + 1, dev_loss=loss, train_seconds=elapsed)
            row.update(status="complete", final_loss=row["curve"][-1]["loss"], train_seconds=elapsed,
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                       student_tokens=row["backward_calls"] * count,
                       microbatch_backward_calls=row["backward_calls"] * (args.batch // args.microbatch),
                       unique_source_tokens=((args.steps + interval - 1) // interval) * count,
                       total_seconds_excluding_teacher_training=elapsed + (construction or {}).get("construction_seconds", 0))
        except FloatingPointError as error:
            model.zero_grad(set_to_none=True)
            row.update(status="failed", error=str(error), train_seconds=elapsed)
        write(output / f"{arm}-order{order}.json", row)
        del cache, previous
        return row

    try:
        results = []
        offset = args.warmup * count
        labels = construction = None
        for arm in args.arms:
            if arm == "teacher_top8":
                labels, construction = build_teacher(offset, 0)
            results.append(run_arm(arm, offset, 0, labels, construction if arm == "teacher_top8" else None))
        baseline = next((row for row in results if row["arm"] == "fresh" and row["status"] == "complete"), None)
        eligible = [row for row in results if row["arm"] != "fresh" and row["status"] == "complete"
                    and baseline is not None and row["final_loss"] < baseline["final_loss"] - 0.02]
        selection = min(eligible, key=lambda row: row["final_loss"])["arm"] if eligible else None
        write(output / "selection.json", {"winner": selection, "rule": "lowest loss at least 0.02 below fresh at fixed updates; not a speed claim"})
        if selection:
            offset += (args.steps + 64) * count
            labels = construction = None
            results.append(run_arm("fresh", offset, 1))
            if selection == "teacher_top8":
                labels, construction = build_teacher(offset, 1)
            results.append(run_arm(selection, offset, 1, labels, construction))
        summary = {"status": "complete", "record_eligible": False, "plan": config,
                   "initial_dev_loss": initial_loss, "selection": selection, "results": results,
                   "limitations": ["Reduced-batch short-horizon development screen", "One shared initialization",
                                   "Teacher training cost must be included in any end-to-end speed claim",
                                   "Cached gradients are a separate training-signal intervention"]}
        write(output / "summary.json", summary)
        report("complete", selection=selection, completed_arms=len(results), finished_unix=time.time())
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if "--output" in sys.argv:
            failure_path = Path(sys.argv[sys.argv.index("--output") + 1]) / "status.json"
            if failure_path.exists():
                failure = json.loads(failure_path.read_text())
                if failure.get("pid") == os.getpid():
                    failure.update(status="failed", error=repr(error), updated_unix=time.time())
                    write(failure_path, failure)
        if dist.is_initialized():
            dist.destroy_process_group()
        raise
