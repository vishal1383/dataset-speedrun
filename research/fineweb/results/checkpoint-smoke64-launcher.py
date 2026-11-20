"""Wrap the pinned upstream learner with data paths, seeds, and complete run records."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f"Upstream source boundary changed: {old!r}")
    return source.replace(old, new, 1)


def render(source, output, train_pattern, seed, updates, validation_tokens, microbatch, compile_model):
    """Only change execution controls. Keep model, optimizer, initialization, and schedule definitions."""
    source = replace_once(source, "mbs = 64", f"mbs = {microbatch}")
    source = replace_once(source, "val_tokens = 20 * 524288", f"val_tokens = {validation_tokens}")
    source = replace_once(source, 'os.makedirs("logs", exist_ok=True)', f'os.makedirs({str(output)!r}, exist_ok=True)')
    source = replace_once(source, 'logfile = f"logs/{uuid.uuid4()}.txt"', f'logfile = {str(output / "upstream.log")!r}')
    source = replace_once(source, '"data/fineweb10B/fineweb_train_*.bin", batch_size', f'{train_pattern!r}, batch_size')
    source = replace_once(source, "for _ in range(num_trials):", f"for _ in range(num_trials):\n    torch.manual_seed({seed})\n    torch.cuda.manual_seed_all({seed})")
    source = replace_once(source, "train_steps = 3250", f"train_steps = 3250\n    run_steps = {updates}")
    source = replace_once(source, "range(train_steps + 1)", "range(run_steps + 1)")
    source = source.replace("if step == train_steps", "if step == run_steps")
    source = replace_once(source, "model.compile(dynamic=False)", "model.compile(dynamic=False)" if compile_model else "# Eager execution: model and optimizer mathematics are unchanged.")
    source = replace_once(source, 'model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()',
                          'model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()\n'
                          'print0(f"parameter_count:{sum(p.numel() for p in model.parameters())}", console=True)')
    # Hash the actual initialized parameters, including BF16 tensors, before any data update.
    source = replace_once(source, "    # create the optimizer(s)",
                          "    import hashlib\n    initial_digest = hashlib.sha256()\n"
                          "    for name, tensor in model.state_dict().items():\n"
                          "        initial_digest.update(name.encode())\n"
                          "        initial_digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())\n"
                          "    print0(f'initial_state_sha256:{initial_digest.hexdigest()}', console=True)\n\n"
                          "    # create the optimizer(s)")
    source = replace_once(source, "            model.train()\n            # start the clock again",
                          "            if step > 0:\n"
                          f"                checkpoint_path = Path({str(output / 'latest.pt')!r})\n"
                          "                temporary = checkpoint_path.with_suffix('.tmp')\n"
                          "                torch.save({'step': step, 'model': model.state_dict(),\n"
                          "                            'optimizers': [opt.state_dict() for opt in optimizers],\n"
                          "                            'cpu_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state(),\n"
                          "                            'training_time': training_time}, temporary)\n"
                          "                temporary.replace(checkpoint_path)\n"
                          "            model.train()\n            # start the clock again")
    return source


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-pattern", default="data/fineweb10B/fineweb_train_*.bin")
    p.add_argument("--seed", type=int, default=17)
    p.add_argument("--updates", type=int, default=3250)
    p.add_argument("--validation-tokens", type=int, default=10485760)
    p.add_argument("--microbatch", type=int, default=64)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    if not 1 <= args.updates <= 3250 or 512 % args.microbatch:
        raise ValueError("Updates must be in [1,3250]; microbatch must divide 512.")
    if args.validation_tokens % (args.microbatch * 1024):
        raise ValueError("Validation must contain complete microbatches.")
    if args.validation_tokens != 10485760 and not args.smoke:
        raise ValueError("Reduced validation is permitted only for explicitly marked smoke runs.")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    pin = json.loads((HERE / "pin.json").read_text())
    upstream = HERE / "upstream/train_gpt_simple.py"
    if digest(upstream) != pin["source_sha256"]:
        raise ValueError("Pinned learner source changed.")
    training = sorted(ROOT.glob(args.train_pattern))
    if not training:
        raise ValueError("No training shards matched the supplied pattern.")
    validation = sorted((ROOT / "data/fineweb10B").glob("fineweb_val_*.bin"))
    if len(validation) != 1:
        raise ValueError("Expected exactly one upstream validation shard.")
    source = render(upstream.read_text(), args.output, args.train_pattern, args.seed, args.updates,
                    args.validation_tokens, args.microbatch, not args.eager)
    generated = args.output / "train.py"
    generated.write_text(source)
    (args.output / "launcher.py").write_bytes(Path(__file__).read_bytes())
    (args.output / "pin.json").write_bytes((HERE / "pin.json").read_bytes())
    configuration = {"upstream": pin, "seed": args.seed, "updates": args.updates, "schedule_updates": 3250,
                     "validation_tokens": args.validation_tokens, "global_batch_tokens": 524288,
                     "microbatch_sequences": args.microbatch, "compile_model": not args.eager,
                     "smoke": args.smoke, "source_sha256": digest(generated), "launcher_sha256": digest(__file__),
                     "training_files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in training},
                     "validation_files": {str(p.relative_to(ROOT)): {"bytes": p.stat().st_size, "sha256": digest(p)} for p in validation},
                     "main_speedrun_record_eligible": False, "status": "running"}
    record = args.output / "result.json"
    record.write_text(json.dumps(configuration, indent=2) + "\n")
    start = time.perf_counter()
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=1", str(generated)]
    with (args.output / "console.log").open("w") as logfile:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            logfile.write(line)
            logfile.flush()
            progress = re.search(r"step:(\d+)/3250 .*train_time:([\d.]+)s", line)
            if progress:
                status = {"step": int(progress[1]), "planned_updates": args.updates,
                          "train_seconds": float(progress[2]), "elapsed_seconds": time.perf_counter()-start,
                          "status": "running", "child_pid": process.pid}
                (args.output / "progress.json").write_text(json.dumps(status, indent=2) + "\n")
        returncode = process.wait()
    text = (args.output / "console.log").read_text()
    curve = [{"step": int(step), "loss": float(loss), "train_seconds": float(seconds)}
             for step, loss, seconds in re.findall(r"step:(\d+)/3250 val_loss:([\d.]+) train_time:([\d.]+)s", text)]
    initials = re.findall(r"initial_state_sha256:([a-f0-9]+)", text)
    configuration.update(status="complete" if returncode == 0 else "failed", exit_code=returncode,
                         elapsed_seconds=time.perf_counter() - start, curve=curve,
                         initial_state_sha256=initials[-1] if initials else None)
    record.write_text(json.dumps(configuration, indent=2) + "\n")
    (args.output / "progress.json").write_text(json.dumps({"status": configuration["status"], "exit_code": returncode,
                                                          "step": args.updates if returncode == 0 else None}, indent=2) + "\n")
    raise SystemExit(returncode)


if __name__ == "__main__":
    main()
