"""Launch the full ordinary self-distillation run against the frozen FineWeb control."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from launch import ROOT, HERE, digest, render, replace_once


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def trainer(output, baseline):
    source = render((HERE/"upstream/train_gpt_simple.py").read_text(), output,
                    "data/fineweb10B/fineweb_train_*.bin", 17, 3250, 10485760, 16, True)
    source = replace_once(source, "    def forward(self, inputs: Tensor, targets: Tensor):",
                          "    def logits(self, inputs):")
    anchor = '        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")'
    source = replace_once(source, anchor, """        return logits

    def forward(self, inputs, targets):
        logits = self.logits(inputs)
        if self.training:
            with torch.no_grad():
                q = teacher.logits(inputs).softmax(-1)
            log_probs = logits.log_softmax(-1)
            hard = -log_probs.gather(-1, targets.unsqueeze(-1)).sum()
            soft = -(q * log_probs).sum()
            return 0.5 * (hard + soft)
""" + anchor)
    source = replace_once(source, 'model.compile(dynamic=False)',
                          'import copy\nteacher = copy.deepcopy(model).requires_grad_(False).eval()\nmodel.compile(dynamic=False)')
    source = replace_once(source, '    # create the optimizer(s)',
                          f'    assert initial_digest.hexdigest() == {baseline["initial_state_sha256"]!r}\n    # create the optimizer(s)')
    source = replace_once(source, '        inputs, targets = next(train_loader)',
                          '        if step % 4 == 0:\n            teacher.load_state_dict(model.state_dict())\n        inputs, targets = next(train_loader)')
    source = replace_once(source, "'cpu_rng': torch.get_rng_state()", "'teacher': teacher.state_dict(), 'cpu_rng': torch.get_rng_state()")
    return source


def main(output):
    output=output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    record=dict(status="preflight", started_unix=time.time(), curve=[], step=0)
    try:
        baseline=json.loads((HERE/"results/compression/baseline-result.json").read_text())
        upstream=HERE/"upstream/train_gpt_simple.py"
        assert digest(upstream)==baseline["upstream"]["source_sha256"]
        for pattern,field in [("fineweb_train_*.bin","training_files"),("fineweb_val_*.bin","validation_files")]:
            actual={str(p.relative_to(ROOT)):{"bytes":p.stat().st_size,"sha256":digest(p)}
                    for p in sorted((ROOT/"data/fineweb10B").glob(pattern))}
            assert actual == baseline[field], field
        generated=output/"train.py"
        generated.write_text(trainer(output,baseline))
        compile(generated.read_text(),str(generated),"exec")
        save(output/"plan.json", dict(seed=17, updates=3250, global_batch_tokens=524288, context=1024,
             validation_tokens=10485760, microbatch_sequences=16, baseline_microbatch_sequences=64,
             refresh_every=4, teacher="own student, detached, ordinary causal prefix",
             objective="equal mix of hard CE and dense teacher CE at temperature 1",
             source_sha256=digest(generated), launcher_sha256=digest(__file__),
             baseline_record_sha256=digest(HERE/"results/compression/baseline-result.json"),
             baseline_final_loss=baseline["curve"][-1]["loss"], expected_initial_state=baseline["initial_state_sha256"]))
        record["status"]="running"
        save(output/"status.json",record)
        start=time.monotonic()
        cmd=[sys.executable,"-m","torch.distributed.run","--standalone","--nproc-per-node=1",str(generated)]
        with (output/"console.log").open("w") as log, subprocess.Popen(cmd,cwd=ROOT,stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,text=True,env=dict(os.environ,PYTHONUNBUFFERED="1")) as proc:
            for line in proc.stdout:
                log.write(line)
                log.flush()
                print(line,end="",flush=True)
                m=re.search(r'step:(\d+)/3250 (?:val_loss:(\S+) )?train_time:([\d.]+)s',line)
                if "initial_state_sha256:" in line:
                    record["initial_state_sha256"]=line.strip().split(":")[-1]
                if m:
                    record.update(step=int(m[1]),train_seconds=float(m[3]),elapsed_seconds=time.monotonic()-start)
                    if m[2]:
                        loss=float(m[2])
                        if not math.isfinite(loss):
                            proc.terminate()
                            raise ValueError("Nonfinite validation loss")
                        record["curve"].append(dict(step=int(m[1]),loss=loss,train_seconds=float(m[3]),elapsed_seconds=record["elapsed_seconds"]))
                    save(output/"status.json",record)
            if proc.wait()!=0:
                raise RuntimeError(f"Trainer exited {proc.returncode}")
        assert record["initial_state_sha256"]==baseline["initial_state_sha256"]
        assert [r["step"] for r in record["curve"]]==[r["step"] for r in baseline["curve"]]
        def threshold(curve):
            return next((r for r in curve if r["loss"]<=3.28),None)
        record.update(status="complete",elapsed_seconds=time.monotonic()-start,
                      final_delta=record["curve"][-1]["loss"]-baseline["curve"][-1]["loss"],
                      threshold=threshold(record["curve"]),baseline_threshold=threshold(baseline["curve"]))
        save(output/"result.json",record)
    except BaseException as exc:
        record.update(status="failed",error=repr(exc))
        raise
    finally:
        save(output/"status.json",record)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    main(parser.parse_args().output)
