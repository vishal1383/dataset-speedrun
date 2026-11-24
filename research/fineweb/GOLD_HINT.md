# Can the observed next token improve a soft target?

Given prefix `c` and observed next token `y`, the teacher sees `[c, y, EOT, c]`.
Its output at the end of the second copy of `c` predicts the same position as the student's output on `c`.
The first copy serves as an answered example. The ordinary teacher receives only `c`.
The shuffled control follows the same prompt format, with the answer taken from another sequence at the same prefix length.
Student evaluation always uses plain text and hard-label next-token loss.

This probe uses two frozen teachers. One is the completed baseline. The other is an exact copy of the student's starting parameters.
All branches start from the same saved warmup model and optimizer state. For teacher distribution `q`, let `r` contain its eight most probable tokens with their original probabilities and zero elsewhere. The student target is `0.5 r + (1 − 0.5 sum(r)) one_hot(y)`.
Targets are fixed across the experiment. SDFT uses demonstration-conditioned on-policy learning. Gold-Conditioned Distillation more closely matches this teacher-forced setup. [SDFT](https://proceedings.mlr.press/v306/shenfeld26a.html) and [Gold-Conditioned Distillation](https://arxiv.org/abs/2610.25497).

Each branch uses the same 65,536 training positions for 16 updates and the same 65,536 development positions for evaluation.
Training and evaluation reset the context every 16 tokens. The stored checkpoint came from longer-context training.
The experiment preserves the original optimizer settings, which were tuned for a much larger batch.
A loss increase from the starting checkpoint limits interpretation to relative behavior in this probe.
The question here is whether including the answer changes the supervision and improves student loss compared with ordinary distillation.

Reproduce either branch group with the pinned environment and the saved checkpoints from the signal screen.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 research/fineweb/gold_hint_probe.py runs/new-gold-probe --teacher mature
python -m torch.distributed.run --standalone --nproc-per-node=1 research/fineweb/gold_hint_probe.py runs/new-self-probe --teacher frozen_student
```

## Development loss after 16 updates

| Teacher | Hard labels | Ordinary teacher | Correct answer in demo | Shuffled answer in demo |
|---|---:|---:|---:|---:|
| Completed baseline | 6.40785 | 6.31509 | 6.34517 | 6.33764 |
| Frozen student | 6.40790 | 6.36286 | 6.36645 | 6.36636 |

All branches started at 6.09596. The correct answer raised the mature teacher's mean probability on gold from 0.150 to 0.347. It raised the frozen student's mean probability on gold from 0.059 to 0.064. Ordinary teacher targets gave the lowest final loss in both groups. This single-window probe provides no evidence of a training speedup. [Mature teacher results](results/gold-hint/mature.json) and [frozen student results](results/gold-hint/frozen-student.json).
