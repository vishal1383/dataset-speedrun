# Answer-conditioned self-distillation from random initialization

Can the observed token help a GPT teach itself during FineWeb pretraining? This pilot compares four runs per seed. Each run starts from random weights and a new optimizer. The only teacher is a frozen copy of that run's own student, refreshed before updates 1, 5, 9, 13, 17 and 21.

| Run | Teacher context for target y |
|---|---|
| Hard labels | No teacher |
| Ordinary self-distillation | Prefix c |
| Answer-conditioned | c, y, end-of-text, c |
| Shuffled hint | c, another sequence's token at the same position, end-of-text, c |

The student always sees c. All teacher runs use half hard cross-entropy and half dense, temperature-one teacher cross-entropy. Targets are detached. The repeated prefix aligns the teacher prediction with y. The shuffled control rotates labels by one sequence within the same training batch. Every run sees identical data in the same order.

The student uses the 162M-parameter baseline architecture and original plateau optimizer settings. This small screen uses 24 updates of 4,096 tokens, context 16 and initialization seeds 17 and 42. Each run evaluates hard next-token loss on the same 65,536 development positions. The screen uses 98,304 training positions per run. The full speedrun uses context 1,024 and 3,250 updates.

The primary comparison requires lower final development loss for answer conditioning than all three controls on both seeds. Recorded training time includes teacher construction and inference. The first host also had another GPU process, so timing is descriptive. The second Spark retained its inference service.

[Fixed plan](results/scratch-hint/plan.json) · [Runner](scratch_hint.py)

Run inside the pinned PyTorch container with the repository's FineWeb training and validation shards available.

```bash
python -m torch.distributed.run --standalone --nproc-per-node=1 research/fineweb/scratch_hint.py runs/new-scratch-hint
```

## Observed step-24 development loss

| Seed | Hard labels | Ordinary | Correct hint | Shuffled hint |
|---:|---:|---:|---:|---:|
| 17 | 7.26967 | 7.20684 | 7.21787 | 7.21803 |
| 42 | 7.26667 | 7.20270 | 7.21089 | 7.21031 |

The correct hint did not pass the specified two-seed comparison. [Complete curves and measured costs](results/scratch-hint/results.json).
