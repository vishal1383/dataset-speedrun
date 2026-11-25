# Ordinary self-distillation at the full FineWeb budget

The seed-17 run finished at **3.28839** validation loss after 3,250 updates. The hard-label baseline reached **3.27822** at the same update. Self-distillation missed 3.28 at every evaluation. Total elapsed time was 19.35 hours against 11.92 hours for the baseline. These timings include the execution differences described below. [Completed run and validation curve](results/full-self-distill/result.json).

The 24-update pilot favored ordinary self-distillation on both seeds. This experiment tests that objective on the full 162M-parameter seed-17 benchmark. The student starts at the baseline's random initialization. The teacher is a detached copy of the same student and refreshes before updates 1, 5, 9 and every fourth update thereafter. Both see the same causal prefix. Each optimizer step minimizes an equal mixture of hard-label and dense teacher cross-entropy at temperature one.

The run uses all 3,250 original optimizer updates, context 1,024, a global batch of 524,288 tokens and the pinned FineWeb stream. Each evaluation measures ordinary hard-label loss on 10,485,760 tokens. The launcher checks the upstream trainer, all shard hashes and the initial model hash. It records the complete validation curve and the first observed crossing of 3.28.

The paired hard-label reference reached 3.27887 at update 3,225 with 39,366.851 seconds of training time. It finished at 3.27822. Baseline total elapsed time was 42,917.53 seconds. Training time excludes evaluations and checkpoints for both runs. The candidate also records elapsed time at each evaluation. Teacher copying and forward passes count toward training time.

The student uses 16 sequences per microbatch on a single GB10, compared with 64 for the historical baseline. The global batch, data order, optimizer, and schedule match. Microbatch size can change floating-point accumulation and execution time. A separate GPU process was present on the candidate host at launch. Time comparisons apply to these executions.

```bash
python research/fineweb/full_self_distill.py runs/new-full-self-distill17
```

Run in the pinned PyTorch environment with the original shards present. The output directory contains live status, the generated trainer, console logs and the latest validation checkpoint. On successful completion, the launcher writes the final paired comparison to `result.json`.

[Run plan](results/full-self-distill/plan.json) · [Baseline record](results/compression/baseline-result.json) · [Short pilot](SCRATCH_HINT.md)
