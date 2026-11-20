# Text and latent compression against the fixed FineWeb learner

The experiment compares three arms at 3,250 updates and identical initialization.
The first arm is the original FineWeb baseline. The second compresses its text pool and changes example order.
The third adds compressed soft-target supervision to exactly the second arm's text stream.

| Arm | Student input | Training objective | Core payload bytes |
|---|---|---|---:|
| Baseline | Original 2-billion-token FineWeb pool | Original hard-token cross-entropy | 4,000,020,480 |
| Text compression | 1,499,996,160 retained tokens plus replay order | Original hard-token cross-entropy | 3,000,009,321 |
| Latent compression | Same compressed text plus conditional target table | Cross-entropy with decoded soft targets | 3,006,301,781 |

Core payload bytes include shard headers and replay order. The latent arm includes every learned table entry.
The manifest and common decoder code are additional small metadata; construction-only block indices are not needed during student training.
The expanded replay stream exposes 3,600,003,072 logical bytes through symlinks to the 3,000,007,680-byte text payload.
Storage compression does not reduce the fixed 1,703,936,000 target-token exposures during training.

The model, optimizer, optimizer hyperparameters, batch size, context, and learning-rate schedule remain fixed in all arms.
Validation always uses original hard-token cross-entropy on the same 10,485,760 FineWeb tokens.
The latent arm changes the training target distribution. It therefore belongs to a separate training-signal track.

## Text method

The builder reads only the first 20 canonical training shards. It never reads validation tokens or validation loss.
It divides each shard into 8,192-token blocks and computes 128-dimensional hashed lexical features.
Features use square-root term frequencies and inverse document frequencies, followed by unit normalization.
The builder samples every sixteenth token for lexical features and measures adjacent repetition across complete blocks.

It checks block hashes for exact duplicates. This pool contained no duplicate complete blocks under that check.
It randomly groups blocks into sets of 64 and removes 16 blocks per group.
Each removal minimizes the remaining group's feature-mean error, with a small preference for removing repetitive blocks.
This is lexical coreset optimization. It is not a neural-gradient matching estimate.

The builder orders retained blocks by standardized unigram surprisal plus fixed Gaussian noise with standard deviation two.
This creates an easy-to-hard curriculum while retaining difficulty diversity throughout the first pass.
The replay pass permutes shard order. The unchanged upstream loader reads the resulting finite stream.
All block boundaries are multiples of the learner's context length before upstream shard packing.
Concatenation can introduce transitions between retained blocks; no loss masking or architecture change hides those transitions.

## Latent method

The builder samples one training trigram per 32 token positions across all 20 shards.
It excludes contexts that cross an end-of-document token and fits empirical conditional next-token distributions.
Categorical maximum likelihood has the closed-form solution `q(v|context) = count(context,v) / count(context)`.
The builder retains 262,144 frequent two-token contexts, each with its top four target IDs and probabilities.
It stores context IDs, target IDs, FP16 probabilities, and support counts in a 6,292,460-byte table.
This construction used 62,319,278 sampled training events and took 36.26 seconds together with text construction.

For a matched context, let `w` be its stored target probabilities multiplied by `support / (support + 32)`.
The student uses this target distribution:

```text
a(step) = 0.25 * (1 - step / 3250)
q_target = (1 - a * sum(w)) * one_hot(gold) + a * w
dL/d(logits) = softmax(logits) - q_target
```

Unstored probability mass stays on the observed target. Missing contexts and the first position use the original target.
The latent contribution decreases during training, reducing short-context bias near the end of the run.
The table can supply alternative targets observed in discarded text blocks.
The student reads neither discarded source text nor a teacher model during training.

The implementation computes this objective as hard-token CE plus a sparse logit correction.
It does not allocate a second dense vocabulary-sized target or probability tensor.
Tests compare the loss and its full logit gradient with dense soft-target cross-entropy.
Tests also check zero-strength equivalence and compiled execution with changing strength.

This first latent method is a compressed conditional distribution model.
It is not meta-learned input embeddings, an optimized synthetic token sequence, or a compressed student checkpoint.
The short-context table can oversmooth useful long-context targets. The paired text-only arm measures that risk directly.

## Frozen experiment and execution

[compression-plan.json](compression-plan.json) freezes source hashes, artifact hashes, the baseline reference, and the comparison metrics.
The selection recipe was fixed before either compression arm produced validation results.
Baseline progress was already visible. This experiment is a pilot, not a prospectively sealed record attempt.

The original FineWeb-Edu comparison was superseded at the user's explicit request.
The running baseline continues with its original generated trainer and launcher.
An explicit source amendment makes the old controller reject its pending Edu run at the next frozen-source check.
The new controller verifies that handoff before it submits another GPU process.

Execution order is:

1. Finish the existing original-data baseline.
2. Run two-update GPU smoke tests for both compression arms.
3. Require finite, decreasing smoke loss and matching initialization hashes.
4. Train the text arm for 3,250 updates.
5. Train the latent arm for 3,250 updates.
6. Save both baseline differences and the latent-minus-text difference.

The controller uses only `spark-a5a1`. It refuses duplicate submission and stops on failed checks.
The original queue's expected source-check failure records cancellation of its pending Edu arm; it does not indicate baseline failure.
The new queue writes its status and results under `runs/fineweb-compression-v1/comparison/`.

```bash
python research/fineweb/compress.py
python research/fineweb/compression_queue.py
```

The second command uses the local frozen baseline and artifact paths. It does not retrain the baseline.
For an independent reconstruction, use the pinned data preparation commands and a new matched baseline run.
Use a new plan and output directory for a new experiment; do not overwrite this pilot's frozen plan.

## Interpretation

The primary metric is lower final canonical FineWeb loss than the paired baseline at exactly 3,250 updates.
The secondary target is loss at most 3.28. Report artifact size, construction cost, training time, and full curves separately.
One favorable seed is preliminary. Fresh paired seeds are required before claiming a reliable improvement.
No compression-arm validation result existed when this experiment was launched.

The code is covered by the repository license. The text derives from the pinned `kjj0/fineweb10B-gpt2` training release.
Source revisions and per-shard hashes are included in the [artifact manifest](results/compression/manifest.json).
Retain the upstream FineWeb dataset attribution and applicable dataset terms when redistributing derived text.
