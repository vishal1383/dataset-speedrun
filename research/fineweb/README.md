# FineWeb data track with a fixed modded-nanoGPT learner

This track changes training data while preserving a published modded-nanoGPT learner.

**Completed pilot:** original FineWeb versus text compression and latent supervision. Neither compression arm improved final loss.
Read [COMPRESSION.md](COMPRESSION.md) and the [frozen compression plan](compression-plan.json).
The text artifact is 25% smaller. The latent arm adds a 6.29 MB conditional target table.
The earlier FineWeb-Edu arm was canceled before training at the user's request.
The completed baseline remains the reference. The construction history below is retained for audit.
It uses the upstream optimization benchmark's tuned Muon+AdamW baseline, identified as result 36.
The source commit, source hash, and dataset revisions are recorded in [pin.json](pin.json).
The upstream source and MIT license are preserved under [upstream](upstream/).

The original modded-nanoGPT tracks prohibit training-data changes.
This is a sibling data benchmark, not an eligible submission to their optimizer or wall-clock leaderboard.
The [main speedrun rules](https://github.com/KellerJordan/modded-nanogpt#rules)
and [optimization benchmark](https://github.com/KellerJordan/modded-nanogpt/tree/master/records/track_3_optimization) define those distinctions.

## Fixed comparison

| Component | Value |
|---|---|
| Model | Upstream 12-layer, width-768 GPT; 162,354,816 parameters |
| Tokenizer | Original GPT-2 token IDs; unchanged padded vocabulary of 50,304 |
| Context | 1,024 tokens |
| Global batch | 524,288 target tokens per optimizer update |
| Microbatch | 64 sequences; eight accumulated microbatches on one Spark |
| Optimizers | Upstream Muon and auxiliary AdamW, with the original parameter groups and hyperparameters |
| Schedule | Upstream 3,250-update schedule, including its original cooldown |
| Loss | Original next-token cross-entropy and original logit transformation |
| Initialization | Original initialization rule; paired deterministic seed |
| Validation | First 10,485,760 target tokens from the canonical FineWeb validation shard |
| Target | At most 3.28 nats/token on that validation stream |
| Hardware | `spark-a5a1`, NVIDIA GB10, using the existing NVIDIA PyTorch container |

The reference trains on the first 20 upstream FineWeb training shards, containing 2 billion tokens.
The superseded open-curation candidate used the first 20 upstream FineWeb-Edu training shards.
Both runs use the canonical FineWeb validation array. FineWeb-Edu validation data is not downloaded or used.
The candidate excludes exact token-document matches against the entire 100-million-token canonical validation shard.
Document reconstruction crosses shard boundaries before exclusion. Near-duplicate exclusion is not implemented.
The frozen candidate excludes 160 matching documents and 152,240 tokens, retaining 1,999,847,760 tokens.
Each run consumes 1,703,936,000 target-token exposures over 3,250 updates.

FineWeb-Edu is a separate public selection from FineWeb, not a subset constrained to the baseline's 2-billion-token pool.
This comparison therefore belongs to an **open-curation data-only** track.
The educational selection and its upstream classifier construction are reused public work; their construction compute is not claimed free.
Read the [FineWeb-Edu dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) for provenance and the ODC-BY license.

The first candidate considered was a conservative repetition filter on the original training pool.
It removed 16 documents and 33,676 tokens, or only 0.001684% of the pool.
That training-only observation led to selecting a substantial data variant before the full paired run.
No full validation loss was used to select FineWeb-Edu.

## What counts as a win

The current compression experiment uses three matched 3,250-update arms with seed 17.
Report the final validation losses, paired difference, token exposures, and complete curves.
One favorable pair is preliminary evidence. It does not establish a compression or speed record.

The eventual speed metric is the common update count at which a frozen recipe passes the 3.28 target across fresh seeds.
Keep the 3,250-step learning-rate schedule fixed when comparing shorter prefixes.
Use a predeclared common stopping point; do not stop different seeds at their individual best validation steps.
Require both evidence for the target and a paired comparison against the reference.
The candidate's variance must be measured rather than assumed equal to an optimizer benchmark's historical variance.

Artifact bytes, teacher or data construction compute, student time, and total compute are separate measurements.
Spark wall-clock results are local comparisons. They are not directly comparable to the main speedrun's 8-H100 timings.
The current main-track code requires eight ranks and contains different architecture and optimizer choices.

## Latent representations

A latent representation is eligible for this fixed-learner track if a decoder produces training token IDs before training.
The learner must consume those tokens through the same data loader and the same cross-entropy objective.
Synthetic text, selected data, repetition, and curricula can use that interface.
Report both materialized token bytes and encoded latent bytes, including every learned decoder parameter.
Report decoder construction and execution costs separately. No uncharged teacher or source-corpus access is allowed during student training.

Soft input embeddings, soft targets, learned sample weights, gradient injection, and parameter-target potentials change the learner interface or objective.
Those methods require separate experimental tracks and matched controls.
In particular, the earlier compressed-checkpoint potential is not a data-only entry here.
The separate latent arm now implements compressed conditional soft targets, as specified in [COMPRESSION.md](COMPRESSION.md).
It does not implement meta-learned input embeddings.

## Verification and run commands

The adapter verifies the pinned upstream source hash before execution.
It saves the complete generated trainer, launcher, source pin, shard hashes, initialization hash, console log, and result record.
AST tests verify that all model and optimizer definitions remain identical to upstream.
Separate tests verify the original optimizer assignments, initialization block, batch size, and learning-rate schedule.
System changes add deterministic seeds, paths, run limits, progress records, and checkpoints after validation.
The text arm permits no learning-rate, loss, optimizer, gradient-scaling, or architecture change.
The separate latent arm changes only the training target distribution and adds its documented strength schedule.

Use the container from the root README. Keep caches in this repository's ignored `cache/` directory.

```bash
python research/fineweb/prepare.py --shards 20
python research/fineweb/prepare.py --dataset finewebedu --shards 20
python research/fineweb/build_edu.py
python research/fineweb/launch.py --output runs/fineweb-reference --seed 17
python research/fineweb/launch.py --output runs/fineweb-edu --seed 17 \
  --train-pattern 'runs/fineweb-v1/finewebedu-decontaminated/fineweb_train_*.bin'
```

The superseded Edu queue used [run_pair.py](run_pair.py) and its historical [plan.json](plan.json).
The current queue uses [compression_queue.py](compression_queue.py) and [compression-plan.json](compression-plan.json).
It checks source and data hashes before each run. It refuses duplicate run directories and stops on failure.
It writes `queue-status.json`, per-run `progress.json`, full logs, and a final `comparison.json`.
Each successful validation writes `latest.pt`, including model, optimizer, and RNG state.
Checkpoints support inspection and future recovery tooling; this version does not automatically resume interrupted runs.

Two-update smoke runs used a reduced validation slice and are explicitly ineligible for the target comparison.
The checkpoint smoke reproduced the first smoke's initialization hash and 8.78785 loss after two updates.
Its warm second update took approximately 12.0 seconds at the original global batch and microbatch sizes.
A full run projects to roughly 11–12 hours including periodic full validation; this is an estimate, not a completed timing.
The single-GPU queue follows the user's instruction to use only the first Spark.
