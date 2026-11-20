# Dataset compression challenge

**Train the same model to the same held-out quality using a smaller training artifact.**

The learner receives 400 updates. The model, tokenizer, optimizer, loss, batch size, and learning-rate schedule stay fixed.
The score is the size of the submitted training artifact, subject to the quality test below.
The full-data reference is an 8,980,666-byte token array. A smaller pool can be sampled repeatedly.
This measures training-data compression at fixed updates. It does not measure faster wall-clock training.

## Submit a dataset

Prepare the source data with `python prepare.py` from the repository root.
Use the reference container from the root README.
The common learner is the 6,884,608-parameter GPT in `configs/tinystories-v0.json`.
Use `python research/round2/prepare.py` to reproduce the second campaign's evaluation arrays.

A submission consists of a JSON manifest and one uncompressed NumPy token array.
Use uint16 tokens with the fixed tokenizer. Submit a one-dimensional stream or an `[N, 257]` window array.
The learner samples 32 windows per update, uniformly with replacement.
A stream is split into 256-target windows with a shared boundary token.
Explicit windows may have arbitrary boundaries. Each window must contain 256 targets and its first input token.
Selection, repetition, packing, and synthetic text are permitted.

Build a starter submission:

```bash
python -m competition.build --fraction 0.75 --output submissions/my-subset.json
```

Replace the selection logic in `competition/build.py` with your method.
Change the manifest's method description and construction-cost fields to describe the actual procedure.
The manifest requires `name`, `track`, `artifact`, `artifact_sha256`, `source`, `license`, and `construction`.
Artifact paths are relative to the repository root.
Keep the artifact outside Git. Attach it to a release or provide a permanent download with its SHA-256 hash.

Run a development attempt:

```bash
python -m competition.runner --artifact runs/submissions/my-subset.npy \
  --evaluation data/round2/dev.npy --seed 17 --output runs/my-subset-dev17
```

Run the paired judge:

```bash
python -m competition.judge --submission submissions/my-subset.json \
  --evaluation data/round2/audit.npy --output runs/my-subset-judge
```

This command trains eight controls and eight candidate models. Each pair uses identical initial weights and the same hardware.
The judge freezes input hashes before training. It writes a result for every run and an aggregate `summary.json`.
Completed runs can be resumed. Changed input hashes are rejected.
Delete no failed or unfavorable results from a submitted cohort.
Record attempts must use the trusted judge from the published commit.

The published audit supports reproduction. It is no longer an untouched test after results are released.
For a new record, freeze the artifact before the maintainer assigns a new audit cohort and confirmation seeds.
The audit text is drawn from a public source. The protocol relies on disclosed provenance and no audit-based selection.
It does not claim cryptographic protection against contamination.

## Quality rule

The candidate must satisfy all three conditions:

1. Its artifact is smaller than the full-data reference.
2. Its mean paired audit-loss difference is at most zero after exactly 400 updates.
3. Its one-sided 95% Student-t upper confidence bound is at most 0.01 nats/token, using at least eight seeds.

The third condition limits unresolved training-seed uncertainty. It does not prove exact equality of population losses.
The confidence interval conditions on the audit documents. It does not quantify uncertainty over all possible corpora.
Report all paired differences and the evaluation token count.
Architecture, optimizer, learning-rate, initialization, and update-budget changes invalidate the data-only entry.
The judge records source hashes and rejects changes to the registered model and configuration.

## Artifact and compute accounting

The main track counts the uncompressed `.npy` payload, including its header.
The fixed tokenizer and common learner are excluded from every entry.
Entropy coding alone cannot set a data-selection record. Do not compare a compressed candidate with an uncompressed baseline.
The learner reads only the submitted token array during training.
Selection IDs that require the original corpus during training are not sufficient submissions.

Use the fixed 20,000-document training pool for the closed-data track.
Teachers may be trained on that pool. Declare all teacher training, scoring, search, and artifact-building costs.
Disclose reused computation, including computation from earlier campaigns.
External pretrained models or external source data require a separate open-data entry.
Do not use audit text, audit gradients, or audit loss to construct an artifact.

Report artifact bytes, construction compute, training time, setup time, hardware, and all quality results separately.
A storage record does not imply an end-to-end compute saving.
Dataset-derived artifacts retain the TinyStories CDLA-Sharing-1.0 license. The benchmark code uses MIT.

## Records and contribution

Open a pull request with the manifest, builder code, complete result records, and an artifact download.
Include the comparison against the full-data control and the exact judge commit.
The maintainer reproduces the submission before adding a record to `competition/leaderboard.json`.
Every accepted record credits its contributors and links its code, artifact, and logs.
Useful negative results can be submitted without claiming a record.

Gradient latents and teacher-target methods remain an experimental training-signal track.
They require separate payload accounting and judging. They cannot enter the data-only leaderboard.
The first campaign's code and failures remain under `research/autoresearch/`.

## Two-worker confirmation

Use separate checkouts, caches, and outputs. Copy the identical frozen manifest to both workers before audit evaluation.
Run `--freeze-only` first. Commit the frozen inputs before launching either worker.
Use `--worker 0 --workers 2` and `--worker 1 --workers 2` to distribute complete candidate/control pairs.
Merge the result directories, preserving their names. Then run:

```bash
python -m competition.judge --summarize --output runs/my-subset-judge
```

## Precedents

[NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt) supplies the competition format and reproducible record history.
[DataComp-LM](https://www.datacomp.ai/dclm/) supplies the fixed-training-recipe precedent for data research.
This challenge adds a training-artifact size objective with a fixed update budget and paired quality checks.
Its TinyStories losses are not comparable to FineWeb losses.
