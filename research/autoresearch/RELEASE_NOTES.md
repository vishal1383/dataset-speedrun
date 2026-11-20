First bounded compression autoresearch campaign: 44 search trials, 12 fresh-seed confirmations, and 3 source-model runs across two DGX Sparks.

The search tested data selection, sparse teacher targets, and low-rank gradient replay with a fixed GPT architecture and Muon/AdamW optimizer.

The half-sized token pool improved development loss but worsened fresh-audit loss from 3.0031 to 3.1014 nats/token.
None of the compressed finalists preserved fresh-audit quality.

One first-step QKV diagnostic found 8.3% raw-gradient error but 97.3% Muon-update error after rank-8 compression.
This is a scoped diagnostic, not a general claim about every Muon update.

The release contains the exact three frozen training-signal artifacts used for confirmation, including the failed gradient candidate.
Artifact hashes and byte counts are in `research/autoresearch/results/release-artifacts.json`.
Methods, costs, raw search histories, and confirmation results are in `research/autoresearch/`.

Source attribution: Ronen Eldan and Yuanzhi Li, TinyStories, https://huggingface.co/datasets/roneneldan/TinyStories.
Dataset-derived release assets use CDLA-Sharing-1.0: https://cdla.dev/sharing-1-0/.
Repository code uses MIT.
