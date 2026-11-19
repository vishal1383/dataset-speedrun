# TinyStories v0 artifacts

`manifest.json` records the dataset revision, preparation time, selection time, and artifact hashes.
`tokenizer.json` contains the fixed tokenizer trained on the original candidate pool.
The `*.ids.json` files contain indices into the deduplicated training document list.

These files contain data artifacts derived from TinyStories, which uses [CDLA-Sharing-1.0](https://cdla.dev/sharing-1-0/).
They are outside the repository's MIT code license.
Source attribution: Ronen Eldan and Yuanzhi Li, [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).

The preparation script regenerates the token arrays. Raw source text is not included in this repository.
An independent preparation run reproduced every file hash before the initial release.
