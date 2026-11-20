# First verified dataset compression record

The `influence75` artifact occupies 6,761,798 bytes, versus 8,980,666 bytes for the full-data reference.
It retains 75% of training windows and reduces serialized storage by 24.7%.

Across eight new training seeds, mean fresh-audit loss was 2.974170 versus 2.974509 for matched full-data controls.
The mean difference was −0.000339 nats/token. Its one-sided 95% upper bound was +0.004146 nats.
The candidate passes the registered mean-quality and +0.01-nat uncertainty criteria.
The result does not establish lower population loss or a training-time saving.

The model, tokenizer, optimizer, loss, batch size, and 400-update schedule remained fixed.
The artifact was frozen in commit `38ebfc5` before audit evaluation.
The audit contains 524,288 target tokens from new source documents.

## Assets

- `influence-f0.750.npy`: the exact submitted uint16 training-window array.
- `release-artifacts.json`: artifact size, SHA-256, and license.

Artifact SHA-256: `da04cd446e9cea2e44ec58bc123edf160126861c2bf8c0d16c6faf9f3c5aae9f`.

The artifact derives from TinyStories by Ronen Eldan and Yuanzhi Li.
Dataset-derived assets use CDLA-Sharing-1.0. Benchmark code uses MIT.

The repository includes exact reconstruction indices, a starter submission builder, a paired judge, and all search and confirmation results.
The audit is now exposed and supports reproduction. New record attempts require another reserved evaluation cohort.
