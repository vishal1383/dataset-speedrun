"""Train fresh-seed controls while search finishes; do not access audit loss."""

from engine import ROOT, PROTOCOL, run

for seed in PROTOCOL["confirmation_seeds"]:
    run(seed, ROOT / "runs/autoresearch-v1/reserved-controls" / f"seed{seed}", checkpoint=True)
