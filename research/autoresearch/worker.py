"""Build a teacher or capture a gradient trajectory before search."""

import argparse
from pathlib import Path

from engine import run, PROTOCOL

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["teacher", "gradient-source"])
parser.add_argument("--root", type=Path, default=Path("runs/autoresearch-v1"))
args = parser.parse_args()
if args.mode == "teacher":
    run(PROTOCOL["teacher_seed"], args.root / "teacher", checkpoint=True)
else:
    run(PROTOCOL["search_seed"], args.root / "gradient-source", capture=True)
