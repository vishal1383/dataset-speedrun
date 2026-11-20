"""Ensure the data-track adapter preserves the pinned learner."""

import ast
import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

from research.fineweb.launch import render, replace_once
from research.fineweb.build_candidate import ShardWriter, repetitive
from research.fineweb.build_edu import documents, doc_hash
from research.fineweb.run_pair import verify_pair

ROOT = Path(__file__).resolve().parents[1]


def source_pair(tmp_path):
    original = (ROOT / "research/fineweb/upstream/train_gpt_simple.py").read_text()
    adapted = render(original, tmp_path, "runs/candidate/train_*.bin", 17, 2500, 10485760, 64, True)
    return original, adapted


def test_pinned_upstream_and_all_learner_definitions_are_unchanged(tmp_path):
    original, adapted = source_pair(tmp_path)
    pin = json.loads((ROOT / "research/fineweb/pin.json").read_text())
    assert hashlib.sha256(original.encode()).hexdigest() == pin["source_sha256"]
    def definitions(text):
        return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(text).body
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))}
    assert definitions(original) == definitions(adapted)


def test_optimizer_and_schedule_assignments_are_unchanged(tmp_path):
    original, adapted = source_pair(tmp_path)
    def selected(text):
        result = []
        for node in ast.walk(ast.parse(text)):
            if isinstance(node, ast.Assign):
                if any(isinstance(target, ast.Name) and target.id in
                       ["optimizer1", "optimizer2", "train_steps", "batch_size", "model"] for target in node.targets):
                    result.append(ast.dump(node, include_attributes=False))
            if isinstance(node, ast.FunctionDef) and node.name == "set_hparams":
                result.append(ast.dump(node, include_attributes=False))
        return sorted(result)
    assert selected(original) == selected(adapted)
    assert "run_steps = 2500" in adapted
    assert "val_tokens = 10485760" in adapted
    assert "mbs = 64" in adapted


def test_initialization_block_is_preserved(tmp_path):
    original, adapted = source_pair(tmp_path)
    def block(text):
        return next(ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(text))
                    if isinstance(n, ast.For) and isinstance(n.target, ast.Tuple)
                    and [getattr(x, "id", None) for x in n.target.elts] == ["name", "p"]
                    and any(isinstance(x, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "w" for t in x.targets)
                            for x in n.body))
    assert block(original) == block(adapted)


def test_source_boundary_changes_fail_closed():
    with pytest.raises(ValueError):
        replace_once("a a", "a", "b")
    with pytest.raises(ValueError):
        replace_once("a", "b", "c")


def test_candidate_shards_preserve_token_stream_across_boundaries(tmp_path):
    writer = ShardWriter(tmp_path / "candidate", capacity=131)
    segments = [np.arange(1, 107, dtype=np.uint16), np.arange(500, 829, dtype=np.uint16)]
    for segment in segments:
        writer.append(segment)
    writer.flush()
    recovered = np.concatenate([np.fromfile(p, dtype=np.uint16, offset=1024)
                                for p in sorted((tmp_path / "candidate").glob("*.bin"))])
    assert np.array_equal(recovered, np.concatenate(segments))
    assert sum(r["tokens"] for r in writer.files.values()) == len(recovered)


def test_repetition_filter_keeps_short_and_diverse_documents():
    assert not repetitive(np.array([50256] + [7] * 30, dtype=np.uint16))
    assert not repetitive(np.array([50256] + list(range(100, 400)), dtype=np.uint16))
    assert repetitive(np.array([50256] + [7] * 300, dtype=np.uint16))


def test_document_reconstruction_survives_source_shard_boundaries(tmp_path):
    writer = ShardWriter(tmp_path / "source", capacity=7)
    original = np.array([99, 98, 50256, 4, 5, 6, 7, 8, 9, 50256, 10, 11, 50256, 12], dtype=np.uint16)
    writer.append(original)
    writer.flush()
    recovered = list(documents(sorted((tmp_path / "source").glob("*.bin"))))
    assert [d.tolist() for d in recovered] == [[99, 98], [50256, 4, 5, 6, 7, 8, 9], [50256, 10, 11], [50256, 12]]
    assert np.array_equal(np.concatenate(recovered), original)
    assert doc_hash(recovered[1]) == doc_hash(original[2:9])


def test_pair_rejects_mismatched_validation_and_incomplete_runs():
    record = {"status": "complete", "smoke": False, "exit_code": 0,
              "curve": [{"step": 3250}], "seed": 17, "updates": 3250,
              "initial_state_sha256": "abc", "validation_files": {"val": "hash"},
              "validation_tokens": 10485760, "schedule_updates": 3250,
              "global_batch_tokens": 524288, "microbatch_sequences": 64,
              "compile_model": True, "upstream": {"commit": "pinned"}}
    runs = [{"seed": 17, "updates": 3250}] * 2
    verify_pair([record, dict(record)], runs)
    for change in [{"validation_files": {"val": "different"}}, {"smoke": True},
                   {"curve": [{"step": 3125}]}, {"initial_state_sha256": None},
                   {"schedule_updates": 3000}]:
        with pytest.raises(ValueError):
            verify_pair([record, dict(record, **change)], runs)
