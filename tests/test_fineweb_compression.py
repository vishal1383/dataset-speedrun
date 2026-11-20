"""Check compressed supervision, coreset selection, and fixed learner boundaries."""

import numpy as np
import ast
from pathlib import Path
import torch
from torch.nn import functional as F

from research.fineweb.compress import make_codebook, select_coreset, VOCAB
from research.fineweb.latent import ConditionalTargets, sparse_target_loss, render_latent
from research.fineweb.launch import render
from research.fineweb.compression_queue import verify_artifact
import json
import hashlib
import pytest


def test_coreset_preserves_balanced_repeated_features():
    features = np.tile(np.eye(4, dtype=np.float32), (16, 1))
    selected, error = select_coreset(features, np.zeros(64, dtype=np.float32))
    assert len(selected) == len(np.unique(selected)) == 48
    np.testing.assert_allclose(features[selected].mean(0), features.mean(0), atol=0.03)
    assert error < 0.06


def test_codebook_retains_true_topk_mass_without_renormalization():
    context = 12 * VOCAB + 34
    targets = np.array([8] * 20 + [9] * 10 + [10] * 5 + [11] * 3 + [12] * 2, dtype=np.uint64)
    events = (np.uint64(context) << np.uint64(16)) | targets
    keys, ids, probabilities, support = make_codebook(events, minimum=16)
    assert keys.tolist() == [context]
    assert ids.tolist() == [[8, 9, 10, 11]]
    np.testing.assert_allclose(probabilities, [[0.5, 0.25, 0.125, 0.075]])
    assert support.tolist() == [40]
    assert probabilities.sum() < 1


def test_codebook_discards_rare_contexts_and_orders_lookup_keys():
    events = np.array([(7 << 16) | 9] * 16 + [(2 << 16) | 5] * 20 + [(1 << 16) | 3] * 2, dtype=np.uint64)
    keys, _, _, _ = make_codebook(events, minimum=16)
    assert keys.tolist() == [2, 7]


def table():
    return ConditionalTargets(torch.tensor([1 * VOCAB + 2, 2 * VOCAB + 3]),
                              torch.tensor([[2, 3, 4, 5], [1, 2, 4, 5]]),
                              torch.tensor([[0.4, 0.2, 0.1, 0.05], [0.3, 0.2, 0.1, 0.05]]),
                              torch.tensor([64., 32.]))


def test_latent_loss_and_gradient_match_dense_soft_target_ce():
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    torch.manual_seed(17)
    logits = torch.randn(1, 4, 7, requires_grad=True)
    ids, weights = table().decode(inputs)
    q = F.one_hot(targets, num_classes=7).float() * (1 - 0.25 * weights.sum(-1, keepdim=True))
    q.scatter_add_(-1, ids, 0.25 * weights)
    torch.testing.assert_close(q.sum(-1), torch.ones_like(targets).float())
    loss = sparse_target_loss(logits, inputs, targets, table(), 0.25)
    dense = -(q * logits.log_softmax(-1)).sum()
    torch.testing.assert_close(loss, dense)
    gradient, = torch.autograd.grad(loss, logits)
    torch.testing.assert_close(gradient, logits.softmax(-1) - q)


def test_missing_context_first_position_and_document_boundaries_get_no_latent_mass():
    inputs = torch.tensor([[1, 2, 50256, 3, 4]])
    _, weights = table().decode(inputs)
    assert weights[0, 1].sum() > 0
    assert weights[0, [0, 2, 3, 4]].sum() == 0


def test_zero_latent_strength_matches_original_ce_and_gradient():
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    logits = torch.randn(1, 4, 7, requires_grad=True)
    loss = sparse_target_loss(logits, inputs, targets, table(), 0.)
    hard = F.cross_entropy(logits.reshape(-1, 7), targets.reshape(-1), reduction="sum")
    torch.testing.assert_close(loss, hard, rtol=0, atol=0)
    a, = torch.autograd.grad(loss, logits, retain_graph=True)
    b, = torch.autograd.grad(hard, logits)
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_latent_adapter_preserves_architecture_optimizer_and_eval_objective(tmp_path):
    root = Path(__file__).resolve().parents[1]
    original = (root / "research/fineweb/upstream/train_gpt_simple.py").read_text()
    hard = render(original, tmp_path, "candidate/*.bin", 17, 3250, 10485760, 64, True)
    soft = render_latent(hard, tmp_path / "latent.npz", root)
    def definitions(source):
        return {n.name: n for n in ast.parse(source).body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    a, b = definitions(hard), definitions(soft)
    for name in a:
        if name != "GPT":
            assert ast.dump(a[name]) == ast.dump(b[name])
    assert ast.dump(a['GPT'].body[0]) == ast.dump(b['GPT'].body[0])  # Parameters unchanged.
    assert ast.dump(a['GPT'].body[1].body[-1]) == ast.dump(b['GPT'].body[1].body[-1])  # Evaluation CE unchanged.
    for name in ['optimizer1', 'optimizer2', 'train_steps', 'batch_size']:
        def assignment(text):
            return [ast.dump(n) for n in ast.walk(ast.parse(text)) if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == name for t in n.targets)]
        assert assignment(hard) == assignment(soft)


def test_compiled_latent_loss_uses_updated_strength():
    inputs = torch.tensor([[1, 2, 3, 4]])
    targets = torch.tensor([[2, 3, 4, 5]])
    logits = torch.randn(1, 4, 7, requires_grad=True)
    strength = torch.tensor(0.25)
    compiled = torch.compile(sparse_target_loss, backend="aot_eager", fullgraph=True)
    for value in [0.25, 0.1, 0.0]:
        strength.fill_(value)
        actual = compiled(logits, inputs, targets, table(), strength)
        expected = sparse_target_loss(logits, inputs, targets, table(), strength)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, logits, retain_graph=True)[0],
                                   torch.autograd.grad(expected, logits)[0])


def test_artifact_verifier_rejects_changed_replay_and_payload(tmp_path):
    (tmp_path / 'text').mkdir()
    (tmp_path / 'stream').mkdir()
    payload = tmp_path / 'text/a.bin'
    payload.write_bytes(b'tokens')
    replay = tmp_path / 'replay.json'
    replay.write_text(json.dumps([{'file': 'fineweb_train_000001.bin', 'payload': 'a.bin'}]))
    link = tmp_path / 'stream/fineweb_train_000001.bin'
    link.symlink_to('../text/a.bin')
    def sha(p):
        return hashlib.sha256(p.read_bytes()).hexdigest()
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'files': {'text/a.bin': {'sha256': sha(payload)}, 'replay.json': {'sha256': sha(replay)}}}))
    expected = sha(manifest)
    verify_artifact(tmp_path, expected)
    link.unlink()
    link.symlink_to('../text/b.bin')
    with pytest.raises(ValueError):
        verify_artifact(tmp_path, expected)
    link.unlink()
    link.symlink_to('../text/a.bin')
    payload.write_bytes(b'changed')
    with pytest.raises(ValueError):
        verify_artifact(tmp_path, expected)
