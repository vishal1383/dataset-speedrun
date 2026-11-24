import torch
from torch.nn import functional as F

from research.fineweb.signal_screen import predicted_gradient, sparse_loss


def test_sparse_teacher_loss_and_gradient_match_dense_distribution():
    torch.manual_seed(19)
    logits = torch.randn(2, 3, 11, dtype=torch.float64, requires_grad=True)
    gold = torch.randint(11, (2, 3))
    ids = torch.tensor([0, 2, 5, 9]).expand(2, 3, 4)
    probs = torch.rand(2, 3, 4, dtype=torch.float64) / 6
    q = F.one_hot(gold, 11).to(torch.float64) * (1 - 0.5 * probs.sum(-1, keepdim=True))
    q.scatter_add_(-1, ids, 0.5 * probs)
    expected = -(q * logits.log_softmax(-1)).sum()
    actual = sparse_loss(logits, gold, ids, probs, 0.5)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, logits)[0], torch.autograd.grad(expected, logits)[0])


def test_predicted_gradient_does_not_alias_cache_and_secant_uses_refresh_interval():
    current = torch.tensor([2.0, 6.0])
    previous = torch.tensor([0.0, 2.0])
    cached = predicted_gradient(current)
    cached.lerp_(torch.zeros_like(cached), 0.95)
    torch.testing.assert_close(current, torch.tensor([2.0, 6.0]))
    torch.testing.assert_close(predicted_gradient(current, previous, 0.5), torch.tensor([3.0, 8.0]))
