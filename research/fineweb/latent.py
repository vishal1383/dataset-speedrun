"""Decode compressed conditional distributions into state-dependent logit gradients."""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F


class ConditionalTargets:
    def __init__(self, keys, ids, probabilities, support):
        self.keys = keys
        self.ids = ids
        self.probabilities = probabilities
        self.support = support

    @classmethod
    def load(cls, path, device="cuda"):
        with np.load(path) as data:
            keys = torch.tensor(data["keys"].astype(np.int64), device=device)
            ids = torch.tensor(data["ids"].astype(np.int64), device=device)
            probabilities = torch.tensor(data["probabilities"].astype(np.float32), device=device)
            support = torch.tensor(data["support"].astype(np.float32), device=device)
        if len(keys) == 0 or not bool((keys[1:] > keys[:-1]).all()):
            raise ValueError("Latent keys must be nonempty, unique, and sorted.")
        if ids.shape != probabilities.shape or ids.shape != (len(keys), 4) or support.shape != keys.shape:
            raise ValueError("Latent table shapes differ from the fixed top-four format.")
        if bool((probabilities < 0).any()) or bool((probabilities.sum(-1) > 1.001).any()):
            raise ValueError("Invalid categorical probability mass.")
        # FP16 rounding can increase a complete row's sum slightly above one.
        probabilities /= probabilities.sum(-1, keepdim=True).clamp_min(1)
        return cls(keys, ids, probabilities, support)

    def decode(self, inputs):
        previous = torch.cat((inputs[:, :1], inputs[:, :-1]), dim=1)
        contexts = (previous.to(torch.int64) * 50304 + inputs.to(torch.int64)).contiguous()
        positions = torch.searchsorted(self.keys, contexts).clamp_max(len(self.keys) - 1)
        valid = (self.keys[positions] == contexts) & (previous != 50256) & (inputs != 50256)
        valid = valid & (torch.arange(inputs.shape[1], device=inputs.device)[None, :] > 0)
        confidence = self.support[positions] / (self.support[positions] + 32)
        weights = self.probabilities[positions] * (valid * confidence)[..., None]
        return self.ids[positions], weights


def sparse_target_loss(logits, inputs, targets, table, strength):
    """Exact CE for q=(1-a*mass)*gold+a*sparse_targets without dense q storage."""
    ids, weights = table.decode(inputs)
    gold = logits.gather(-1, targets[..., None]).squeeze(-1)
    alternatives = logits.gather(-1, ids)
    correction = (weights * (gold[..., None] - alternatives)).sum()
    hard = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum")
    return hard + strength * correction


def render_latent(source, codebook, root):
    try:
        from .launch import replace_once
    except ImportError:
        from launch import replace_once
    source = replace_once(source, "import torch.distributed as dist", "import torch.distributed as dist\n"
                          f"sys.path.insert(0, {str(root)!r})\n"
                          "from research.fineweb.latent import ConditionalTargets, sparse_target_loss")
    source = replace_once(source, "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()",
                          f"_latent_table = ConditionalTargets.load({str(Path(codebook).resolve())!r})\n"
                          "_latent_strength = torch.tensor(0.25, device='cuda')\n"
                          "model = GPT(vocab_size=50304, num_layers=12, model_dim=768).cuda()")
    original = '        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")'
    source = replace_once(source, original,
                          "        if self.training:\n"
                          "            return sparse_target_loss(logits, inputs, targets, _latent_table, _latent_strength)\n" + original)
    source = replace_once(source, "        inputs, targets = next(train_loader)",
                          "        _latent_strength.fill_(0.25 * (1 - step / train_steps))\n"
                          "        inputs, targets = next(train_loader)")
    return source
