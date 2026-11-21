"""Prevent pilot selection from turning a failed or partial result into a winner."""

import pytest

from research.fineweb.replicate import choose_winner


def summary(text=3.27, latent=3.275):
    return {"baseline_loss": 3.27822, "arms": [{"arm": "text", "final_loss": text},
                                              {"arm": "latent", "final_loss": latent}]}


def test_selects_lowest_final_loss_and_prefers_text_on_ties():
    assert choose_winner(summary()) == "text"
    assert choose_winner(summary(latent=3.26)) == "latent"
    assert choose_winner(summary(text=3.27, latent=3.27)) == "text"


def test_no_winner_when_both_arms_fail_to_improve():
    assert choose_winner(summary(text=3.28, latent=3.29)) is None
    assert choose_winner(summary(text=3.27822, latent=3.27822)) is None


def test_rejects_partial_or_nonfinite_results():
    for data in [summary(text=float('nan')), summary(latent=float('inf')),
                 {'baseline_loss': 3.28, 'arms': [{'arm': 'text', 'final_loss': 3.27}]}]:
        with pytest.raises(ValueError):
            choose_winner(data)
