"""Tests for the masked-diffusion lab. Green = you have written masked diffusion.

Run:  python3 -m pytest -q
"""
import numpy as np
import pytest

from masked_diffusion import (MASK, ddim_step, forward_mask, nelbo_weight,
                              reveal_prob, reverse_step, weighted_mlm_loss)


# ---------------------------------------------------------------- reveal ----
def test_reveal_prob_quiz_numbers():
    # the explainer's quiz question: alpha_s = 0.75, alpha_t = 0.25 -> 2/3
    assert reveal_prob(0.75, 0.25) == pytest.approx(2 / 3)


def test_reveal_prob_endpoints():
    # a final step to alpha_s = 1 reveals everything still masked
    assert reveal_prob(1.0, 0.4) == pytest.approx(1.0)
    # from a fully-noised state (alpha_t = 0), reveal prob is alpha_s itself
    assert reveal_prob(0.6, 0.0) == pytest.approx(0.6)


# --------------------------------------------------------------- forward ----
def test_forward_mask_survival_rate():
    rng = np.random.default_rng(0)
    x = np.arange(20000) % 50
    z = forward_mask(x, alpha_t=0.3, rng=rng)
    survived = z != MASK
    assert abs(survived.mean() - 0.3) < 0.02          # ~alpha_t survive
    assert np.all(z[survived] == x[survived])          # survivors unchanged
    assert np.all(z[~survived] == MASK)                # the rest are MASK


def test_forward_mask_does_not_mutate_input():
    rng = np.random.default_rng(1)
    x = np.arange(100)
    x_before = x.copy()
    forward_mask(x, 0.5, rng)
    assert np.array_equal(x, x_before)


# --------------------------------------------------------------- reverse ----
def test_reverse_step_carries_over_unmasked():
    rng = np.random.default_rng(2)
    z = np.array([5, MASK, 7, MASK])
    x_probs = np.full((4, 10), 0.1)
    out = reverse_step(z, x_probs, alpha_s=0.9, alpha_t=0.1, rng=rng)
    assert out[0] == 5 and out[2] == 7                 # visible tokens frozen


def test_reverse_step_reveal_fraction():
    rng = np.random.default_rng(3)
    n, v = 20000, 8
    z = np.full(n, MASK)
    x_probs = np.full((n, v), 1.0 / v)
    out = reverse_step(z, x_probs, alpha_s=0.75, alpha_t=0.25, rng=rng)
    frac = (out != MASK).mean()
    assert abs(frac - 2 / 3) < 0.02                    # (0.75-0.25)/(1-0.25)


def test_reverse_step_samples_from_model():
    rng = np.random.default_rng(4)
    n = 5000
    z = np.full(n, MASK)
    x_probs = np.zeros((n, 6))
    x_probs[:, 4] = 1.0                                # model is certain: token 4
    out = reverse_step(z, x_probs, alpha_s=1.0, alpha_t=0.0, rng=rng)
    assert np.all(out == 4)                            # revealed = model's choice


def test_reverse_chain_reconstructs_with_oracle():
    # with an oracle x_theta (one-hot on the truth), the full reverse chain
    # from all-masked must reconstruct the sentence exactly
    rng = np.random.default_rng(5)
    x = np.array([3, 1, 4, 1, 5, 9, 2, 6])
    v = 10
    x_probs = np.eye(v)[x]                             # oracle predictions
    z = np.full_like(x, MASK)
    alphas = np.linspace(0.0, 1.0, 9)                  # t: 1 -> 0, alpha: 0 -> 1
    for a_t, a_s in zip(alphas[:-1], alphas[1:]):
        z = reverse_step(z, x_probs, alpha_s=a_s, alpha_t=a_t, rng=rng)
    assert np.array_equal(z, x)


# ------------------------------------------------------------- objective ----
def test_nelbo_weight_linear_schedule():
    assert nelbo_weight(0.5) == pytest.approx(2.0)
    assert nelbo_weight(0.25) == pytest.approx(4.0)


def test_weighted_mlm_loss_hand_computed():
    # 3 tokens, vocab 4; positions 0 and 2 masked; t = 0.5 -> weight 2
    logp = np.log(np.array([
        [0.7, 0.1, 0.1, 0.1],
        [0.25, 0.25, 0.25, 0.25],
        [0.1, 0.1, 0.6, 0.2],
    ]))
    x = np.array([0, 1, 2])
    z = np.array([MASK, 1, MASK])
    expected = 2.0 * (-(np.log(0.7)) + -(np.log(0.6)))
    assert weighted_mlm_loss(logp, x, z, t=0.5) == pytest.approx(expected)


# ------------------------------------------------------------------ ddim ----
def test_ddim_step_moves_to_correct_noise_level():
    # construct x_t exactly from (x0, eps); one step with the TRUE eps must
    # land exactly on the (x0, eps) mixture at the new level
    rng = np.random.default_rng(6)
    x0 = rng.standard_normal(64)
    eps = rng.standard_normal(64)
    ab_t, ab_s = 0.36, 0.81
    x_t = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * eps
    x_s = ddim_step(x_t, eps, ab_t, ab_s)
    np.testing.assert_allclose(x_s, np.sqrt(ab_s) * x0 + np.sqrt(1 - ab_s) * eps, atol=1e-12)


def test_ddim_step_final_recovers_clean():
    rng = np.random.default_rng(7)
    x0 = rng.standard_normal(32)
    eps = rng.standard_normal(32)
    ab_t = 0.25
    x_t = np.sqrt(ab_t) * x0 + np.sqrt(1 - ab_t) * eps
    np.testing.assert_allclose(ddim_step(x_t, eps, ab_t, 1.0), x0, atol=1e-12)
