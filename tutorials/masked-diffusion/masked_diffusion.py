"""Masked discrete diffusion, from scratch.

Your job: replace every `raise NotImplementedError` with a working
implementation until `python3 -m pytest -q` is green. Everything you need was
derived in the explainer's Maths II section (plus one DDIM step from Maths I):
docs/2026-08-01-diffusion-language-models.html

Stuck on ONE function? Peek at solutions/masked_diffusion.py — one function,
not the file.
"""
import numpy as np

MASK = -1  # the [MASK] token id


def reveal_prob(alpha_s: float, alpha_t: float) -> float:
    """Probability that a token masked at time t is revealed when stepping to
    the less-noisy time s (alpha_s > alpha_t).

    Hint: it is a ratio — the chance of surviving to s but not to t, given the
    token is masked at t. Derived in "The reverse posterior" (Maths II).
    """
    raise NotImplementedError


def forward_mask(x: np.ndarray, alpha_t: float, rng: np.random.Generator) -> np.ndarray:
    """One-jump forward corruption q(z_t | x).

    Each token independently SURVIVES with probability alpha_t and is replaced
    by MASK otherwise. Return a NEW array; do not modify x.

    Hint: rng.random(x.shape) gives you one uniform draw per token.
    """
    raise NotImplementedError


def reverse_step(z: np.ndarray, x_probs: np.ndarray, alpha_s: float, alpha_t: float,
                 rng: np.random.Generator) -> np.ndarray:
    """One ancestral step of the reverse process from time t to time s (s < t,
    alpha_s > alpha_t).

    - Unmasked positions carry over unchanged (probability 1).
    - Each masked position is revealed with probability reveal_prob(alpha_s,
      alpha_t); when revealed, sample its token from that position's row of
      x_probs (shape: len(z) x vocab). Return a NEW array.

    Hint: rng.choice(vocab_size, p=row) samples from one categorical row.
    """
    raise NotImplementedError


def nelbo_weight(t: float) -> float:
    """The continuous-time NELBO weight w(t) = -alpha'_t / (1 - alpha_t) for
    the LINEAR schedule alpha_t = 1 - t. Simplify it by hand first.
    """
    raise NotImplementedError


def weighted_mlm_loss(logp: np.ndarray, x: np.ndarray, z: np.ndarray, t: float) -> float:
    """MDLM's integrand at corruption level t (linear schedule):

        w(t) * sum over masked positions of  -log p(true token at that position)

    logp: (n, vocab) log-probabilities from the model; x: (n,) true tokens;
    z: (n,) corrupted tokens (MASK where hidden). Return a float.
    """
    raise NotImplementedError


def ddim_step(x_t: np.ndarray, eps: np.ndarray, ab_t: float, ab_s: float) -> np.ndarray:
    """One deterministic DDIM step (sigma = 0) from noise level ab_t (= alpha-bar
    at time t) to ab_s.

    Hint: form the clean estimate x0_hat from (x_t, eps) at level ab_t, then
    re-mix it with the same eps at level ab_s. Derived in Maths I.
    """
    raise NotImplementedError
