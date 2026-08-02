"""Masked discrete diffusion, from scratch — REFERENCE SOLUTION.

Everything here is the numpy translation of the explainer's Maths II section
(plus one DDIM step from Maths I). Peek at ONE function when stuck, not the file.
"""
import numpy as np

MASK = -1  # the [MASK] token id


def reveal_prob(alpha_s: float, alpha_t: float) -> float:
    """Probability that a token masked at time t is revealed when stepping to
    the less-noisy time s (alpha_s > alpha_t).

        P(reveal) = (alpha_s - alpha_t) / (1 - alpha_t)
    """
    return (alpha_s - alpha_t) / (1.0 - alpha_t)


def forward_mask(x: np.ndarray, alpha_t: float, rng: np.random.Generator) -> np.ndarray:
    """One-jump forward corruption q(z_t | x).

    Each token independently SURVIVES with probability alpha_t and is replaced
    by MASK otherwise. Returns a new array; does not modify x.
    """
    survive = rng.random(x.shape) < alpha_t
    z = np.where(survive, x, MASK)
    return z


def reverse_step(z: np.ndarray, x_probs: np.ndarray, alpha_s: float, alpha_t: float,
                 rng: np.random.Generator) -> np.ndarray:
    """One ancestral step of the reverse process from time t to time s (s < t,
    alpha_s > alpha_t).

    - Unmasked positions carry over unchanged (probability 1).
    - Each masked position is revealed with probability reveal_prob(alpha_s,
      alpha_t); when revealed, its token is sampled from that position's row of
      x_probs (shape: len(z) x vocab).
    Returns a new array.
    """
    z = z.copy()
    p = reveal_prob(alpha_s, alpha_t)
    masked = np.where(z == MASK)[0]
    for i in masked:
        if rng.random() < p:
            z[i] = rng.choice(len(x_probs[i]), p=x_probs[i])
    return z


def nelbo_weight(t: float) -> float:
    """The continuous-time NELBO weight w(t) = -alpha'_t / (1 - alpha_t) for
    the LINEAR schedule alpha_t = 1 - t.  (It simplifies — do the algebra.)
    """
    return 1.0 / t


def weighted_mlm_loss(logp: np.ndarray, x: np.ndarray, z: np.ndarray, t: float) -> float:
    """MDLM's integrand at corruption level t (linear schedule):

        w(t) * sum over masked positions of  -log p(true token at that position)

    logp: (n, vocab) log-probabilities from the model; x: (n,) true tokens;
    z: (n,) corrupted tokens (MASK where hidden).
    """
    masked = z == MASK
    return float(nelbo_weight(t) * np.sum(-logp[masked, x[masked]]))


def ddim_step(x_t: np.ndarray, eps: np.ndarray, ab_t: float, ab_s: float) -> np.ndarray:
    """One deterministic DDIM step (sigma = 0) from noise level ab_t (= alpha-bar
    at time t) to ab_s:

        x0_hat = (x_t - sqrt(1 - ab_t) * eps) / sqrt(ab_t)
        x_s    = sqrt(ab_s) * x0_hat + sqrt(1 - ab_s) * eps
    """
    x0_hat = (x_t - np.sqrt(1.0 - ab_t) * eps) / np.sqrt(ab_t)
    return np.sqrt(ab_s) * x0_hat + np.sqrt(1.0 - ab_s) * eps
