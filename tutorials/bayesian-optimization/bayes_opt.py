"""Bayesian optimization, from scratch.

Your job: replace every `raise NotImplementedError` with a working
implementation until `python3 -m pytest -q` is green. Everything you need is
derived in the explainer's maths sections:
docs/2026-08-02-bayesian-optimization.html

Hints live in each docstring. Stuck on ONE function? Peek at
solutions/bayes_opt.py — one function, not the file.
"""
import math

import numpy as np

# Hint: the standard normal CDF, vectorized —
#   Phi = np.vectorize(lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def rbf_kernel(A, B, lengthscale):
    """Squared-exponential kernel matrix between row-vectors A (n,d) and B (m,d):

        k(x, x') = exp(-||x - x'||^2 / (2 * lengthscale^2))
    """
    raise NotImplementedError


def gp_posterior(Xs, X, y, lengthscale, noise=1e-6):
    """GP posterior at query points Xs (m,d) given data (X (n,d), y (n,)).

    Returns (mu, sd), each shape (m,). Unit signal variance, zero prior mean:
        mu    = k*^T (K + noise I)^-1 y
        sd^2  = 1 + noise - k*^T (K + noise I)^-1 k*   (clipped at >= 1e-12)
    """
    raise NotImplementedError


def expected_improvement(mu, sd, best):
    """Closed-form EI for maximization: (mu-best)*Phi(z) + sd*phi(z), z=(mu-best)/sd."""
    raise NotImplementedError


def ucb(mu, sd, beta):
    """Upper confidence bound: mu + sqrt(beta) * sd."""
    raise NotImplementedError


def propose_next(candidates, X, y, lengthscale, noise=1e-6):
    """The acquisition step: return the candidate row maximizing EI."""
    raise NotImplementedError


def run_bo(f, X_init, n_steps, lengthscale, rng, noise=1e-6):
    """Full BO loop (maximization). f maps (n,d) -> (n,); X_init is (k,d).

    Each step: propose_next over 256 fresh uniform candidates in [0,1]^d,
    evaluate f there, augment. Returns (X, y) of everything evaluated.
    """
    raise NotImplementedError

