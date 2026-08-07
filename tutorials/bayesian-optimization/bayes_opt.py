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



def knowledge_gradient(candidates, X, y, lengthscale, noise=1e-6, n_fantasy=64, rng=None):
    """Monte-Carlo knowledge gradient for each row of `candidates`.

        KG(x) = E[ max_x' mu_{n+1}(x') | sampled at x ] - max_x' mu_n(x')

    where mu_{n+1} is the posterior mean AFTER observing a value at x. That
    value is unknown, so average over `n_fantasy` draws from the current
    posterior at x (this is the "fantasy" sampling described in Maths II).

    The inner max is over `candidates` — the discretization that makes KG
    affordable. Steps for one candidate x:
      1. current best posterior mean over candidates  -> baseline
      2. for each fantasy value y~ ~ N(mu(x), sd(x)^2):
           refit the posterior with (x, y~) appended, take max over candidates
      3. KG(x) = mean(those maxima) - baseline   (clip at >= 0)

    Returns an array of shape (len(candidates),). Use `rng` for the draws so
    the tests are reproducible.
    """
    raise NotImplementedError
