"""Bayesian optimization, from scratch — REFERENCE SOLUTION.

The numpy translation of the explainer's maths sections: RBF kernel, GP
posterior by Gaussian conditioning, closed-form expected improvement, UCB,
and the full BO loop. Peek at ONE function when stuck, not the file.
"""
import math

import numpy as np

_PHI = np.vectorize(lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))


def rbf_kernel(A, B, lengthscale):
    """Squared-exponential kernel matrix between row-vectors A (n,d) and B (m,d):

        k(x, x') = exp(-||x - x'||^2 / (2 * lengthscale^2))
    """
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / lengthscale ** 2)


def gp_posterior(Xs, X, y, lengthscale, noise=1e-6):
    """GP posterior at query points Xs (m,d) given data (X (n,d), y (n,)).

    Returns (mu, sd), each shape (m,). Unit signal variance, zero prior mean:
        mu    = k*^T (K + noise I)^-1 y
        sd^2  = 1 + noise - k*^T (K + noise I)^-1 k*   (clipped at >= 1e-12)
    """
    K = rbf_kernel(X, X, lengthscale) + noise * np.eye(len(X))
    Ks = rbf_kernel(Xs, X, lengthscale)
    Ki = np.linalg.inv(K)
    mu = Ks @ Ki @ y
    var = np.clip(1.0 + noise - np.einsum("ij,jk,ik->i", Ks, Ki, Ks), 1e-12, None)
    return mu, np.sqrt(var)


def expected_improvement(mu, sd, best):
    """Closed-form EI for maximization: (mu-best)*Phi(z) + sd*phi(z), z=(mu-best)/sd."""
    z = (mu - best) / sd
    phi = np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    return (mu - best) * _PHI(z) + sd * phi


def ucb(mu, sd, beta):
    """Upper confidence bound: mu + sqrt(beta) * sd."""
    return mu + math.sqrt(beta) * sd


def propose_next(candidates, X, y, lengthscale, noise=1e-6):
    """The acquisition step: return the candidate row maximizing EI."""
    mu, sd = gp_posterior(candidates, X, y, lengthscale, noise)
    return candidates[int(np.argmax(expected_improvement(mu, sd, y.max())))]


def run_bo(f, X_init, n_steps, lengthscale, rng, noise=1e-6):
    """Full BO loop (maximization). f maps (n,d) -> (n,); X_init is (k,d).

    Each step: propose_next over 256 fresh uniform candidates in [0,1]^d,
    evaluate f there, augment. Returns (X, y) of everything evaluated.
    """
    X = np.array(X_init, dtype=float)
    y = f(X)
    d = X.shape[1]
    for _ in range(n_steps):
        C = rng.uniform(0, 1, (256, d))
        x_next = propose_next(C, X, y, lengthscale, noise)
        X = np.vstack([X, x_next])
        y = np.append(y, f(x_next[None, :]))
    return X, y
