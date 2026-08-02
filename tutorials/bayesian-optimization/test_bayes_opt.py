"""Tests for the Bayesian optimization tutorial. Green = you wrote BO.

Run:  python3 -m pytest -q
"""
import math

import numpy as np
import pytest

from bayes_opt import (expected_improvement, gp_posterior, propose_next,
                       rbf_kernel, run_bo, ucb)


# ---------------------------------------------------------------- kernel ---
def test_rbf_diagonal_and_symmetry():
    X = np.array([[0.1], [0.5], [0.9]])
    K = rbf_kernel(X, X, lengthscale=0.3)
    assert np.allclose(np.diag(K), 1.0)          # k(x,x) = 1
    assert np.allclose(K, K.T)                    # symmetric


def test_rbf_known_value():
    A = np.array([[0.0]]); B = np.array([[1.0]])
    # ||x-x'|| = 1, lengthscale = 1  ->  exp(-1/2)
    assert rbf_kernel(A, B, 1.0)[0, 0] == pytest.approx(math.exp(-0.5))


def test_rbf_lengthscale_controls_reach():
    A = np.array([[0.0]]); B = np.array([[0.5]])
    assert rbf_kernel(A, B, 1.0)[0, 0] > rbf_kernel(A, B, 0.1)[0, 0]


# ------------------------------------------------------------- posterior ---
def test_posterior_single_point_quiz_numbers():
    # the explainer's quiz: one noiseless obs y=2; at k(x,x1)=0.6:
    # mu = 1.2, var = 1 - 0.36 = 0.64  ->  sd = 0.8
    X = np.array([[0.0]]); y = np.array([2.0])
    ls = 1.0
    # choose xq so that k(xq, 0) = 0.6  ->  d = sqrt(-2 ln 0.6)
    d = math.sqrt(-2 * math.log(0.6))
    mu, sd = gp_posterior(np.array([[d]]), X, y, ls, noise=0.0)
    assert mu[0] == pytest.approx(1.2, abs=1e-6)
    assert sd[0] == pytest.approx(0.8, abs=1e-6)


def test_posterior_interpolates_noiseless_data():
    X = np.array([[0.2], [0.5], [0.8]])
    y = np.array([1.0, -0.5, 0.3])
    mu, sd = gp_posterior(X, X, y, lengthscale=0.2, noise=1e-10)
    assert np.allclose(mu, y, atol=1e-4)          # mean hits the data
    assert np.all(sd < 1e-3)                       # ...with ~zero uncertainty


def test_posterior_reverts_to_prior_far_away():
    X = np.array([[0.5]]); y = np.array([3.0])
    mu, sd = gp_posterior(np.array([[50.0]]), X, y, lengthscale=0.1)
    assert abs(mu[0]) < 1e-6                       # prior mean 0
    assert sd[0] == pytest.approx(1.0, abs=1e-3)   # prior sd 1


# ------------------------------------------------------------ acquisition ---
def test_ei_at_z_zero_is_04_sigma():
    # mu == best  ->  EI = sd * phi(0) = sd * 0.39894...
    v = expected_improvement(np.array([1.0]), np.array([0.5]), best=1.0)
    assert v[0] == pytest.approx(0.5 / math.sqrt(2 * math.pi), abs=1e-9)


def test_ei_at_z_one():
    # z = 1: EI = sd * (Phi(1) + phi(1)) = sd * 1.08332...
    v = expected_improvement(np.array([2.0]), np.array([1.0]), best=1.0)
    expected = 1.0 * (0.8413447460685429 + 0.24197072451914337)
    assert v[0] == pytest.approx(expected, abs=1e-6)


def test_ei_never_negative_and_grows_with_sd():
    mu = np.array([-1.0, -1.0]); sd = np.array([0.1, 1.0])
    v = expected_improvement(mu, sd, best=0.0)
    assert np.all(v >= 0)
    assert v[1] > v[0]                             # more uncertainty, more EI


def test_ucb_formula():
    assert ucb(np.array([1.0]), np.array([0.5]), beta=4.0)[0] == pytest.approx(2.0)


# ------------------------------------------------------------------ loop ---
def test_propose_next_picks_max_ei_candidate():
    X = np.array([[0.2], [0.8]]); y = np.array([0.0, 1.0])
    C = np.linspace(0, 1, 101)[:, None]
    x = propose_next(C, X, y, lengthscale=0.15)
    mu, sd = gp_posterior(C, X, y, 0.15)
    e = expected_improvement(mu, sd, y.max())
    assert x[0] == pytest.approx(C[np.argmax(e), 0])


def test_run_bo_finds_the_peak():
    # f has a single peak at x = 0.62; 2 inits + 10 BO steps must land within 0.05
    f = lambda X: 1.0 - 4.0 * (X[:, 0] - 0.62) ** 2
    rng = np.random.default_rng(0)
    X, y = run_bo(f, X_init=[[0.1], [0.9]], n_steps=10, lengthscale=0.15, rng=rng)
    assert len(y) == 12
    assert abs(X[np.argmax(y), 0] - 0.62) < 0.05


def test_run_bo_beats_random_at_equal_budget():
    # same 12-evaluation budget on a wiggly function, 5 seeds: BO wins on average
    f = lambda X: np.sin(5.3 * X[:, 0] + 1.1) + 0.6 * np.cos(11.0 * X[:, 0])
    gaps = []
    for s in range(5):
        rng = np.random.default_rng(s)
        _, y = run_bo(f, X_init=[[0.25], [0.75]], n_steps=10, lengthscale=0.12, rng=rng)
        y_rand = f(np.random.default_rng(100 + s).uniform(0, 1, (12, 1)))
        gaps.append(y.max() - y_rand.max())
    assert np.mean(gaps) > 0
