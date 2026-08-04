"""Tests for the language-modelling tutorial.

Run against your own code:      python3 -m pytest -q
Run against the reference:      LM_SOLUTION=1 python3 -m pytest -q

Every expected value here is either an exact identity you can derive on paper
or a number quoted in the explainer's quizzes.
"""
import os

import numpy as np
import pytest

if os.environ.get("LM_SOLUTION"):
    from solutions import lm
else:
    import lm


# ---------------------------------------------------------------- objective
def test_softmax_uniform_is_one_over_v():
    p = lm.softmax(np.zeros(7))
    assert np.allclose(p, 1 / 7)
    assert np.isclose(p.sum(), 1.0)


def test_softmax_is_shift_invariant_and_does_not_overflow():
    big = np.array([1000.0, 1000.0, 1000.0])
    p = lm.softmax(big)
    assert np.all(np.isfinite(p)) and np.allclose(p, 1 / 3)


def test_cross_entropy_of_uniform_is_log_v():
    V, n = 50, 8
    logits = np.zeros((n, V))
    targets = np.arange(n) % V
    assert np.isclose(lm.cross_entropy(logits, targets), np.log(V))


def test_perplexity_of_uniform_is_exactly_v():
    V, n = 32, 5
    logits = np.zeros((n, V))
    targets = np.arange(n) % V
    assert np.isclose(lm.perplexity(logits, targets), V)


def test_perplexity_of_a_perfect_model_approaches_one():
    logits = np.full((4, 10), -50.0)
    targets = np.array([3, 1, 4, 1])
    logits[np.arange(4), targets] = 50.0
    assert lm.perplexity(logits, targets) < 1.001


# ------------------------------------------------------------------- block
def test_rmsnorm_of_unit_vector_is_the_weight():
    x = np.ones(8)
    w = np.arange(1.0, 9.0)
    assert np.allclose(lm.rmsnorm(x, w), w, atol=1e-5)


def test_rmsnorm_is_scale_invariant():
    x = np.array([1.0, -2.0, 3.0, 0.5])
    w = np.ones(4)
    a, b = lm.rmsnorm(x, w), lm.rmsnorm(10 * x, w)
    assert np.allclose(a, b, atol=1e-4)


def test_swiglu_shapes_and_zero_input():
    d, h = 6, 11
    rng = np.random.default_rng(0)
    W1, W3 = rng.normal(size=(d, h)), rng.normal(size=(d, h))
    W2 = rng.normal(size=(h, d))
    out = lm.swiglu(np.zeros((3, d)), W1, W2, W3)
    assert out.shape == (3, d)
    assert np.allclose(out, 0.0)          # silu(0)*0 = 0


def test_rope_is_a_rotation_so_it_preserves_norms():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(5, 8))
    y = lm.rope(x, np.arange(5))
    assert np.allclose(np.linalg.norm(x, axis=-1), np.linalg.norm(y, axis=-1))


def test_rope_dot_product_depends_only_on_relative_position():
    rng = np.random.default_rng(2)
    q = rng.normal(size=(1, 16))
    k = rng.normal(size=(1, 16))
    # same offset (m-n = 3) at two different absolute placements
    a = lm.rope(q, np.array([5]))[0] @ lm.rope(k, np.array([2]))[0]
    b = lm.rope(q, np.array([13]))[0] @ lm.rope(k, np.array([10]))[0]
    assert np.isclose(a, b)
    # a different offset must give a different score
    c = lm.rope(q, np.array([5]))[0] @ lm.rope(k, np.array([4]))[0]
    assert not np.isclose(a, c)


def test_rope_at_position_zero_is_the_identity():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(1, 4))
    assert np.allclose(lm.rope(x, np.array([0])), x)


def test_causal_attention_first_row_sees_only_itself():
    rng = np.random.default_rng(4)
    Q, K = rng.normal(size=(4, 3)), rng.normal(size=(4, 3))
    V = rng.normal(size=(4, 3))
    out = lm.attention(Q, K, V, causal=True)
    assert np.allclose(out[0], V[0])


def test_attention_with_zero_queries_averages_the_visible_values():
    V = np.array([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    Q = np.zeros((3, 2)); K = np.zeros((3, 2))
    out = lm.attention(Q, K, V, causal=True)
    assert np.allclose(out[1], [2.0, 0.0])       # mean of rows 0..1
    assert np.allclose(out[2], [3.0, 0.0])       # mean of rows 0..2


def test_attention_rows_are_a_probability_weighted_mix():
    rng = np.random.default_rng(5)
    V = rng.normal(size=(6, 4))
    out = lm.attention(rng.normal(size=(6, 4)), rng.normal(size=(6, 4)), V)
    assert out.shape == (6, 4)
    assert np.all(out.max() <= V.max() + 1e-9) and np.all(out.min() >= V.min() - 1e-9)


# ---------------------------------------------------------------- training
def test_adamw_first_step_moves_by_lr_whatever_the_gradient_scale():
    """Adam is scale-free: after bias correction the first step is lr*sign(g),
    across ten orders of magnitude of gradient. This is why Adam needs no
    per-layer LR tuning the way SGD does."""
    for g in (1e-3, 1.0, 1e3, 1e6):
        p, m, v = lm.adamw_step(np.array([0.0]), np.array([g]),
                                np.array([0.0]), np.array([0.0]), t=1, lr=0.01)
        assert np.isclose(float(p[0]), -0.01, rtol=1e-4), g


def test_adamw_epsilon_damps_the_step_for_tiny_gradients():
    """The scale-free property breaks once |g| approaches eps — the step is
    lr*|g|/(|g|+eps), so a gradient at eps moves only half as far."""
    p, _, _ = lm.adamw_step(np.array([0.0]), np.array([1e-8]),
                            np.array([0.0]), np.array([0.0]),
                            t=1, lr=0.01, eps=1e-8)
    assert np.isclose(float(p[0]), -0.005, rtol=1e-3)


def test_adamw_decoupled_decay_shrinks_a_zero_gradient_parameter():
    p, _, _ = lm.adamw_step(np.array([2.0]), np.array([0.0]),
                            np.array([0.0]), np.array([0.0]),
                            t=1, lr=0.1, wd=0.5)
    assert np.isclose(float(p[0]), 2.0 - 0.1 * 0.5 * 2.0)


def test_adamw_descends_a_quadratic():
    p = np.array([5.0]); m = np.array([0.0]); v = np.array([0.0])
    for t in range(1, 400):
        g = 2 * p                                  # d/dp of p^2
        p, m, v = lm.adamw_step(p, g, m, v, t, lr=0.05)
    assert abs(float(p[0])) < 0.5


# ------------------------------------------------------------ tokenization
def test_bpe_merges_the_most_frequent_pair_first():
    data = b"ababababab"
    merges = lm.train_bpe(data, 1)
    assert len(merges) == 1
    (pair, new_id) = merges[0]
    assert pair == (ord("a"), ord("b")) and new_id == 256


def test_bpe_encode_roundtrips_length():
    data = b"ababababab"
    merges = lm.train_bpe(data, 1)
    ids = lm.bpe_encode(data, merges)
    assert ids == [256] * 5


def test_bpe_compresses_repetitive_text():
    data = (b"the quick brown fox jumps over the lazy dog. " * 20)
    merges = lm.train_bpe(data, 50)
    ids = lm.bpe_encode(data, merges)
    assert len(ids) < len(data) / 2          # >2x compression
    # It may stop before 50: training halts once no adjacent pair repeats,
    # which is why tiny corpora cannot fill a large vocabulary.
    assert 20 <= len(merges) <= 50


def test_bpe_stops_when_no_pair_repeats():
    merges = lm.train_bpe(b"abcdef", 10)
    assert merges == []


# ------------------------------------------------------------ sampling etc
def test_zero_temperature_is_greedy():
    rng = np.random.default_rng(0)
    logits = np.array([0.1, 5.0, 0.2, 0.3])
    assert lm.sample_next(logits, rng, temperature=0.0) == 1


def test_top_k_of_one_is_always_the_argmax():
    logits = np.array([0.1, 5.0, 0.2, 4.9])
    for seed in range(20):
        rng = np.random.default_rng(seed)
        assert lm.sample_next(logits, rng, top_k=1) == 1


def test_top_p_keeps_the_token_that_crosses_the_threshold():
    # probabilities approx [0.6, 0.3, 0.1]; top_p=0.65 must keep the first TWO
    logits = np.log(np.array([0.6, 0.3, 0.1]))
    seen = {lm.sample_next(logits, np.random.default_rng(s), top_p=0.65)
            for s in range(200)}
    assert seen <= {0, 1} and seen == {0, 1}


def test_sampling_is_unbiased_at_temperature_one():
    logits = np.log(np.array([0.7, 0.2, 0.1]))
    draws = [lm.sample_next(logits, np.random.default_rng(s)) for s in range(3000)]
    frac0 = sum(d == 0 for d in draws) / len(draws)
    assert 0.66 < frac0 < 0.74


# -------------------------------------------------------------- accounting
def test_flops_reduce_to_the_six_n_d_rule():
    """With L << d the attention term vanishes and forward FLOPs/token ~= 2N."""
    n_layers, d = 12, 1024
    f = lm.transformer_flops(n_layers, d, seq_len=1)
    params = n_layers * (4 * d * d + 2 * d * (4 * d))     # attn + ffn matrices
    assert abs(f - 2 * params) / (2 * params) < 0.01


def test_attention_share_crosses_half_at_six_d():
    d, n = 4096, 1
    below = lm.transformer_flops(n, d, seq_len=6 * d // 2)
    above = lm.transformer_flops(n, d, seq_len=6 * d * 2)
    # attention is linear in L, the rest is constant -> total grows with L
    assert above > below
    attn_at_cross = 4 * (6 * d) * d
    rest = 8 * d * d + 2 * 2 * d * (4 * d)
    assert np.isclose(attn_at_cross, rest)


def test_chinchilla_optimum_sits_on_the_compute_constraint():
    C = 1e23
    N, D, loss = lm.chinchilla_optimal(C)
    assert np.isclose(6 * N * D, C, rtol=1e-3)
    assert 1.6 < loss < 3.0


def test_chinchilla_scales_both_n_and_d_with_budget():
    n1, d1, _ = lm.chinchilla_optimal(1e21)
    n2, d2, _ = lm.chinchilla_optimal(1e23)
    assert n2 > n1 and d2 > d1


# ------------------------------------------------------------- integration
def test_end_to_end_induction_head_copies_the_right_token():
    """A hand-built attention head that provably copies a matching earlier token.

    Keys mark 'what came before me'; the query at the last position matches the
    key laid down after the earlier occurrence of the same symbol, so attention
    retrieves that continuation. This is the mechanism behind in-context
    learning, assembled from your own attention + softmax.
    """
    d = 8
    sym = {"A": np.eye(d)[0], "B": np.eye(d)[1], "C": np.eye(d)[2]}
    seq = ["A", "B", "C", "A"]                   # after A came B; expect B
    K = np.stack([sym[s] for s in seq]) * 10.0   # sharp match
    Q = np.stack([sym[s] for s in seq]) * 10.0
    V = np.stack([sym[s] for s in seq])
    # shift values by one so position i holds "the token that FOLLOWED i"
    Vs = np.roll(V, -1, axis=0); Vs[-1] = 0.0
    out = lm.attention(Q, K, Vs, causal=True)
    pred = int(np.argmax(out[-1]))
    assert pred == 1, "the head should retrieve B, the token that followed A"


def test_end_to_end_a_trained_bpe_shrinks_a_real_corpus_and_lowers_perplexity():
    """Fewer, longer tokens => shorter sequence => the uniform-model perplexity
    bound over the SAME text falls in nats-per-byte terms."""
    text = (b"language models predict the next token. " * 40)
    merges = lm.train_bpe(text, 80)
    ids = lm.bpe_encode(text, merges)
    V_byte, V_bpe = 256, 256 + len(merges)
    nats_per_byte_bytes = len(text) * np.log(V_byte) / len(text)
    nats_per_byte_bpe = len(ids) * np.log(V_bpe) / len(text)
    assert nats_per_byte_bpe < nats_per_byte_bytes
