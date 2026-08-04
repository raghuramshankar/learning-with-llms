"""A language model, from scratch, in numpy.

Your job: replace every `raise NotImplementedError` with a working
implementation until `python3 -m pytest -q` is green. Everything you need is
derived in the explainer's maths sections:
docs/2026-08-04-language-modelling.html

The order in the README is easiest -> hardest. Hints live in each docstring.
Stuck on ONE function? Peek at solutions/lm.py — one function, not the file.
"""
from collections import Counter

import numpy as np

# ---------------------------------------------------------------------------
# Part 1 — the objective (Maths I)
# ---------------------------------------------------------------------------


def softmax(logits, axis=-1):
    """Numerically stable softmax.

    Subtract the max along `axis` BEFORE exponentiating: exp(1000) overflows to
    inf, but softmax is invariant to a constant shift per row, so
    softmax(z) == softmax(z - max z). Getting this wrong is the single most
    common source of NaNs in a hand-written LM.
    """
    raise NotImplementedError


def cross_entropy(logits, targets):
    """Mean token-level cross entropy in NATS.

    logits  (n, V) unnormalized scores
    targets (n,)   integer class indices

        H = -(1/n) * sum_i log softmax(logits_i)[targets_i]

    Compute it via log-sum-exp, not log(softmax(...)) — same stability issue.
    Sanity check you can do in your head: uniform logits over V classes must
    give exactly log(V).
    """
    raise NotImplementedError


def perplexity(logits, targets):
    """exp(cross_entropy). The 'effective number of equally likely choices'.

    A model that is perfectly uniform over V tokens has perplexity exactly V.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 2 — the transformer block (Maths II)
# ---------------------------------------------------------------------------


def rmsnorm(x, weight, eps=1e-6):
    """Root-mean-square layer norm over the LAST axis.

        rmsnorm(x) = x / sqrt(mean(x^2) + eps) * weight

    Note what is missing versus LayerNorm: no mean subtraction and no bias.
    That is the whole point — it is cheaper and works just as well.
    """
    raise NotImplementedError


def swiglu(x, W1, W2, W3):
    """SwiGLU feed-forward network.

        SwiGLU(x) = ( silu(x @ W1) * (x @ W3) ) @ W2

    where silu(z) = z * sigmoid(z). The elementwise product of a gated branch
    and a linear branch is the "GLU" part; using silu as the gate is the "Swi".
    Three matrices instead of two is why the hidden dim shrinks to ~8/3 d.
    """
    raise NotImplementedError


def rope(x, positions, base=10000.0):
    """Rotary position embedding.

    x         (..., seq, dim) with dim EVEN
    positions (seq,) integer positions

    Treat consecutive pairs (x[..., 0::2], x[..., 1::2]) as 2-D vectors and
    rotate pair i at position p by angle p * theta_i, where

        theta_i = base ** (-2i / dim),  i = 0 .. dim/2 - 1

    Two properties your tests will check, and both are the reason RoPE works:
      * rotation is orthogonal, so ||rope(x)|| == ||x||
      * <rope(q, m), rope(k, n)> depends only on (m - n), never on m and n
        separately — relative position falls out of absolute rotation.
    """
    raise NotImplementedError


def attention(Q, K, V, causal=True):
    """Scaled dot-product attention.

        A = softmax( Q K^T / sqrt(d_k)  + mask )  ;  out = A V

    Q, K, V are (seq, d_k). With causal=True, position i may attend only to
    j <= i: add -inf to the upper triangle BEFORE the softmax (masking after
    the softmax breaks normalization).

    The 1/sqrt(d_k) is not cosmetic — see the derivation in Maths II. Without
    it the dot products have variance d_k, and softmax saturates as d_k grows.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 3 — training (Maths III)
# ---------------------------------------------------------------------------


def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    """One AdamW update. Returns (p_new, m_new, v_new).

        m = b1*m + (1-b1)*g
        v = b2*v + (1-b2)*g^2
        m_hat = m / (1 - b1^t)          # t is 1-based
        v_hat = v / (1 - b2^t)
        p = p - lr * m_hat / (sqrt(v_hat) + eps)
        p = p - lr * wd * p             # DECOUPLED: not added to the gradient

    That last line is the entire difference between Adam and AdamW, and it is
    why AdamW is what everyone actually trains with.

    Worth predicting before you test it: on the very first step with wd=0, what
    is the update magnitude for a scalar parameter? It does not depend on |g|.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 4 — tokenization (Maths I)
# ---------------------------------------------------------------------------


def train_bpe(data: bytes, n_merges: int):
    """Byte-level BPE. Returns a list of ((a, b), new_id) in merge order.

    Start from the 256 raw byte values. Repeatedly find the most frequent
    adjacent pair and merge it into a new id (256, 257, ...). Ties: pick the
    pair `Counter.most_common` returns first, so results are deterministic.
    Stop early if the best pair occurs fewer than twice.
    """
    raise NotImplementedError


def bpe_encode(data: bytes, merges):
    """Apply `merges` (in order) to `data`, returning a list of token ids."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Part 5 — sampling (Maths II) and resource accounting (Maths III)
# ---------------------------------------------------------------------------


def sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None):
    """Sample one token id from `logits` (V,).

    Order matters: temperature, then top-k, then top-p, then renormalize.
      temperature: divide the logits (t -> 0 approaches argmax)
      top_k:       keep only the k highest-scoring tokens
      top_p:       keep the smallest set whose probability mass >= top_p,
                   INCLUDING the token that crosses the threshold
    Use rng.random() so the tests are reproducible.
    """
    raise NotImplementedError


def transformer_flops(n_layers, d_model, seq_len, d_ff=None, ffn_matrices=2):
    """Forward FLOPs per token for a decoder-only stack (mult+add = 2 FLOPs).

    Per layer:  qkv 6*d^2 | out proj 2*d^2 | scores 2*L*d | attn@V 2*L*d
                ffn ffn_matrices * 2 * d * d_ff        (d_ff defaults to 4*d)

    Return the total across all layers. This is the accounting behind the
    C ~= 6ND rule of thumb, which you derive in Maths III.
    """
    raise NotImplementedError


def chinchilla_optimal(C, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    """Compute-optimal (N, D) for budget C under L(N,D) = E + A/N^a + B/D^b.

    Minimize L subject to C = 6*N*D. A log-spaced search over N from 1e7 to
    1e13 with ~20000 points is accurate enough for the tests.
    Return (N, D, loss).
    """
    raise NotImplementedError
