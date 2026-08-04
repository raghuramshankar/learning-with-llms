"""Reference solutions for the language-modelling tutorial."""
from collections import Counter

import numpy as np


def softmax(logits, axis=-1):
    z = logits - np.max(logits, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=axis, keepdims=True)


def _logsumexp(x, axis=-1):
    m = np.max(x, axis=axis, keepdims=True)
    return (m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))).squeeze(axis)


def cross_entropy(logits, targets):
    logits = np.asarray(logits, dtype=float)
    targets = np.asarray(targets, dtype=int)
    lse = _logsumexp(logits, axis=-1)
    picked = logits[np.arange(len(targets)), targets]
    return float(np.mean(lse - picked))


def perplexity(logits, targets):
    return float(np.exp(cross_entropy(logits, targets)))


def rmsnorm(x, weight, eps=1e-6):
    x = np.asarray(x, dtype=float)
    rms = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)
    return x / rms * weight


def _silu(z):
    return z / (1.0 + np.exp(-z))


def swiglu(x, W1, W2, W3):
    return (_silu(x @ W1) * (x @ W3)) @ W2


def rope(x, positions, base=10000.0):
    x = np.asarray(x, dtype=float)
    dim = x.shape[-1]
    if dim % 2:
        raise ValueError("rope needs an even head dimension")
    i = np.arange(dim // 2)
    theta = base ** (-2.0 * i / dim)                 # (dim/2,)
    ang = np.asarray(positions, dtype=float)[:, None] * theta[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xe * sin + xo * cos
    return out


def attention(Q, K, V, causal=True):
    Q, K, V = (np.asarray(a, dtype=float) for a in (Q, K, V))
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    if causal:
        n = scores.shape[0]
        mask = np.triu(np.ones((n, scores.shape[1]), dtype=bool), k=1)
        scores = np.where(mask, -np.inf, scores)
    return softmax(scores, axis=-1) @ V


def adamw_step(p, g, m, v, t, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    p = p - lr * mhat / (np.sqrt(vhat) + eps)
    p = p - lr * wd * p                       # decoupled
    return p, m, v


def train_bpe(data: bytes, n_merges: int):
    ids = list(data)
    merges = []
    for i in range(n_merges):
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs:
            break
        top, count = pairs.most_common(1)[0]
        if count < 2:
            break
        new_id = 256 + i
        merges.append((top, new_id))
        out, j = [], 0
        while j < len(ids):
            if j < len(ids) - 1 and (ids[j], ids[j + 1]) == top:
                out.append(new_id); j += 2
            else:
                out.append(ids[j]); j += 1
        ids = out
    return merges


def bpe_encode(data: bytes, merges):
    ids = list(data)
    for pair, new_id in merges:
        out, j = [], 0
        while j < len(ids):
            if j < len(ids) - 1 and (ids[j], ids[j + 1]) == pair:
                out.append(new_id); j += 2
            else:
                out.append(ids[j]); j += 1
        ids = out
    return ids


def sample_next(logits, rng, temperature=1.0, top_k=None, top_p=None):
    logits = np.asarray(logits, dtype=float).copy()
    if temperature is not None and temperature > 0:
        logits = logits / temperature
    else:                                     # t -> 0 is greedy
        return int(np.argmax(logits))
    if top_k is not None and top_k < len(logits):
        cut = np.partition(logits, -top_k)[-top_k]
        logits[logits < cut] = -np.inf
    probs = softmax(logits)
    if top_p is not None and 0 < top_p < 1.0:
        order = np.argsort(-probs)
        csum = np.cumsum(probs[order])
        k = int(np.searchsorted(csum, top_p) + 1)     # include the crosser
        keep = order[:k]
        mask = np.zeros_like(probs, dtype=bool)
        mask[keep] = True
        probs = np.where(mask, probs, 0.0)
        probs = probs / probs.sum()
    u = rng.random()
    return int(np.searchsorted(np.cumsum(probs), u, side="right"))


def transformer_flops(n_layers, d_model, seq_len, d_ff=None, ffn_matrices=2):
    d = d_model
    d_ff = d_ff if d_ff else 4 * d
    per_layer = (6 * d * d + 2 * d * d          # qkv + output projection
                 + 4 * seq_len * d              # scores + attn @ V
                 + ffn_matrices * 2 * d * d_ff)
    return n_layers * per_layer


def chinchilla_optimal(C, E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28):
    N = np.logspace(7, 13, 20000)
    D = C / (6.0 * N)
    L = E + A / N ** alpha + B / D ** beta
    i = int(np.argmin(L))
    return float(N[i]), float(D[i]), float(L[i])
