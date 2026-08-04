#!/usr/bin/env python3
"""Plotly figures for the language modelling explainer.

Everything here is computed, not asserted:
  * fig_bpe        trains a real byte-level BPE on a real corpus and measures
                   the compression curve
  * fig_scaling    numerically minimizes the Chinchilla parametric loss under
                   an iso-FLOP constraint to recover the compute-optimal frontier
  * fig_flops      exact per-token FLOP accounting for a decoder-only block,
                   showing where attention overtakes the feed-forward network
  * fig_lr         the actual learning-rate schedules (cosine / WSD / linear)
  * fig_kv         KV-cache bytes vs context length for MHA / GQA / MLA
"""
import math
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import style, save_div, BLUE, GREEN, RED, GREY, FONT

OUT = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)
REPO = Path(__file__).resolve().parents[2]


def rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"


# ----------------------------------------------------------------------------
# 1. A real byte-level BPE trainer, and the compression curve it produces.
# ----------------------------------------------------------------------------
def train_bpe(data: bytes, n_merges: int):
    """Textbook byte-level BPE. Returns (merges, compression_curve).

    compression_curve[i] = bytes-per-token after i merges, measured by actually
    re-encoding the corpus, not estimated.
    """
    ids = list(data)
    n_bytes = len(ids)
    curve = [(0, n_bytes / len(ids))]
    merges = []
    for i in range(n_merges):
        pairs = Counter(zip(ids, ids[1:]))
        if not pairs:
            break
        top, count = pairs.most_common(1)[0]
        if count < 2:
            break
        new_id = 256 + i
        merges.append((top, new_id, count))
        # apply the merge
        out, j = [], 0
        while j < len(ids):
            if j < len(ids) - 1 and (ids[j], ids[j + 1]) == top:
                out.append(new_id)
                j += 2
            else:
                out.append(ids[j])
                j += 1
        ids = out
        curve.append((i + 1, n_bytes / len(ids)))
    return merges, curve


def fig_bpe():
    # Corpus: this repo's own prose (README + the skill that generated the page).
    # Committed files, so the figure is reproducible by anyone who clones.
    parts = []
    for p in [REPO / "README.md",
              REPO / "skills" / "learning-new-topic" / "SKILL.md"]:
        if p.exists():
            parts.append(p.read_text())
    corpus = ("\n".join(parts) * 3).encode("utf-8")
    merges, curve = train_bpe(corpus, 600)

    xs = [c[0] for c in curve]
    ys = [c[1] for c in curve]
    vocab = [256 + m for m in xs]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=vocab, y=ys, mode="lines", line=dict(color=BLUE, width=2.4),
        name="measured", hovertemplate="vocab %{x}<br>%{y:.3f} bytes/token<extra></extra>"))
    # Annotate the first few merges — they are always the obvious English digraphs.
    top3 = merges[:3]
    labels = []
    for (pair, _nid, cnt) in top3:
        try:
            s = bytes(pair).decode("utf-8")
        except Exception:
            s = str(pair)
        labels.append(f"{s!r} ({cnt}x)")
    fig.add_annotation(x=vocab[3], y=ys[3], text="first merges: " + ", ".join(labels),
                       showarrow=True, arrowhead=0, ax=90, ay=-40,
                       font=dict(color=FONT, size=11), arrowcolor=GREY)
    fig.add_hline(y=1.0, line=dict(color=GREY, dash="dot"),
                  annotation_text="1 byte/token (no compression)",
                  annotation_font=dict(color=FONT, size=11))
    style(fig, height=380)
    fig.update_xaxes(title="vocabulary size (256 bytes + merges)")
    fig.update_yaxes(title="bytes per token")
    save_div(fig, OUT / "fig_bpe.html", "fig-bpe")
    return merges, curve


# ----------------------------------------------------------------------------
# 2. Chinchilla: minimize the parametric loss under an iso-FLOP constraint.
# ----------------------------------------------------------------------------
# Hoffmann et al. (2022), Approach 3, exactly as published.
CHIN = dict(E=1.69, A=406.4, B=410.7, alpha=0.34, beta=0.28)
# Besiroglu et al. replication re-estimate. Included because the published
# Approach-3 constants do NOT reproduce Chinchilla's own 70B/1.4T
# recommendation when you actually minimize them — the replication's do.
CHIN_R = dict(E=1.82, A=482.0, B=2085.0, alpha=0.35, beta=0.37)
# The model Hoffmann et al. actually trained, for reference.
CHIN_N, CHIN_D = 70e9, 1.4e12


def chinchilla_loss(N, D, p=CHIN):
    return p["E"] + p["A"] / np.power(N, p["alpha"]) + p["B"] / np.power(D, p["beta"])


def optimal_split(C, p=CHIN, grid=4000):
    """argmin_N L(N, D) s.t. 6ND = C. Solved numerically on a log grid."""
    N = np.logspace(7, 12, grid)
    D = C / (6.0 * N)
    L = chinchilla_loss(N, D, p)
    i = int(np.argmin(L))
    return N[i], D[i], L[i]


def fig_scaling():
    budgets = np.logspace(18, 25, 60)
    Ns, Ds, Ls = [], [], []
    for C in budgets:
        n, d, l = optimal_split(C)
        Ns.append(n); Ds.append(d); Ls.append(l)
    Ns, Ds = np.array(Ns), np.array(Ds)

    fig = go.Figure()
    # iso-FLOP loss curves: for three budgets, sweep N and show the U shape
    for C, col in [(1e21, GREY), (1e23, GREEN), (1e25, BLUE)]:
        N = np.logspace(7.5, 11.5, 300)
        D = C / (6.0 * N)
        L = chinchilla_loss(N, D)
        keep = (D > 1e8)
        fig.add_trace(go.Scatter(
            x=N[keep], y=L[keep], mode="lines",
            line=dict(color=col, width=2.2),
            name=f"C = {C:.0e} FLOPs",
            hovertemplate="N=%{x:.3s}<br>loss %{y:.3f}<extra></extra>"))
        n, d, l = optimal_split(C)
        fig.add_trace(go.Scatter(
            x=[n], y=[l], mode="markers",
            marker=dict(color=col, size=11, symbol="circle",
                        line=dict(color="rgba(128,140,160,.8)", width=1)),
            showlegend=False,
            hovertemplate=("optimum<br>N=%.2e<br>D=%.2e<br>D/N=%.1f<extra></extra>"
                           % (n, d, d / n))))
    fig.add_hline(y=CHIN["E"], line=dict(color=GREY, dash="dot"),
                  annotation_text="E = 1.69 — irreducible entropy of the data",
                  annotation_font=dict(color=FONT, size=11))
    style(fig, height=400, legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(type="log", title="parameters N")
    fig.update_yaxes(title="loss L(N, D)")
    save_div(fig, OUT / "fig_scaling.html", "fig-scaling")

    # the tokens-per-parameter ratio along the frontier, under BOTH fits
    Nr, Dr = [], []
    for C in budgets:
        n, d, _ = optimal_split(C, CHIN_R)
        Nr.append(n); Dr.append(d)
    Nr, Dr = np.array(Nr), np.array(Dr)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=budgets, y=Ds / Ns, mode="lines", line=dict(color=RED, width=2.4),
        name="Hoffmann Approach 3, as published",
        hovertemplate="C=%{x:.1e}<br>D/N = %{y:.1f}<extra></extra>"))
    fig2.add_trace(go.Scatter(
        x=budgets, y=Dr / Nr, mode="lines", line=dict(color=GREEN, width=2.4),
        name="Besiroglu et al. re-estimate",
        hovertemplate="C=%{x:.1e}<br>D/N = %{y:.1f}<extra></extra>"))
    fig2.add_hline(y=20, line=dict(color=BLUE, dash="dash"),
                   annotation_text="the famous ≈20 tokens per parameter",
                   annotation_font=dict(color=FONT, size=11))
    fig2.add_trace(go.Scatter(
        x=[6 * CHIN_N * CHIN_D], y=[CHIN_D / CHIN_N], mode="markers+text",
        marker=dict(color=BLUE, size=13, symbol="star"),
        text=["Chinchilla itself (70B, 1.4T)"], textposition="bottom right",
        textfont=dict(color=FONT, size=11), showlegend=False,
        hovertemplate="the model they actually trained<br>D/N = 20<extra></extra>"))
    style(fig2, height=380, legend=dict(orientation="h", y=1.16, x=0))
    fig2.update_xaxes(type="log", title="training compute C (FLOPs)")
    fig2.update_yaxes(type="log", title="optimal tokens per parameter (D/N)")
    save_div(fig2, OUT / "fig_ratio.html", "fig-ratio")
    return Ns, Ds


# ----------------------------------------------------------------------------
# 3. Exact FLOP accounting: when does attention actually dominate?
# ----------------------------------------------------------------------------
def block_flops(d, L, d_ff=None, ffn_mats=2):
    """Forward FLOPs per token for one decoder block (mult+add = 2 FLOPs).

    qkv       3 * 2d^2
    scores    2 * L * d      (causal: L is the average context, see caption)
    attn@V    2 * L * d
    out proj  2d^2
    ffn       ffn_mats * 2 * d * d_ff
    """
    d_ff = d_ff if d_ff else 4 * d
    proj = 6 * d * d + 2 * d * d
    attn = 4 * L * d
    ffn = ffn_mats * 2 * d * d_ff
    return proj, attn, ffn


def fig_flops():
    Ls = np.logspace(2, 6, 200)
    fig = go.Figure()
    for d, col in [(1024, GREY), (4096, GREEN), (8192, BLUE)]:
        proj, attn, ffn = block_flops(d, Ls)
        total = proj + attn + ffn
        fig.add_trace(go.Scatter(
            x=Ls, y=attn / total * 100, mode="lines",
            line=dict(color=col, width=2.4), name=f"d_model = {d}",
            hovertemplate="context %{x:.0f}<br>attention = %{y:.1f}% of FLOPs<extra></extra>"))
        # crossover: attention == everything else
        cross = 6 * d  # 4Ld = 24d^2  ->  L = 6d   (with d_ff = 4d, 2 matrices)
        fig.add_vline(x=cross, line=dict(color=col, dash="dot"))
    fig.add_hline(y=50, line=dict(color=GREY, dash="dash"),
                  annotation_text="attention = half the compute",
                  annotation_font=dict(color=FONT, size=11))
    style(fig, height=390, legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(type="log", title="context length L (tokens)")
    fig.update_yaxes(title="attention share of block FLOPs (%)", range=[0, 100])
    save_div(fig, OUT / "fig_flops.html", "fig-flops")


# ----------------------------------------------------------------------------
# 4. Learning-rate schedules (the real functions).
# ----------------------------------------------------------------------------
def fig_lr():
    T = 1000
    t = np.arange(T)
    warm = 0.03 * T

    def cosine(t, peak=1.0, final=0.1):
        lr = np.where(t < warm, t / warm,
                      final + 0.5 * (1 - final) *
                      (1 + np.cos(np.pi * np.clip((t - warm) / (T - warm), 0, 1))))
        return peak * lr

    def wsd(t, peak=1.0, decay_frac=0.2):
        d0 = T * (1 - decay_frac)
        lr = np.where(t < warm, t / warm,
                      np.where(t < d0, 1.0,
                               np.clip(1 - (t - d0) / (T - d0), 0, 1) ** 1.0))
        return peak * lr

    def linear(t, peak=1.0):
        return peak * np.where(t < warm, t / warm,
                               np.clip(1 - (t - warm) / (T - warm), 0, 1))

    fig = go.Figure()
    for fn, name, col in [(cosine, "cosine (GPT-3, Llama)", BLUE),
                          (wsd, "WSD / warmup-stable-decay", GREEN),
                          (linear, "linear decay", GREY)]:
        fig.add_trace(go.Scatter(x=t, y=fn(t), mode="lines",
                                 line=dict(color=col, width=2.4), name=name))
    fig.add_vrect(x0=0, x1=warm, fillcolor=rgba(RED, 0.10), line_width=0,
                  annotation_text="warmup", annotation_position="top left",
                  annotation_font=dict(color=FONT, size=11))
    style(fig, height=340, legend=dict(orientation="h", y=1.14, x=0))
    fig.update_xaxes(title="training step")
    fig.update_yaxes(title="learning rate (fraction of peak)")
    save_div(fig, OUT / "fig_lr.html", "fig-lr")


# ----------------------------------------------------------------------------
# 5. KV cache: the real memory formula for MHA / GQA / MLA.
# ----------------------------------------------------------------------------
def fig_kv():
    """bytes = 2 (K and V) * layers * kv_heads * head_dim * seq * dtype_bytes"""
    layers, heads, head_dim, dbytes = 80, 64, 128, 2   # ~70B-class, bf16
    Ls = np.logspace(2, 6, 200)
    d = heads * head_dim

    variants = [
        ("MHA (64 kv heads)", 2 * layers * heads * head_dim * dbytes, BLUE),
        ("GQA (8 kv groups)", 2 * layers * 8 * head_dim * dbytes, GREEN),
        ("MLA (latent 512)", layers * 512 * dbytes, RED),
    ]
    fig = go.Figure()
    for name, per_tok, col in variants:
        gb = per_tok * Ls / 1e9
        fig.add_trace(go.Scatter(
            x=Ls, y=gb, mode="lines", line=dict(color=col, width=2.4), name=name,
            hovertemplate=name + "<br>%{x:.0f} tokens → %{y:.2f} GB<extra></extra>"))
    fig.add_hline(y=80, line=dict(color=GREY, dash="dash"),
                  annotation_text="one 80 GB accelerator",
                  annotation_font=dict(color=FONT, size=11))
    style(fig, height=380, legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(type="log", title="context length (tokens)")
    fig.update_yaxes(type="log", title="KV cache (GB, one sequence)")
    save_div(fig, OUT / "fig_kv.html", "fig-kv")


if __name__ == "__main__":
    merges, curve = fig_bpe()
    print(f"  BPE: {len(merges)} merges, {curve[0][1]:.3f} -> {curve[-1][1]:.3f} bytes/token")
    Ns, Ds = fig_scaling()
    Cc = 6 * CHIN_N * CHIN_D
    for label, p in (("published ", CHIN), ("replicated", CHIN_R)):
        n, d, l = optimal_split(Cc, p)
        print(f"  {label} @ Chinchilla's own C={Cc:.2e}: "
              f"N={n/1e9:6.1f}B D={d/1e12:5.2f}T D/N={d/n:6.1f}")
    print(f"  {'actual    '} @ Chinchilla's own C={Cc:.2e}: "
          f"N={CHIN_N/1e9:6.1f}B D={CHIN_D/1e12:5.2f}T D/N={CHIN_D/CHIN_N:6.1f}")
    fig_flops()
    for d in (1024, 4096, 8192):
        print(f"  d={d}: attention overtakes the rest at L={6*d}")
    fig_lr()
    fig_kv()
