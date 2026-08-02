#!/usr/bin/env python3
"""Plotly figures for the Bayesian optimization explainer.

Runs a real GP-EI Bayesian optimizer in numpy (vs random and grid search) for
the sample-efficiency figure; everything is precomputed, nothing asserted.
"""
import math
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import style, save_div, BLUE, GREEN, RED, GREY, FONT

OUT = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)

PHI = np.vectorize(lambda z: 0.5 * (1.0 + math.erf(z / math.sqrt(2.0))))
phi = lambda z: np.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)

def rgba(hexc, a):
    h = hexc.lstrip("#")
    return f"rgba({int(h[0:2],16)},{int(h[2:4],16)},{int(h[4:6],16)},{a})"

def rbf(A, B, ls):
    d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / ls ** 2)

def gp_post(Xs, X, y, ls, noise=1e-4):
    K = rbf(X, X, ls) + noise * np.eye(len(X))
    Ks = rbf(Xs, X, ls)
    Ki = np.linalg.inv(K)
    mu = Ks @ Ki @ y
    var = np.clip(1.0 - np.einsum("ij,jk,ik->i", Ks, Ki, Ks), 1e-12, None)
    return mu, np.sqrt(var)

def ei(mu, sd, best):
    z = (mu - best) / sd
    return (mu - best) * PHI(z) + sd * phi(z)

# ---- objective: negated, rescaled Branin on [0,1]^2 (maximize) ----------
def f2(X):
    x = 15 * X[:, 0] - 5
    y = 15 * X[:, 1]
    b = (y - 5.1 / (4 * np.pi ** 2) * x ** 2 + 5 / np.pi * x - 6) ** 2 \
        + 10 * (1 - 1 / (8 * np.pi)) * np.cos(x) + 10
    return -b / 60.0  # max ≈ -0.0066 at the three Branin minima

# ---------------------------------------------------------- fig_regret ---
SEEDS, T, LS = 20, 30, 0.2
curves = {"BO (GP + EI)": [], "random search": [], "grid search": []}
gx = np.linspace(0.05, 0.95, 6)
grid_pts = np.array([[a, b] for a in gx for b in np.linspace(0.05, 0.95, 5)])[:T]
for s in range(SEEDS):
    rng = np.random.default_rng(s)
    # BO
    X = rng.uniform(0, 1, (3, 2)); y = f2(X)
    for _ in range(T - 3):
        C = rng.uniform(0, 1, (512, 2))
        mu, sd = gp_post(C, X, y, LS)
        nxt = C[np.argmax(ei(mu, sd, y.max()))]
        if np.min(((X - nxt) ** 2).sum(1)) < 1e-6:      # avoid duplicate points
            nxt = rng.uniform(0, 1, 2)
        X = np.vstack([X, nxt])
        y = f2(X)
    curves["BO (GP + EI)"].append(np.maximum.accumulate(y))
    # random
    yr = f2(rng.uniform(0, 1, (T, 2)))
    curves["random search"].append(np.maximum.accumulate(yr))
    # grid (fixed order, same for all seeds)
    curves["grid search"].append(np.maximum.accumulate(f2(grid_pts)))

fig = go.Figure()
COLORS = {"BO (GP + EI)": GREEN, "random search": BLUE, "grid search": RED}
for name, runs in curves.items():
    R = np.array(runs); m, sdv = R.mean(0), R.std(0)
    xs = np.arange(1, T + 1)
    c = COLORS[name]
    fig.add_trace(go.Scatter(x=np.r_[xs, xs[::-1]], y=np.r_[m + sdv, (m - sdv)[::-1]],
                             fill="toself", fillcolor=rgba(c, 0.13) if c.startswith("#") else c,
                             line=dict(width=0), hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=m, name=name, line=dict(color=c, width=2.5),
                             hovertemplate="eval %{x}: best = %{y:.3f}<extra>" + name + "</extra>"))
fig.update_xaxes(title="function evaluations")
fig.update_yaxes(title="best value found (higher is better)")
style(fig, height=360, legend=dict(x=0.55, y=0.15, bgcolor="rgba(0,0,0,0)"),
      title=dict(text="Same budget, different strategies — 20 repetitions, negated Branin",
                 font=dict(size=15)))
save_div(fig, OUT / "fig_regret.html", "fig-regret")

# ------------------------------------------------------ fig_ls_samples ---
xs = np.linspace(0, 1, 220)[:, None]
rngs = np.random.default_rng(3)
LSS = [0.03, 0.1, 0.3, 1.0]
frames = []
for ls in LSS:
    K = rbf(xs, xs, ls) + 1e-8 * np.eye(len(xs))
    L = np.linalg.cholesky(K)
    Z = np.random.default_rng(5).standard_normal((len(xs), 4))
    S = L @ Z
    frames.append(go.Frame(name=str(ls), data=[
        go.Scatter(x=xs[:, 0], y=S[:, i], mode="lines",
                   line=dict(width=2, color=[BLUE, GREEN, RED, GREY][i]),
                   hovertemplate="x=%{x:.2f} f=%{y:.2f}<extra></extra>")
        for i in range(4)]))
fig = go.Figure(data=frames[1].data, frames=frames)
fig.update_layout(sliders=[dict(active=1, pad=dict(t=8), font=dict(color=FONT),
    currentvalue=dict(prefix="lengthscale ℓ = "),
    steps=[dict(method="animate", label=str(ls),
                args=[[str(ls)], dict(mode="immediate", frame=dict(duration=200, redraw=True),
                                      transition=dict(duration=150))]) for ls in LSS])],
    showlegend=False,
    title=dict(text="Four draws from a GP prior — the lengthscale is a hypothesis about smoothness",
               font=dict(size=15)))
fig.update_xaxes(title="x")
fig.update_yaxes(title="f(x)", range=[-3, 3])
style(fig, height=380)
save_div(fig, OUT / "fig_ls_samples.html", "fig-ls-samples")

# ----------------------------------------------------- fig_acq_anatomy ---
X1 = np.array([[0.05], [0.22], [0.4], [0.62], [0.78], [0.95]])
f1 = lambda x: np.sin(6.2 * x[:, 0]) + 0.35 * np.cos(13 * x[:, 0])
y1 = f1(X1)
xs = np.linspace(0, 1, 400)[:, None]
mu, sd = gp_post(xs, X1, y1, 0.12)
best = y1.max()
ACQ = {"EI": ei(mu, sd, best),
       "PI": PHI((mu - best) / sd),
       "UCB (β=2)": mu + 2 * sd}
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.62, 0.38],
                    vertical_spacing=0.06)
fig.add_trace(go.Scatter(x=np.r_[xs[:, 0], xs[::-1, 0]], y=np.r_[mu + 2 * sd, (mu - 2 * sd)[::-1]],
                         fill="toself", fillcolor="rgba(91,141,239,0.15)", line=dict(width=0),
                         hoverinfo="skip", showlegend=False), row=1, col=1)
fig.add_trace(go.Scatter(x=xs[:, 0], y=mu, name="posterior mean μ(x)",
                         line=dict(color=BLUE, width=2.5),
                         hovertemplate="μ=%{y:.2f}<extra></extra>"), row=1, col=1)
fig.add_trace(go.Scatter(x=X1[:, 0], y=y1, mode="markers", name="observations",
                         marker=dict(color=GREEN, size=9),
                         hovertemplate="obs (%{x:.2f}, %{y:.2f})<extra></extra>"), row=1, col=1)
for i, (name, a) in enumerate(ACQ.items()):
    an = (a - a.min()) / (a.max() - a.min() + 1e-12)
    fig.add_trace(go.Scatter(x=xs[:, 0], y=an, name=name, visible=(i == 0),
                             line=dict(color=GREEN, width=2.5),
                             hovertemplate=name + " (scaled)=%{y:.2f}<extra></extra>"),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=[xs[np.argmax(a), 0]], y=[1.0], mode="markers",
                             marker=dict(color=RED, size=11, symbol="triangle-down"),
                             name="argmax", visible=(i == 0), showlegend=False,
                             hovertemplate="next evaluation<extra></extra>"), row=2, col=1)
buttons = []
for i, name in enumerate(ACQ):
    vis = [True, True, True] + [False] * (2 * len(ACQ))
    vis[3 + 2 * i] = vis[4 + 2 * i] = True
    buttons.append(dict(label=name, method="update", args=[{"visible": vis}]))
fig.update_layout(updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center",
                                    y=1.14, bgcolor="rgba(0,0,0,0)", font=dict(color=FONT),
                                    buttons=buttons)],
                  legend=dict(x=0.995, xanchor="right", y=0.99, bgcolor="rgba(0,0,0,0)"))
fig.update_yaxes(title="f(x)", row=1, col=1)
fig.update_yaxes(title="acquisition (scaled)", row=2, col=1)
fig.update_xaxes(title="x", row=2, col=1)
style(fig, height=520)
save_div(fig, OUT / "fig_acq_anatomy.html", "fig-acq-anatomy")

# --------------------------------------------------------- fig_turbo -----
# Simplified single-trust-region TuRBO on negated Branin: success/failure
# counters drive the region side length L (double on 3 successes, halve on
# 3 failures, restart below L_min keeping only the incumbent).
rng = np.random.default_rng(7)
T, LS_T = 42, 0.2
L, L_INIT, L_MAX, L_MIN = 0.4, 0.4, 0.8, 0.03
TAU_S, TAU_F = 3, 3
X = rng.uniform(0, 1, (4, 2)); y = f2(X)
succ = fail = 0
best_hist = list(np.maximum.accumulate(y))
L_hist = [L] * len(y)
restarts = []
while len(best_hist) < T:
    c = X[np.argmax(y)]
    C = np.clip(c + L * (rng.uniform(size=(256, 2)) - 0.5), 0, 1)
    mu, sd = gp_post(C, X, y, LS_T)
    nxt = C[np.argmax(ei(mu, sd, y.max()))]
    y_new = f2(nxt[None, :])[0]
    improved = y_new > y.max() + 1e-4
    X = np.vstack([X, nxt]); y = np.append(y, y_new)
    best_hist.append(max(best_hist[-1], y_new)); L_hist.append(L)
    if improved: succ += 1; fail = 0
    else: fail += 1; succ = 0
    if succ >= TAU_S: L = min(L_MAX, 2 * L); succ = 0
    if fail >= TAU_F: L = L / 2; fail = 0
    if L < L_MIN and len(best_hist) < T - 4:
        restarts.append(len(y)); L = L_INIT
        keep = int(np.argmax(y))
        Xr = rng.uniform(0, 1, (3, 2)); yr = f2(Xr)
        X = np.vstack([X[keep][None, :], Xr]); y = np.append(y[keep], yr)
        best_hist.extend([max(best_hist[-1], v) for v in yr])
        L_hist.extend([L] * 3)
evals = np.arange(1, len(best_hist) + 1)
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.55, 0.45],
                    vertical_spacing=0.08)
fig.add_trace(go.Scatter(x=evals, y=best_hist, name="best value found",
                         line=dict(color=GREEN, width=2.5),
                         hovertemplate="eval %{x}: best = %{y:.3f}<extra></extra>"),
              row=1, col=1)
fig.add_trace(go.Scatter(x=evals, y=L_hist, name="trust-region side L",
                         line=dict(color=RED, width=2.5, shape="hv"),
                         hovertemplate="eval %{x}: L = %{y:.3f}<extra></extra>"),
              row=2, col=1)
for rx in restarts:
    fig.add_vline(x=rx, line=dict(color=GREY, dash="dot", width=1.5))
fig.update_yaxes(title="best f", row=1, col=1)
fig.update_yaxes(title="side length L", row=2, col=1)
fig.update_xaxes(title="function evaluations", row=2, col=1)
style(fig, height=430, showlegend=False,
      title=dict(text="Simplified TuRBO, one run: the trust region breathes",
                 font=dict(size=15)))
save_div(fig, OUT / "fig_turbo.html", "fig-turbo")
print("all figures written")

