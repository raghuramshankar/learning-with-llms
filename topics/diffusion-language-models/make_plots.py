#!/usr/bin/env python3
"""Generate Plotly figure snippets (div + script, no library) for the dLLM explainer.

Runs the real DDIM sampler in numpy for a sweep of step counts, then writes:
  plots/plotly_lib.js      - inlined plotly.min.js (embedded once in <head>)
  plots/fig_latency.html   - AR vs dLLM sequential-pass scaling
  plots/fig_ddim_sweep.html- animated DDIM final samples, slider over S
  plots/fig_quality.html   - sample quality (NLL) vs step count S
  plots/fig_sched.html     - schedule alpha_t and NELBO weight w(t), buttons
  plots/fig_reveal.html    - reveal-probability heatmap over (alpha_t, alpha_s)
"""
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from plot_style import style, save_div, write_plotly_lib, FONT, GRID, BLUE, GREEN, RED, GREY

OUT = Path(__file__).parent / "plots"
OUT.mkdir(exist_ok=True)
DOCS = Path(__file__).resolve().parents[2] / "docs"


# ---------------------------------------------------------------- latency --
N = np.arange(1, 4001)
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=N, y=N, name="autoregressive: N passes", line=dict(color=RED, width=2.5),
    hovertemplate="N = %{x} tokens → %{y} sequential passes<extra>AR</extra>"))
for s, dashed in ((16, "dot"), (64, "dash")):
    fig.add_trace(go.Scatter(
        x=N, y=np.full_like(N, s), name=f"dLLM: {s} passes",
        line=dict(color=BLUE, width=2.5, dash=dashed),
        hovertemplate="N = %{x} tokens → " + str(s) + " sequential passes<extra>dLLM</extra>"))
fig.update_yaxes(type="log", title="sequential forward passes (log)")
fig.update_xaxes(title="output length N (tokens)")
style(fig, height=340, legend=dict(x=0.02, y=0.98, bgcolor="rgba(0,0,0,0)"),
      title=dict(text="Why agents care: sequential passes vs output length", font=dict(size=15)))
save_div(fig, OUT / "fig_latency.html", "fig-latency")

# ------------------------------------------------- DDIM sweep + quality ----
# mixture (same face as the JS lab)
mus = [(-0.42, 0.38), (0.42, 0.38)]
sds = [0.07, 0.07]
ws = [0.14, 0.14]
for i in range(6):
    a = np.deg2rad(205 + i * 26)
    mus.append((0.62 * np.cos(a), 0.12 + 0.62 * np.sin(a)))
    sds.append(0.06)
    ws.append(0.72 / 6)
MU = np.array(mus)          # (K,2)
SD = np.array(sds)          # (K,)
W = np.array(ws)            # (K,)

def eps_hat(X, ab):
    """Exact eps for the Gaussian-mixture marginal at noise level ab. X: (n,2)."""
    sa, nb = np.sqrt(ab), 1.0 - ab
    var = ab * SD**2 + nb                                  # (K,)
    d = X[:, None, :] - sa * MU[None, :, :]                # (n,K,2)
    logg = -np.sum(d**2, axis=2) / (2 * var) - np.log(var) + np.log(W)
    g = np.exp(logg - logg.max(axis=1, keepdims=True))
    r = g / g.sum(axis=1, keepdims=True)                   # responsibilities
    score = -np.sum(r[:, :, None] * d / var[None, :, None], axis=1)
    return -np.sqrt(nb) * score

def ddim_run(S, X0):
    ab = lambda t: np.cos(np.pi * t / 2) ** 2
    ts = 0.985 * (1 - np.arange(S + 1) / S) + 0.005
    X = X0.copy()
    for i in range(S):
        a0, a1 = ab(ts[i]), ab(ts[i + 1])
        e = eps_hat(X, a0)
        x0 = (X - np.sqrt(1 - a0) * e) / np.sqrt(a0)
        X = np.sqrt(a1) * x0 + np.sqrt(1 - a1) * e
    return X

def nll(X):
    var = SD**2
    d = X[:, None, :] - MU[None, :, :]
    logg = -np.sum(d**2, axis=2) / (2 * var) - np.log(2 * np.pi * var) + np.log(W)
    m = logg.max(axis=1, keepdims=True)
    return float(-np.mean(m.squeeze() + np.log(np.exp(logg - m).sum(axis=1))))

rng = np.random.default_rng(21)
X0 = rng.standard_normal((600, 2))
target = MU[rng.choice(len(W), 600, p=W)] + rng.standard_normal((600, 2)) * \
         SD[rng.choice(len(W), 600, p=W)][:, None]
# (resample target properly: component index shared between mean and sd)
idx = rng.choice(len(W), 600, p=W)
target = MU[idx] + rng.standard_normal((600, 2)) * SD[idx][:, None]

SVALS = [1, 2, 4, 8, 16, 32, 64]
finals, quality = {}, []
for S in SVALS:
    XS = ddim_run(S, X0)
    finals[S] = XS
    quality.append(nll(XS))

AXIS = dict(range=[-1.35, 1.35], showgrid=False, zeroline=False, visible=False)
frames = [go.Frame(name=str(S), data=[
    go.Scatter(x=target[:, 0], y=target[:, 1], mode="markers",
               marker=dict(size=3.5, color=GREY), name="target",
               hoverinfo="skip"),
    go.Scatter(x=finals[S][:, 0], y=finals[S][:, 1], mode="markers",
               marker=dict(size=4.5, color=BLUE), name="DDIM samples",
               hovertemplate="(%{x:.2f}, %{y:.2f})<extra>S=" + str(S) + "</extra>"),
]) for S in SVALS]
fig = go.Figure(data=frames[2].data, frames=frames)   # start at S=4
steps = [dict(method="animate", label="S = " + str(S),
              args=[[str(S)], dict(mode="immediate",
                                   frame=dict(duration=250, redraw=True),
                                   transition=dict(duration=250))])
         for S in SVALS]
fig.update_layout(
    sliders=[dict(active=2, steps=steps, currentvalue=dict(prefix="step count: "),
                  pad=dict(t=8), font=dict(color=FONT))],
    updatemenus=[dict(type="buttons", x=0.02, y=1.12, direction="right",
                      bgcolor="rgba(0,0,0,0)", font=dict(color=FONT),
                      buttons=[dict(label="▶ sweep S", method="animate",
                                    args=[[str(S) for S in SVALS],
                                          dict(mode="immediate",
                                               frame=dict(duration=600, redraw=True),
                                               transition=dict(duration=300))])])],
    showlegend=False,
    title=dict(text="Same model, same noise — final samples as the step dial turns",
               font=dict(size=15)),
)
fig.update_xaxes(**AXIS)
fig.update_yaxes(**AXIS, scaleanchor="x")
style(fig, height=470)
save_div(fig, OUT / "fig_ddim_sweep.html", "fig-ddim-sweep")

fig = go.Figure(go.Scatter(
    x=SVALS, y=quality, mode="lines+markers", line=dict(color=GREEN, width=2.5),
    marker=dict(size=8),
    hovertemplate="S = %{x} steps → avg NLL %{y:.2f}<extra></extra>"))
fig.update_xaxes(type="log", title="DDIM steps S (log)", tickvals=SVALS)
fig.update_yaxes(title="avg negative log-likelihood of samples")
style(fig, height=320,
      title=dict(text="Quality vs steps: computed, not asserted", font=dict(size=15)))
save_div(fig, OUT / "fig_quality.html", "fig-quality")

# ---------------------------------------------------------------- sched ----
t = np.linspace(0.001, 0.999, 400)
SCHEDS = {
    "linear: 1−t": (1 - t, np.ones_like(t)),
    "cosine: cos²(πt/2)": (np.cos(np.pi * t / 2) ** 2, np.pi / 2 * np.sin(np.pi * t)),
    "quadratic: 1−t²": (1 - t**2, 2 * t),
}
WCAP = 6.0
fig = go.Figure()
for i, (name, (a, nda)) in enumerate(SCHEDS.items()):
    w = np.minimum(WCAP, nda / np.maximum(1e-9, 1 - a))
    fig.add_trace(go.Scatter(x=t, y=a, name="α_t", visible=(i == 0),
                             line=dict(color=BLUE, width=2.5),
                             hovertemplate="t=%{x:.2f}  α=%{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=t, y=w, name="w(t) = −α′/(1−α)", visible=(i == 0),
                             line=dict(color=GREEN, width=2.5),
                             hovertemplate="t=%{x:.2f}  w=%{y:.2f}<extra></extra>"))
buttons = []
for i, name in enumerate(SCHEDS):
    vis = [False] * (2 * len(SCHEDS))
    vis[2 * i] = vis[2 * i + 1] = True
    buttons.append(dict(label=name, method="update", args=[{"visible": vis}]))
fig.update_layout(
    updatemenus=[dict(type="buttons", direction="right", x=0.5, xanchor="center",
                      y=1.22, bgcolor="rgba(0,0,0,0)", font=dict(color=FONT),
                      buttons=buttons)],
    legend=dict(x=0.72, y=0.98, bgcolor="rgba(0,0,0,0)"),
)
fig.update_xaxes(title="t (corruption level)")
fig.update_yaxes(title="value", range=[0, WCAP * 1.02])
style(fig, height=380)
save_div(fig, OUT / "fig_sched.html", "fig-sched")

# ------------------------------------------------------------- reveal ------
n = 120
at = np.linspace(0.0, 0.98, n)
as_ = np.linspace(0.01, 1.0, n)
AT, AS = np.meshgrid(at, as_)
Z = np.where(AS > AT, (AS - AT) / (1 - AT), np.nan)
fig = go.Figure(go.Heatmap(
    x=at, y=as_, z=Z, colorscale="Blues", zmin=0, zmax=1,
    colorbar=dict(title="P(reveal)", tickfont=dict(color=FONT)),
    hovertemplate="α_t=%{x:.2f}  α_s=%{y:.2f}  P(reveal)=%{z:.3f}<extra></extra>"))
fig.update_xaxes(title="α_t (noisier time)")
fig.update_yaxes(title="α_s (less-noisy time)")
style(fig, height=400,
      title=dict(text="The whole reveal-probability landscape (blank: s must be less noisy)",
                 font=dict(size=15)))
save_div(fig, OUT / "fig_reveal.html", "fig-reveal")

# ---------------------------------------------------------------- library --
write_plotly_lib(DOCS / "plotly.min.js")
