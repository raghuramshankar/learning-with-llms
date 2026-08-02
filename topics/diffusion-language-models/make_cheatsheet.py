#!/usr/bin/env python3
"""Build the one-page dLLM cheat sheet spec and render it."""
import json
import subprocess
import sys
from pathlib import Path

def frac(num, den):
    return ("<span class='frac'><span class='num'>" + num +
            "</span><span class='den'>" + den + "</span></span>")

AB = "&#8113;"  # alpha with macron

CARDS = """
<div class='grid2'>

<div class='cheat-card'><h4>The two regimes</h4>
<p><strong>Typewriter (AR):</strong> N tokens = N sequential passes, left&rarr;right, write-once,
memory-bandwidth-bound.</p>
<p><strong>Editor (dLLM):</strong> k parallel refinement passes, k &Lt; N, bidirectional context,
revisable. <em>Fewer passes than tokens is the entire speed story.</em></p></div>

<div class='cheat-card'><h4>Continuous forward process</h4>
<div class='math'>q(x<sub>t</sub> | x<sub>0</sub>) = &#119977;( &radic;""" + AB + """<sub>t</sub>&middot;x<sub>0</sub>, (1&minus;""" + AB + """<sub>t</sub>)&middot;I )</div>
<p>One-jump corruption to any noise level; variance-preserving. Training never simulates the
chain.</p></div>

<div class='cheat-card'><h4>Training loss (DDPM)</h4>
<div class='math'>L<sub>simple</sub> = &#120124; &Vert; &epsilon; &minus; &epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) &Vert;&sup2;</div>
<p>&ldquo;Guess the noise I added.&rdquo; The ELBO's Gaussian KLs, collapsed and unweighted.</p></div>

<div class='cheat-card'><h4>Score link</h4>
<div class='math'>s<sub>&theta;</sub>(x<sub>t</sub>) = &minus;&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) / &radic;(1&minus;""" + AB + """<sub>t</sub>)</div>
<p>Denoising = gradient ascent on log-density. Bridge to the SDE view.</p></div>

<div class='cheat-card'><h4>DDIM update (&sigma; = 0)</h4>
<div class='math'>x&#770;<sub>0</sub> = """ + frac("x<sub>t</sub> &minus; &radic;(1&minus;" + AB + "<sub>t</sub>)&middot;&epsilon;<sub>&theta;</sub>", "&radic;" + AB + "<sub>t</sub>") + """;&nbsp;
x<sub>t&minus;1</sub> = &radic;""" + AB + """<sub>t&minus;1</sub>&middot;x&#770;<sub>0</sub> + &radic;(1&minus;""" + AB + """<sub>t&minus;1</sub>)&middot;&epsilon;<sub>&theta;</sub></div>
<p>Deterministic, same marginals, same network &mdash; so you can skip steps. <em>Step count is a
dial, not architecture.</em></p></div>

<div class='cheat-card'><h4>Discrete forward process</h4>
<div class='math'>q(z<sub>t</sub> | x) = Cat( &alpha;<sub>t</sub>&middot;x + (1&minus;&alpha;<sub>t</sub>)&middot;m )</div>
<p>Noise for text = masking. Absorbing mask state; independent per position.</p></div>

<div class='cheat-card'><h4>Reveal posterior</h4>
<div class='math'>P(reveal) = """ + frac("&alpha;<sub>s</sub> &minus; &alpha;<sub>t</sub>", "1 &minus; &alpha;<sub>t</sub>") + """</div>
<p>Masked tokens reveal (to the true token) with this probability; visible tokens carry over with
probability 1 &mdash; the frozen-token theorem ReMDM later breaks.</p></div>

<div class='cheat-card'><h4>MDLM objective</h4>
<div class='math'>L<sub>&infin;</sub> = &int;<sub>0</sub><sup>1</sup> """ + frac("&minus;&alpha;&prime;<sub>t</sub>", "1&minus;&alpha;<sub>t</sub>") + """ &middot; (BERT loss at level t) dt</div>
<p>Fill-in-the-blank, integrated over all masking ratios, is a bound on log-likelihood. Linear
schedule &rArr; weight 1/t. Value is schedule-invariant.</p></div>

<div class='cheat-card'><h4>Mean-field error</h4>
<p>Tokens committed in one pass are sampled <em>independently</em> from per-position marginals
(&ldquo;hot cream&rdquo;). Fewer steps &rArr; more simultaneous commits &rArr; more error.
<em>The step dial trades coordination for speed; T&rarr;&infin; is exact.</em></p></div>

<div class='cheat-card'><h4>The five dLLM fixes</h4>
<p><strong>MDLM</strong>: clean objective, likelihoods competitive.<br>
<strong>Block Diffusion</strong>: AR across blocks &mdash; variable length + KV cache.<br>
<strong>ReMDM</strong>: marginal-preserving remasking &mdash; revision, no retraining.<br>
<strong>Guidance</strong>: (p<sub>cond</sub>/p<sub>uncond</sub>)<sup>&gamma;</sup> per position &mdash; steering.<br>
<strong>d1</strong>: SFT + diffu-GRPO (one-step masked log-probs) &mdash; reasoning.</p></div>

<div class='cheat-card'><h4>The foundations</h4>
<p><strong>DDIM</strong> (2020): few-step sampling.
<strong>Decision Transformer</strong> (2021): RL as sequence modeling.
<strong>FlashAttention</strong> (2022): exact attention, IO-aware tiling.
<strong>DPO</strong> (2023): r = &beta;&middot;log(&pi;/&pi;<sub>ref</sub>) &mdash; alignment without
a reward model.</p></div>

<div class='cheat-card'><h4>Mercury, in one breath</h4>
<p>Inception Labs' commercial dLLM family: MDLM training + Block Diffusion caching + ReMDM
revision + guidance + d1-style reasoning, on FlashAttention-class kernels &mdash; claiming ~5&times;
AR speed. Frontier quality remains the open bet.</p></div>

</div>

<p>Papers: <a href='https://arxiv.org/abs/2010.02502'>DDIM</a> &middot;
<a href='https://arxiv.org/abs/2106.01345'>Decision Transformer</a> &middot;
<a href='https://arxiv.org/abs/2205.14135'>FlashAttention</a> &middot;
<a href='https://arxiv.org/abs/2305.18290'>DPO</a> &middot;
<a href='https://arxiv.org/abs/2406.07524'>MDLM</a> &middot;
<a href='https://arxiv.org/abs/2412.10193'>Guidance</a> &middot;
<a href='https://arxiv.org/abs/2503.09573'>Block Diffusion</a> &middot;
<a href='https://remdm.github.io/'>ReMDM</a> &middot;
<a href='https://arxiv.org/abs/2504.12216'>d1</a>
&nbsp;&mdash;&nbsp; full story: <a href='2026-08-01-diffusion-language-models.html'>the explainer</a>
&middot; drills: <a href='2026-08-02-diffusion-review.html'>review deck</a></p>
"""

spec = {
    "title": "dLLM Cheat Sheet",
    "subtitle": "Every formula from the diffusion-language-models explainer, with its punchline",
    "slug": "dllm-cheatsheet",
    "date": "2026-08-02",
    "repo": "companion to the Inception Labs papers explainer",
    "sections": [{"id": "sheet", "title": "The One-Pager", "html": CARDS}],
    "site_title": "\u2190 Diffusion Language Models",
    "nav": [["Explainer", "2026-08-01-diffusion-language-models.html"],
            ["Review deck", "2026-08-02-diffusion-review.html"],
            ["Lab", "https://github.com/raghuramshankar/learning-with-llms/tree/main/labs/masked-diffusion"]],
    "generator": {
        "skill": "learning-new-topic",
        "skill_url": "https://github.com/raghuramshankar/learning-with-llms/blob/main/skills/learning-new-topic/SKILL.md",
        "model": "Claude Fable 5",
    },
}

here = Path(__file__).parent
(here / "cheatsheet_spec.json").write_text(json.dumps(spec))
subprocess.run([sys.executable, str(here.parents[1] / "tools" / "render.py"),
                str(here / "cheatsheet_spec.json"),
                "-o", str(here.parents[1] / "docs")], check=True)
