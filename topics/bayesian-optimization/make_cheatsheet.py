#!/usr/bin/env python3
"""Build the one-page Bayesian optimization cheat sheet and render it."""
import json
import subprocess
import sys
from pathlib import Path

def frac(num, den):
    return ("<span class='frac'><span class='num'>" + num +
            "</span><span class='den'>" + den + "</span></span>")

CARDS = """
<div class='grid2'>

<div class='cheat-card'><h4>The loop</h4>
<p>Fit GP posterior on all data &rarr; maximize acquisition on the model (cheap, thousands of
queries) &rarr; evaluate f at the argmax (expensive, once) &rarr; repeat. <em>Model computation
buys evaluation frugality.</em></p></div>

<div class='cheat-card'><h4>RBF kernel</h4>
<div class='math'>k(x, x&prime;) = exp( &minus;""" + frac("&Vert;x&minus;x&prime;&Vert;&sup2;", "2&#8467;&sup2;") + """ )</div>
<p>&#8467; is a hypothesis about how fast f varies. Too long &rArr; confidently smooths over
hidden peaks (the dangerous direction); too short &rArr; ignorance everywhere.</p></div>

<div class='cheat-card'><h4>GP posterior (the two numbers)</h4>
<div class='math'>&mu;<sub>n</sub>(x) = k<sub>*</sub><sup>T</sup>(K+&sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup>y</div>
<div class='math'>&sigma;<sub>n</sub>&sup2;(x) = k(x,x) &minus; k<sub>*</sub><sup>T</sup>(K+&sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup>k<sub>*</sub></div>
<p>Exact Gaussian conditioning. &sigma;&sup2; depends on where you measured, not what you saw.
Cost: one O(n&sup3;) factorization, then O(n) / O(n&sup2;) per query.</p></div>

<div class='cheat-card'><h4>Marginal likelihood (fitting &#8467;, &sigma;<sub>n</sub>)</h4>
<div class='math'>log p(y|X,&theta;) = &minus;&frac12;y<sup>T</sup>K&#7488;&#8315;&#185;y &minus; &frac12;log det K&#7488; &minus; """ + frac("n", "2") + """log 2&pi;</div>
<p>Fit term vs complexity term (log-det) = automatic Occam&rsquo;s razor. Surface can be
multi-modal &mdash; restart or integrate (Snoek 2012).</p></div>

<div class='cheat-card'><h4>Expected improvement</h4>
<div class='math'>z = """ + frac("&mu;<sub>n</sub>(x) &minus; f<sup>*</sup><sub>n</sub>", "&sigma;<sub>n</sub>(x)") + """;&nbsp;&nbsp;
EI = (&mu;<sub>n</sub>&minus;f<sup>*</sup><sub>n</sub>)&Phi;(z) + &sigma;<sub>n</sub>&phi;(z)</div>
<p>Exploit term + explore term. <strong>At z = 0: EI = &sigma;&phi;(0) &asymp; 0.40&sigma;</strong>
&mdash; uncertainty alone has cash value. Default since EGO (1998).</p></div>

<div class='cheat-card'><h4>PI &amp; UCB</h4>
<div class='math'>PI = &Phi;(z);&nbsp;&nbsp;&nbsp;UCB = &mu;<sub>n</sub>(x) + &beta;<sup>&frac12;</sup>&sigma;<sub>n</sub>(x)</div>
<p>PI ignores improvement size (hugs the incumbent). UCB is explicit optimism; with
&beta;<sub>n</sub> ~ log n, GP-UCB proves sublinear regret. All one-step criteria are
myopic.</p></div>

<div class='cheat-card'><h4>LogEI (NeurIPS 2023)</h4>
<p>EI and its gradients underflow where improvement is unlikely &rarr; the acquisition
optimizer silently stalls. Optimizing a numerically careful log EI fixes it. <em>Decades of
&ldquo;EI underperforms&rdquo; was partly this bug.</em> Modern BoTorch default.</p></div>

<div class='cheat-card'><h4>Batch: qEI</h4>
<div class='math'>qEI = &#120124;[ max( max<sub>j</sub> f(x<sub>j</sub>) &minus; f<sup>*</sup><sub>n</sub>, 0 ) ]</div>
<p>Joint expectation via Monte-Carlo posterior samples (reparameterized, so gradients flow).
The jointness is what stops q near-duplicate proposals.</p></div>

<div class='cheat-card'><h4>Failure modes</h4>
<p><strong>Myopia</strong> (one-step lookahead); <strong>model mismatch</strong> (wrong &#8467;
&rArr; confident blindness); <strong>dimensionality</strong> (&gt;~20D: contested territory).
And if evaluations are cheap &mdash; just use random search.</p></div>

<div class='cheat-card'><h4>The modern line</h4>
<p><strong>TPE/SMAC</strong>: non-GP surrogates (Optuna). <strong>TuRBO</strong>: trust
regions. <strong>SAASBO</strong>: sparse dimension priors. <strong>&pi;BO</strong>: user
beliefs. <strong>qEHVI</strong>: parallel multi-objective. <strong>PFNs4BO/OptFormer/GIT-BO</strong>:
pretrained transformer surrogates. <strong>LLAMBO</strong>: LLM warm-starts.</p></div>

<div class='cheat-card'><h4>The live fights (2024&ndash;26)</h4>
<p>High dimensions: vanilla GPs match specialized methods (ICML&rsquo;24, ICLR&rsquo;25) vs
&ldquo;we still don&rsquo;t understand&rdquo; (Nov&rsquo;25). LLMs in the loop: good for priors
and warm starts; budget-matched studies say not yet a surrogate.</p></div>

<div class='cheat-card'><h4>When to reach for BO</h4>
<p>Evaluations expensive (minutes+), budget small (tens), dimensions modest (&le;~20),
continuous-ish inputs. Otherwise: random search (cheap f), CMA-ES (cheap f, many dims),
Optuna/TPE (conditional spaces).</p></div>

</div>

<p>Papers: <a href='https://arxiv.org/abs/0912.3995'>GP-UCB</a> &middot;
<a href='https://arxiv.org/abs/1206.2944'>Snoek 2012</a> &middot;
<a href='https://arxiv.org/abs/1807.02811'>Frazier tutorial</a> &middot;
<a href='https://arxiv.org/abs/1910.01739'>TuRBO</a> &middot;
<a href='https://arxiv.org/abs/2103.00349'>SAASBO</a> &middot;
<a href='https://arxiv.org/abs/2310.20708'>LogEI</a> &middot;
<a href='https://arxiv.org/abs/2305.17535'>PFNs4BO</a> &middot;
<a href='https://arxiv.org/abs/2402.03921'>LLAMBO</a> &middot;
<a href='https://bayesoptbook.com/'>Garnett (book)</a>
&nbsp;&mdash;&nbsp; full story: <a href='2026-08-02-bayesian-optimization.html'>the explainer</a>
&middot; drills: <a href='2026-08-02-bayesian-optimization-review.html'>review deck</a></p>
"""

spec = {
    "title": "Bayesian Optimization Cheat Sheet",
    "subtitle": "Every formula from the Bayesian optimization explainer, with its punchline",
    "slug": "bo-cheatsheet",
    "date": "2026-08-02",
    "repo": "companion to the Bayesian optimization explainer",
    "site_title": "← Bayesian Optimization",
    "nav": [["Explainer", "2026-08-02-bayesian-optimization.html"],
            ["Review deck", "2026-08-02-bayesian-optimization-review.html"],
            ["Tutorials", "https://github.com/raghuramshankar/learning-with-llms/tree/main/tutorials/bayesian-optimization"]],
    "generator": {
        "skill": "learning-new-topic",
        "skill_url": "https://github.com/raghuramshankar/learning-with-llms/blob/main/skills/learning-new-topic/SKILL.md",
        "model": "Claude Fable 5",
    },
    "sections": [{"id": "sheet", "title": "The One-Pager", "html": CARDS}],
}

here = Path(__file__).parent
(here / "cheatsheet_spec.json").write_text(json.dumps(spec))
subprocess.run([sys.executable, str(here.parents[1] / "tools" / "render.py"),
                str(here / "cheatsheet_spec.json"),
                "-o", str(here.parents[1] / "docs")], check=True)
