#!/usr/bin/env python3
"""Build the one-page language modelling cheat sheet and render it."""
import json
import subprocess
import sys
from pathlib import Path

DATE = "2026-08-04"


def frac(num, den):
    return ("<span class='frac'><span class='num'>" + num +
            "</span><span class='den'>" + den + "</span></span>")


CARDS = ("""
<div class='grid2'>

<div class='cheat-card'><h4>The whole object</h4>
<div class='math'>P(x&#8321;..x<sub>T</sub>) = &prod;<sub>t</sub> P(x<sub>t</sub> | x<sub>&lt;t</sub>)</div>
<p>An identity, not an approximation. Learn one conditional, apply it T times.
Labels are free: the target at t is the token at t+1.</p></div>

<div class='cheat-card'><h4>Objective and its units</h4>
<div class='math'>L = &minus;""" + frac("1", "T") + """&sum;<sub>t</sub> log p<sub>t</sub>[x<sub>t</sub>]
&nbsp;&nbsp;&nbsp; PPL = e<sup>L</sup></div>
<p>Uniform over V &rArr; PPL = V exactly. Floor is H(data), the E in the scaling law.
<strong>Compare models in bits/byte, never per-token perplexity across tokenizers.</strong></p></div>

<div class='cheat-card'><h4>BPE</h4>
<p>Merge the most frequent adjacent pair; repeat. Vocab = 256 + merges.
More merges &rArr; shorter sequences but a harder per-token prediction and a
bigger embedding table. Stops when no pair repeats.</p></div>

<div class='cheat-card'><h4>Attention</h4>
<div class='math'>softmax( """ + frac("QK<sup>T</sup>", "&radic;d<sub>k</sub>") + """ + M ) V</div>
<p>Mask <em>before</em> the softmax. Var(q&middot;k) = d<sub>k</sub>, so without
the &radic;d<sub>k</sub> the softmax saturates and its Jacobian
p<sub>i</sub>(&delta;<sub>ij</sub>&minus;p<sub>j</sub>) vanishes &mdash; no gradient.</p></div>

<div class='cheat-card'><h4>RoPE</h4>
<div class='math'>&theta;<sub>i</sub> = base<sup>&minus;2i/d</sup>;&nbsp;
&lang;R(m)q, R(n)k&rang; = f(q,k,m&minus;n)</div>
<p>Rotate pairs by position&times;angle. Norm-preserving, and the score sees only
relative distance. Position 0 is the identity.</p></div>

<div class='cheat-card'><h4>Block extras</h4>
<div class='math'>RMSNorm(x) = x / &radic;(mean(x&sup2;)+&epsilon;) &middot; g</div>
<p>No mean subtraction, no bias; pre-norm keeps a clean residual path.
SwiGLU = (swish(xW&#8321;) &odot; xW&#8323;)W&#8322;, three matrices, so
d<sub>ff</sub> = 8/3&middot;d holds parameters fixed.</p></div>

<div class='cheat-card'><h4>Cost accounting</h4>
<div class='math'>C &asymp; 6ND</div>
<p>2 forward + 2 grad-wrt-input + 2 grad-wrt-weight, per parameter per token.
Excludes attention scores (4Ld per layer), which stay a minority until
<strong>L &gt; 6d</strong>.</p></div>

<div class='cheat-card'><h4>Memory per parameter</h4>
<div class='math'>2 + 2 + 4 + 4 + 4 = 16 bytes</div>
<p>bf16 weights + bf16 grads + fp32 master + Adam m + Adam v. A 70B run needs
~1.1 TB of state. ZeRO shards it: 16&Psi; &rarr; 16&Psi;/N<sub>d</sub>.</p></div>

<div class='cheat-card'><h4>AdamW</h4>
<div class='math'>&theta; &larr; &theta; &minus; &eta;""" +
    frac("m&#770;", "&radic;v&#770;+&epsilon;") + """ &minus; &eta;&lambda;&theta;</div>
<p>Scale-free: first step is &eta;&middot;sign(g) for any gradient magnitude
(until |g| approaches &epsilon;). Decay is <em>decoupled</em> &mdash; applied to
weights, not added to the gradient.</p></div>

<div class='cheat-card'><h4>Chinchilla</h4>
<div class='math'>L = E + """ + frac("A", "N<sup>&alpha;</sup>") + """ + """ +
    frac("B", "D<sup>&beta;</sup>") + """ &nbsp; s.t. &nbsp; C = 6ND</div>
<div class='math'>N* &prop; C<sup>&beta;/(&alpha;+&beta;)</sup>,&nbsp;
D* &prop; C<sup>&alpha;/(&alpha;+&beta;)</sup></div>
<p>&alpha;&asymp;&beta; &rArr; both exponents &frac12; &rArr; scale N and D together
(&asymp;20 tokens/param). <strong>Minimizing the published Approach-3 constants does
not reproduce the paper&rsquo;s own 70B/1.4T</strong> &mdash; the replication&rsquo;s
re-estimate does.</p></div>

<div class='cheat-card'><h4>KV cache</h4>
<div class='math'>2 &middot; n<sub>layers</sub> &middot; n<sub>kv</sub> &middot;
d<sub>head</sub> &middot; L &middot; 2 bytes</div>
<p>Linear in context, and it is what decoding is bandwidth-bound on. MQA: one KV
head. GQA: g groups. MLA: cache a low-rank latent instead.</p></div>

<div class='cheat-card'><h4>FlashAttention</h4>
<p>Exact attention; only the memory traffic changes. Tile Q/K/V into SRAM and carry
a running max and sum (online softmax) so the T&times;T matrix never reaches HBM.
Attention is memory-bound, not compute-bound.</p></div>

<div class='cheat-card'><h4>Mixture of experts</h4>
<div class='math'>y = &sum;<sub>i&isin;TopK</sub> g<sub>i</sub>(x)&middot;FFN<sub>i</sub>(x)</div>
<p>Params scale with E, FLOPs with k. Failure mode is routing collapse: fix with an
auxiliary balance loss, or a per-expert routing bias that leaves the main gradient
alone.</p></div>

<div class='cheat-card'><h4>DPO</h4>
<div class='math'>&minus;log &sigma;( &beta; log """ +
    frac("&pi;(y<sub>w</sub>)", "&pi;<sub>ref</sub>(y<sub>w</sub>)") + """ &minus; &beta; log """ +
    frac("&pi;(y<sub>l</sub>)", "&pi;<sub>ref</sub>(y<sub>l</sub>)") + """ )</div>
<p>The KL-regularized RLHF optimum inverts to give the reward as a policy log-ratio;
substituting cancels the reward model. Keep &pi;<sub>ref</sub> frozen.</p></div>

<div class='cheat-card'><h4>GRPO</h4>
<div class='math'>A&#770;<sub>i</sub> = """ +
    frac("r<sub>i</sub> &minus; mean(r)", "std(r)") + """</div>
<p>The group of sampled completions is its own baseline, so PPO&rsquo;s value network
disappears. With a verifier as the reward, the reward model goes too.</p></div>

<div class='cheat-card'><h4>Open questions</h4>
<p>Does RLVR add capability or sharpen sampling? Do byte-level models retire the
tokenizer? Is aggressive dedup always good (FineWeb says no)? Are emergent abilities
real transitions or a scoring artifact?</p></div>

</div>

<p style='text-align:center;margin-top:1.4rem;font-size:.85rem'>
<a href='https://arxiv.org/abs/1706.03762'>Attention</a> &middot;
<a href='https://arxiv.org/abs/2203.15556'>Chinchilla</a> &middot;
<a href='https://arxiv.org/abs/2404.10102'>replication</a> &middot;
<a href='https://arxiv.org/abs/2205.14135'>FlashAttention</a> &middot;
<a href='https://arxiv.org/abs/2305.18290'>DPO</a> &middot;
<a href='https://stanford-cs336.github.io/spring2025/'>CS336</a>
&nbsp;&mdash;&nbsp; full story: <a href='""" + DATE + """-language-modelling.html'>the explainer</a>
&middot; drills: <a href='""" + DATE + """-language-modelling-review.html'>review deck</a></p>
""")

spec = {
    "title": "Language Modelling Cheat Sheet",
    "subtitle": "Every formula from the language modelling explainer, with its punchline",
    "slug": "lm-cheatsheet",
    "date": DATE,
    "repo": "companion to the language modelling explainer",
    "topic_title": "Language Modelling",
    "topic_href": DATE + "-language-modelling.html",
    "nav": [["explainer", DATE + "-language-modelling.html"],
            ["review deck", DATE + "-language-modelling-review.html"],
            ["tutorials", "https://github.com/raghuramshankar/learning-with-llms/tree/main/tutorials/language-modelling"]],
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
