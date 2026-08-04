#!/usr/bin/env python3
"""Build the JSON content spec for the language modelling explainer.

Citations are not written from memory: `cite()` reads survey/verified.json,
which is produced by survey/verify_batch.py against the arXiv API. Anything
the API did not confirm is rendered with an explicit "unverified" marker
rather than a plausible-looking identifier.
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / "tools"
DOCS = HERE.parents[1] / "docs"
DATE = "2026-08-04"

VERIFIED = {}
_vp = HERE / "survey" / "verified.json"
if _vp.exists():
    VERIFIED = {k: v for k, v in json.loads(_vp.read_text()).items() if v}


def frac(num, den):
    return ("<span class='frac'><span class='num'>" + num +
            "</span><span class='den'>" + den + "</span></span>")


def cite(title, short=None):
    """Render an inline citation from API-verified data only."""
    rec = VERIFIED.get(title)
    label = short or title
    if not rec:
        return ("<span class='cite-unver'>" + label +
                " <em>(arXiv id unverified)</em></span>")
    return ("<a href='https://arxiv.org/abs/" + rec["id"].split("v")[0] +
            "'>" + label + "</a> <span class='cite-id'>" +
            rec["id"].split("v")[0] + "</span>")


def refrow(title, note):
    rec = VERIFIED.get(title)
    if not rec:
        return ("<tr><td>" + title + "</td><td><em>not verified against the API"
                "</em></td><td><em>" + note + "</em></td></tr>")
    aid = rec["id"].split("v")[0]
    return ("<tr><td><a href='https://arxiv.org/abs/" + aid + "'>" + rec["title"] +
            "</a></td><td>" + rec["first_author"] + " et al., " +
            rec["published"][:4] + " &middot; arXiv:" + aid +
            "</td><td><em>" + note + "</em></td></tr>")


# ===========================================================================
BACKGROUND = ("""
<p>A language model does one thing: given some text, it puts a probability on
what comes next. Everything else &mdash; writing code, answering questions,
translating, refusing &mdash; is downstream of that single mechanic. This page
builds the whole object from bytes upward, in the order Stanford&rsquo;s CS336
builds it: tokenizer, architecture, optimizer, systems, scaling laws, data,
alignment. Every formula is derived rather than quoted, and every paper is
checked against the arXiv API before it is cited.</p>
<p>If you already know what a softmax is and why gradient descent needs a
differentiable loss, skip to the second subtopic.</p>

<h3>Deep background: what a model of text even is</h3>
<p>Suppose you want to assign a probability to an entire sentence. Directly
tabulating <em>P(the whole sentence)</em> is hopeless: there are more possible
50-word English sentences than atoms in the observable universe, so no corpus
could ever list them. The escape is the chain rule of probability, which is an
identity, not an approximation:</p>
<div class='math'>P(x&#8321;, &hellip;, x<sub>T</sub>) = &prod;<sub>t=1..T</sub>
P(x<sub>t</sub> | x<sub>&lt;t</sub>)</div>
<p>The joint probability of a sequence factorizes exactly into a product of
next-token probabilities. That single line is why modern language modelling is
possible: instead of learning an astronomically large table, you learn one
conditional distribution &mdash; <em>what comes next, given everything so
far</em> &mdash; and apply it repeatedly. A model that is good at that one
prediction is, by the identity above, a model of all text.</p>

<div class='diagram'>
  <div class='flow'>
    <div class='box accent'>bytes<small>raw text</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box'>tokens<small>BPE merges</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box'>transformer<small>attention + FFN</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box ok'>P(next token)<small>a vector over the vocabulary</small></div>
  </div>
  <div class='caption'>The entire pipeline. Every part of this page is one of
  these four boxes, or the machinery needed to fit the third one at scale.</div>
</div>

<h3>Deep background: learning by prediction</h3>
<p>To <em>learn</em> that conditional distribution you need three things: a
family of functions with adjustable parameters, a number saying how wrong the
current parameters are, and a way to change them to make that number smaller.
The function family is a neural network; the number is the negative log
probability the model assigned to the token that actually occurred; and the
update is gradient descent, which works because every operation in the network
is differentiable.</p>
<p>The remarkable part is that the objective is free. Nobody labels anything:
the correct answer at position t is simply the token at position t+1, already
sitting there in the text. This is why the internet could be turned into
training data, and why the field&rsquo;s bottleneck became compute and data
curation rather than annotation.</p>

<h3>The tension: prediction is easy to state and brutal to scale</h3>
<p>So the objective is trivial to write down and the data is free. Where did
the difficulty go? It went into <em>scale</em>, and it did not go quietly.
Three walls appear immediately, and the rest of this page is the story of
each.</p>
<p>First, text is not a sequence of words &mdash; it is a sequence of bytes,
and how you group them into units changes both the sequence length and the
difficulty of every prediction. Second, the obvious architecture for sequences
processes them one step at a time, which cannot use a modern accelerator;
making it parallel was the 2017 breakthrough. Third, once training runs cost
millions of dollars, you can no longer discover the right model size by trying
several &mdash; you need a theory that tells you, before you start, how to
split a fixed budget between model size and data.</p>

<div class='callout'>
<p><strong>The puzzle this page resolves.</strong> A language model is trained
to do exactly one thing: minimize the average surprise of the next token. It
is never taught to reason, translate, or write code. Yet those capabilities
appear as the same objective is optimized harder, on more data, with more
parameters. What is the mechanism by which a next-token predictor becomes a
general-purpose system &mdash; and how much of that is understood versus
merely observed?</p>
</div>
""")

QUIZ_BACKGROUND = [
    {"question": "Why does the chain-rule factorization make language modelling tractable at all?",
     "options": [
        {"text": "It replaces one astronomically large table with a single reusable conditional.",
         "correct": True,
         "explanation": "The joint over sequences has combinatorially many entries; the factorization means you only ever learn P(next | prefix) and apply it T times. It is an exact identity, so nothing is lost."},
        {"text": "It approximates the joint distribution well when tokens are nearly independent.",
         "explanation": "No approximation is involved and no independence is assumed — the chain rule is an identity that holds for any joint distribution."},
        {"text": "It lets the model condition on the future as well as the past.",
         "explanation": "That describes a bidirectional (masked) model like BERT. The autoregressive factorization deliberately conditions only on the prefix."},
        {"text": "It removes the need for a normalized probability distribution.",
         "explanation": "Each factor is still a normalized distribution over the vocabulary — the softmax exists precisely to guarantee that."}]},
    {"question": "What makes next-token prediction 'self-supervised' rather than supervised?",
     "options": [
        {"text": "The label at each position is the next token, already present in the text.",
         "correct": True,
         "explanation": "No annotator is involved: the supervision signal is extracted from the data's own ordering, which is why raw web text works as training data."},
        {"text": "The model generates its own labels from an auxiliary reward model.",
         "explanation": "That is RLHF, a post-training stage covered in Part 6 — it needs human preference data and comes long after pretraining."},
        {"text": "Gradients are estimated without backpropagation.",
         "explanation": "Backpropagation is used throughout; 'self-supervised' refers to where labels come from, not how gradients are computed."},
        {"text": "The loss requires no normalization over the vocabulary.",
         "explanation": "Cross entropy is computed against a normalized softmax over the whole vocabulary — that normalization is the expensive part."}]},
    {"question": "Which of these is NOT one of the three scaling walls the page identifies?",
     "options": [
        {"text": "Gradient descent cannot optimize a non-convex objective.",
         "correct": True,
         "explanation": "Non-convexity is real but is not a barrier in practice — SGD variants find useful minima routinely. The three walls named are tokenization, sequential architectures, and budget allocation."},
        {"text": "Text arrives as bytes that must be grouped into units.",
         "explanation": "This is the first wall: the grouping choice sets sequence length and per-token difficulty. Part 3 derives it."},
        {"text": "Recurrent processing cannot saturate a modern accelerator.",
         "explanation": "This is the second wall, and the reason attention replaced recurrence in 2017."},
        {"text": "A fixed compute budget must be split between parameters and tokens.",
         "explanation": "This is the third wall, and exactly what the Chinchilla analysis in Part 5 answers."}]},
    {"question": "A model assigns probability 0.5 to the actual next token at every position of a 10-token sequence. What probability does it assign to the whole sequence?",
     "options": [
        {"text": "0.5^10 ≈ 0.00098",
         "correct": True,
         "explanation": "By the chain rule the joint is the product of the per-token conditionals: 0.5 multiplied ten times. Sequence probabilities shrink geometrically, which is exactly why the field works in log space."},
        {"text": "0.5, since the per-token probability is constant.",
         "explanation": "That would be true only for a single token — the joint multiplies across all ten positions."},
        {"text": "10 × 0.5 = 5",
         "explanation": "Probabilities multiply rather than add under the chain rule, and a probability can never exceed 1."},
        {"text": "It cannot be determined without the vocabulary size.",
         "explanation": "The joint depends only on the assigned conditionals, which are given. Vocabulary size would matter for a uniform baseline, not here."}]},
    {"question": "Why did the free availability of the training objective shift the field's bottleneck?",
     "options": [
        {"text": "It moved the constraint onto compute and data curation instead of annotation.",
         "correct": True,
         "explanation": "With labels free, the limits became how much text you can gather and clean and how many FLOPs you can spend — which is why Parts 5 and 7 exist at all."},
        {"text": "It made supervised fine-tuning unnecessary.",
         "explanation": "Instruction tuning and preference optimization remain essential to make a raw predictor usable; Part 6 covers them."},
        {"text": "It eliminated the need for held-out evaluation.",
         "explanation": "Evaluation became harder, not easier — contamination and benchmark saturation are active problems."},
        {"text": "It removed overfitting as a concern at any scale.",
         "explanation": "Overfitting is milder at web scale but data repetition still degrades models, which is the subject of data-constrained scaling work."}]},
]

# ===========================================================================
INTUITION = ("""
<p>Before any equations, three pictures. The Maths sections that follow will
formalize precisely these, and nothing else.</p>

<h3>The next-token game, played by hand</h3>
<p>Fix a vocabulary and a context. The model outputs one number per vocabulary
entry, and those numbers are turned into a distribution by a softmax. Training
nudges the number for the token that actually appeared upward, and the rest
down. Generation samples from the result, appends, and repeats.</p>
<p>The sampler is not a detail. The same model is a different writer depending
on how you draw from it, and the knobs below are the ones every deployed
system exposes. Set the temperature near zero and it becomes deterministic and
repetitive; open it up and it becomes surprising and occasionally incoherent.
Watch the empirical bars converge onto the theoretical distribution as you
draw.</p>

<div class='widget' id='w-sample'>
  <div class='wctl'>
    <label>temperature <input type='range' id='w-sample-t' min='1' max='200' value='100'></label>
    <label>top-k <input type='range' id='w-sample-k' min='1' max='10' value='10'></label>
    <label>top-p <input type='range' id='w-sample-p' min='10' max='100' value='100'></label>
  </div>
  <div class='wstat' id='w-sample-cfg'></div>
  <div id='w-sample-bars'></div>
  <div class='wctl'>
    <button class='wbtn' id='w-sample-1'>draw 1</button>
    <button class='wbtn' id='w-sample-200'>draw 200</button>
    <button class='wbtn' id='w-sample-reset'>reset</button>
  </div>
  <div class='wstat' id='w-sample-stat'></div>
  <div class='caption'>A real sampler over a fixed ten-token distribution. Note
  what top-k and top-p do differently: top-k always keeps the same number of
  candidates, while top-p keeps however many are needed to cover the mass, so
  it adapts to how confident the model is.</div>
</div>

<h3>Attention as soft lookup</h3>
<p>Recurrence processes a sequence by carrying a hidden state forward one step
at a time. Two things go wrong: information from far back has to survive many
overwrites, and step t cannot be computed until step t&minus;1 is done, so a
GPU with tens of thousands of cores runs one position at a time.</p>
<p>Attention replaces the carried state with direct access. Every position
emits a <em>query</em> (&ldquo;what am I looking for?&rdquo;), every position
emits a <em>key</em> (&ldquo;what do I offer?&rdquo;) and a <em>value</em>
(&ldquo;what I will hand over&rdquo;). Each query is compared against all keys
at once, the comparisons are softmaxed into weights, and the output is the
weighted average of values. Every position is computed independently, so the
whole sequence goes through in one parallel pass.</p>
<p>The name to keep: <strong>attention trades memory for parallelism</strong>.
It removes the sequential dependency, but it must materialize a comparison
between every pair of positions, and that is quadratic. Part 5 shows exactly
when that quadratic term starts to hurt &mdash; the answer is later than most
people assume.</p>

<h3>What scale actually buys</h3>
<p>Make the model bigger and the loss goes down along a startlingly smooth
power law, across many orders of magnitude. There is no visible threshold
where the model &ldquo;starts understanding&rdquo;; the curve is boring, and
its boringness is what makes budget planning possible at all.</p>
<p>What is <em>not</em> boring is that specific capabilities &mdash; doing
arithmetic, following an instruction &mdash; appear to switch on abruptly as
the smooth loss keeps sliding. Whether those jumps are real transitions or an
artifact of scoring answers as right-or-wrong is genuinely contested, and Part
8 gives both sides.</p>

<div class='callout warn'>
<p><strong>Honesty box.</strong> Several things on this page are load-bearing
engineering with thin theory underneath.</p>
<ul>
<li><strong>Why attention works</strong> is not derived from first principles.
We can say what it computes and why it parallelizes; we cannot say why this
inductive bias beats alternatives on language.</li>
<li><strong>Scaling-law constants are fitted, not predicted</strong>, and they
move with data quality and architecture. Part 5 shows a published fit that
does not reproduce its own paper&rsquo;s headline recommendation.</li>
<li><strong>Emergence is contested.</strong> The measurements are real; the
interpretation is disputed.</li>
<li><strong>Alignment does not make a model truthful.</strong> It optimizes a
proxy &mdash; human preference or a verifier &mdash; and models learn to
exploit proxies.</li>
</ul>
</div>
""")

QUIZ_INTUITION = [
    {"question": "What is the essential difference between top-k and top-p (nucleus) sampling?",
     "options": [
        {"text": "top-k keeps a fixed count; top-p keeps a count that varies with model confidence.",
         "correct": True,
         "explanation": "top-p keeps the smallest set covering probability mass p, so a confident step keeps one or two candidates and an uncertain step keeps many. top-k is blind to the shape of the distribution."},
        {"text": "top-k renormalizes the distribution; top-p does not.",
         "explanation": "Both renormalize after truncation — otherwise the surviving probabilities would not sum to one."},
        {"text": "top-p applies before the temperature; top-k applies after.",
         "explanation": "Temperature is applied first in both cases; it changes the distribution that the truncation then acts on."},
        {"text": "top-k is stochastic; top-p is deterministic.",
         "explanation": "Both are stochastic — they restrict the candidate set and then sample from it."}]},
    {"question": "Which limitation of recurrence does attention NOT fix?",
     "options": [
        {"text": "Memory cost grows with the square of the sequence length.",
         "correct": True,
         "explanation": "This is a cost attention *introduces*, not one it fixes: every query-key pair is compared. Recurrence has O(1) memory per step. Everything else listed is a genuine recurrence problem attention removes."},
        {"text": "Distant information must survive many sequential overwrites.",
         "explanation": "Attention fixes exactly this — position t reads position 1 directly, in one hop, with no intervening state updates."},
        {"text": "Positions cannot be computed in parallel during training.",
         "explanation": "Attention fixes this: all positions are computed in one pass, which is what lets it saturate an accelerator."},
        {"text": "The path length between two positions grows with their distance.",
         "explanation": "Attention makes every path length 1, which is the property that motivated it."}]},
    {"question": "The loss follows a smooth power law in scale. Why does that matter practically?",
     "options": [
        {"text": "It lets you predict a large run's loss from a ladder of small ones.",
         "correct": True,
         "explanation": "Smoothness is what makes extrapolation legitimate: you fit on cheap runs and forecast the expensive one, which is the entire methodology behind budget allocation."},
        {"text": "It proves the model has learned the true data distribution.",
         "explanation": "The loss is bounded below by the data's own entropy, and a power law approaching that floor says nothing about having reached it."},
        {"text": "It guarantees that specific capabilities improve smoothly too.",
         "explanation": "This is precisely what is contested — aggregate loss is smooth while individual benchmark scores can look discontinuous."},
        {"text": "It means larger models need fewer training tokens.",
         "explanation": "The opposite: compute-optimal training scales tokens up along with parameters, as Part 5 derives."}]},
    {"question": "Attention is described as trading memory for parallelism. What exactly is traded?",
     "options": [
        {"text": "A sequential dependency is removed at the cost of an all-pairs comparison.",
         "correct": True,
         "explanation": "Recurrence is O(T) sequential steps with small state; attention is one parallel step with a T×T score matrix. That matrix is the price, and FlashAttention in Part 5 is about never writing it to slow memory."},
        {"text": "Parameters are traded for activations at fixed total memory.",
         "explanation": "Attention's parameter count is independent of sequence length; the trade is about the dependency structure of the computation, not parameter placement."},
        {"text": "Precision is reduced to allow larger batches.",
         "explanation": "That describes mixed-precision training, an unrelated systems technique."},
        {"text": "Depth is traded for width at constant FLOPs.",
         "explanation": "Depth/width balance is an architecture hyperparameter question, not what attention changes relative to recurrence."}]},
    {"question": "According to the honesty box, which statement is best supported?",
     "options": [
        {"text": "We can describe what attention computes but not derive why it suits language.",
         "correct": True,
         "explanation": "The mechanism is fully specified and the empirical result is overwhelming; the inductive-bias justification is post hoc. That gap is stated plainly rather than papered over."},
        {"text": "Scaling-law constants are derived from information theory.",
         "explanation": "They are fitted to observed runs, and they shift with data quality and architecture — which is why a replication can disagree."},
        {"text": "Alignment training makes models truthful.",
         "explanation": "It optimizes a proxy for human approval; reward hacking is the well-documented failure mode."},
        {"text": "Emergent abilities have been shown to be measurement artifacts.",
         "explanation": "That is one side of a live argument, not a settled finding — the page presents both."}]},
]

# ===========================================================================
MATH_TOKEN = ("""
<p>Two things get formalized here: how text becomes a sequence of integers,
and what number training actually minimizes. They are linked, because the
tokenizer decides both how long the sequence is and how hard each prediction
is &mdash; and those pull in opposite directions.</p>

<h3>From bytes to tokens: BPE</h3>
<p>Start from raw bytes: 256 possible values, every text representable, no
unknown-token problem ever. The cost is length &mdash; one English word is
four to five bytes, so sequences are long and attention is quadratic in
length.</p>
<p><strong>Byte-pair encoding</strong> buys length back. Count every adjacent
pair in the corpus, merge the most frequent one into a new symbol, and repeat.
Frequent sequences (<code>&nbsp;the</code>, <code>ing</code>) become single
tokens; rare ones stay in pieces. After k merges the vocabulary is 256 + k.</p>

<div class='widget' id='w-bpe'>
  <div class='wctl'>
    <label>merges <input type='range' id='w-bpe-n' min='0' max='80' value='0'></label>
  </div>
  <div class='wchips' id='w-bpe-chips'></div>
  <div class='wstat' id='w-bpe-stat'></div>
  <div class='caption'>A real BPE trainer running in your browser on the
  sentence above. Drag from zero: the first merges are always the boring
  frequent digraphs, and compression improves fast at first and then flattens
  &mdash; the same diminishing return the measured curve below shows.</div>
</div>

<div class='widget'>
__FIG_BPE__
<div class='caption'>BPE trained in numpy on this repository&rsquo;s own prose,
re-encoding the corpus after every merge to <em>measure</em> compression rather
than estimate it. The curve is steep then flat: the first few hundred merges do
most of the work, which is why vocabularies of ~32k&ndash;128k are a
reasonable stopping point rather than a deep truth.</div>
</div>

<p>The trade is now visible. More merges means fewer tokens per document
(cheaper attention, more text per context) but a larger vocabulary, and a
larger vocabulary means a harder prediction at every step and a bigger
embedding matrix. Compression and per-token difficulty move together.</p>

<h3>The objective: cross entropy</h3>
<p>The model produces a vector of logits z &isin; &#8477;<sup>V</sup>; the
softmax turns it into a distribution, and the loss is the negative log
probability of the token that actually occurred:</p>
<div class='math'>p<sub>i</sub> = """ + frac("e<sup>z<sub>i</sub></sup>",
    "&sum;<sub>j</sub> e<sup>z<sub>j</sub></sup>") + """
&nbsp;&nbsp;&nbsp;&nbsp; L = &minus;""" + frac("1", "T") + """
&sum;<sub>t</sub> log p<sub>t</sub>[x<sub>t</sub>]</div>
<p class='where'><b>z</b> the logit vector the network emits &middot; <b>V</b> the vocabulary size &middot; <b>T</b> the number of tokens scored &middot; <b>p<sub>t</sub>[x<sub>t</sub>]</b> the probability the model gave the token that actually occurred at position t</p>
<div class='callout'>
<p><strong>Why this matters.</strong> Minimizing this is exactly maximizing
the log-likelihood the model assigns to the corpus, which by the chain rule is
the probability of the entire dataset. Cross entropy is not a heuristic loss
chosen for convenience &mdash; it <em>is</em> the sequence probability, in log
space.</p>
</div>
<p>Log space is not optional. The joint probability of a 1000-token document is
around 10<sup>&minus;2000</sup>, which underflows any float; sums of logs do
not. The same concern reappears one level down, inside the softmax itself,
where you must subtract the maximum logit before exponentiating or
<code>exp</code> overflows to infinity.</p>

<h3>Perplexity, and what it is bounded by</h3>
<p>Cross entropy in nats is hard to feel. Exponentiate it and you get
<strong>perplexity</strong>, which has an interpretation: the effective number
of equally-likely options the model is choosing between.</p>
<div class='math'>PPL = exp(L)</div>

<div class='deriv'>
  <div class='deriv-head'>
    <span class='deriv-title'>Faded derivation: the uniform bound, and why perplexity is comparable only within a tokenizer</span>
    <button class='wbtn deriv-practice'>practice (hide all)</button>
    <button class='wbtn deriv-worked'>worked (show all)</button>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>1</span><span class='dstep-goal'>Compute the loss of a model that has learned nothing: uniform over V tokens.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>p<sub>t</sub>[x<sub>t</sub>] = 1/V &nbsp;&rArr;&nbsp; L = &minus;log(1/V) = log V &nbsp;&rArr;&nbsp; PPL = V</div><p>The uninformed baseline has perplexity exactly equal to the vocabulary size. This is the number every real model is measured against.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>2</span><span class='dstep-goal'>State the lower bound. What stops the loss reaching zero?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>L &ge; H(data) &nbsp;&nbsp;(equality iff the model equals the true distribution)</div><p>Cross entropy decomposes as H(p) + KL(p &#8214; q): the data&rsquo;s own entropy plus the model&rsquo;s divergence from it. Only the second term is trainable. Text is genuinely stochastic, so H &gt; 0 &mdash; this is the E term in the scaling law of Part 5.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>3</span><span class='dstep-goal'>Now change the tokenizer, keeping the text fixed. What happens to per-token perplexity?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>L<sub>token</sub> = """ + frac("total nats for the document", "number of tokens") + """</div><p>Merging more aggressively shrinks the denominator while the numerator &mdash; the information content of the document &mdash; is unchanged. So per-token loss <em>rises</em> even though the model got no worse.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>4</span><span class='dstep-goal'>Fix the comparison. What quantity is tokenizer-independent?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>bits per byte = """ + frac("L<sub>token</sub> &middot; N<sub>tokens</sub>", "N<sub>bytes</sub> &middot; ln 2") + """</div><p><strong>Normalize by bytes, not tokens, and perplexity numbers become comparable across tokenizers.</strong> This is why serious evaluations report bits-per-byte: a model can &ldquo;win&rdquo; on perplexity purely by tokenizing more coarsely.</p></div>
  </div>
  <div class='caption'>Step 4 is the one that bites in practice. Two papers
  reporting perplexity on the same dataset with different tokenizers are not
  comparable, and the difference can exceed the modelling improvement being
  claimed.</div>
</div>
""")

QUIZ_MATH_TOKEN = [
    {"question": "A model is uniform over a 50,257-token vocabulary. What is its perplexity?",
     "options": [
        {"text": "50,257",
         "correct": True,
         "explanation": "Uniform gives L = log V, so PPL = exp(log V) = V exactly. This identity is the sanity check the tutorial's test suite encodes."},
        {"text": "log(50,257) ≈ 10.82",
         "explanation": "That is the cross entropy in nats. Perplexity is its exponential."},
        {"text": "1, because every token is equally likely.",
         "explanation": "Perplexity 1 means total certainty — a model that always assigns probability 1 to the correct token."},
        {"text": "It depends on the sequence length.",
         "explanation": "Per-token perplexity is an average, so it is independent of how many tokens are averaged over."}]},
    {"question": "Team A reports perplexity 12.0 with a 32k vocabulary; Team B reports 15.0 with a 128k vocabulary, same test text. What can you conclude?",
     "options": [
        {"text": "Nothing yet — per-token perplexity is not comparable across tokenizers.",
         "correct": True,
         "explanation": "A coarser tokenizer produces fewer, harder tokens, raising per-token loss without the model being worse. Convert both to bits per byte before comparing."},
        {"text": "Team A's model is better by 3.0 perplexity.",
         "explanation": "This is exactly the invalid comparison step 4 of the derivation warns about — the denominators differ."},
        {"text": "Team B's model is better because it handles a larger vocabulary.",
         "explanation": "Vocabulary size alone says nothing about model quality; it changes the units the loss is measured in."},
        {"text": "They are equivalent, since perplexity is normalized.",
         "explanation": "It is normalized per token, and 'per token' means something different for each tokenizer."}]},
    {"question": "Why must you subtract the maximum logit before exponentiating in a softmax?",
     "options": [
        {"text": "exp overflows for large logits, and softmax is invariant to a constant shift.",
         "correct": True,
         "explanation": "exp(1000) is inf in float32, giving inf/inf = NaN. Since softmax(z) = softmax(z − c) exactly, subtracting the max costs nothing and bounds the largest exponent at exp(0) = 1."},
        {"text": "It makes the resulting probabilities sum to one.",
         "explanation": "The denominator guarantees normalization regardless of any shift — that is not what the subtraction is for."},
        {"text": "It centers the gradients, which speeds up convergence.",
         "explanation": "The output is mathematically identical after the shift, so gradients are unchanged. The motivation is purely numerical range."},
        {"text": "It prevents the loss from becoming negative.",
         "explanation": "Cross entropy of a normalized distribution is non-negative anyway, since probabilities are at most 1."}]},
    {"question": "In L = H(data) + KL(data ‖ model), which part can training reduce?",
     "options": [
        {"text": "Only the KL term; H is a property of the data.",
         "correct": True,
         "explanation": "The entropy of the text itself is a floor no model can go below — it reappears as the irreducible constant E in the Chinchilla loss form."},
        {"text": "Only H, since KL is fixed by the architecture.",
         "explanation": "Backwards: H is fixed by the data, KL is what the parameters control."},
        {"text": "Both, in proportion to model size.",
         "explanation": "No amount of capacity reduces the data's own entropy — that would mean predicting genuinely random continuations."},
        {"text": "Neither; cross entropy is minimized by the tokenizer.",
         "explanation": "The tokenizer changes the units of the loss, not the model's ability to reduce divergence."}]},
    {"question": "What does a BPE trainer do when no adjacent pair occurs more than once?",
     "options": [
        {"text": "It stops, because no merge would compress anything.",
         "correct": True,
         "explanation": "A pair appearing once yields no net saving, so training halts — which is why a small corpus cannot fill a large vocabulary. The tutorial has a test for exactly this."},
        {"text": "It merges an arbitrary pair to reach the target vocabulary size.",
         "explanation": "That would add symbols that never appear again, wasting embedding parameters."},
        {"text": "It falls back to character-level splitting.",
         "explanation": "The base vocabulary is already bytes; there is nothing finer to fall back to."},
        {"text": "It restarts with a different merge ordering.",
         "explanation": "Standard BPE is greedy and deterministic — it has no backtracking."}]},
]

# ===========================================================================
MATH_ARCH = ("""
<p>One block, four ideas: mix information across positions (attention), give
positions an identity (RoPE), keep activations well-scaled (RMSNorm), and do
the per-position computation (SwiGLU). Then count what it costs.</p>

<h3>Attention, derived</h3>
<p>Each position emits three vectors by linear projection: a query
q&nbsp;=&nbsp;xW<sub>Q</sub>, a key k&nbsp;=&nbsp;xW<sub>K</sub>, and a value
v&nbsp;=&nbsp;xW<sub>V</sub>. The compatibility of query i with key j is their
dot product; softmax over j turns compatibilities into weights that sum to
one; the output is the weighted average of values.</p>
<div class='math'>Attention(Q,K,V) = softmax( """ +
    frac("QK<sup>T</sup>", "&radic;d<sub>k</sub>") + """ + M ) V</div>
<p class='where'><b>Q, K, V</b> the stacked query, key and value matrices, one
row per position &middot; <b>d<sub>k</sub></b> the head dimension (the length
of a single query or key vector) &middot; <b>M</b> the causal mask</p>
<p>M is the causal mask: 0 where j &le; i and &minus;&infin; above the
diagonal, so position i cannot read the future. The mask must be added
<em>before</em> the softmax; zeroing weights afterwards would leave the rows
unnormalized.</p>
<div class='callout'>
<p><strong>Why this matters.</strong> Every dependency in the sequence is now
one hop away and every position is computed in parallel. The price is the
T&times;T matrix inside the softmax &mdash; the only term in the whole
architecture that is quadratic in sequence length.</p>
</div>

<h3>Why the 1/&radic;d<sub>k</sub> is not cosmetic</h3>
<p>The scaling factor is the single most-copied and least-explained line in
the architecture. It has a short derivation, and the derivation predicts
exactly what the widget below shows.</p>

<div class='deriv'>
  <div class='deriv-head'>
    <span class='deriv-title'>Faded derivation: the temperature of a dot product</span>
    <button class='wbtn deriv-practice'>practice (hide all)</button>
    <button class='wbtn deriv-worked'>worked (show all)</button>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>1</span><span class='dstep-goal'>Model q and k as independent vectors with zero-mean, unit-variance components. What is E[q&middot;k]?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&#120124;[q&middot;k] = &sum;<sub>i</sub> &#120124;[q<sub>i</sub>]&#120124;[k<sub>i</sub>] = 0</div><p>Independence lets the expectation factor; each factor is zero.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>2</span><span class='dstep-goal'>Now the variance. This is the step that matters.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>Var(q&middot;k) = &sum;<sub>i</sub> Var(q<sub>i</sub>k<sub>i</sub>) = &sum;<sub>i</sub> 1 = d<sub>k</sub></div><p>Variances of independent terms add. So the logits have standard deviation &radic;d<sub>k</sub> &mdash; they grow as the head gets wider.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>3</span><span class='dstep-goal'>What does a softmax do to inputs whose spread keeps growing?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><p>It saturates. Softmax is scale-sensitive: multiply all logits by c &gt; 1 and the distribution sharpens; as c &rarr; &infin; it becomes one-hot. With spread &radic;d<sub>k</sub>, a d<sub>k</sub>&nbsp;=&nbsp;128 head has logits several times larger than a d<sub>k</sub>&nbsp;=&nbsp;8 head, purely from dimensionality.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>4</span><span class='dstep-goal'>Why is saturation fatal for training specifically?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&part;softmax<sub>i</sub>/&part;z<sub>j</sub> = p<sub>i</sub>(&delta;<sub>ij</sub> &minus; p<sub>j</sub>)</div><p class='where'><b>p</b> the softmax output &middot; <b>z</b> its input logits &middot; <b>&delta;<sub>ij</sub></b> the Kronecker delta, 1 when i = j and 0 otherwise</p><p>When p is one-hot, every entry of that Jacobian is ~0: p<sub>i</sub>&nbsp;&asymp;&nbsp;0 kills most terms and p<sub>i</sub>&nbsp;&asymp;&nbsp;1 gives p(1&minus;p)&nbsp;&asymp;&nbsp;0. <strong>No gradient flows back through a saturated softmax, so the attention pattern cannot be learned.</strong></p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>5</span><span class='dstep-goal'>Choose the divisor that makes the logit variance independent of width.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>Var( """ + frac("q&middot;k", "&radic;d<sub>k</sub>") + """ ) = """ + frac("d<sub>k</sub>", "d<sub>k</sub>") + """ = 1</div><p>Dividing by &radic;d<sub>k</sub> holds the entering variance at 1 for every head width, so the softmax operates in the same regime whether the head is 8 or 128 wide. This is exactly the justification given in the original paper.</p></div>
  </div>
  <div class='caption'>Step 2 is the one to remember: <em>variances add, so dot
  products grow like &radic;d</em>. The same argument governs initialization
  schemes and is worth having in reflex memory.</div>
</div>

<p>This is not a reconstruction after the fact. The original paper gives the
same argument in a footnote, suspecting that &ldquo;for large values of
d<sub>k</sub>, the dot products grow large in magnitude, pushing the softmax
function into regions where it has extremely small gradients&rdquo; &mdash;
steps 2 through 4 above, in one sentence.</p>

<div class='widget' id='w-attn'>
  <div class='wctl' id='w-attn-predict'>
    <label>Predict first &mdash; with scaling OFF, as d<sub>k</sub> grows the attention becomes:</label>
    <button class='wbtn' data-pred='flatter'>flatter (more uniform)</button>
    <button class='wbtn' data-pred='sharper'>sharper (one-hot)</button>
    <button class='wbtn' data-pred='same'>unchanged</button>
  </div>
  <div class='wstat' id='w-attn-predfb' style='display:none'></div>
  <div class='wctl'>
    <label>d<sub>k</sub> <input type='range' id='w-attn-d' min='2' max='256' value='16'></label>
    <button class='wbtn' id='w-attn-scale'>scaling: 1/&radic;d&#8342; (on)</button>
  </div>
  <canvas id='w-attn-cv' style='width:100%;height:190px'></canvas>
  <div class='wstat' id='w-attn-stat'></div>
  <div class='caption'>Real attention over 12 positions with seeded random
  queries and keys; darker means more weight. Turn the scaling off and drag
  d<sub>k</sub>: the causal rows collapse onto a single bright cell and the
  entropy falls toward zero &mdash; the saturation predicted in step 3.</div>
</div>

<h3>Position without recurrence: RoPE</h3>
<p>Attention as written is permutation-equivariant: shuffle the input and the
outputs shuffle with it. Nothing so far knows about order. The modern answer
is <strong>rotary position embedding</strong>: rotate each consecutive pair of
coordinates of q and k by an angle proportional to the position.</p>
<div class='math'>&theta;<sub>i</sub> = base<sup>&minus;2i/d</sup>,&nbsp;&nbsp;
R(p) rotates pair i by p&middot;&theta;<sub>i</sub></div>
<p class='where'><b>d</b> the head dimension &middot; <b>i</b> indexes the d/2 coordinate pairs &middot; <b>base</b> a fixed constant (10,000 originally; 500,000 in Llama&nbsp;3) setting how slowly the angles decay &middot; <b>p</b> the token's position</p>
<p>The property that earns its place: because a rotation by
p&theta; followed by an inverse rotation by n&theta; is a rotation by
(p&minus;n)&theta;, the resulting attention score</p>
<div class='math'>&lang; R(m)q, R(n)k &rang; = f(q, k, m &minus; n)</div>
<p>depends only on the <em>relative</em> distance m&nbsp;&minus;&nbsp;n, even
though each vector was rotated by its absolute position. Rotations also
preserve norms, so nothing is rescaled. Both properties are asserted as tests
in the tutorial, and both fail loudly if you get the pairing wrong.</p>

<h3>The rest of the block: RMSNorm and SwiGLU</h3>
<p><strong>RMSNorm</strong> divides by the root-mean-square of the activation
vector and rescales, dropping LayerNorm&rsquo;s mean subtraction and bias:</p>
<div class='math'>RMSNorm(x) = """ + frac("x", "&radic;( (1/d)&sum;x<sub>i</sub>&sup2; + &epsilon; )") + """ &middot; g</div>
<p class='where'><b>x</b> the activation vector for one position &middot; <b>d</b> its length &middot; <b>g</b> a learned per-channel gain &middot; <b>&epsilon;</b> a small constant guarding against division by zero</p>
<p>The empirical finding is that re-centering was never doing the work &mdash;
only re-scaling was. Applying it <em>before</em> each sublayer (pre-norm)
rather than after leaves a clean residual path from input to loss, which is
what makes deep stacks trainable without careful warmup.</p>
<p><strong>SwiGLU</strong> replaces the two-matrix feed-forward network with a
gated three-matrix version:</p>
<div class='math'>SwiGLU(x) = ( swish(xW<sub>1</sub>) &odot; xW<sub>3</sub> ) W<sub>2</sub></div>
<p class='where'><b>W<sub>1</sub>, W<sub>3</sub></b> project up to the hidden width; <b>W<sub>2</sub></b> projects back down &middot; <b>&odot;</b> elementwise product &middot; <b>swish(z) = z&middot;&sigma;(z)</b>, a smooth gate</p>
<p>Three matrices instead of two means the hidden dimension is shrunk by 2/3
&mdash; the familiar 8/3&nbsp;d rather than 4d &mdash; to hold the parameter
count fixed, which is where that odd-looking constant in every modern config
file comes from.</p>

<h3>Counting what it costs</h3>
<p>Per layer, with hidden size d and feed-forward size d<sub>ff</sub>, the
forward pass costs (in FLOPs per token, counting a multiply-add as 2):</p>
<div class='math'>6d&sup2; <span class='dim'>(QKV)</span> + 2d&sup2;
<span class='dim'>(output)</span> + 4Ld <span class='dim'>(scores and
values)</span> + 2&middot;n<sub>mat</sub>&middot;d&middot;d<sub>ff</sub>
<span class='dim'>(FFN)</span></div>
<p class='where'><b>d</b> the model width &middot; <b>d<sub>ff</sub></b> the feed-forward hidden width &middot; <b>L</b> the context length &middot; <b>n<sub>mat</sub></b> the number of FFN matrices (2 classic, 3 for SwiGLU)</p>
<p>Set d<sub>ff</sub>&nbsp;=&nbsp;4d with two matrices and the quadratic term
overtakes everything else only when 4Ld &gt; 24d&sup2;, that is
<strong>L&nbsp;&gt;&nbsp;6d</strong>.</p>

<div class='widget'>
__FIG_FLOPS__
<div class='caption'>Attention&rsquo;s share of block FLOPs, computed exactly
from the expression above. For a 4096-wide model the crossover is at about
24,000 tokens of context. Below that, <em>the feed-forward network dominates
and attention is not the bottleneck</em> &mdash; which is why the first wave
of efficient-attention work often failed to speed anything up in practice.</div>
</div>
""")

QUIZ_MATH_ARCH = [
    {"question": "Two random query/key vectors have iid unit-variance components. What is the standard deviation of their dot product before scaling?",
     "options": [
        {"text": "√d_k",
         "correct": True,
         "explanation": "Variance of a sum of d_k independent unit-variance products is d_k, so the standard deviation is √d_k. This is step 2 of the derivation and the entire reason for the scaling factor."},
        {"text": "d_k",
         "explanation": "That is the variance, not the standard deviation — you need its square root."},
        {"text": "1, since each component has unit variance.",
         "explanation": "Each *term* has unit variance, but d_k of them are summed, and variances add."},
        {"text": "1/√d_k",
         "explanation": "That is the correction factor applied to fix the problem, not the size of the problem."}]},
    {"question": "Why is a saturated softmax specifically fatal during training rather than merely inaccurate?",
     "options": [
        {"text": "Its Jacobian p_i(δ_ij − p_j) vanishes, so no gradient reaches the attention weights.",
         "correct": True,
         "explanation": "At a one-hot p every entry of the Jacobian is near zero, so the attention pattern receives no learning signal at all — it is frozen wherever initialization put it."},
        {"text": "It produces NaNs from exponentiating large logits.",
         "explanation": "That is a separate numerical issue fixed by subtracting the max; a saturated softmax can be perfectly finite and still untrainable."},
        {"text": "It makes the output values exceed the range of the value vectors.",
         "explanation": "The output stays a convex combination of values regardless of sharpness, so it can never leave their range."},
        {"text": "It breaks the causal mask.",
         "explanation": "Masking is applied to the logits before the softmax and is unaffected by how sharp the resulting distribution is."}]},
    {"question": "A model has d_model = 8192 and d_ff = 4d with two FFN matrices. Above roughly what context length does attention exceed all other FLOPs in a block?",
     "options": [
        {"text": "About 49,000 tokens",
         "correct": True,
         "explanation": "The crossover is at L = 6d = 6 × 8192 = 49,152. Below that the feed-forward network dominates, which is why attention is often not the bottleneck people assume it is."},
        {"text": "About 8,000 tokens",
         "explanation": "That is d itself; the crossover is six times larger because the non-attention terms total 24d² against attention's 4Ld."},
        {"text": "About 2,000 tokens",
         "explanation": "Far too low — at 2k context attention is a small minority of the compute for a model this wide."},
        {"text": "About 500,000 tokens",
         "explanation": "An order of magnitude too high; that would require d ≈ 83,000."}]},
    {"question": "What property makes RoPE encode relative position despite rotating by absolute position?",
     "options": [
        {"text": "Composing a rotation by mθ with the inverse of nθ leaves a rotation by (m−n)θ.",
         "correct": True,
         "explanation": "The inner product ⟨R(m)q, R(n)k⟩ therefore depends only on m−n. Absolute rotations go in; relative distance comes out of the dot product."},
        {"text": "The rotation angles are learned during training.",
         "explanation": "RoPE's angles are fixed by the formula base^(−2i/d); nothing about it is learned."},
        {"text": "Position embeddings are added to the values as well as queries and keys.",
         "explanation": "RoPE touches only queries and keys — values are left alone, which is part of why it composes cleanly with a KV cache."},
        {"text": "It normalizes each vector to unit length first.",
         "explanation": "Rotations preserve norms rather than imposing them; no normalization step is involved."}]},
    {"question": "Why is the SwiGLU hidden dimension conventionally 8/3·d rather than 4d?",
     "options": [
        {"text": "It uses three matrices instead of two, so 2/3 the width keeps parameters equal.",
         "correct": True,
         "explanation": "4d × (2/3) = 8/3·d. The odd constant in modern configs is pure bookkeeping to make the gated variant a fair swap for the classic FFN."},
        {"text": "8/3 is the ratio that minimizes the attention/FFN FLOP crossover.",
         "explanation": "The crossover follows from the ratio rather than motivating it, and nobody chose the width to tune that."},
        {"text": "It matches the head dimension of 128 used in most models.",
         "explanation": "Head dimension is independent of the FFN width; the two are set separately."},
        {"text": "Gated activations require a hidden size divisible by 3.",
         "explanation": "There is no divisibility requirement — the value comes from a parameter-count equivalence."}]},
]

# ===========================================================================
MATH_TRAIN = ("""
<p>The architecture is settled; now it has to be fitted, and at this scale the
binding constraints are arithmetic and memory rather than statistics. Three
numbers govern everything: how many FLOPs a step costs, how many bytes a
parameter occupies, and how to split a fixed budget between model and data.</p>

<h3>AdamW, and why the W matters</h3>
<p>Adam keeps running estimates of the gradient&rsquo;s first and second
moments and steps by their ratio:</p>
<div class='math'>m<sub>t</sub> = &beta;&#8321;m<sub>t&minus;1</sub> + (1&minus;&beta;&#8321;)g<sub>t</sub>
&nbsp;&nbsp;&nbsp; v<sub>t</sub> = &beta;&#8322;v<sub>t&minus;1</sub> + (1&minus;&beta;&#8322;)g<sub>t</sub>&sup2;</div>
<div class='math'>&theta;<sub>t</sub> = &theta;<sub>t&minus;1</sub> &minus; &eta;
""" + frac("m&#770;<sub>t</sub>", "&radic;v&#770;<sub>t</sub> + &epsilon;") + """
&minus; &eta;&lambda;&theta;<sub>t&minus;1</sub></div>
<p class='where'><b>g<sub>t</sub></b> the gradient &middot; <b>m, v</b> running estimates of its mean and squared magnitude &middot; <b>&beta;&#8321;, &beta;&#8322;</b> their decay rates (0.9, 0.95&ndash;0.999) &middot; <b>m&#770;, v&#770;</b> the same after bias correction &middot; <b>&eta;</b> the learning rate &middot; <b>&lambda;</b> the weight decay &middot; <b>&epsilon;</b> a small constant (~10<sup>&minus;8</sup>) keeping the denominator finite</p>
<p>The ratio makes the step <em>scale-free</em>: multiply every gradient by a
million and the first step is unchanged, because the numerator and denominator
scale together. That is why Adam needs no per-layer learning-rate tuning.</p>
<div class='callout'>
<p><strong>Why the last term is separate.</strong> Classic L2 regularization
adds &lambda;&theta; to the gradient, so it passes through the
1/&radic;v&#770; normalizer and gets <em>divided by the gradient magnitude</em>
&mdash; parameters with large gradients end up barely regularized. AdamW
applies the decay directly to the weights instead, decoupled from the adaptive
scaling. That one change is the difference between Adam and the optimizer
every large model is actually trained with.</p>
</div>

<div class='widget'>
__FIG_LR__
<div class='caption'>The schedules themselves, exactly as implemented. Warmup
exists because the second-moment estimate v is meaningless for the first few
steps &mdash; a full-size step on a near-zero denominator is how runs diverge
in the first hundred iterations. WSD holds a plateau and decays late, which
lets you branch a run at any point rather than committing to a horizon in
advance.</div>
</div>

<h3>The 6ND rule</h3>
<div class='deriv'>
  <div class='deriv-head'>
    <span class='deriv-title'>Faded derivation: why training costs ≈ 6 FLOPs per parameter per token</span>
    <button class='wbtn deriv-practice'>practice (hide all)</button>
    <button class='wbtn deriv-worked'>worked (show all)</button>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>1</span><span class='dstep-goal'>Count the forward cost of one weight in one matrix multiply, per token.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>1 multiply + 1 add = 2 FLOPs</div><p>Every weight participates in exactly one multiply-accumulate per token, so the forward pass costs 2N FLOPs per token for N parameters.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>2</span><span class='dstep-goal'>The backward pass computes two different gradients. Name them.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><p>The gradient with respect to the layer&rsquo;s <em>inputs</em> (to keep propagating backwards) and with respect to its <em>weights</em> (to update them). Each is a matrix product of the same shape class as the forward one.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>3</span><span class='dstep-goal'>Total the three passes.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>2N <span class='dim'>forward</span> + 2N <span class='dim'>grad wrt input</span> + 2N <span class='dim'>grad wrt weight</span> = 6N per token</div><p><strong>Over D tokens the run costs C &asymp; 6ND FLOPs.</strong> Backward is twice forward, which is the fact worth remembering.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>4</span><span class='dstep-goal'>What does this expression ignore, and when does that bite?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><p>The attention score computation, which has no parameters and so contributes nothing to N, yet costs 4Ld per layer per token. From the previous part that stays a minority until L &gt; 6d. <em>6ND is an excellent approximation at ordinary context lengths and an increasingly bad one at very long ones.</em></p></div>
  </div>
</div>

<h3>Splitting the budget: Chinchilla</h3>
<p>Given C FLOPs, should you train a big model on few tokens or a small model
on many? Fit the loss as a parametric form and it becomes a constrained
optimization with a closed-form answer:</p>
<div class='math'>L(N, D) = E + """ + frac("A", "N<sup>&alpha;</sup>") + """ + """
    + frac("B", "D<sup>&beta;</sup>") + """
&nbsp;&nbsp;&nbsp; subject to &nbsp; C = 6ND</div>
<p class='where'><b>N</b> parameters &middot; <b>D</b> training tokens &middot; <b>C</b> the compute budget in FLOPs &middot; <b>E</b> the irreducible entropy of the text &middot; <b>A, B</b> fitted scale constants &middot; <b>&alpha;, &beta;</b> fitted exponents saying how fast loss falls with parameters and with data</p>
<p>E is the irreducible entropy of the text &mdash; the floor from Part 3.
Substituting the constraint and differentiating gives</p>
<div class='math'>N* &prop; C<sup>&beta;/(&alpha;+&beta;)</sup>,&nbsp;&nbsp;&nbsp;
D* &prop; C<sup>&alpha;/(&alpha;+&beta;)</sup></div>
<p>so the two exponents sum to one, as they must for the constraint to hold.
When &alpha;&nbsp;&asymp;&nbsp;&beta; both exponents are &frac12;: model and
data should scale <em>together</em>, in a fixed ratio. That is the Chinchilla
result, and it overturned a previous generation of models that were far too
large for the data they saw.</p>

<div class='widget'>
__FIG_SCALING__
<div class='caption'>Iso-FLOP curves from the published parametric fit: for
each budget, sweeping model size traces a U whose minimum is the
compute-optimal point. The floor E = 1.69 is the entropy term no amount of
compute removes.</div>
</div>

<div class='callout warn'>
<p><strong>An honest wrinkle, checkable in one line of code.</strong> If you
take Hoffmann et al.&rsquo;s published Approach-3 constants and actually
minimize them under the constraint, at their own budget
(C&nbsp;=&nbsp;5.9&times;10<sup>23</sup>) you get roughly a
<strong>33B model on 3.0T tokens</strong> &mdash; not the 70B on 1.4T they
recommended and trained. The exponents &alpha;&nbsp;=&nbsp;0.34 and
&beta;&nbsp;=&nbsp;0.28 differ enough that the optimal tokens-per-parameter
ratio <em>grows</em> with compute rather than staying near 20. The replication
by """ + cite("Chinchilla Scaling: A replication attempt", "Besiroglu et al.") + """
re-estimates the constants and recovers &alpha;&nbsp;&asymp;&nbsp;&beta;, which
reproduces both the 20:1 rule and the model they actually built. The figure
below plots both.</p>
</div>

<div class='widget'>
__FIG_RATIO__
<div class='caption'>Optimal tokens-per-parameter under each fit, with the
Chinchilla model itself marked. The re-estimated constants pass essentially
through it; the published ones do not. <strong>The famous 20:1 is a fitted
number, not a law of nature</strong> &mdash; and it is inference cost, not
this curve, that pushes production models far past it.</div>
</div>

<div class='widget' id='w-budget'>
  <div class='wctl'>
    <label>compute budget (log&#8321;&#8320; FLOPs) <input type='range' id='w-budget-c' min='190' max='260' value='230'></label>
  </div>
  <div id='w-budget-out'></div>
  <div class='caption'>Both fits, minimized live under C = 6ND. Slide up to a
  frontier-scale budget and watch the two recommendations diverge by more than
  a factor of two in model size.</div>
</div>

<h3>The memory wall</h3>
<p>FLOPs are rarely what stops you; bytes are. A mixed-precision AdamW run
holds, per parameter: 2 bytes of bf16 weights, 2 of gradients, and an fp32
master copy plus two optimizer moments at 4 bytes each.</p>
<div class='math'>2 + 2 + 4 + 4 + 4 = <strong>16 bytes per parameter</strong></div>
<p>A 70B model therefore needs about 1.1 TB before a single activation is
stored &mdash; roughly fourteen 80&nbsp;GB accelerators just to hold state.
""" + cite("ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", "ZeRO") + """
attacks this by sharding rather than replicating: stage 1 partitions optimizer
state, stage 2 adds gradients, stage 3 adds the parameters themselves, taking
per-device state from 16&Psi; down to 16&Psi;/N<sub>d</sub> at the cost of
extra collective communication.</p>

<h3>FlashAttention: the same maths, different memory traffic</h3>
<p>Attention is memory-bound, not compute-bound. The naive implementation
writes the T&times;T score matrix to HBM, reads it back for the softmax,
writes it again, and reads it once more to multiply by V.
""" + cite("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "FlashAttention") + """
never materializes it: it tiles Q, K and V into blocks that fit in on-chip
SRAM and carries a running maximum and running sum so the softmax can be
completed incrementally &mdash; the online-softmax trick of
""" + cite("Online normalizer calculation for softmax", "Milakov &amp; Gimelshein") + """.</p>
<p>The output is bit-for-bit the same attention. What changes is HBM traffic,
and since SRAM bandwidth is roughly an order of magnitude higher than HBM, the
wall-clock effect is large. <strong>It is an exact algorithm that is faster
purely because of where the bytes live</strong> &mdash; the clearest example
in the field of arithmetic intensity mattering more than FLOP count.</p>
""")

QUIZ_MATH_TRAIN = [
    {"question": "You train a 7B-parameter model on 2T tokens. Roughly what compute does this take?",
     "options": [
        {"text": "≈8.4 × 10²² FLOPs",
         "correct": True,
         "explanation": "C ≈ 6ND = 6 × 7e9 × 2e12 = 8.4e22. This one multiplication is the most useful arithmetic in the field — it converts any model card into a compute figure."},
        {"text": "≈1.4 × 10²² FLOPs",
         "explanation": "That is 1·N·D, forgetting the factor of 6 from forward plus the two backward products."},
        {"text": "≈2.8 × 10²² FLOPs",
         "explanation": "That uses a factor of 2 — the forward pass alone. Backward costs twice as much again."},
        {"text": "≈5.0 × 10²⁵ FLOPs",
         "explanation": "Three orders of magnitude too large; this would be a frontier-scale run rather than a 7B model."}]},
    {"question": "Why is the factor 6 rather than 2 in C ≈ 6ND?",
     "options": [
        {"text": "Backward computes gradients for both inputs and weights, costing twice the forward pass.",
         "correct": True,
         "explanation": "Forward is 2N per token; the two backward matrix products add 2N each, giving 6N total. This is step 3 of the derivation."},
        {"text": "Three tokens are processed per parameter update on average.",
         "explanation": "Token count is D and is already a separate factor; the 6 has nothing to do with batching."},
        {"text": "Mixed precision requires three copies of every weight.",
         "explanation": "That is a memory cost (the 16 bytes/parameter figure), not a FLOP cost."},
        {"text": "Attention triples the cost of every matrix multiply.",
         "explanation": "Attention's cost is a separate additive term with no parameters, and 6ND explicitly excludes it."}]},
    {"question": "What does AdamW change relative to Adam with L2 regularization?",
     "options": [
        {"text": "Decay is applied to the weights directly instead of passing through the adaptive normalizer.",
         "correct": True,
         "explanation": "Under Adam, L2 added to the gradient gets divided by √v̂, so heavily-updated parameters are barely regularized. Decoupling makes the decay uniform, which is why AdamW became standard."},
        {"text": "It removes the second-moment estimate entirely.",
         "explanation": "v is still maintained and still normalizes the step — that is what makes Adam scale-free."},
        {"text": "It applies weight decay only during warmup.",
         "explanation": "Decay runs throughout training; the schedule is a separate concern."},
        {"text": "It replaces the exponential moving averages with plain sums.",
         "explanation": "Both moments remain exponential moving averages with the usual β₁, β₂ decay rates."}]},
    {"question": "A 13B model trains with mixed-precision AdamW. Roughly how much memory does the persistent state need, before activations?",
     "options": [
        {"text": "≈208 GB",
         "correct": True,
         "explanation": "16 bytes per parameter × 13e9 = 2.08e11 bytes ≈ 208 GB — already more than two 80 GB accelerators for state alone, which is why sharding exists."},
        {"text": "≈26 GB",
         "explanation": "That counts only the 2-byte bf16 weights and ignores gradients, the fp32 master copy, and both optimizer moments."},
        {"text": "≈52 GB",
         "explanation": "That is weights plus gradients only (4 bytes/param); the 12 bytes of fp32 optimizer state dominate."},
        {"text": "≈13 GB",
         "explanation": "This would be one byte per parameter — below even pure int8 weights, and it ignores all training state."}]},
    {"question": "FlashAttention is faster than a naive implementation. What does it change?",
     "options": [
        {"text": "The volume of HBM traffic; the computed result is identical.",
         "correct": True,
         "explanation": "It is exact attention, tiled so the T×T matrix never reaches HBM, with online softmax keeping running max and sum. The speedup is entirely a memory-hierarchy effect."},
        {"text": "It approximates the softmax with a low-rank factorization.",
         "explanation": "That describes approximate-attention methods like Performer; FlashAttention deliberately gives up nothing."},
        {"text": "It reduces the asymptotic FLOP count below quadratic.",
         "explanation": "The FLOP count is unchanged and still quadratic in sequence length."},
        {"text": "It sparsifies the attention matrix to skip masked entries.",
         "explanation": "Causal masking does save work in good implementations, but that is not the mechanism the paper introduces."}]},
]

# ===========================================================================
MATH_SOTA = ("""
<p>Four ideas that separate a 2026 model from the 2020 template. Each is one
equation changed &mdash; a routing rule, a cache layout, or a training
objective &mdash; and each is shown here as that equation rather than as a
description of what the paper claims.</p>

<h3>Mixture of experts: parameters without proportional FLOPs</h3>
<p>Replace the single feed-forward network with E of them plus a router. The
router scores experts, keeps the top k, and mixes their outputs:</p>
<div class='math'>y = &sum;<sub>i&isin;TopK(g(x))</sub> g<sub>i</sub>(x) &middot; FFN<sub>i</sub>(x),
&nbsp;&nbsp; g(x) = softmax(xW<sub>r</sub>)</div>
<p class='where'><b>E</b> the number of experts &middot; <b>k</b> how many are used per token &middot; <b>W<sub>r</sub></b> the router's projection &middot; <b>g<sub>i</sub>(x)</b> the gate weight expert i receives</p>
<p>Total parameters scale with E; FLOPs per token scale with k. """ +
    cite("DeepSeek-V3 Technical Report", "DeepSeek-V3") + """ carries 671B
parameters but activates 37B per token &mdash; an 18&times; gap between what
the model knows and what any single token pays for.</p>
<p>The failure mode is routing collapse: nothing in the loss stops the router
from sending everything to one expert. The classical fix adds an auxiliary
load-balancing loss, which fights the language-modelling objective. The newer
approach adds a per-expert bias to the routing scores and adjusts it based on
observed load, balancing without touching the gradient of the main loss at
all.</p>

<h3>Shrinking the KV cache: MQA, GQA, MLA</h3>
<p>During generation every past key and value must be kept. For multi-head
attention that is</p>
<div class='math'>2 &middot; n<sub>layers</sub> &middot; n<sub>heads</sub>
&middot; d<sub>head</sub> &middot; L &middot; 2 bytes</div>
<p class='where'>the leading <b>2</b> counts keys and values &middot; <b>n<sub>layers</sub>, n<sub>heads</sub>, d<sub>head</sub></b> the model's depth, head count and head width &middot; <b>L</b> the tokens cached so far &middot; the trailing <b>2 bytes</b> is bf16</p>
<p>which grows linearly with context and quickly exceeds the weights
themselves. """ + cite("Fast Transformer Decoding: One Write-Head is All You Need", "MQA") +
    """ shares one key/value head across all query heads &mdash; a large saving
and a real quality loss. """ +
    cite("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "GQA") +
    """ interpolates: g groups of query heads share a key/value head, recovering
most of the quality at a fraction of the cache. <strong>Multi-head latent
attention</strong> instead caches a low-rank latent that keys and values are
reconstructed from, decoupling cache size from head count entirely.</p>

<div class='widget' id='w-kv'>
  <div class='wctl'>
    <label>context (log&#8321;&#8320; tokens) <input type='range' id='w-kv-ctx' min='20' max='60' value='40'></label>
    <label>batch <input type='range' id='w-kv-b' min='1' max='64' value='1'></label>
  </div>
  <div id='w-kv-out'></div>
  <div class='caption'>The formula above, evaluated live for a 70B-class
  configuration. Push the context to 100k with a batch of 32 and multi-head
  attention needs more memory for the cache than for the model.</div>
</div>

<div class='widget'>
__FIG_KV__
<div class='caption'>The same three variants across four orders of magnitude
of context. The gap is not a constant factor to be engineered away later
&mdash; it decides whether long-context serving is possible on a given
machine at all.</div>
</div>

<h3>DPO: preference learning without the RL</h3>
<p>The RLHF pipeline of """ +
    cite("Training language models to follow instructions with human feedback", "Ouyang et al.") +
    """ trains a reward model, then optimizes the policy against it with PPO and
a KL penalty:</p>
<div class='math'>max<sub>&pi;</sub> &#120124;[ r(x,y) ] &minus; &beta;
&middot; KL( &pi; &#8214; &pi;<sub>ref</sub> )</div>
<p class='where'><b>&pi;</b> the policy being trained &middot; <b>&pi;<sub>ref</sub></b> the frozen starting model &middot; <b>r(x,y)</b> the learned reward for response y to prompt x &middot; <b>&beta;</b> how hard the KL term pulls &pi; back toward &pi;<sub>ref</sub></p>
<p>""" + cite("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "DPO") + """
observes that this objective has a closed-form optimum,
&pi;*(y|x)&nbsp;&prop;&nbsp;&pi;<sub>ref</sub>(y|x)&nbsp;exp(r(x,y)/&beta;).
Invert it to express the reward in terms of the policy, substitute into the
preference likelihood, and the reward model cancels out. What remains is a
plain classification loss on preference pairs:</p>
<div class='math'>&#8466;<sub>DPO</sub> = &minus;log &sigma;( &beta; log """ +
    frac("&pi;(y<sub>w</sub>|x)", "&pi;<sub>ref</sub>(y<sub>w</sub>|x)") + """
&minus; &beta; log """ + frac("&pi;(y<sub>l</sub>|x)", "&pi;<sub>ref</sub>(y<sub>l</sub>|x)") + """ )</div>
<p class='where'><b>y<sub>w</sub>, y<sub>l</sub></b> the preferred and rejected responses to prompt x &middot; <b>&sigma;</b> the logistic function &middot; <b>&beta;</b> the same KL strength as above, here acting as a temperature on the log-ratios</p>
<div class='callout'>
<p><strong>Why this matters.</strong> No reward model, no sampling loop, no
value network &mdash; just supervised learning on pairs. The implicit reward
is the log-ratio of the policy to its reference, which is why the reference
model must be kept around and frozen.</p>
</div>

<h3>GRPO and verifiable rewards</h3>
<p>PPO needs a learned value network to compute advantages, which is a second
model of the same size. """ +
    cite("DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "GRPO") + """
removes it: sample a group of G completions for the same prompt and let the
group be its own baseline.</p>
<div class='math'>A&#770;<sub>i</sub> = """ +
    frac("r<sub>i</sub> &minus; mean(r&#8321;..r<sub>G</sub>)",
         "std(r&#8321;..r<sub>G</sub>)") + """</div>
<p class='where'><b>G</b> the number of completions sampled for one prompt &middot; <b>r<sub>i</sub></b> the reward for completion i &middot; <b>A&#770;<sub>i</sub></b> its advantage, standardized within the group</p>
<p>When the reward is a verifier rather than a learned model &mdash; does the
program pass its tests, is the final answer correct &mdash; the whole reward
model disappears too. That is the recipe behind """ +
    cite("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", "DeepSeek-R1") + """.</p>
<div class='callout warn'>
<p><strong>Genuinely unresolved.</strong> """ +
    cite("Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?",
         "Yue et al. (2025)") + """ argue that RLVR sharpens sampling toward
solutions the base model could already produce rather than creating new
capability: at large pass@k the base model can match or exceed the RL-trained
one. The counter-argument is that reliably surfacing a rare correct answer is
itself the capability that matters in deployment. Both sides agree on the
measurements and disagree on what &ldquo;new capability&rdquo; means &mdash;
worth watching rather than settling.</p>
</div>
""")

QUIZ_MATH_SOTA = [
    {"question": "A MoE model has 671B total parameters and activates 37B per token. What does this buy?",
     "options": [
        {"text": "Capacity scaling with total parameters at a FLOP cost set by the activated subset.",
         "correct": True,
         "explanation": "Memory and knowledge scale with the full 671B; per-token compute tracks the 37B. The costs paid are memory footprint and routing complexity, not FLOPs."},
        {"text": "An 18× reduction in the memory needed to serve the model.",
         "explanation": "The opposite — all 671B parameters must be resident, which is precisely the deployment difficulty of MoE."},
        {"text": "An 18× longer usable context window.",
         "explanation": "Context length is governed by attention and the KV cache; routing has no bearing on it."},
        {"text": "Elimination of the need for load balancing across experts.",
         "explanation": "Routing collapse is the central MoE failure mode and needs an explicit fix, whether an auxiliary loss or a bias-adjustment scheme."}]},
    {"question": "In the DPO derivation, what exactly cancels?",
     "options": [
        {"text": "The reward model, once the reward is rewritten as a log-ratio of policy to reference.",
         "correct": True,
         "explanation": "The KL-regularized optimum π* ∝ π_ref·exp(r/β) inverts to give r in terms of π and π_ref; substituting into the Bradley-Terry preference likelihood leaves a loss containing only the two policies."},
        {"text": "The reference policy, leaving a loss on the trained policy alone.",
         "explanation": "The reference is essential and must stay frozen — it defines the implicit reward's zero point."},
        {"text": "The KL penalty, which is why DPO needs no β.",
         "explanation": "β survives as the temperature multiplying the log-ratios and is DPO's main hyperparameter."},
        {"text": "The preference labels, replaced by a verifiable reward.",
         "explanation": "DPO trains directly on preference pairs; verifiable rewards belong to the RLVR/GRPO line instead."}]},
    {"question": "What does GRPO remove relative to PPO, and what replaces it?",
     "options": [
        {"text": "The value network; a group of sampled completions supplies the baseline.",
         "correct": True,
         "explanation": "Advantages are computed by normalizing rewards within the group, eliminating a second model of comparable size — the memory saving the paper emphasizes."},
        {"text": "The policy gradient; a supervised loss replaces it.",
         "explanation": "That is DPO's approach. GRPO remains a policy-gradient method with sampling."},
        {"text": "The KL penalty; group normalization makes it unnecessary.",
         "explanation": "GRPO implementations typically retain a KL term against the reference policy."},
        {"text": "The reward model; group voting determines correctness.",
         "explanation": "The reward can be a model or a verifier — that choice is orthogonal to GRPO's advantage estimator."}]},
    {"question": "A 70B-class model with 80 layers, 64 heads, head dim 128, bf16. Roughly how large is the MHA KV cache for 100k tokens, one sequence?",
     "options": [
        {"text": "≈262 GB",
         "correct": True,
         "explanation": "2 × 80 × 64 × 128 × 2 bytes = 2.62 MB per token; × 100,000 ≈ 262 GB — several times one accelerator, and far more than the model's own weights. This is the entire motivation for GQA and MLA."},
        {"text": "≈2.6 GB",
         "explanation": "That is the cache for about 1,000 tokens; the per-token figure is 2.62 MB, so 100k tokens is a hundred times larger."},
        {"text": "≈33 GB",
         "explanation": "That is the GQA-with-8-groups figure — an eighth of the multi-head cost, since 8 KV groups replace 64 KV heads."},
        {"text": "≈140 GB",
         "explanation": "That is roughly the bf16 weights of a 70B model. At this context the cache is larger than the weights, which is the point of the widget."}]},
    {"question": "What is the actual disagreement in the RLVR capability debate?",
     "options": [
        {"text": "Whether reliably surfacing an answer the base model could rarely sample counts as new capability.",
         "correct": True,
         "explanation": "Both sides accept the pass@k measurements. The dispute is definitional and practical — sharpening a distribution versus extending its support."},
        {"text": "Whether RLVR improves benchmark scores at all.",
         "explanation": "Score improvements are not disputed by either side."},
        {"text": "Whether verifiable rewards can be computed for mathematics.",
         "explanation": "Verification for maths and code is the least controversial part of the setup."},
        {"text": "Whether GRPO is more memory-efficient than PPO.",
         "explanation": "That is a settled implementation fact, unrelated to the capability question."}]},
]

# ===========================================================================
CONCEPT_MAP = ("""
<p>Every idea on this page and how it connects. Hover a node to isolate its
edges; click for a one-breath recap and a link back to the section that derives
it. The outlined nodes are the load-bearing hubs.</p>
<div class='widget' id='w-cmap'>
  <svg viewBox='0 0 740 360' style='width:100%;height:auto'></svg>
  <div class='wstat' id='w-cmap-info'></div>
</div>
<p>The self-test: <strong>every edge should be one sentence you can produce
out loud.</strong> &ldquo;Cross entropy connects to Chinchilla allocation
because&nbsp;&hellip;&rdquo; If an edge is silent, that is the section to
reread.</p>
""")

KEEP_LEARNING = ("""
<p>Reading this page once will not keep it. Four things will.</p>

<h3>Spaced review (retention)</h3>
<p>The <a href='""" + DATE + """-language-modelling-review.html'>review deck</a>
runs the whole quiz bank as Leitner boxes with 1/3/7/14/30-day intervals: a
first-try correct answer promotes a card, a miss demotes it and brings it back
later in the same session. An <a href='language-modelling.apkg'>Anki deck</a>
is there too if you already have a review habit.</p>

<h3>Build it yourself (transfer)</h3>
<p>The <a href='https://github.com/raghuramshankar/learning-with-llms/tree/main/tutorials/language-modelling'>tutorial</a>
is twelve numpy functions with 32 failing tests: BPE, cross entropy, RMSNorm,
SwiGLU, RoPE, causal attention, AdamW, sampling, and the FLOP accounting. The
two integration tests are the point &mdash; one builds an induction head by
hand and shows it retrieving the right token, the other shows your own
tokenizer paying for itself in bits per byte.</p>

<h3>Teach it back (generation)</h3>
<div class='callout'>
<p>Paste this into a fresh Claude session. Explaining is where you find out
what you actually know.</p>
<pre data-copy>I have just worked through an explainer on language modelling from scratch
(tokenization, the cross-entropy objective, attention and the transformer
block, AdamW, the 6ND rule and Chinchilla allocation, the memory wall,
FlashAttention, MoE, GQA/MLA, DPO and GRPO).

Play a curious student. Ask me to explain ONE core idea at a time, starting
with the objective and moving to architecture, then scale, then alignment.
After each answer, probe with a "why" follow-up. Do not explain anything to me
unless I am stuck twice, and then only hint.

When we have covered six ideas, grade me on mechanism, formulas and caveats,
list my weak spots, and write three new questions targeting exactly those.</pre>
</div>

<h3>Going deeper (the sources)</h3>
<p><strong>Stanford CS336, &ldquo;Language Modeling from Scratch&rdquo;</strong>
is the direct inspiration and the natural next step: five assignments that
build a tokenizer and transformer, a Triton FlashAttention-2, a scaling-law
fit, a Common Crawl pipeline, and an RL fine-tune. Reach for it when you want
the implementation rather than the derivation.</p>
<p>The <a href='""" + DATE + """-lm-cheatsheet.html'>cheat sheet</a> is every
formula on one printable page.</p>
""")

# ===========================================================================
PAPERS = ("""
<p>Everything cited, why it mattered, and how it was checked.</p>

<div class='callout'>
<p><strong>Survey methodology.</strong> Papers were verified in three ways,
recorded per entry in the build: a direct arXiv Atom API round-trip; fetching
the paper&rsquo;s own arxiv.org abstract page and reading the title, first
author and submission date off it; or verification by a survey subagent whose
method is recorded in <code>topics/language-modelling/survey/</code>. The API
rate-limited this session heavily, which is why the second route carries much
of the load. <strong>No identifier on this page was written from memory</strong>
&mdash; <code>build_spec.py</code> renders citations from
<code>survey/verified.json</code> and marks anything unconfirmed as unverified
rather than guessing. Access date: """ + DATE + """. The maths follows the
notation of the original papers where possible. Errors of interpretation are
this page&rsquo;s, not theirs.</p>
</div>

<h3>The foundations</h3>
<p>""" + cite("Attention Is All You Need", "Vaswani et al. (2017)") + """
removed recurrence entirely, which is what let sequence models saturate an
accelerator; the 1/&radic;d<sub>k</sub> derived in Part 4 is their footnote.
""" + cite("Language Models are Few-Shot Learners", "Brown et al. (2020)") + """
showed that scale alone turns a next-token predictor into a system you can
instruct without gradient updates. """ +
    cite("Neural Machine Translation of Rare Words with Subword Units", "Sennrich et al. (2015)") + """
brought BPE from compression into NLP, ending the unknown-word problem, and
""" + cite("SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", "Kudo &amp; Richardson (2018)") + """
made it language-agnostic and reversible.</p>

<h3>The modern block</h3>
<p>""" + cite("Root Mean Square Layer Normalization", "RMSNorm") + """ showed
re-centering was never load-bearing; """ +
    cite("RoFormer: Enhanced Transformer with Rotary Position Embedding", "RoPE") + """
made relative position fall out of an absolute rotation; """ +
    cite("GLU Variants Improve Transformer", "Shazeer (2020)") + """ found the
gated feed-forward variant that every current model uses, and with it the 8/3
width convention. On the cache side, """ +
    cite("Fast Transformer Decoding: One Write-Head is All You Need", "MQA") +
    """ and """ + cite("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "GQA") + """
traded quality against KV-cache size, and """ +
    cite("Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", "Shazeer et al. (2017)") +
    """ then """ + cite("Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", "Switch") + """
made conditional computation work at scale.</p>

<h3>Scale, systems, and alignment</h3>
<p>""" + cite("Scaling Laws for Neural Language Models", "Kaplan et al. (2020)") + """
established that loss follows smooth power laws; """ +
    cite("Training Compute-Optimal Large Language Models", "Hoffmann et al. (2022)") + """
corrected the allocation and made a generation of models look badly
undertrained; """ + cite("Chinchilla Scaling: A replication attempt", "Besiroglu et al. (2024)") + """
then showed the published Approach-3 constants do not reproduce that
paper&rsquo;s own recommendation. """ +
    cite("Decoupled Weight Decay Regularization", "AdamW") + """ fixed
regularization under adaptive scaling, and """ +
    cite("Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", "muP") + """
made hyperparameters transfer from small models to large ones.
""" + cite("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "FlashAttention") + """
reframed attention as an IO problem; """ +
    cite("ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", "ZeRO") + """,
""" + cite("Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", "Megatron-LM") + """
and """ + cite("GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", "GPipe") + """
divide the model three different ways. Finally """ +
    cite("Training language models to follow instructions with human feedback", "InstructGPT") + """
made raw predictors usable, """ +
    cite("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "DPO") + """
removed the RL from RLHF, and """ +
    cite("DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "GRPO") + """
removed the value network.</p>

<h3>Three live threads (2024 &rarr; mid-2026)</h3>
<p><strong>1. Is the tokenizer removable?</strong> """ +
    cite("Byte Latent Transformer: Patches Scale Better Than Tokens", "Byte Latent Transformer (2024)") + """
learns dynamic byte patches and matches token-based models at scale. Whether
byte-level models displace BPE or remain a research direction is open.</p>
<p><strong>2. Data is now the binding constraint.</strong> """ +
    cite("The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale", "FineWeb") + """
and """ + cite("DataComp-LM: In search of the next generation of training sets for language models", "DCLM") + """
show curation beating scale, while """ +
    cite("Scaling Data-Constrained Language Models", "Muennighoff et al.") + """
studies what happens when you run out and must repeat data. """ +
    cite("Deduplicating Training Data Makes Language Models Better", "Lee et al.") + """
established deduplication as mandatory &mdash; though FineWeb later found more
aggressive global dedup can hurt, so even this is not simple.</p>
<p><strong>3. Does RL add capability?</strong> """ +
    cite("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", "DeepSeek-R1") + """
made reasoning RL mainstream; """ +
    cite("Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", "Yue et al.") + """
argues it sharpens rather than extends. Unresolved, and the most consequential
open question on this page.</p>

<h3>The complete reference list</h3>
<table>
<thead><tr><th>work</th><th>authors &middot; identifier</th><th>why it is here</th></tr></thead>
<tbody>
__REFTABLE__
</tbody>
</table>

<h3>Suggested reading order</h3>
<p>Start with Attention Is All You Need for the mechanism, then Chinchilla for
the economics, then FlashAttention for how systems thinking changes an
algorithm without changing its output. If you are implementing rather than
reading, do CS336&rsquo;s assignments in order instead &mdash; the papers make
much more sense after you have written the thing.</p>
""")

REFS = [
    ("Attention Is All You Need", "The transformer. Removes recurrence; source of the scaled dot-product formula."),
    ("Language Models are Few-Shot Learners", "GPT-3: in-context learning emerges from scale alone."),
    ("Neural Machine Translation of Rare Words with Subword Units", "Brings BPE into NLP; ends the unknown-word problem."),
    ("SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing", "Language-agnostic, reversible tokenization."),
    ("Byte Latent Transformer: Patches Scale Better Than Tokens", "Dynamic byte patches; the strongest tokenizer-free result."),
    ("Root Mean Square Layer Normalization", "Drops mean subtraction; cheaper and equally effective."),
    ("RoFormer: Enhanced Transformer with Rotary Position Embedding", "Relative position from absolute rotation."),
    ("GLU Variants Improve Transformer", "SwiGLU, and the 8/3 hidden-width convention."),
    ("Fast Transformer Decoding: One Write-Head is All You Need", "MQA: one KV head for the whole layer."),
    ("GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", "Interpolates MHA and MQA; the current default."),
    ("Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", "Conditional computation: parameters without proportional FLOPs."),
    ("Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity", "Top-1 routing makes MoE simple and stable."),
    ("DeepSeek-V3 Technical Report", "MLA plus auxiliary-loss-free balancing at 671B/37B."),
    ("Scaling Laws for Neural Language Models", "Loss follows smooth power laws in N, D and C."),
    ("Training Compute-Optimal Large Language Models", "Chinchilla: scale parameters and tokens together."),
    ("Chinchilla Scaling: A replication attempt", "Shows the published Approach-3 constants are internally inconsistent."),
    ("Scaling Data-Constrained Language Models", "What to do when the data, not the compute, runs out."),
    ("Decoupled Weight Decay Regularization", "AdamW: decay the weights, not the gradient."),
    ("Tensor Programs V: Tuning Large Neural Networks via Zero-Shot Hyperparameter Transfer", "muP: tune small, transfer to large."),
    ("FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness", "Exact attention, restructured around memory traffic."),
    ("FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", "Better work partitioning; roughly doubles throughput."),
    ("Online normalizer calculation for softmax", "The streaming-softmax trick FlashAttention is built on."),
    ("Training Deep Nets with Sublinear Memory Cost", "Gradient checkpointing: √n memory for one extra forward pass."),
    ("ZeRO: Memory Optimizations Toward Training Trillion Parameter Models", "Shard optimizer state, gradients, then parameters."),
    ("Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism", "Tensor parallelism inside the layer."),
    ("GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism", "Pipeline parallelism with micro-batches."),
    ("Ring Attention with Blockwise Transformers for Near-Infinite Context", "Context parallelism with communication hidden behind compute."),
    ("Training language models to follow instructions with human feedback", "InstructGPT: the RLHF pipeline that made models usable."),
    ("Direct Preference Optimization: Your Language Model is Secretly a Reward Model", "Closed-form solution removes the reward model and the RL loop."),
    ("DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models", "GRPO: the group replaces the value network."),
    ("DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning", "Reasoning RL with verifiable rewards, at scale."),
    ("Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model?", "Argues RLVR sharpens sampling rather than adding capability."),
    ("The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale", "Open, documented web curation at 15T tokens."),
    ("DataComp-LM: In search of the next generation of training sets for language models", "Controlled benchmark showing curation beats raw scale."),
    ("Deduplicating Training Data Makes Language Models Better", "Deduplication as a first-class training-data operation."),
    ("The Pile: An 800GB Dataset of Diverse Text for Language Modeling", "The first widely-used open diverse pretraining corpus."),
    ("Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", "T5 and C4; the cleaned-Common-Crawl template."),
    ("DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining", "Learns domain weights instead of guessing them."),
]

QUIZ_PAPERS = [
    {"question": "Which pairing of paper to contribution is correct?",
     "options": [
        {"text": "Hoffmann et al. (2022) — parameters and tokens should scale together for a fixed budget.",
         "correct": True,
         "explanation": "That is Chinchilla. Kaplan et al. (2020) established the power laws but favoured spending on parameters; Besiroglu et al. (2024) is the replication; Ouyang et al. (2022) is InstructGPT."},
        {"text": "Hoffmann et al. (2022) — the first demonstration that loss follows a power law in scale.",
         "explanation": "That is Kaplan et al. (2020), two years earlier."},
        {"text": "Hoffmann et al. (2022) — introduced rotary position embeddings.",
         "explanation": "RoPE is the RoFormer paper, Su et al. (2021)."},
        {"text": "Hoffmann et al. (2022) — showed RLHF makes small models preferred over much larger ones.",
         "explanation": "That is the InstructGPT result, Ouyang et al. (2022)."}]},
    {"question": "What did the Chinchilla replication attempt actually find?",
     "options": [
        {"text": "The published Approach-3 constants are inconsistent with the paper's own other approaches.",
         "correct": True,
         "explanation": "Re-estimating gives exponents that are near-equal and that reproduce both the 20:1 rule and the 70B/1.4T model — which minimizing the published constants does not."},
        {"text": "Chinchilla's core recommendation was wrong and models should be larger.",
         "explanation": "The replication supports the headline recommendation; it disputes one set of fitted constants."},
        {"text": "The 6ND compute approximation was shown to be invalid.",
         "explanation": "6ND is unrelated to the fitting dispute and is not challenged."},
        {"text": "Scaling laws do not extrapolate beyond the fitted range.",
         "explanation": "Extrapolation limits are a separate and more general concern, not this paper's finding."}]},
    {"question": "Why does this page fetch arXiv abstract pages instead of trusting remembered identifiers?",
     "options": [
        {"text": "A plausible-looking identifier can be confidently wrong, so citations are rendered only from checked records.",
         "correct": True,
         "explanation": "The build reads survey/verified.json and explicitly marks anything unconfirmed rather than emitting a guess — the failure mode being designed against is a citation that looks right and is not."},
        {"text": "arXiv identifiers change over time and must be refreshed.",
         "explanation": "Identifiers are permanent; only version suffixes are appended."},
        {"text": "The abstract page is the only place the author list appears.",
         "explanation": "The API returns authors too — it was simply rate-limited during this build."},
        {"text": "Papers are frequently withdrawn and the check detects this.",
         "explanation": "Withdrawal is rare and is not what the verification step is for."}]},
    {"question": "Which of these is a genuinely open question rather than a settled result?",
     "options": [
        {"text": "Whether RL with verifiable rewards adds capability or sharpens existing sampling.",
         "correct": True,
         "explanation": "Both camps accept the pass@k measurements and disagree on interpretation; the page presents both rather than picking."},
        {"text": "Whether backward costs about twice the forward pass.",
         "explanation": "This follows from counting the two backward matrix products — it is arithmetic, not opinion."},
        {"text": "Whether FlashAttention changes the numerical result of attention.",
         "explanation": "It is exact by construction; only memory traffic changes."},
        {"text": "Whether decoupled weight decay differs from L2 under Adam.",
         "explanation": "The two are mathematically different updates, which is the whole point of AdamW."}]},
    {"question": "Deduplication is described as mandatory, yet the page flags a complication. What is it?",
     "options": [
        {"text": "FineWeb found that more aggressive global dedup can reduce model quality.",
         "correct": True,
         "explanation": "Removing too much can strip well-written repeated content and skew the surviving distribution, so dedup strength is a tuned parameter rather than a monotone good."},
        {"text": "Deduplication is computationally impossible above a trillion tokens.",
         "explanation": "MinHash/LSH pipelines run at that scale routinely."},
        {"text": "It conflicts with the 6ND compute estimate.",
         "explanation": "Data curation and FLOP accounting are unrelated concerns."},
        {"text": "It only helps for code, not natural language.",
         "explanation": "The original result was demonstrated on natural-language corpora."}]},
]

# ===========================================================================
spec = {
    "title": "A Language Model, From Scratch",
    "subtitle": "Tokenizers, attention, scaling laws and alignment — every formula derived, every paper checked, in the order CS336 builds them",
    "slug": "language-modelling",
    "date": DATE,
    "multipage": True,
    "topic_title": "Language Modelling",
    "topic_href": DATE + "-language-modelling.html",
    "nav": [["review deck", DATE + "-language-modelling-review.html"],
            ["cheat sheet", DATE + "-lm-cheatsheet.html"],
            ["tutorials", "https://github.com/raghuramshankar/learning-with-llms/tree/main/tutorials/language-modelling"]],
    "generator": {
        "skill": "learning-new-topic",
        "skill_url": "https://github.com/raghuramshankar/learning-with-llms/blob/main/skills/learning-new-topic/SKILL.md",
        "model": "Claude Fable 5",
    },
    "intro": """
<p>This is a deep dive into how a language model is actually built, from raw
bytes to a system that follows instructions. It follows the arc of
Stanford&rsquo;s CS336 &mdash; tokenizer, architecture, optimizer, systems,
scaling laws, data, alignment &mdash; but derives the mathematics rather than
presenting it, and checks every citation against arXiv rather than quoting
from memory.</p>
<p>It is meant to be worked through, not skimmed. Each part ends with a hard
five-question quiz; the maths parts carry faded derivations to attempt on
paper first; and the simulations ask you to commit to a prediction before you
run them. The four derivations are deliberately spent on the results that
compound &mdash; the perplexity bound, the 1/&radic;d<sub>k</sub> scaling, the
6ND rule, and the Chinchilla allocation &mdash; rather than on methods that
will look dated in two years.</p>
<p>Read the parts in order. Parts 1, 2 and 8 stand alone if you want the story
without the equations; the attention widget in Part 4 is the heart of the
page.</p>
""",
    "sections": [
        {"id": "background", "title": "Background", "html": BACKGROUND, "quiz": QUIZ_BACKGROUND},
        {"id": "intuition", "title": "Intuition", "html": INTUITION, "quiz": QUIZ_INTUITION},
        {"id": "math-token", "title": "The Maths I: Tokens and the Objective", "html": MATH_TOKEN, "quiz": QUIZ_MATH_TOKEN},
        {"id": "math-arch", "title": "The Maths II: Inside the Transformer", "html": MATH_ARCH, "quiz": QUIZ_MATH_ARCH},
        {"id": "math-train", "title": "The Maths III: Training at Scale", "html": MATH_TRAIN, "quiz": QUIZ_MATH_TRAIN},
        {"id": "math-sota", "title": "The Maths IV: Inside the SOTA Methods", "html": MATH_SOTA, "quiz": QUIZ_MATH_SOTA},
        {"id": "concept-map", "title": "The Concept Map", "html": CONCEPT_MAP},
        {"id": "keep-learning", "title": "Keep Learning", "html": KEEP_LEARNING},
        {"id": "papers", "title": "The Papers & Sources", "html": PAPERS, "quiz": QUIZ_PAPERS},
    ],
    "scripts": [(TOOLS / "widgets_lib.js").read_text(),
                (HERE / "widgets.js").read_text()],
}
if "--inline" in sys.argv:
    spec["head_scripts"] = [(DOCS / "plotly.min.js").read_text()]
else:
    spec["head_script_srcs"] = ["plotly.min.js"]

PLOTS = HERE / "plots"
subs = {
    "__FIG_BPE__": (PLOTS / "fig_bpe.html").read_text(),
    "__FIG_SCALING__": (PLOTS / "fig_scaling.html").read_text(),
    "__FIG_RATIO__": (PLOTS / "fig_ratio.html").read_text(),
    "__FIG_FLOPS__": (PLOTS / "fig_flops.html").read_text(),
    "__FIG_LR__": (PLOTS / "fig_lr.html").read_text(),
    "__FIG_KV__": (PLOTS / "fig_kv.html").read_text(),
    "__REFTABLE__": "\n".join(refrow(t, n) for t, n in REFS),
}
for s in spec["sections"]:
    for k, v in subs.items():
        s["html"] = s["html"].replace(k, v)

missing = [t for t, _ in REFS if t not in VERIFIED]
if missing:
    print("WARNING: %d reference(s) unverified, rendered as such:" % len(missing))
    for m in missing:
        print("   -", m)

out = HERE / "spec.json"
out.write_text(json.dumps(spec, indent=1))
print(out, "|", len(spec["sections"]), "sections |",
      sum(len(s.get("quiz") or []) for s in spec["sections"]), "questions |",
      len(VERIFIED), "verified citations")
