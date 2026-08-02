#!/usr/bin/env python3
"""Build the JSON content spec for the diffusion-LLM explainer (v2: math deep dives + per-section quizzes)."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / "tools"
DOCS = HERE.parents[1] / "docs"

def toks(*items):
    """items: (text, cls) tuples -> row of token boxes."""
    out = []
    for text, cls in items:
        c = ("box " + cls).strip()
        out.append("<div class='" + c + "'>" + text + "</div>")
    return "".join(out)

def frac(num, den):
    return ("<span class='frac'><span class='num'>" + num +
            "</span><span class='den'>" + den + "</span></span>")

BACKGROUND = """
<p>Inception Labs' <a href='https://www.inceptionlabs.ai/about'>about page</a> lists nine papers
under &ldquo;Some of the technologies we&rsquo;ve developed.&rdquo; They aren&rsquo;t a random reading list:
they are the research trail of the company&rsquo;s founders &mdash; Stefano Ermon, Volodymyr Kuleshov, and
Aditya Grover &mdash; and together they tell one coherent story: <strong>how to make language models that
generate text the way image generators paint images, instead of typing it out one word at a time.</strong>
This page walks that story from the beginning. If you already know how LLMs generate text, skip to
the second half of this section.</p>

<h3>Deep background: what a language model does</h3>
<p>A language model is a neural network (in practice, a <em>transformer</em>) trained on an enormous
amount of text to do one deceptively simple thing: given a piece of text, assign a probability to
what comes next. Text is first chopped into <strong>tokens</strong> &mdash; short chunks, roughly
word-sized, drawn from a fixed vocabulary of tens of thousands. Given the tokens so far, the model
outputs a probability for every token in the vocabulary being the next one. Training simply pushes
the model to give high probability to the token that actually came next in real text; do this over
trillions of tokens and the model absorbs grammar, facts, and reasoning patterns along the way.</p>

<h3>Autoregressive generation: the typewriter</h3>
<p>Almost every LLM you have used &mdash; GPT, Claude, Gemini, Llama &mdash; generates text
<strong>autoregressively</strong>: it predicts one token, appends it, feeds the extended text back
in, predicts the next, and so on, strictly left to right. Mathematically it factorizes the
probability of a sentence as a chain of next-token predictions:</p>
<div class='math'>P(x) = P(x<sub>1</sub>) &middot; P(x<sub>2</sub> | x<sub>1</sub>) &middot; P(x<sub>3</sub> | x<sub>1</sub>, x<sub>2</sub>) &middot;&thinsp;&ctdot;&thinsp;&middot; P(x<sub>N</sub> | x<sub>1</sub>, &hellip;, x<sub>N&minus;1</sub>)</div>
<p>Note something important: this factorization is <em>fully general</em>. The chain rule of
probability lets you decompose <em>any</em> joint distribution this way. Autoregression&rsquo;s
limits are not about what it can represent &mdash; they are about how it must compute.</p>

<div class='diagram'>
  <div class='flow'>
    <div class='box ok'>The</div><span class='arr'>&rarr;</span>
    <div class='box ok'>cat</div><span class='arr'>&rarr;</span>
    <div class='box ok'>sat</div><span class='arr'>&rarr;</span>
    <div class='box ok'>on</div><span class='arr'>&rarr;</span>
    <div class='box ok'>the</div><span class='arr'>&rarr;</span>
    <div class='box accent'>mat</div>
  </div>
  <div class='caption'>The typewriter: six tokens means six full forward passes through the network,
  each one waiting for the previous to finish. Nothing already typed can ever be changed.</div>
</div>

<p>Two properties of this design matter for our story:</p>
<ul>
  <li><strong>It is inherently sequential.</strong> Token 500 cannot be computed until token 499
  exists &mdash; its <em>sampled value</em> is an input to the next pass. (During training you can
  evaluate all positions in parallel because the true text is known; at inference the inputs
  don&rsquo;t exist until you sample them.) Generating <em>N</em> tokens costs <em>N</em> forward
  passes, one after another. Worse, each decode step is <em>memory-bandwidth-bound</em>: the GPU
  spends its time streaming billions of weights and the KV cache from memory to compute a single
  token&rsquo;s worth of arithmetic. The chip is starved for data, not arithmetic.</li>
  <li><strong>It is write-once.</strong> Once a token is emitted it is frozen. If the model commits to
  a word that turns out to be a poor fit three words later, it cannot go back &mdash; it can only try
  to recover in the text that follows.</li>
</ul>

<p>For a chatbot answering a single question, this was tolerable. But the field is moving toward
<em>agents</em> and <em>reasoning models</em> that generate thousands of tokens of intermediate
thinking, call tools, and chain many steps together. Every extra token of &ldquo;thinking&rdquo; is paid
for in wall-clock time, multiplied across every step of a workflow. (Note the one thing faster
decoding cannot fix: agent round <em>k</em>+1 still has to wait for round <em>k</em> &mdash; which is
precisely why the per-round latency is the lever that matters.) Inception&rsquo;s framing on the very
page we&rsquo;re studying: &ldquo;AI shouldn&rsquo;t behave like a one-way typewriter&rdquo; &mdash; it
should behave like an editor.</p>

<p>Here is the whole argument as one picture &mdash; hover to compare (the y-axis is
logarithmic):</p>

<div class='widget'>
__FIG_LATENCY__
<div class='caption'>An AR model&rsquo;s sequential passes grow linearly with output length; a
dLLM&rsquo;s stay flat at its step count. At N = 1,000 tokens the gap is 1,000 passes versus a few
dozen &mdash; and this ratio, not raw FLOPs, is what sets wall-clock latency in the
bandwidth-bound decode regime described above.</div>
</div>

<h3>Meanwhile, in image generation: diffusion</h3>
<p>While language modeling settled on the typewriter, image generation took a completely different
path. Systems like Stable Diffusion and Midjourney are <strong>diffusion models</strong>. The recipe:
take a real image and gradually corrupt it with random noise over many steps until nothing but static
remains (the <em>forward process</em>). Then train a network to undo one step of that corruption
(the <em>reverse process</em>). To generate, start from pure static and repeatedly denoise:</p>

<div class='diagram'>
  <div class='flow'>
    <div class='box dim'>pure noise<small>static</small></div><span class='arr'>&rarr;</span>
    <div class='box dim'>vague shapes<small>composition emerges</small></div><span class='arr'>&rarr;</span>
    <div class='box'>cat-like blob<small>structure</small></div><span class='arr'>&rarr;</span>
    <div class='box ok'>sharp cat photo<small>details</small></div>
  </div>
  <div class='caption'>Coarse-to-fine: every pixel is updated <em>in parallel</em> at every step.
  The image gets globally better each pass, rather than being painted left to right.</div>
</div>

<p>Here is that forward process live, on an actual (very small) image. Drag the corruption level
yourself:</p>

<div class='widget' id='w-noise'>
  <div class='wctl'>
    <button class='wbtn' id='w-noise-play'>&#9654; Corrupt</button>
    <button class='wbtn' id='w-noise-rev'>&#9664; Rewind</button>
    <label>t</label>
    <input type='range' id='w-noise-t' min='0' max='1000' value='0'>
  </div>
  <canvas id='w-noise-cv' width='260' height='260'></canvas>
  <div class='wstat' id='w-noise-stat'></div>
  <div class='caption'>The display is exactly x<sub>t</sub> = &radic;&#8113;&middot;x<sub>0</sub> +
  &radic;(1&minus;&#8113;)&middot;&epsilon; with a cosine schedule and one <em>fixed</em> draw of
  &epsilon;, which is why scrubbing is perfectly repeatable. Honesty note: dragging left is the
  movie in reverse, not a trained denoiser &mdash; training the denoiser is what the maths sections
  are about.</div>
</div>

<p>Notice what diffusion gives you that the typewriter cannot: the whole output exists (in rough
form) from the very first step, every step refines <em>everything at once</em>, and the number of
steps is a dial you can turn &mdash; a handful of steps for a fast draft, more for higher quality.
Early diffusion models needed ~1,000 denoising steps; the 2020 <strong>DDIM</strong> paper (the
&ldquo;Diffusion Models&rdquo; link on Inception&rsquo;s page, co-authored by Ermon) showed
how to restructure sampling so that a few dozen steps suffice &mdash; using the same trained network,
no retraining. That obsession with cutting sampling steps is the company&rsquo;s DNA.</p>

<div class='callout'>
<p><strong>The obvious question:</strong> if diffusion is so good for images, why not text? The
blocker is that images are <em>continuous</em> &mdash; you can add a little Gaussian noise to a pixel
value. Text is <em>discrete</em>: token #4711 is &ldquo;cat&rdquo;, and there is no such thing as
&ldquo;cat plus 0.1 noise&rdquo;. Finding the right notion of noise for text is where the intuition
section picks up.</p>
</div>
"""

QUIZ_BACKGROUND = [
    {
        "question": "With KV caching enabled, what dominates per-token latency during autoregressive decoding on a modern GPU?",
        "options": [
            {"text": "Streaming the model weights and KV cache from GPU memory on every step.",
             "correct": True,
             "explanation": "Decode is memory-bandwidth-bound: each new token requires re-reading billions of parameters to do a single token's worth of arithmetic. The compute units sit mostly idle — which is exactly the waste dLLMs attack with big parallel refinement passes."},
            {"text": "The floating-point cost of the matrix multiplications in each layer.",
             "explanation": "At batch-of-one decode the matmuls are tiny relative to the chip's throughput; the GPU is starved for data, not arithmetic. That's what memory-bandwidth-bound means."},
            {"text": "Recomputing attention over the entire prefix from scratch at each step.",
             "explanation": "That recomputation is precisely what the KV cache eliminates — keys and values for the prefix are stored and reused."},
            {"text": "Sampling the next token from the vocabulary-sized output distribution.",
             "explanation": "A softmax over ~100k entries is trivial next to moving gigabytes of weights through memory per token."},
        ],
    },
    {
        "question": "Which statement about the chain-rule factorization <span class='m'>P(x) = &prod;<sub>t</sub> P(x<sub>t</sub> | x<sub>&lt;t</sub>)</span> is true?",
        "options": [
            {"text": "It can represent any distribution over sequences; the cost is sequential decoding.",
             "correct": True,
             "explanation": "The chain rule is an identity — every joint distribution factorizes this way (in any order). Autoregression's weakness is computational (N sequential passes, no revision), not representational. That's why dLLMs compete on speed and editability, not on expressiveness."},
            {"text": "It can only represent distributions whose dependencies run left to right.",
             "explanation": "Conditioning on the full prefix captures arbitrary dependencies — the factorization is fully general. The left-to-right structure constrains computation, not expressiveness."},
            {"text": "It assumes each token depends only on the k most recent tokens before it.",
             "explanation": "That's a Markov assumption; transformers condition on the entire prefix, and the chain rule itself makes no independence assumption at all."},
            {"text": "It requires the training corpus itself to be processed left to right.",
             "explanation": "The factorization is a property of the model's decomposition of probability, not of corpus ordering or training order."},
        ],
    },
    {
        "question": "Why can't autoregressive generation be parallelized by simply predicting all N tokens in one forward pass?",
        "options": [
            {"text": "Each position must condition on sampled values of earlier positions, which don't exist yet.",
             "correct": True,
             "explanation": "Training parallelizes fine (teacher forcing: true tokens are known). At inference, position t's input is the sampled token at t−1. Predicting everything at once gives N independent marginals that ignore inter-token dependence — which, notably, is exactly the approximation dLLMs make within a step, then repair by iterating."},
            {"text": "The causal attention mask physically prevents computing more than one output per pass.",
             "explanation": "The mask doesn't stop parallel computation — during training, all positions' predictions are computed in one masked pass. The blocker at inference is missing inputs, not the mask."},
            {"text": "The KV cache is only able to append one new entry per forward pass.",
             "explanation": "Cache mechanics are a consequence of sequential sampling, not its cause."},
            {"text": "The softmax normalization couples all N output positions to each other.",
             "explanation": "Softmax normalizes over the vocabulary independently at each position; positions aren't coupled through it."},
        ],
    },
    {
        "question": "DDPM sampling needed ~1,000 steps. Which route did DDIM take to cut that down?",
        "options": [
            {"text": "Reformulate sampling as a deterministic process sharing DDPM's marginals — no retraining.",
             "correct": True,
             "explanation": "DDIM's insight: the training loss only depends on the noisy marginals q(x<sub>t</sub>|x<sub>0</sub>), so a whole family of samplers — including a deterministic one that can skip steps — reuses the same trained network. The maths section derives it."},
            {"text": "Distill the slow sampler into a student network that takes far fewer steps.",
             "explanation": "That's progressive distillation (Salimans & Ho, 2022) — a later, different line of work that does require additional training."},
            {"text": "Learn the noise schedule β<sub>t</sub> jointly with the model to shorten the chain.",
             "explanation": "Schedule learning is the Improved-DDPM / Variational Diffusion Models direction; DDIM changes the sampler, not the schedule or the training."},
            {"text": "Replace the U-Net backbone with a transformer to make each pass cheaper.",
             "explanation": "Architecture swaps (e.g. DiT) change per-step cost. DDIM attacks the number of steps, which is where the 10–50× came from."},
        ],
    },
    {
        "question": "A coding agent runs k sequential tool-calling rounds, each generating n tokens. Which cost does switching from an AR model to a dLLM NOT reduce?",
        "options": [
            {"text": "The k-round sequential dependency — round i+1 still waits for round i.",
             "correct": True,
             "explanation": "The workflow chain stays sequential no matter how generation works: round i+1's input is round i's output. dLLMs shrink each round from n sequential passes to a few dozen — that per-round lever is the only one available, which is why Inception leads with the agent use case."},
            {"text": "The number of sequential model passes inside each n-token generation.",
             "explanation": "This is exactly what dLLMs reduce: a few dozen refinement passes instead of n token-by-token passes."},
            {"text": "The wall-clock latency of each individual round of the workflow.",
             "explanation": "Per-round latency drops roughly in proportion to the pass count — that's the headline speedup."},
            {"text": "GPU idle time caused by one-token-at-a-time decoding.",
             "explanation": "Each dLLM pass processes the whole sequence — big parallel work that keeps the compute units busy instead of bandwidth-starved."},
        ],
    },
]

INTUITION = """
<p>The core idea that makes diffusion work for text is almost embarrassingly simple once you see it:
<strong>for discrete tokens, the analogue of noise is masking.</strong></p>

<h3>Noising text = hiding it</h3>
<p>The forward process doesn&rsquo;t add static to a sentence &mdash; it progressively <em>hides</em>
it, replacing tokens with a special <code>[MASK]</code> symbol (shown as &#9634; below) until nothing
is left:</p>

<div class='diagram'>
  <div class='flow vertical'>
    <div class='flow'><span class='tag'>clean</span>__ROW_CLEAN__</div>
    <span class='arr'>&darr; mask a few</span>
    <div class='flow'><span class='tag'>partly&nbsp;masked</span>__ROW_PART__</div>
    <span class='arr'>&darr; mask more</span>
    <div class='flow'><span class='tag'>fully&nbsp;masked</span>__ROW_FULL__</div>
  </div>
  <div class='caption'>The forward process: corruption by masking. Training shows the model sentences
  at every corruption level and asks it to recover what&rsquo;s hidden &mdash; like BERT&rsquo;s
  fill-in-the-blank task, but at every masking ratio from 0% to 100%.</div>
</div>

<p>Scrub it yourself &mdash; and watch the expected-versus-actual masked count while you do
(this is exactly the quantity the first quiz question below asks about):</p>

<div class='widget' id='w-mask'>
  <div class='wctl'>
    <label>corruption t</label>
    <input type='range' id='w-mask-t' min='0' max='1000' value='0'>
  </div>
  <div class='flow' id='w-mask-row'></div>
  <div class='wstat' id='w-mask-stat'></div>
  <div class='caption'>Linear schedule &alpha;<sub>t</sub> = 1&minus;t. Each token has a fixed random
  threshold, so scrubbing back and forth is consistent &mdash; you are conditioning on one draw of
  the forward process, the same trick training uses.</div>
</div>

<p>Generation is the reverse movie. Start from a row of blanks. In <strong>one forward pass</strong>,
the model predicts a probability distribution for <em>every</em> blank simultaneously &mdash; this is
the crucial difference from the typewriter, which predicts exactly one position per pass. Then commit
the predictions the model is most confident about, keep the rest masked, and repeat:</p>

<div class='diagram'>
  <div class='flow vertical'>
    <div class='flow'><span class='tag'>start</span>__GEN_ROW0__</div>
    <span class='arr'>&darr; pass 1: predict all six slots, keep the two most confident</span>
    <div class='flow'><span class='tag'>step&nbsp;1</span>__GEN_ROW1__</div>
    <span class='arr'>&darr; pass 2</span>
    <div class='flow'><span class='tag'>step&nbsp;2</span>__GEN_ROW2__</div>
    <span class='arr'>&darr; pass 3</span>
    <div class='flow'><span class='tag'>done</span>__GEN_ROW3__</div>
  </div>
  <div class='caption'>Blue = committed this pass &middot; green = committed earlier &middot;
  &#9634; = still masked. Six tokens in three passes instead of six. At real scale the ratio is far
  more dramatic: a thousand tokens in a few dozen passes, with each pass a big, GPU-friendly
  parallel computation.</div>
</div>

<div class='callout'>
<p><strong>Definition.</strong> A <strong>diffusion language model (dLLM)</strong> is a language model
trained to reverse a corruption process over discrete tokens (in practice, masking), generating text
by iteratively refining the whole sequence in parallel over a small number of steps &mdash; rather
than emitting one token per forward pass, left to right. Fewer passes than tokens is the entire
source of the speed advantage.</p>
</div>

<p>Here are the two regimes racing on the same sentence. Every tick of the clock is one forward
pass for both models:</p>

<div class='widget' id='w-race'>
  <div class='wctl'>
    <button class='wbtn' id='w-race-play'>&#9654; Race</button>
    <button class='wbtn' id='w-race-reset'>Reset</button>
  </div>
  <p style='margin:.2rem 0 0'><span class='tag'>typewriter (autoregressive)</span></p>
  <div class='flow' id='w-race-ar'></div>
  <div class='wstat' id='w-race-ar-stat'>passes: 0</div>
  <p style='margin:.6rem 0 0'><span class='tag'>editor (diffusion)</span></p>
  <div class='flow' id='w-race-dl'></div>
  <div class='wstat' id='w-race-dl-stat'>passes: 0</div>
  <div class='caption'>The dLLM commits several tokens per pass (easiest first, the commit order is
  scripted here) and finishes in 4 passes; the typewriter needs 16. Real deployments report the
  same shape: a few dozen passes for a thousand tokens.</div>
</div>

<h3>The catch: parallel commits are independent commits</h3>
<p>There is a price for committing several tokens in one pass, and it is worth naming precisely
because the maths section will formalize it. Within a single pass, the model produces a
<em>separate</em> distribution for each blank, and sampling them simultaneously treats them as
independent. Suppose two adjacent blanks should be either &ldquo;hot dog&rdquo; or &ldquo;ice
cream&rdquo;, each with probability &frac12;. Independent sampling happily produces &ldquo;hot
cream&rdquo; a quarter of the time. Committing fewer tokens per pass (more passes) shrinks this
coupling error; committing more (fewer passes) grows it. <strong>The step count is exactly the dial
that trades speed against coordination.</strong></p>

<p>Don&rsquo;t take the 25% on faith &mdash; run the experiment. Draw samples in both modes and
watch the tallies converge:</p>

<div class='widget' id='w-indep'>
  <div class='wctl' id='w-indep-predict'>
    <label>Predict first &mdash; in one-pass mode, what fraction will be coherent?</label>
    <button class='wbtn' data-pred='100'>100%</button>
    <button class='wbtn' data-pred='75'>75%</button>
    <button class='wbtn' data-pred='50'>50%</button>
  </div>
  <div class='wstat' id='w-indep-predfb' style='display:none'></div>
  <div class='wctl'>
    <button class='wbtn active' id='w-indep-par'>both blanks in one pass</button>
    <button class='wbtn' id='w-indep-seq'>one blank per pass (2 passes)</button>
  </div>
  <div class='wctl'>
    <button class='wbtn' id='w-indep-1'>draw 1</button>
    <button class='wbtn' id='w-indep-50'>draw 50</button>
    <button class='wbtn' id='w-indep-reset'>reset</button>
  </div>
  <div class='wstat' id='w-indep-stat'>no samples yet</div>
  <div class='wchips' id='w-indep-chips'></div>
  <div class='caption'>Target: &ldquo;hot dog&rdquo; or &ldquo;ice cream&rdquo;, 50/50. Each
  word&rsquo;s marginal is a fair coin flip, so the one-pass mode settles near 50% coherent
  (&frac14; &ldquo;hot cream&rdquo; + &frac14; &ldquo;ice dog&rdquo;), while spending a second pass
  conditions the second word on the first and never errs. Speed bought with coordination, measured.</div>
</div>

<h3>Bidirectional sight</h3>
<p>A subtler gift: when the typewriter predicts a token, it can only look <em>backward</em> &mdash;
the future doesn&rsquo;t exist yet. A dLLM filling in a blank sees committed tokens on <em>both
sides</em>. Predicting the blank in &ldquo;the &#9634; sat on the mat&rdquo; is much easier when
&ldquo;sat on the mat&rdquo; is already visible. Architecturally this is a small change: drop the
causal attention mask so every position attends to every other &mdash; the transformer itself is
otherwise the same machine.</p>

<h3>The editor: fixing your own mistakes</h3>
<p>Plain masked diffusion has one typewriter habit left: once a blank is filled, it stays filled.
That can wedge the model into a corner. Watch what happens when it confidently commits the article
before choosing the noun:</p>

<div class='diagram'>
  <div class='flow vertical'>
    <div class='flow'>__FIX_ROW1__</div>
    <span class='arr'>&darr; next pass wants &ldquo;apple&rdquo; &mdash; but &ldquo;a apple&rdquo; is wrong, and &ldquo;a&rdquo; is frozen</span>
    <div class='flow'>__FIX_ROW2__</div>
    <span class='arr'>&darr; remasking: put the offending token back into the noise</span>
    <div class='flow'>__FIX_ROW3__</div>
    <span class='arr'>&darr; re-predict it with full context</span>
    <div class='flow'>__FIX_ROW4__</div>
  </div>
  <div class='caption'>Remasking turns generation into editing: any committed token can be sent back
  to &#9634; and re-decided in light of everything else. An autoregressive model has no equivalent
  move &mdash; its only option is to keep typing and hope.</div>
</div>

<p>This is the &ldquo;editor, not typewriter&rdquo; idea made concrete, and it is exactly what the
<strong>Remasking Diffusion</strong> paper contributes. It also creates a genuinely new dial:
spend more passes &rarr; more rounds of self-correction &rarr; better output. Quality becomes
something you can buy with inference-time compute, smoothly.</p>

<h3>Head-to-head</h3>
<table>
<tr><th></th><th>Autoregressive (typewriter)</th><th>Masked diffusion (editor)</th></tr>
<tr><td>Generation order</td><td>Strictly left &rarr; right</td><td>Any order; easiest-first, coarse-to-fine</td></tr>
<tr><td>Forward passes for N tokens</td><td>N, strictly sequential</td><td>k steps with k &ll; N; k is tunable</td></tr>
<tr><td>Context per prediction</td><td>Past tokens only</td><td>Both directions</td></tr>
<tr><td>Revising committed text</td><td>Impossible</td><td>Yes, via remasking</td></tr>
<tr><td>Speed&ndash;quality trade-off</td><td>Fixed by length</td><td>Dial: number of steps</td></tr>
<tr><td>Variable length + KV caching</td><td>Natural</td><td>Needs Block Diffusion (see the papers)</td></tr>
</table>

<div class='callout warn'>
<p><strong>Honesty box.</strong> Autoregressive models still hold the overall quality crown at the
frontier, and vanilla masked diffusion arrives with real handicaps: it generates fixed-length
canvases, it can&rsquo;t reuse cached computation the way AR decoding does, and its likelihoods
historically lagged. The reason Inception&rsquo;s paper list is interesting is that nearly every entry
knocks down one of these handicaps.</p>
</div>

<p>That is the intuition in full. The next two sections re-tell the same story with the equations
attached: first the continuous case, where diffusion and its vocabulary were invented, then the
discrete, masked case that a dLLM actually runs. They are the most demanding sections of this page
&mdash; and the most rewarding, because after them the nine papers read like straightforward
engineering decisions rather than magic.</p>
"""

QUIZ_INTUITION = [
    {
        "question": "In a masked-diffusion forward process, α<sub>t</sub> = 0.25 is the probability a token survives unmasked at time t. For a 12-token sentence, how many tokens do you expect to remain visible?",
        "options": [
            {"text": "3", "correct": True,
             "explanation": "Each token survives independently with probability α<sub>t</sub>, so the expected count is 12 × 0.25 = 3. Keeping the α convention straight (survival, not corruption) pays off in the maths sections."},
            {"text": "9", "explanation": "That's 12 × (1 − α<sub>t</sub>) — the expected number of <em>masked</em> tokens. α<sub>t</sub> is the survival probability."},
            {"text": "6", "explanation": "That would be α<sub>t</sub> = 0.5. Here each token survives with probability 0.25, so 12 × 0.25 = 3."},
            {"text": "4", "explanation": "12 × 0.25 = 3, not 4 — the expected visible count is sentence length times the survival probability α<sub>t</sub>."},
        ],
    },
    {
        "question": "When a dLLM commits several tokens in a single reverse step, what is the precise statistical error being made?",
        "options": [
            {"text": "Tokens are drawn from independent per-position marginals, ignoring their joint dependence.",
             "correct": True,
             "explanation": "Within one pass the model outputs a separate distribution per blank; sampling them together is a mean-field (factorized) approximation of the true joint. Hence “hot cream”: two individually reasonable marginals, jointly wrong."},
            {"text": "Tokens are drawn from the prior over the vocabulary rather than from the posterior.",
             "explanation": "Each marginal genuinely is the model's posterior for that position given the current context — the flaw is sampling them jointly as if independent, not using the wrong distribution per position."},
            {"text": "The step reuses stale logits that were computed before the previous step's commits.",
             "explanation": "Each step runs a fresh forward pass over the updated sequence; the error is within-step independence, not staleness across steps."},
            {"text": "Low-probability tokens can never be committed, biasing output toward frequent words.",
             "explanation": "Confidence-based ordering changes <em>when</em> a position is decided, not what can be sampled there; rare tokens still win when context demands them."},
        ],
    },
    {
        "question": "Architecturally, what is the key difference between a masked dLLM and a GPT-style autoregressive transformer?",
        "options": [
            {"text": "It drops the causal attention mask, letting every position attend to every other.",
             "correct": True,
             "explanation": "That's essentially it — same transformer, bidirectional attention, trained to predict masked slots instead of next tokens. The magnitude of the behavioral change (parallel, revisable generation) versus the smallness of the architectural change is the striking part."},
            {"text": "It adds a separate encoder whose output feeds cross-attention into a decoder.",
             "explanation": "No encoder–decoder split is needed; a single bidirectional stack does everything."},
            {"text": "It replaces learned positional information with the masking schedule α<sub>t</sub>.",
             "explanation": "Positional encodings remain; the schedule governs corruption during training and sampling, not position."},
            {"text": "It widens the output head to predict an entire block of tokens jointly.",
             "explanation": "The head is still per-position over the vocabulary — that per-position independence is precisely the approximation discussed above."},
        ],
    },
    {
        "question": "Why can a vanilla masked-diffusion sampler never revise a token it has committed?",
        "options": [
            {"text": "Its reverse posterior assigns probability 1 to carrying over any revealed token.",
             "correct": True,
             "explanation": "In the reverse process of absorbing-state diffusion, an unmasked token stays itself with certainty — mask is absorbing forward, revealed is absorbing backward. ReMDM modifies the <em>sampler's</em> kernel to reopen that door."},
            {"text": "Remasked inputs would be out-of-distribution, so revising requires retraining first.",
             "explanation": "Instructively false: a remasked token is indistinguishable from an ordinarily masked one, which is exactly why ReMDM works on a pretrained model with no retraining."},
            {"text": "The attention mask hides committed positions from all later reverse steps.",
             "explanation": "Attention is fully bidirectional — committed tokens are visible everywhere; they just can't transition back to mask under the vanilla kernel."},
            {"text": "Any revision would invalidate the ELBO bound that was used during training.",
             "explanation": "The training bound constrains the learned model, not the sampler you run afterwards; ReMDM changes only the sampling procedure."},
        ],
    },
    {
        "question": "You cut a dLLM sampler from 64 steps to 16 for a 512-token generation. What necessarily happens?",
        "options": [
            {"text": "About 4× more tokens are committed per step, compounding independence errors.",
             "correct": True,
             "explanation": "512 tokens over 16 steps means ~32 commits per step instead of ~8 — more simultaneous, independently-sampled decisions per pass, hence more “hot cream”. That's the speed–quality dial in mechanical terms."},
            {"text": "The model must be fine-tuned on the shorter schedule before quality recovers.",
             "explanation": "Step count is an inference-time choice; the same network serves any number of steps (MDLM's objective is even invariant to the schedule — see the maths)."},
            {"text": "The training NELBO increases because fewer corruption levels get integrated over.",
             "explanation": "The training objective integrates over continuous corruption levels and never sees the sampler's step count."},
            {"text": "Latency stays roughly the same, since each step must now do 4× the work.",
             "explanation": "Each step is one full-sequence forward pass regardless of how many tokens it commits — so 16 steps costs about a quarter of 64. That's the whole point."},
        ],
    },
]

MATH1 = """
<p>The pictures so far are honest, but they hide the machinery. This section makes the continuous
(image) case precise; the next one carries the same skeleton over to text. The only prerequisites
are Gaussians, expectations, and a tolerance for subscripts.</p>

<h3>The forward process</h3>
<p>Fix a <em>noise schedule</em> &beta;<sub>1</sub>, &hellip;, &beta;<sub>T</sub> &isin; (0,1) of small,
typically increasing values, and let data be <span class='m'>x<sub>0</sub> ~ q(x<sub>0</sub>)</span>.
The forward process is a Markov chain that corrupts one step at a time:</p>
<div class='math'>q(x<sub>t</sub> | x<sub>t&minus;1</sub>) = &#119977;( x<sub>t</sub> ; &radic;(1&minus;&beta;<sub>t</sub>) &middot; x<sub>t&minus;1</sub> , &beta;<sub>t</sub>&middot;I )</div>
<p>The odd-looking <span class='m'>&radic;(1&minus;&beta;<sub>t</sub>)</span> is deliberate: if
<span class='m'>Var(x<sub>t&minus;1</sub>) = I</span>, then
<span class='m'>Var(x<sub>t</sub>) = (1&minus;&beta;<sub>t</sub>)&middot;I + &beta;<sub>t</sub>&middot;I = I</span>.
Signal shrinks at exactly the rate noise grows &mdash; a <em>variance-preserving</em> chain that
neither explodes nor collapses. Because Gaussians compose, the whole chain telescopes into a single
jump. Writing <span class='m'>&alpha;<sub>t</sub> = 1&minus;&beta;<sub>t</sub></span> and
<span class='m'>&#8113;<sub>t</sub> = &alpha;<sub>1</sub>&alpha;<sub>2</sub>&ctdot;&alpha;<sub>t</sub></span>:</p>
<div class='math'>q(x<sub>t</sub> | x<sub>0</sub>) = &#119977;( &radic;&#8113;<sub>t</sub>&middot;x<sub>0</sub> , (1&minus;&#8113;<sub>t</sub>)&middot;I )
&nbsp;&nbsp;&hArr;&nbsp;&nbsp; x<sub>t</sub> = &radic;&#8113;<sub>t</sub>&middot;x<sub>0</sub> + &radic;(1&minus;&#8113;<sub>t</sub>)&middot;&epsilon;,&nbsp;&nbsp;&epsilon; ~ &#119977;(0, I)</div>
<div class='callout'>
<p><strong>Why this closed form matters.</strong> Training never has to simulate the chain: draw a
random t, draw &epsilon;, and you have a corrupted sample at exactly noise level t in one line of
code. This is what makes diffusion training as cheap per example as ordinary supervised learning.
By design <span class='m'>&#8113;<sub>T</sub> &asymp; 0</span>, so
<span class='m'>x<sub>T</sub> ~ &#119977;(0, I)</span>: all information about the data is destroyed,
and generation can start from pure noise.</p>
</div>

<h3>The reverse process and the ELBO</h3>
<p>Generation needs <span class='m'>q(x<sub>t&minus;1</sub> | x<sub>t</sub>)</span>, which is
intractable &mdash; it implicitly marginalizes over the entire data distribution. Two facts rescue
us. First, for small &beta;<sub>t</sub> the true reversal is itself nearly Gaussian, so a Gaussian
model <span class='m'>p<sub>&theta;</sub>(x<sub>t&minus;1</sub> | x<sub>t</sub>)</span> is
well-specified. Second, conditioned <em>additionally on the clean image</em>, the reversal is exactly
Gaussian with closed form:</p>
<div class='math'>q(x<sub>t&minus;1</sub> | x<sub>t</sub>, x<sub>0</sub>) = &#119977;( &mu;&#771;<sub>t</sub>, &beta;&#771;<sub>t</sub>&middot;I ),&nbsp;&nbsp;&nbsp;
&mu;&#771;<sub>t</sub> = __FRAC_MU1__&middot;x<sub>0</sub> &nbsp;+&nbsp; __FRAC_MU2__&middot;x<sub>t</sub>,&nbsp;&nbsp;&nbsp;
&beta;&#771;<sub>t</sub> = __FRAC_BETA__&middot;&beta;<sub>t</sub></div>
<p>The standard variational argument (the same one behind VAEs) then bounds the negative
log-likelihood &mdash; the <strong>NELBO</strong> &mdash; and the bound splits cleanly per timestep:</p>
<div class='math'>&minus;log p<sub>&theta;</sub>(x<sub>0</sub>) &le;
&#120124;<sub>q</sub>[ &minus;log p<sub>&theta;</sub>(x<sub>0</sub> | x<sub>1</sub>) ]
&nbsp;+&nbsp; &sum;<sub>t=2</sub><sup>T</sup> &#120124;<sub>q</sub> KL( q(x<sub>t&minus;1</sub> | x<sub>t</sub>, x<sub>0</sub>) &Vert; p<sub>&theta;</sub>(x<sub>t&minus;1</sub> | x<sub>t</sub>) )
&nbsp;+&nbsp; KL( q(x<sub>T</sub> | x<sub>0</sub>) &Vert; &#119977;(0, I) )</div>
<p>Read the three terms left to right: a <strong>reconstruction</strong> term for the final tiny
step; the <strong>denoising-matching</strong> sum &mdash; the workhorse, demanding that at every noise
level the model's one-step denoiser match the tractable posterior above; and the
<strong>prior</strong> term, which contains no parameters at all and is &asymp;0 whenever the
schedule truly destroys the signal. Every KL is between Gaussians, hence closed-form.</p>
<p>One reparameterization turns this into the loss everyone actually ships. Predict the
<em>noise</em> &epsilon; rather than the mean:</p>
<div class='math'>&mu;<sub>&theta;</sub>(x<sub>t</sub>, t) = __FRAC_MUTHETA1__&middot;( x<sub>t</sub> &minus; __FRAC_MUTHETA2__&middot;&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) )</div>
<p>after which the KL terms collapse into weighted squared errors on &epsilon;. DDPM famously drops
the weights (it trains better) and arrives at:</p>
<div class='math'>L<sub>simple</sub> = &#120124;<sub>t, x<sub>0</sub>, &epsilon;</sub> &Vert; &epsilon; &minus; &epsilon;<sub>&theta;</sub>( &radic;&#8113;<sub>t</sub>&middot;x<sub>0</sub> + &radic;(1&minus;&#8113;<sub>t</sub>)&middot;&epsilon;, &nbsp;t ) &Vert;&sup2;</div>
<p>&ldquo;Guess the noise I added&rdquo; &mdash; that single regression, averaged over all noise
levels, trains the entire generative hierarchy.</p>
<div class='callout'>
<p><strong>The score view.</strong> A direct computation from the closed form gives
<span class='m'>&nabla;<sub>x<sub>t</sub></sub> log q(x<sub>t</sub> | x<sub>0</sub>) =
&minus;&epsilon; / &radic;(1&minus;&#8113;<sub>t</sub>)</span>, so the trained network estimates the
<em>score</em> (gradient of the log-density) of each noisy marginal:
<span class='m'>s<sub>&theta;</sub>(x<sub>t</sub>) = &minus;&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) / &radic;(1&minus;&#8113;<sub>t</sub>)</span>.
Denoising is gradient ascent on log-probability; this identification (score-based generative
modeling, developed extensively in Ermon&rsquo;s group) unifies diffusion with stochastic
differential equations and is where samplers, guidance, and much else are derived from.</p>
</div>

<div class='deriv'>
  <div class='deriv-head'>
    <span class='deriv-title'>Faded derivation: forward kernel &rarr; L<sub>simple</sub></span>
    <button class='wbtn deriv-practice'>practice (hide all)</button>
    <button class='wbtn deriv-worked'>worked (show all)</button>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>1</span><span class='dstep-goal'>Compose the per-step Gaussians into a one-jump marginal q(x<sub>t</sub> | x<sub>0</sub>). What are its mean and variance?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>q(x<sub>t</sub> | x<sub>0</sub>) = &#119977;( &radic;&#8113;<sub>t</sub>&middot;x<sub>0</sub>, (1&minus;&#8113;<sub>t</sub>)&middot;I ),&nbsp;&nbsp;&#8113;<sub>t</sub> = &prod;<sub>s&le;t</sub>(1&minus;&beta;<sub>s</sub>)</div><p>Gaussians compose: means multiply, variances accumulate; variance preservation keeps the total at I.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>2</span><span class='dstep-goal'>Use Bayes to get the reverse step conditioned on the clean image, q(x<sub>t&minus;1</sub> | x<sub>t</sub>, x<sub>0</sub>). What form does it take?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&#119977;( &mu;&#771;<sub>t</sub>, &beta;&#771;<sub>t</sub>&middot;I ),&nbsp;&nbsp;&mu;&#771;<sub>t</sub> = __FRAC_MU1__&middot;x<sub>0</sub> + __FRAC_MU2__&middot;x<sub>t</sub>,&nbsp;&nbsp;&beta;&#771;<sub>t</sub> = __FRAC_BETA__&middot;&beta;<sub>t</sub></div><p>Product of two Gaussians in x<sub>t&minus;1</sub> is Gaussian; complete the square to read off mean and variance.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>3</span><span class='dstep-goal'>Write the NELBO and split it per timestep. Which three kinds of term appear?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&#120124;<sub>q</sub>[&minus;log p<sub>&theta;</sub>(x<sub>0</sub>|x<sub>1</sub>)] + &sum;<sub>t&ge;2</sub> &#120124;<sub>q</sub> KL( q(x<sub>t&minus;1</sub>|x<sub>t</sub>,x<sub>0</sub>) &Vert; p<sub>&theta;</sub>(x<sub>t&minus;1</sub>|x<sub>t</sub>) ) + KL( q(x<sub>T</sub>|x<sub>0</sub>) &Vert; &#119977;(0,I) )</div><p>Reconstruction + denoising-matching (the workhorse) + parameter-free prior (&asymp;0 by schedule design).</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>4</span><span class='dstep-goal'>Reparameterize the model mean &mu;<sub>&theta;</sub> in terms of a noise prediction &epsilon;<sub>&theta;</sub>.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&mu;<sub>&theta;</sub>(x<sub>t</sub>, t) = __FRAC_MUTHETA1__&middot;( x<sub>t</sub> &minus; __FRAC_MUTHETA2__&middot;&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) )</div><p>Solve x<sub>t</sub> = &radic;&#8113;<sub>t</sub>x<sub>0</sub> + &radic;(1&minus;&#8113;<sub>t</sub>)&epsilon; for x<sub>0</sub> and substitute into &mu;&#771;<sub>t</sub>.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>5</span><span class='dstep-goal'>Collapse the Gaussian KLs and drop the weights. What loss remains?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>L<sub>simple</sub> = &#120124;<sub>t, x<sub>0</sub>, &epsilon;</sub> &Vert; &epsilon; &minus; &epsilon;<sub>&theta;</sub>( &radic;&#8113;<sub>t</sub>&middot;x<sub>0</sub> + &radic;(1&minus;&#8113;<sub>t</sub>)&middot;&epsilon;, t ) &Vert;&sup2;</div><p>KL between Gaussians with shared variance is a scaled squared distance between means; DDPM found the unweighted version trains better.</p></div>
  </div>
  <div class='caption'>Try to produce each step on paper before revealing it. When you can go 1&rarr;5
  unaided, you own this derivation. (Steps start hidden; &ldquo;worked&rdquo; shows everything.)</div>
</div>

<h3>DDIM: same training, a fraction of the steps</h3>
<p>Now the observation that gives DDIM its power: <em>L<sub>simple</sub> depends on the forward
process only through the marginals</em> <span class='m'>q(x<sub>t</sub> | x<sub>0</sub>)</span>.
Song, Meng &amp; Ermon construct an entire family of <em>non-Markovian</em> forward processes,
indexed by noise scales &sigma;<sub>t</sub> &ge; 0, all sharing exactly those marginals &mdash; so a
network trained once serves every member of the family. Sampling first forms the model&rsquo;s
current best guess of the clean image:</p>
<div class='math'>x&#770;<sub>0</sub> = __FRAC_XHAT__</div>
<p>then jumps toward it:</p>
<div class='math'>x<sub>t&minus;1</sub> = &radic;&#8113;<sub>t&minus;1</sub>&middot;x&#770;<sub>0</sub>
&nbsp;+&nbsp; &radic;(1&minus;&#8113;<sub>t&minus;1</sub>&minus;&sigma;<sub>t</sub>&sup2;)&middot;&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t)
&nbsp;+&nbsp; &sigma;<sub>t</sub>&middot;z,&nbsp;&nbsp;&nbsp;z ~ &#119977;(0, I)</div>
<p>Three ingredients: rescaled <em>signal</em> (the clean estimate), a <em>direction</em> term that
re-applies the predicted noise at the new level, and <em>fresh randomness</em> &sigma;<sub>t</sub>z.
One particular choice of &sigma;<sub>t</sub> recovers stochastic DDPM sampling exactly. The
interesting extreme is <strong>&sigma;<sub>t</sub> = 0</strong>: the update becomes deterministic
&mdash; in the limit, the discretization of an ordinary differential equation &mdash; and
deterministic maps tolerate coarse discretization. You can sample on a sparse subsequence of
timesteps &tau;<sub>1</sub> &lt; &ctdot; &lt; &tau;<sub>S</sub> with S &asymp; 20&ndash;50 instead of
1,000, with the very same network and no retraining.</p>

<p>You can run this yourself. The widget below is a <em>genuine</em> DDIM sampler (&sigma; = 0) on a
toy two-dimensional data distribution &mdash; a Gaussian mixture arranged as a face, chosen because
its exact score is computable in closed form, so it plays the role of a perfectly-trained
&epsilon;<sub>&theta;</sub>. Pick a step count and watch the same &ldquo;model&rdquo;, from the same
starting noise, land differently:</p>

<div class='widget' id='w-ddim'>
  <div class='wctl' id='w-ddim-predict'>
    <label>Predict first &mdash; at S&nbsp;=&nbsp;1, the samples will form:</label>
    <button class='wbtn' data-pred='face'>a crisp face</button>
    <button class='wbtn' data-pred='blob'>a mush between the modes</button>
    <button class='wbtn' data-pred='ring'>uniform noise</button>
  </div>
  <div class='wstat' id='w-ddim-predfb' style='display:none'></div>
  <div class='wctl'>
    <label>steps S:</label>
    <button class='wbtn' data-s='1'>1</button>
    <button class='wbtn' data-s='2'>2</button>
    <button class='wbtn active' data-s='4'>4</button>
    <button class='wbtn' data-s='8'>8</button>
    <button class='wbtn' data-s='32'>32</button>
    <button class='wbtn' id='w-ddim-run'>&#9654; Run</button>
    <button class='wbtn' id='w-ddim-reset'>New noise</button>
  </div>
  <canvas id='w-ddim-cv' width='420' height='420'></canvas>
  <div class='wstat' id='w-ddim-stat'></div>
  <div class='caption'>Grey: true samples from the target mixture. Blue: the sampler&rsquo;s 420
  particles. With S = 1 every particle jumps straight to a posterior average &mdash; a mush between
  the modes. S = 4 finds the structure; S = 32 is crisp. Nothing about the model changed between
  runs: <em>only the sampler&rsquo;s step count</em>. This is the speed&ndash;quality dial made
  visible &mdash; and because &sigma; = 0 is deterministic, rerunning at the same S reproduces the
  identical picture.</div>
</div>

<p>The widget above runs live in your browser; the two figures below were <em>precomputed in
Python</em> (numpy running the identical sampler, 600 particles, S swept from 1 to 64). Drag the
slider to scrub the dial itself, or press play to sweep it:</p>

<div class='widget'>
__FIG_DDIM_SWEEP__
</div>

<p>And because this sampler is exact, we can put a number on what your eyes see &mdash; the average
negative log-likelihood of the final samples under the true mixture, as a function of S:</p>

<div class='widget'>
__FIG_QUALITY__
<div class='caption'>Hover for values. Quality improves steeply up to a few tens of steps, then
saturates &mdash; which is precisely why &ldquo;a few dozen steps&rdquo; became the DDIM regime, and
why a dLLM can afford to be 10&ndash;50&times; more parallel than a typewriter without giving the
quality back.</div>
</div>

<p>Hold on to the shape of this result: <em>the trained model fixes what is learnable; the sampler
decides how many steps you pay for.</em> The dLLM thesis is this punchline transplanted to text
&mdash; which requires rebuilding everything above for discrete tokens. That is the next section.</p>
"""

QUIZ_MATH1 = [
    {
        "question": "Suppose &#8113;<sub>t</sub> = 0.64. Which is the correct one-jump corruption distribution q(x<sub>t</sub> | x<sub>0</sub>)?",
        "options": [
            {"text": "&#119977;(0.8&middot;x<sub>0</sub>, 0.36&middot;I)", "correct": True,
             "explanation": "q(x<sub>t</sub>|x<sub>0</sub>) = 𝒩(√&#8113;<sub>t</sub>·x<sub>0</sub>, (1−&#8113;<sub>t</sub>)·I): the mean uses √0.64 = 0.8 while the variance uses 1 − 0.64 = 0.36. Mixing up where the square root lives is the classic slip."},
            {"text": "&#119977;(0.64&middot;x<sub>0</sub>, 0.36&middot;I)",
             "explanation": "The variance is right, but the mean coefficient is √&#8113;<sub>t</sub> = 0.8, not &#8113;<sub>t</sub> itself."},
            {"text": "&#119977;(0.8&middot;x<sub>0</sub>, 0.6&middot;I)",
             "explanation": "The mean is right, but the variance is 1 − &#8113;<sub>t</sub> = 0.36 — no square root on the variance term."},
            {"text": "&#119977;(0.64&middot;x<sub>0</sub>, 0.8&middot;I)",
             "explanation": "Both slots are off: mean √&#8113;<sub>t</sub> = 0.8, variance 1 − &#8113;<sub>t</sub> = 0.36."},
        ],
    },
    {
        "question": "What does the &radic;(1&minus;&beta;<sub>t</sub>) factor in the forward kernel accomplish?",
        "options": [
            {"text": "It keeps the chain's variance at I, shrinking signal exactly as noise grows.",
             "correct": True,
             "explanation": "Var(x<sub>t</sub>) = (1−β<sub>t</sub>)·I + β<sub>t</sub>·I = I — the variance-preserving property. Without the shrink, variance would blow up additively step after step."},
            {"text": "It guarantees the exact reverse of each step is also a Gaussian distribution.",
             "explanation": "Near-Gaussian reversals come from β<sub>t</sub> being small (many gentle steps), not from the scaling factor."},
            {"text": "It forces the chain's mean to zero regardless of what the data mean is.",
             "explanation": "The mean decays because the factor is < 1 compounded over steps, but the factor is chosen for the variance bookkeeping, not centering."},
            {"text": "It makes the noise schedule's cumulative product &#8113;<sub>t</sub> increase monotonically.",
             "explanation": "&#8113;<sub>t</sub> <em>decreases</em> monotonically toward 0 — that's the point: signal must die by time T."},
        ],
    },
    {
        "question": "In the ELBO decomposition, the prior term KL( q(x<sub>T</sub>|x<sub>0</sub>) &Vert; &#119977;(0, I) ) is special because it is...",
        "options": [
            {"text": "Parameter-free, and &asymp;0 whenever the schedule fully destroys the signal.",
             "correct": True,
             "explanation": "Neither q(x<sub>T</sub>|x<sub>0</sub>) nor 𝒩(0,I) contains θ, so it contributes no gradient; with &#8113;<sub>T</sub> ≈ 0 the two distributions nearly coincide and the KL vanishes. It's a schedule-design sanity check, not a training signal."},
            {"text": "The largest of the three terms, and therefore minimized first in training.",
             "explanation": "It's typically the <em>smallest</em> (≈0 by design), and with no parameters inside there is nothing to minimize."},
            {"text": "The only term that must be estimated by Monte-Carlo sampling.",
             "explanation": "All terms are estimated by sampling t and ε during training; this one happens to be computable in closed form and ignorable."},
            {"text": "Equal to the reconstruction term by the time-symmetry of the chain.",
             "explanation": "There is no such symmetry — reconstruction concerns the near-clean end (t=1), the prior term the fully-noised end (t=T)."},
        ],
    },
    {
        "question": "Set &sigma;<sub>t</sub> = 0 in the DDIM family. Which update rule results?",
        "options": [
            {"text": "x<sub>t&minus;1</sub> = &radic;&#8113;<sub>t&minus;1</sub>&middot;x&#770;<sub>0</sub> + &radic;(1&minus;&#8113;<sub>t&minus;1</sub>)&middot;&epsilon;<sub>&theta;</sub>",
             "correct": True,
             "explanation": "With σ = 0 the fresh-noise term dies and √(1−&#8113;<sub>t−1</sub>−σ²) becomes √(1−&#8113;<sub>t−1</sub>): rescaled clean estimate plus rescaled predicted noise, fully deterministic — the property that lets the sampler skip steps."},
            {"text": "x<sub>t&minus;1</sub> = &radic;&#8113;<sub>t&minus;1</sub>&middot;x<sub>t</sub> + &radic;(1&minus;&#8113;<sub>t&minus;1</sub>)&middot;&epsilon;<sub>&theta;</sub>",
             "explanation": "This reuses the noisy x<sub>t</sub> where the clean estimate x&#770;<sub>0</sub> belongs — the signal term must be rebuilt from the model's denoised guess."},
            {"text": "x<sub>t&minus;1</sub> = &radic;(1&minus;&#8113;<sub>t&minus;1</sub>)&middot;x&#770;<sub>0</sub> + &radic;&#8113;<sub>t&minus;1</sub>&middot;&epsilon;<sub>&theta;</sub>",
             "explanation": "Coefficients swapped: signal carries √&#8113;<sub>t−1</sub>, noise direction carries √(1−&#8113;<sub>t−1</sub>) — same bookkeeping as the forward marginal."},
            {"text": "x<sub>t&minus;1</sub> = x&#770;<sub>0</sub> + &radic;(1&minus;&#8113;<sub>t&minus;1</sub>)&middot;&epsilon;<sub>&theta;</sub>",
             "explanation": "The clean estimate must be rescaled by √&#8113;<sub>t−1</sub> to sit at the right signal level for time t−1; without it the marginals no longer match."},
        ],
    },
    {
        "question": "How is the trained noise-prediction network related to the score of the noisy marginal?",
        "options": [
            {"text": "s<sub>&theta;</sub>(x<sub>t</sub>) = &minus;&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) / &radic;(1&minus;&#8113;<sub>t</sub>)",
             "correct": True,
             "explanation": "From q(x<sub>t</sub>|x<sub>0</sub>): ∇ log q = −(x<sub>t</sub> − √&#8113;<sub>t</sub>x<sub>0</sub>)/(1−&#8113;<sub>t</sub>) = −ε/√(1−&#8113;<sub>t</sub>). Predicting noise is estimating the gradient field of the log-density — the bridge between DDPM and score-based SDEs."},
            {"text": "s<sub>&theta;</sub>(x<sub>t</sub>) = &minus;&radic;(1&minus;&#8113;<sub>t</sub>) &middot; &epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t)",
             "explanation": "Off by the square of the factor — you divide by √(1−&#8113;<sub>t</sub>), not multiply. As noise grows the score flattens, which the division captures."},
            {"text": "s<sub>&theta;</sub>(x<sub>t</sub>) = &epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t) / &radic;&#8113;<sub>t</sub>",
             "explanation": "Wrong sign and wrong normalizer: the score points <em>against</em> the added noise and is scaled by the noise level, not the signal level."},
            {"text": "s<sub>&theta;</sub>(x<sub>t</sub>) = &minus;&nabla;<sub>x</sub> &epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t)",
             "explanation": "No differentiation of the network is involved — the relation is a simple rescaling that falls out of the Gaussian closed form."},
        ],
    },
]

MATH2 = """
<p>Everything in the previous section assumed you can add and scale &mdash; operations tokens
don&rsquo;t support. This section rebuilds the pipeline for discrete data, following the notation of
the MDLM paper (Kuleshov&rsquo;s lab), which is the cleanest formulation and the one the later papers
build on. Time now runs continuously from t = 0 (clean) to t = 1 (fully corrupted).</p>

<h3>From Gaussians to categoricals</h3>
<p>Represent a token as a one-hot vector <span class='m'>x</span> over the vocabulary extended with a
special mask symbol <span class='m'>m</span>. Choose a strictly decreasing schedule
<span class='m'>&alpha;<sub>t</sub></span> with <span class='m'>&alpha;<sub>0</sub> &asymp; 1</span>
and <span class='m'>&alpha;<sub>1</sub> &asymp; 0</span>. The forward process corrupts every position
independently by interpolating, in probability, between the token and the mask:</p>
<div class='math'>q(z<sub>t</sub> | x) = Cat( z<sub>t</sub> ; &alpha;<sub>t</sub>&middot;x + (1&minus;&alpha;<sub>t</sub>)&middot;m )</div>
<p>In words: at time t the token has survived with probability &alpha;<sub>t</sub> and has been
absorbed into <code>[MASK]</code> with probability 1&minus;&alpha;<sub>t</sub> &mdash; and once
masked, always masked (the mask state is <em>absorbing</em>). This is the direct analogue of
<span class='m'>q(x<sub>t</sub> | x<sub>0</sub>)</span>: a one-jump formula requiring no simulation.
The general theory (D3PM) allows arbitrary transition matrices &mdash; uniform noise, nearest-neighbor
swaps &mdash; but the masking kernel is the one that turned out simple <em>and</em> strong.</p>

<h3>The reverse posterior</h3>
<p>Exactly as before, the reversal conditioned on the clean token is available in closed form. For
times s &lt; t (so &alpha;<sub>s</sub> &gt; &alpha;<sub>t</sub>), two cases:</p>
<ul>
<li><strong>Token visible at t:</strong> it stays itself with probability 1 (<em>carry-over</em>).
This is the frozen-token property, now visible as a theorem rather than a habit.</li>
<li><strong>Token masked at t:</strong></li>
</ul>
<div class='math'>q(z<sub>s</sub> | z<sub>t</sub> = m, x) = Cat( z<sub>s</sub> ;&nbsp; __FRAC_STAY__&middot;m &nbsp;+&nbsp; __FRAC_REVEAL__&middot;x )</div>
<p>So a masked token is <em>revealed</em> with probability
<span class='m'>(&alpha;<sub>s</sub>&minus;&alpha;<sub>t</sub>)/(1&minus;&alpha;<sub>t</sub>)</span>
&mdash; and when revealed, it is always the true token. At generation time we don&rsquo;t know
<span class='m'>x</span>, so the model supplies a prediction
<span class='m'>x<sub>&theta;</sub>(z<sub>t</sub>, t)</span>: a probability vector over the
vocabulary at every masked position, computed by one bidirectional transformer pass. MDLM constrains
this network in two ways, jointly called <strong>SUBS</strong>: <em>zero masking probability</em>
(the prediction never puts mass on the mask symbol) and <em>carry-over unmasking</em> (visible tokens
are copied through unchanged). These aren&rsquo;t cosmetic &mdash; substituting them into the ELBO
makes several terms vanish identically, a Rao-Blackwellization that removes estimation variance
rather than merely reducing it.</p>

<p>Play with the reveal probability directly &mdash; it is the formula your sampler applies at
every step, and the quiz below will ask you to compute it:</p>

<div class='widget' id='w-reveal'>
  <div class='wctl'>
    <label>&alpha;<sub>s</sub></label><input type='range' id='w-rev-as' min='0' max='100' value='75'>
    <label>&alpha;<sub>t</sub></label><input type='range' id='w-rev-at' min='0' max='100' value='25'>
  </div>
  <canvas id='w-rev-cv' width='560' height='68'></canvas>
  <div class='wstat' id='w-rev-stat'></div>
  <div class='caption'>The fate of a token that is masked at time t when the sampler steps to the
  less-noisy time s. The sliders clamp to keep &alpha;<sub>s</sub> &gt; &alpha;<sub>t</sub>. Note
  the endpoint behavior: as &alpha;<sub>t</sub> &rarr; 0 (very noisy) the reveal probability
  approaches &alpha;<sub>s</sub> itself, and a final step to &alpha;<sub>s</sub> = 1 reveals
  everything still masked.</div>
</div>

<p>The sliders show one point at a time; here is the entire landscape at once (hover anywhere for
the exact value; the blank triangle is the forbidden region where s would be noisier than t):</p>

<div class='widget'>
__FIG_REVEAL__
</div>

<h3>The objective: BERT, integrated</h3>
<p>With SUBS in place, the discrete-time ELBO telescopes, and taking the number of corruption levels
to infinity gives MDLM&rsquo;s continuous-time objective:</p>
<div class='math'>L<sub>&infin;</sub> = &int;<sub>0</sub><sup>1</sup> __FRAC_WEIGHT__ &middot;
&#120124;<sub>q</sub>&#8202;[ &sum;<sub>&#8467;: masked</sub> &minus;log &lang;x<sub>&theta;,&#8467;</sub>(z<sub>t</sub>, t), x<sub>&#8467;</sub>&rang; ] &nbsp;dt</div>
<p>Unpack it: sample a corruption level t, mask the sentence accordingly, and charge the model
cross-entropy for every masked position &mdash; <em>that inner sum is exactly BERT&rsquo;s
masked-language-modeling loss</em>. The integral averages it over all masking ratios, weighted by
<span class='m'>w(t) = &minus;&alpha;&prime;<sub>t</sub>/(1&minus;&alpha;<sub>t</sub>)</span>. For the
linear schedule &alpha;<sub>t</sub> = 1&minus;t this weight is 1/t: lightly-masked sentences carry
large per-token weight but contain few masked tokens, and the two effects balance. The punchline
deserves its own sentence: <strong>the venerable fill-in-the-blank objective, integrated over all
masking ratios, <em>is</em> a variational bound on log-likelihood.</strong> That is what lets dLLMs
report honest perplexities comparable against AR models &mdash; and MDLM used it to close most of the
historical gap.</p>
<div class='callout'>
<p><strong>Invariance theorem.</strong> MDLM proves the continuous-time NELBO&rsquo;s <em>value</em>
does not depend on the functional form of &alpha;<sub>t</sub> at all &mdash; only the endpoints
matter. Schedules still differ in gradient variance and in how you'd like to sample, but not in the
objective they optimize. (This is why cutting sampler steps needs no retraining: the model was never
trained for a step count in the first place.)</p>
</div>

<p>See the invariance from the other side &mdash; how different the schedules <em>look</em> while
optimizing the identical objective:</p>

<div class='widget'>
__FIG_SCHED__
<div class='caption'>Switch schedules with the buttons; hover for exact values (w is clipped at 6).
Every schedule blows up its weight near t = 0 (few masked tokens, each heavily weighted) and
differs everywhere else &mdash; yet the NELBO&rsquo;s value is identical for all three. What the
schedule really redistributes is <em>which corruption levels you practice most</em>, i.e. gradient
variance, not the optimization target.</div>
</div>

<div class='deriv'>
  <div class='deriv-head'>
    <span class='deriv-title'>Faded derivation: reveal probability &amp; the NELBO weight</span>
    <button class='wbtn deriv-practice'>practice (hide all)</button>
    <button class='wbtn deriv-worked'>worked (show all)</button>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>1</span><span class='dstep-goal'>Write the one-jump masking marginal q(z<sub>t</sub> | x).</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>q(z<sub>t</sub> | x) = Cat( z<sub>t</sub> ; &alpha;<sub>t</sub>&middot;x + (1&minus;&alpha;<sub>t</sub>)&middot;m )</div><p>Survive with probability &alpha;<sub>t</sub>, absorb into the mask otherwise; independent per position.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>2</span><span class='dstep-goal'>For s &lt; t, apply Bayes to a token that is masked at t: q(z<sub>s</sub> | z<sub>t</sub> = m, x) &prop; q(z<sub>t</sub> = m | z<sub>s</sub>) &middot; q(z<sub>s</sub> | x). Enumerate the two cases for z<sub>s</sub>.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><p>Case z<sub>s</sub> = x (still visible at s, masked between s and t): weight &alpha;<sub>s</sub> &middot; (probability of masking in (s,t]) = &alpha;<sub>s</sub>&middot;(1 &minus; &alpha;<sub>t</sub>/&alpha;<sub>s</sub>) = &alpha;<sub>s</sub> &minus; &alpha;<sub>t</sub>. Case z<sub>s</sub> = m (already masked by s): weight 1 &minus; &alpha;<sub>s</sub> (mask is absorbing, so it stays masked at t with probability 1).</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>3</span><span class='dstep-goal'>Normalize the two weights. What are the stay and reveal probabilities?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>P(stay) = __FRAC_STAY__,&nbsp;&nbsp;&nbsp;P(reveal) = __FRAC_REVEAL__</div><p>The weights sum to 1 &minus; &alpha;<sub>t</sub>, the total probability of being masked at t. When revealed, the token is always the true x.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>4</span><span class='dstep-goal'>Write the continuous-time NELBO as a weighted integral of masked cross-entropies. What is the weight?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>L<sub>&infin;</sub> = &int;<sub>0</sub><sup>1</sup> __FRAC_WEIGHT__ &middot; &#120124;<sub>q</sub>[ &sum;<sub>&#8467;: masked</sub> &minus;log &lang;x<sub>&theta;,&#8467;</sub>(z<sub>t</sub>, t), x<sub>&#8467;</sub>&rang; ] dt</div><p>The discrete-time bound telescopes; SUBS zeroes the other terms; the infinitesimal limit leaves w(t) = &minus;&alpha;&prime;<sub>t</sub>/(1&minus;&alpha;<sub>t</sub>).</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>5</span><span class='dstep-goal'>Specialize to the linear schedule &alpha;<sub>t</sub> = 1 &minus; t. What does the weight become, and why doesn&rsquo;t the blow-up at t &rarr; 0 hurt?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>w(t) = 1/t</div><p>&minus;&alpha;&prime;<sub>t</sub> = 1 and 1 &minus; &alpha;<sub>t</sub> = t. Near t = 0 each masked token is weighted heavily, but an expected fraction t of tokens is masked &mdash; the two factors cancel in expectation, keeping the integrand finite.</p></div>
  </div>
  <div class='caption'>Same drill: write each step before revealing. Step 2 is the one worth
  sweating &mdash; it is the only genuine derivation on this page that fits on an index card.</div>
</div>

<h3>Where the speed&ndash;quality trade-off lives, formally</h3>
<p>To generate with T steps, discretize t = 1 &rarr; 0 and repeatedly apply the reverse posterior
with <span class='m'>x<sub>&theta;</sub></span> substituted for x. Within one step, each masked
position reveals independently with probability
<span class='m'>(&alpha;<sub>s</sub>&minus;&alpha;<sub>t</sub>)/(1&minus;&alpha;<sub>t</sub>)</span>,
each sampling from its own marginal &mdash; the mean-field approximation from the Intuition section,
now with coordinates. As T &rarr; &infin; the per-step reveal probability shrinks until at most one
token flips per step and the factorization becomes exact; small T means many simultaneous,
uncoordinated reveals. <em>The step count is precisely the dial on mean-field error.</em></p>

<p>The modern extensions &mdash; Block Diffusion, ReMDM, guidance, diffu-GRPO &mdash; each
modify exactly one piece of the machinery above. Their equations get their own section next.</p>
"""

QUIZ_MATH2 = [
    {
        "question": "Take &alpha;<sub>s</sub> = 0.75 and &alpha;<sub>t</sub> = 0.25 (s the less-noisy time). A token masked at t is revealed when stepping to s with probability...",
        "options": [
            {"text": "2/3",
             "correct": True,
             "explanation": "(α<sub>s</sub> − α<sub>t</sub>)/(1 − α<sub>t</sub>) = 0.5/0.75 = 2/3. The denominator conditions on being masked at t — forgetting it is the standard error."},
            {"text": "1/3",
             "explanation": "That's the complementary stay-masked probability, (1 − α<sub>s</sub>)/(1 − α<sub>t</sub>) = 0.25/0.75."},
            {"text": "1/2",
             "explanation": "That's the bare difference α<sub>s</sub> − α<sub>t</sub> without normalizing by the probability 1 − α<sub>t</sub> of being masked at t in the first place."},
            {"text": "3/4",
             "explanation": "That's α<sub>s</sub> itself — the unconditional survival probability at s, not the conditional reveal probability for a token currently masked."},
        ],
    },
    {
        "question": "What do MDLM's two SUBS constraints (zero mask-probability and carry-over unmasking) actually buy?",
        "options": [
            {"text": "Several ELBO terms vanish identically, leaving a lower-variance weighted MLM loss.",
             "correct": True,
             "explanation": "Substituting the constraints makes whole terms of the bound zero <em>analytically</em> — Rao-Blackwellization. What survives is exactly the weighted masked cross-entropy, simple to implement and lower-variance to estimate."},
            {"text": "The sampler gains the ability to remask and revise already-committed tokens.",
             "explanation": "Revision is ReMDM's contribution, and it lives in the sampler. SUBS is about making the training objective exact and tight."},
            {"text": "The forward process no longer requires an absorbing mask state to work.",
             "explanation": "The absorbing mask state is the heart of the formulation; SUBS shapes the reverse-side parameterization built on top of it."},
            {"text": "The model's likelihood becomes exactly equal to that of an AR factorization.",
             "explanation": "The NELBO stays an upper bound on −log p; MDLM narrows the gap to AR models but no equality is claimed."},
        ],
    },
    {
        "question": "MDLM's continuous-time NELBO is provably invariant to...",
        "options": [
            {"text": "The functional form of the schedule &alpha;<sub>t</sub>, endpoints held fixed.",
             "correct": True,
             "explanation": "A change of variables absorbs the schedule: any strictly-decreasing α<sub>t</sub> from ≈1 to ≈0 yields the same objective value. Schedules still affect gradient variance — but not what is being optimized."},
            {"text": "The length of the sequences that the model is trained on.",
             "explanation": "Length isn't invariant-out — the loss sums over positions; longer sequences mean more terms."},
            {"text": "The size of the vocabulary the tokenizer produces.",
             "explanation": "Cross-entropy depends directly on the support size; per-token likelihoods are not vocabulary-invariant."},
            {"text": "The parameterization chosen for the denoiser x<sub>&theta;</sub>.",
             "explanation": "Parameterization matters a great deal — SUBS is precisely a parameterization choice that tightens and simplifies the bound."},
        ],
    },
    {
        "question": "With the linear schedule &alpha;<sub>t</sub> = 1 &minus; t, the weight w(t) multiplying the masked cross-entropy at corruption level t is...",
        "options": [
            {"text": "1/t",
             "correct": True,
             "explanation": "w(t) = −α′<sub>t</sub>/(1 − α<sub>t</sub>) = 1/t for the linear schedule. Lightly-masked examples (small t) get a large per-token weight but contribute few masked tokens — the effects balance across the integral."},
            {"text": "t",
             "explanation": "Inverted: the weight <em>decreases</em> with corruption level under the linear schedule — heavily-masked examples get weight ≈1, not ≈0."},
            {"text": "1/(1&minus;t)",
             "explanation": "That's 1/α<sub>t</sub> — the survival probability's reciprocal, not −α′<sub>t</sub>/(1 − α<sub>t</sub>) = 1/t."},
            {"text": "e<sup>&minus;t</sup>",
             "explanation": "Nothing exponential appears: −α′<sub>t</sub> = 1 and 1 − α<sub>t</sub> = t give exactly 1/t."},
        ],
    },
    {
        "question": "Under the linear schedule the NELBO weight w(t) = 1/t blows up as t &rarr; 0. Why doesn't the training objective diverge?",
        "options": [
            {"text": "The expected number of masked tokens shrinks like t, cancelling the 1/t weight.",
             "correct": True,
             "explanation": "At corruption level t an expected fraction t of tokens is masked, so the inner sum has ~tL terms while each carries weight 1/t — the product stays finite. Lightly-masked sentences are rare practice but heavily-weighted practice, and the two effects balance in expectation."},
            {"text": "The integral is truncated at a minimum corruption level t_min > 0 in practice.",
             "explanation": "No truncation is needed — implementations sample t uniformly on (0,1); the estimator has finite expectation precisely because of the masked-count cancellation."},
            {"text": "SUBS removes the small-t terms from the bound entirely.",
             "explanation": "SUBS (zero mask-probability + carry-over) zeroes different terms — those for unmasked positions — at every t, not the small-t region."},
            {"text": "The schedule's derivative &alpha;&prime;<sub>t</sub> vanishes at t = 0, taming the ratio.",
             "explanation": "For the linear schedule α′<sub>t</sub> = −1 everywhere — constant, not vanishing. The rescue comes from the data (few masked tokens), not the schedule."},
        ],
    },
]

MATH_SOTA = """
<p>The Papers &amp; Sources part tells the story of the modern diffusion-LM methods; this
section shows their actual mathematics. Each of the four ideas below modifies exactly one piece
of the Maths II machinery &mdash; the factorization, the sampler&rsquo;s kernel, the per-step
distribution, or the training signal.</p>

<h3>Block Diffusion: the block-factorized bound</h3>
<p>Chop the sequence into blocks x<sup>(1)</sup>, x<sup>(2)</sup>, &hellip; of length
L&prime; and put the autoregressive chain rule at block granularity only:</p>
<div class='math'>&minus;log p<sub>&theta;</sub>(x) &nbsp;&le;&nbsp; &sum;<sub>b</sub> L<sub>MDLM</sub>( x<sup>(b)</sup> | x<sup>(&lt;b)</sup> )</div>
<p>Each summand is the Maths II NELBO for one block, conditioned on all previous blocks through
attention to their (frozen, cacheable) keys and values. The block length is a genuine dial
between the two worlds: <strong>at L&prime; = 1 each &ldquo;block&rdquo; is a single token whose
only possible reveal is next-token prediction &mdash; the bound collapses exactly to the
autoregressive cross-entropy</strong>; at L&prime; = n you recover vanilla MDLM. In between:
variable-length generation, KV caching across blocks, parallel diffusion within them. Why the
cache is legal here and not in vanilla MDLM is worth stating precisely: cached K/V are valid
only if the tokens that produced them never change. Vanilla masked diffusion re-edits the whole
canvas every pass; BD3-LM&rsquo;s committed blocks are frozen by construction. (The paper adds
engineering: a vectorized two-forward-pass training scheme, and data-driven noise schedules
tuned per block length to cut gradient variance.)</p>

<h3>ReMDM: the remasking kernel, with its feasibility bound</h3>
<p>Vanilla masked diffusion&rsquo;s reverse kernel carries revealed tokens over with probability
1. ReMDM replaces the sampler&rsquo;s kernel: a token currently revealed as x returns to
<code>[MASK]</code> with probability &sigma;<sub>t</sub>, and stays with probability
1&minus;&sigma;<sub>t</sub>. For the pretrained model to remain valid, the per-token forward
marginals Cat(&alpha;<sub>s</sub>x + (1&minus;&alpha;<sub>s</sub>)m) must be preserved &mdash;
which forces the reveal probability for masked tokens up to compensate:</p>
<div class='math'>P(reveal at s) = """ + frac("&alpha;<sub>s</sub> &minus; (1&minus;&sigma;<sub>t</sub>)&thinsp;&alpha;<sub>t</sub>", "1 &minus; &alpha;<sub>t</sub>") + """</div>
<p>Requiring this to be a probability (&le; 1) caps the remasking rate:</p>
<div class='math'>0 &le; &sigma;<sub>t</sub> &le; min( 1, """ + frac("1 &minus; &alpha;<sub>s</sub>", "&alpha;<sub>t</sub>") + """ )</div>
<p>Within that band, any schedule works with the same trained network &mdash; a remasked token
is statistically indistinguishable from an ordinarily masked one. Read it as a
predictor&ndash;corrector loop for text (re-noise a little, re-denoise better): the remasking
budget converts extra sampling steps into extra rounds of self-correction, which is where
dLLMs&rsquo; inference-time scaling comes from.</p>

<h3>Discrete guidance: sharpening by ratio</h3>
<p>Classifier-free guidance ports to discrete diffusion per position, per step. With a
conditional and an unconditional prediction for slot &#8467;, sample from the renormalized
sharpened product:</p>
<div class='math'>p&#771;<sub>&gamma;</sub>(z<sub>s,&#8467;</sub> | z<sub>t</sub>, c) &nbsp;&prop;&nbsp;
p<sub>&theta;</sub>(z<sub>s,&#8467;</sub> | z<sub>t</sub>) &middot;
( p<sub>&theta;</sub>(z<sub>s,&#8467;</sub> | z<sub>t</sub>, c) / p<sub>&theta;</sub>(z<sub>s,&#8467;</sub> | z<sub>t</sub>) )<sup>&gamma;</sup></div>
<p>Equivalently in logits: &#8467;&#771; = &#8467;<sub>u</sub> + &gamma;(&#8467;<sub>c</sub> &minus; &#8467;<sub>u</sub>).
The dial reads: &gamma; = 0 ignores the condition, &gamma; = 1 is ordinary conditional sampling,
&gamma; &gt; 1 exaggerates whatever direction the condition pulls &mdash; a token whose
conditional-to-unconditional ratio is 2 gets its weight multiplied by 2<sup>&gamma;</sup> before
renormalization. Because tokens are discrete there is no gradient to follow, so the
classifier-<em>based</em> variant instead reweights each candidate token by a classifier&rsquo;s
judgment p<sub>&phi;</sub>(c | z with that token)<sup>&gamma;</sup> &mdash; same trick whether
c is a sentiment, a molecular property, or a DNA motif.</p>

<h3>diffu-GRPO: policy gradients without left-to-right log-probs</h3>
<p>GRPO&rsquo;s objective needs per-token importance ratios
&rho;<sub>k</sub> = &pi;<sub>&theta;</sub>(o<sub>k</sub>|q) / &pi;<sub>old</sub>(o<sub>k</sub>|q)
and group-relative advantages computed over G sampled completions of the same prompt:</p>
<div class='math'>A&#770;<sub>i</sub> = """ + frac("r<sub>i</sub> &minus; mean(r<sub>1..G</sub>)", "std(r<sub>1..G</sub>)") + """</div>
<p>&mdash; no value network. The obstacle: a dLLM defines no left-to-right factorization
&pi;<sub>&theta;</sub>(o<sub>k</sub> | o<sub>&lt;k</sub>). d1&rsquo;s estimator replaces it with
a <em>one-step mean-field</em> probe: mask the entire completion, run one forward pass, and read
each token&rsquo;s probability from its own position&rsquo;s softmax. Cheap enough to sit inside
an RL loop; biased, but consistently so across the ratio. The second trick: re-mask the
<em>prompt</em> randomly at each gradient step, so successive log-prob estimates decorrelate
&mdash; acting as a regularizer and letting one batch of rollouts support many updates. The
result is the standard capability playbook (SFT, then RL) running natively on diffusion.</p>

<p>You now hold every equation the story needs. What remains is to see how the nine papers
deploy them &mdash; and that reading is now almost leisurely.</p>
"""

QUIZ_MATH_SOTA = [
    {
        "question": "Set the block length L&prime; = 1 in Block Diffusion. What does the training objective become?",
        "options": [
            {"text": "Exactly the autoregressive next-token cross-entropy.",
             "correct": True,
             "explanation": "A one-token block's only masked-diffusion event is revealing that token given everything before it — the block NELBO collapses to −log p(x_k | x_<k), summed: the AR chain rule. That exactness is what makes BD3-LM a true interpolation, not an approximation, between the two model families."},
            {"text": "Exactly the vanilla MDLM NELBO over the whole sequence.",
             "explanation": "That's the OTHER endpoint: L′ = n (one block spanning everything). The dial runs AR at L′=1 to vanilla diffusion at L′=n."},
            {"text": "A KL divergence between the AR and diffusion posteriors.",
             "explanation": "No such cross-model term appears — the bound is a sum of per-block diffusion losses whose limit happens to BE the AR loss."},
            {"text": "An undefined quantity, since diffusion needs at least two tokens per block.",
             "explanation": "Nothing breaks at one token; the masking process over a single position is perfectly well-defined, and its NELBO is the standard cross-entropy."},
        ],
    },
    {
        "question": "ReMDM caps the remasking rate at &sigma;<sub>t</sub> &le; (1&minus;&alpha;<sub>s</sub>)/&alpha;<sub>t</sub>. What breaks beyond the cap?",
        "options": [
            {"text": "The compensating reveal probability would have to exceed 1 to keep the marginals.",
             "correct": True,
             "explanation": "Marginal preservation forces P(reveal) = (α_s − (1−σ_t)α_t)/(1−α_t); push σ_t past the cap and this exceeds 1 — no valid kernel exists. The bound is exactly where 'remask freely' collides with 'stay statistically indistinguishable from training'."},
            {"text": "The KV cache of previously committed blocks is invalidated.",
             "explanation": "Cache mechanics belong to Block Diffusion; vanilla ReMDM re-reads the canvas anyway. The cap is a probability constraint, not an engineering one."},
            {"text": "The training NELBO diverges and the model must be re-fit.",
             "explanation": "ReMDM never touches training — it's a sampler modification; the failure past the cap is that the sampler's distribution stops being a distribution."},
            {"text": "Guidance and remasking begin to conflict at the shared positions.",
             "explanation": "Guidance composes with any reveal distribution — the two mechanisms are orthogonal; nothing about the cap involves conditioning."},
        ],
    },
    {
        "question": "At one position the conditional/unconditional ratio is 2 and the guidance scale is &gamma; = 2. Before renormalization, the unconditional probability gets multiplied by:",
        "options": [
            {"text": "4",
             "correct": True,
             "explanation": "The reweighting is (p_c/p_u)^γ = 2² = 4. In logit form: ℓ_u + γ(ℓ_c − ℓ_u) adds twice the log-ratio, same thing. γ exponentiates the ratio — it doesn't scale it linearly."},
            {"text": "2",
             "explanation": "That's γ = 1 — ordinary conditional sampling. Guidance at γ = 2 squares the ratio."},
            {"text": "8",
             "explanation": "8 would be ratio 2 at γ = 3 (2³); the exponent is γ itself, not γ+1."},
            {"text": "&radic;2",
             "explanation": "√2 is γ = ½ — a SOFTENING of the condition, the opposite direction of the sharpening asked about."},
        ],
    },
    {
        "question": "diffu-GRPO (the d1 paper) needs per-token log-probs, which a dLLM doesn't define left-to-right. How does it estimate them?",
        "options": [
            {"text": "One forward pass with the completion fully masked; read per-position probabilities.",
             "correct": True,
             "explanation": "A mean-field one-step estimate — cheap enough to run inside an RL loop. Random re-masking of the prompt at each gradient step decorrelates successive estimates and regularizes, enabling several updates per batch of rollouts."},
            {"text": "Averaging log-probabilities over full multi-step reverse trajectories.",
             "explanation": "That's the expensive quasi-exact route the one-step estimator exists to avoid — running it inside RL would multiply training cost by the step count."},
            {"text": "A separate learned value network that scores each token's contribution.",
             "explanation": "GRPO's defining feature is having no value network — advantages are group-relative statistics of sampled completions' rewards."},
            {"text": "Chain-rule decomposition after ordering tokens by model confidence.",
             "explanation": "No ordering rescues a left-to-right factorization the model doesn't define; d1 embraces the parallel mean-field estimate instead."},
        ],
    },
    {
        "question": "Why is KV caching sound across Block Diffusion's blocks but not within vanilla masked diffusion's passes?",
        "options": [
            {"text": "Cached keys/values stay valid only for tokens that never change; committed blocks are frozen, the vanilla canvas is not.",
             "correct": True,
             "explanation": "Bidirectional attention over a canvas that is re-edited every pass would make every cached K/V stale immediately. BD3-LM's block-causal structure guarantees prefix blocks never change after commitment — exactly the invariant a cache needs."},
            {"text": "Masked positions produce no keys or values, so vanilla models have nothing to cache.",
             "explanation": "Masked tokens participate in attention like any other token (that's how blanks see context); the problem is their values CHANGING across passes, not their absence."},
            {"text": "KV caching mathematically requires a strictly causal attention mask.",
             "explanation": "Close but too strong: caching requires immutability of the attended-to tokens, which block-level causality provides while full causality is merely one sufficient case — BD3 caches with bidirectional attention inside blocks."},
            {"text": "Vanilla diffusion could cache, but the memory cost exceeds recomputation.",
             "explanation": "It's a correctness issue, not a cost trade-off — stale K/V from a re-edited canvas give wrong attention outputs."},
        ],
    },
]

PAPERS = """
<p>One section for the whole literature: the narrative first, then the complete reference list.
Methodology: this page was built from Inception Labs&rsquo;
<a href='https://www.inceptionlabs.ai/about'>about page</a> (accessed 2026-08-01) &mdash; the
source of the nine-paper &ldquo;Some of the technologies we&rsquo;ve developed&rdquo; list, the
&ldquo;typewriter/editor&rdquo; framing, and the Mercury claims (first commercially available
dLLM family; ~5&times; speed), reported here as the company&rsquo;s own. The math sections
follow DDIM&rsquo;s notation (continuous case) and MDLM&rsquo;s (discrete case). Errors of
interpretation are this page&rsquo;s, not the sources&rsquo;.</p>
<p>The nine links split cleanly into two groups: four papers that are general-purpose LLM
foundations from the founders&rsquo; earlier careers, and five that form the diffusion-LLM
research line proper &mdash; each one removing an obstacle you can now name in equations
(<a href='#math-sota'>Maths III</a> derives them).</p>

<h3>Group 1: the foundations</h3>

<h4>Diffusion Models &mdash; DDIM (Song, Meng &amp; Ermon, 2020)</h4>
<p>The paper behind the page&rsquo;s &ldquo;Diffusion Models&rdquo; link is
<a href='https://arxiv.org/abs/2010.02502'>Denoising Diffusion Implicit Models</a> &mdash; the
&sigma;<sub>t</sub>-indexed sampler family derived at the end of Maths I. Original diffusion models
(DDPM) needed on the order of a thousand denoising steps per image; DDIM&rsquo;s deterministic member
of the family produces comparable quality in a few dozen, with the same trained network. Thematically
it&rsquo;s the company thesis in embryo: <em>the step count is a dial, and driving it down is how you
win on speed.</em></p>

<h4>FlashAttention (Dao et al., 2022)</h4>
<p><a href='https://arxiv.org/abs/2205.14135'>FlashAttention</a> (Ermon is a co-author) computes
<em>exact</em> transformer attention in a way that respects the GPU memory hierarchy: Q, K, V are
processed in tiles sized to fit on-chip SRAM, and the N&times;N attention matrix is never
materialized in slow HBM (the backward pass recomputes what it needs rather than storing it). No
approximation, no sparsity &mdash; just radically less memory traffic. Its presence on the list
signals that Inception&rsquo;s speed claims rest on systems engineering as much as on modeling: a
dLLM&rsquo;s big parallel refinement passes are only fast if the kernels are.</p>

<h4>Decision Transformers (Chen et al., 2021)</h4>
<p><a href='https://arxiv.org/abs/2106.01345'>Decision Transformer</a> (Grover is a co-author)
reframed reinforcement learning as sequence modeling: condition a transformer on the
<em>return-to-go</em> you want, and it predicts the actions that achieve it. No value functions, no
policy-gradient machinery &mdash; just next-token prediction over trajectories. It matters here as an
early demonstration that transformers are general sequence engines whose training paradigm can absorb
whole other fields &mdash; the same conviction that says text generation need not be autoregressive.</p>

<h4>Direct Preference Optimization (Rafailov et al., 2023)</h4>
<p><a href='https://arxiv.org/abs/2305.18290'>DPO</a> (Ermon again) made aligning models to human
preferences radically simpler. Classic RLHF fits a reward model to preference pairs and then runs PPO
against it. DPO notices that the KL-regularized RL objective has a closed-form optimal policy,
<span class='m'>&pi;*(y|x) &prop; &pi;<sub>ref</sub>(y|x)&middot;exp(r(x,y)/&beta;)</span>, inverts it
to express the reward as a log-ratio of policies, and substitutes that into the Bradley&ndash;Terry
preference likelihood &mdash; collapsing the whole pipeline into one classification-style loss on
preference pairs. No reward model, no RL loop. Any production model family, Mercury included, needs
an alignment recipe; DPO is the founders&rsquo; contribution to that toolbox.</p>

<h3>Group 2: the diffusion-LLM line</h3>
<p>Read these five in this order &mdash; each solves a problem the previous one exposes.</p>

<h4>1. Masked Diffusion &mdash; MDLM (Sahoo et al., 2024)</h4>
<p><a href='https://arxiv.org/abs/2406.07524'>Simple and Effective Masked Diffusion Language
Models</a> is the workhorse recipe &mdash; the entire Maths II section up to the extensions.
Its contributions: the SUBS parameterization that Rao-Blackwellizes the bound, the clean
weighted-MLM continuous-time objective, and the schedule-invariance theorem. With them, masked
diffusion went from mathematically intricate and badly lagging to simple and nearly
perplexity-competitive with AR training &mdash; the proof that the approach deserved scaling.</p>

<h4>2. Block Diffusion &mdash; BD3-LM (Arriola et al., 2025)</h4>
<p>Vanilla masked diffusion paints on a fixed-size canvas and re-reads the whole canvas every pass,
forfeiting KV caching. <a href='https://arxiv.org/abs/2503.09573'>Block Diffusion</a> puts the AR
chain rule at block granularity &mdash; <span class='m'>&sum;<sub>b</sub> L<sub>MDLM</sub>(x<sup>(b)</sup> | x<sup>(&lt;b)</sup>)</span>
&mdash; so each block diffuses in parallel while conditioning on cached previous blocks.</p>

<div class='diagram'>
  <div class='flow'>
    <div class='box ok'>Block 1<small>done &middot; cached</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box accent'>Block 2<small>diffusing in parallel now</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box dim'>Block 3<small>not started</small></div>
  </div>
  <div class='caption'>Autoregressive <em>across</em> blocks (variable length, KV caching),
  diffusion <em>within</em> blocks (parallel speed). Block size 1 is pure AR; block size = sequence
  length is vanilla diffusion; production lives in between.</div>
</div>

<h4>3. Remasking Diffusion &mdash; ReMDM (Wang et al., 2025)</h4>
<p><a href='https://remdm.github.io/'>Remasking Discrete Diffusion Models</a> fixes the frozen-token
theorem: a modified sampler kernel that returns committed tokens to <code>[MASK]</code> with
probability &sigma;<sub>t</sub>, constructed to preserve the forward marginals &mdash; which is
precisely why it needs no retraining. This is what makes a dLLM a true editor, and it gives dLLMs an
inference-time scaling law: more sampling steps &rarr; more self-correction &rarr; measurably better
text.</p>

<h4>4. Discrete Diffusion Guidance (Schiff et al., 2024)</h4>
<p><a href='https://arxiv.org/abs/2412.10193'>Simple Guidance Mechanisms for Discrete Diffusion
Models</a> ports image diffusion&rsquo;s best control trick to discrete data: the
&gamma;-exponentiated conditional/unconditional ratio derived in Maths III, applied per position per step,
plus a classifier-based variant for when no conditional model exists. Turn &gamma; and you push
generation toward a sentiment, a style, or drug-likeness when the &ldquo;tokens&rdquo; are molecule
fragments &mdash; at sampling time, no retraining.</p>

<h4>5. d1 Reasoning (Zhao et al., 2025)</h4>
<p><a href='https://arxiv.org/abs/2504.12216'>d1</a> (from Grover&rsquo;s lab) answers the
frontier-relevance question: can dLLMs do the long chain-of-thought reasoning that made o1-style
models famous? The recipe: supervised fine-tuning on worked reasoning traces, then diffu-GRPO
&mdash; the one-step log-prob estimator and group-relative advantages derived in Maths III. It works:
reasoning improves substantially, showing the modern capability playbook transfers to diffusion.
Note the pleasing symmetry with the foundations group: RL-as-sequence-modeling (Decision
Transformer) and preference training (DPO) were both founder groundwork for this step.</p>

<h3>How it adds up</h3>
<p>Stack the five contributions and you get the spec sheet for <strong>Mercury</strong>,
Inception&rsquo;s commercial model family and, by their description, the first commercially available
diffusion LLM: MDLM&rsquo;s training recipe, Block Diffusion&rsquo;s variable-length caching,
ReMDM&rsquo;s self-correction, guidance for control, and d1&rsquo;s reasoning recipe &mdash; running
on FlashAttention-class kernels, aligned with DPO-class methods, chasing DDIM&rsquo;s goal of
ever-fewer steps. Inception claims roughly 5&times; the speed of comparable AR models; whether dLLMs
reach frontier quality is still an open bet, but you now know exactly which obstacles have been
cleared, which papers cleared them, and with which equations.</p>

<h3>The complete reference list</h3>
<ol>
<li>Inception Labs, <a href='https://www.inceptionlabs.ai/about'>&ldquo;About&rdquo;</a> &mdash;
inceptionlabs.ai/about (accessed 2026-08-01). <em>The primary source: the nine-paper list and
the Mercury claims.</em></li>
<li>Jiaming Song, Chenlin Meng, Stefano Ermon (2020).
<a href='https://arxiv.org/abs/2010.02502'>Denoising Diffusion Implicit Models</a>. ICLR 2021.
arXiv:2010.02502.</li>
<li>Lili Chen, Kevin Lu, Aravind Rajeswaran, Kimin Lee, Aditya Grover, Michael Laskin,
Pieter Abbeel, Aravind Srinivas, Igor Mordatch (2021).
<a href='https://arxiv.org/abs/2106.01345'>Decision Transformer: Reinforcement Learning via
Sequence Modeling</a>. NeurIPS 2021. arXiv:2106.01345.</li>
<li>Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, Christopher R&eacute; (2022).
<a href='https://arxiv.org/abs/2205.14135'>FlashAttention: Fast and Memory-Efficient Exact
Attention with IO-Awareness</a>. NeurIPS 2022. arXiv:2205.14135.</li>
<li>Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning,
Chelsea Finn (2023). <a href='https://arxiv.org/abs/2305.18290'>Direct Preference Optimization:
Your Language Model is Secretly a Reward Model</a>. NeurIPS 2023. arXiv:2305.18290.</li>
<li>Subham Sekhar Sahoo, Marianne Arriola, Yair Schiff, Aaron Gokaslan, Edgar Marroquin,
Justin T. Chiu, Alexander Rush, Volodymyr Kuleshov (2024).
<a href='https://arxiv.org/abs/2406.07524'>Simple and Effective Masked Diffusion Language
Models</a>. NeurIPS 2024. arXiv:2406.07524.</li>
<li>Yair Schiff, Subham Sekhar Sahoo, Hao Phung, Guanghan Wang, Sam Boshar, Hugo Dalla-torre,
Bernardo P. de Almeida, Alexander Rush, Thomas Pierrot, Volodymyr Kuleshov (2024).
<a href='https://arxiv.org/abs/2412.10193'>Simple Guidance Mechanisms for Discrete Diffusion
Models</a>. ICLR 2025. arXiv:2412.10193.</li>
<li>Marianne Arriola, Aaron Gokaslan, Justin T. Chiu, Zhihan Yang, Zhixuan Qi, Jiaqi Han,
Subham Sekhar Sahoo, Volodymyr Kuleshov (2025).
<a href='https://arxiv.org/abs/2503.09573'>Block Diffusion: Interpolating Between Autoregressive
and Diffusion Language Models</a>. ICLR 2025. arXiv:2503.09573.</li>
<li>Guanghan Wang, Yair Schiff, Subham Sekhar Sahoo, Volodymyr Kuleshov (2025).
<a href='https://remdm.github.io/'>Remasking Discrete Diffusion Models</a> (project page, as
linked by the primary source). arXiv preprint, 2025.</li>
<li>Siyan Zhao, Devaansh Gupta, Qinqing Zheng, Aditya Grover (2025).
<a href='https://arxiv.org/abs/2504.12216'>d1: Scaling Reasoning in Diffusion Large Language
Models via Reinforcement Learning</a>. arXiv preprint, 2025.</li>
<li>Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, Surya Ganguli (2015).
<a href='https://arxiv.org/abs/1503.03585'>Deep Unsupervised Learning using Nonequilibrium
Thermodynamics</a>. ICML 2015. <em>The origin of the diffusion idea.</em></li>
<li>Jonathan Ho, Ajay Jain, Pieter Abbeel (2020).
<a href='https://arxiv.org/abs/2006.11239'>Denoising Diffusion Probabilistic Models</a>.
NeurIPS 2020. <em>The forward process, posterior, and L<sub>simple</sub> of Maths&nbsp;I.</em></li>
<li>Yang Song, Stefano Ermon (2019).
<a href='https://arxiv.org/abs/1907.05600'>Generative Modeling by Estimating Gradients of the
Data Distribution</a>. NeurIPS 2019. <em>The score-based view.</em></li>
<li>Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon,
Ben Poole (2020). <a href='https://arxiv.org/abs/2011.13456'>Score-Based Generative Modeling
through Stochastic Differential Equations</a>. ICLR 2021. <em>The SDE unification in the score
callout.</em></li>
<li>Jacob Austin, Daniel D. Johnson, Jonathan Ho, Daniel Tarlow, Rianne van den Berg (2021).
<a href='https://arxiv.org/abs/2107.03006'>Structured Denoising Diffusion Models in Discrete
State-Spaces</a> (D3PM). NeurIPS 2021. <em>The general discrete framework; masking is its
absorbing-state case.</em></li>
<li>Jacob Devlin, Ming-Wei Chang, Kenton Lee, Kristina Toutanova (2018).
<a href='https://arxiv.org/abs/1810.04805'>BERT: Pre-training of Deep Bidirectional Transformers
for Language Understanding</a>. NAACL 2019. <em>The MLM loss that MDLM integrates.</em></li>
<li>Jonathan Ho, Tim Salimans (2022).
<a href='https://arxiv.org/abs/2207.12598'>Classifier-Free Diffusion Guidance</a>.
<em>The mechanism ported to discrete diffusion; a quiz reference.</em></li>
<li>Tim Salimans, Jonathan Ho (2022).
<a href='https://arxiv.org/abs/2202.00512'>Progressive Distillation for Fast Sampling of
Diffusion Models</a>. ICLR 2022. <em>The other route to fewer steps; a quiz distractor.</em></li>
<li>Alex Nichol, Prafulla Dhariwal (2021).
<a href='https://arxiv.org/abs/2102.09672'>Improved Denoising Diffusion Probabilistic
Models</a>. ICML 2021. <em>Schedule learning; a quiz distractor.</em></li>
<li>Diederik P. Kingma, Tim Salimans, Ben Poole, Jonathan Ho (2021).
<a href='https://arxiv.org/abs/2107.00630'>Variational Diffusion Models</a>. NeurIPS 2021.
<em>Likelihood-focused training; a quiz distractor.</em></li>
<li>William Peebles, Saining Xie (2022).
<a href='https://arxiv.org/abs/2212.09748'>Scalable Diffusion Models with Transformers</a>
(DiT). ICCV 2023. <em>Architecture as the other speed axis; a quiz distractor.</em></li>
<li>Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y.K. Li,
Y. Wu, Daya Guo (2024). <a href='https://arxiv.org/abs/2402.03300'>DeepSeekMath: Pushing the
Limits of Mathematical Reasoning in Open Language Models</a>. <em>The GRPO that diffu-GRPO
adapts.</em></li>
<li>Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin,
Ji-Rong Wen, Chongxuan Li (2025). <a href='https://arxiv.org/abs/2502.09992'>Large Language
Diffusion Models</a> (LLaDA). <em>The pretrained dLLM d1 builds on.</em></li>
</ol>

<p>Suggested reading order if you want to go to the sources: skim <strong>MDLM</strong> first
(its intro is the best short statement of the whole field, and Maths II is your map of its
notation), then <strong>Block Diffusion</strong> and <strong>ReMDM</strong> for the two big
fixes (Maths III has their equations), <strong>d1</strong> for where this is headed, and keep
<strong>DDIM</strong> for when you want the image-diffusion roots in full mathematical dress.</p>
"""

QUIZ_PAPERS = [
    {
        "question": "Which pairing of vanilla-masked-diffusion limitation → fixing paper is correct?",
        "options": [
            {"text": "Fixed-length canvas and lost KV caching → Block Diffusion.",
             "correct": True,
             "explanation": "AR-across-blocks restores variable length and cached prefixes; diffusion-within-blocks keeps the parallelism. The other three pairings each swap in the wrong paper: frozen tokens is ReMDM's target, steering is Guidance, likelihood competitiveness is MDLM."},
            {"text": "Frozen committed tokens → Discrete Diffusion Guidance.",
             "explanation": "Frozen tokens are ReMDM's territory (the marginal-preserving remasking sampler). Guidance is about steering generation toward attributes."},
            {"text": "No steerability toward attributes → Remasking Diffusion.",
             "explanation": "Steering is the Guidance paper's contribution (the γ-sharpened conditional ratio). ReMDM is about revising committed tokens."},
            {"text": "Uncompetitive likelihoods versus AR models → d1 Reasoning.",
             "explanation": "Closing the perplexity gap was MDLM's achievement; d1 is about adding reasoning through SFT + diffusion-adapted RL."},
        ],
    },
    {
        "question": "Why does ReMDM work with an off-the-shelf pretrained MDLM, with no retraining at all?",
        "options": [
            {"text": "Its sampler preserves the forward marginals, so every visited state looks like training data.",
             "correct": True,
             "explanation": "The remasking kernel is constructed so per-token marginals stay α<sub>t</sub>x + (1−α<sub>t</sub>)m — and a remasked token is indistinguishable from an ordinarily masked one. In-distribution states are the whole license."},
            {"text": "It only ever remasks the tokens the model originally predicted with low confidence.",
             "explanation": "Remasking policies can be confidence-aware, but that's a heuristic layered on top — the no-retraining guarantee comes from marginal preservation, not from which tokens get picked."},
            {"text": "It attaches a small trained adapter that specializes in handling remasked inputs.",
             "explanation": "No new parameters are involved anywhere — ReMDM is purely a change to the sampling procedure."},
            {"text": "MDLM training already includes remasking as one of its data augmentations.",
             "explanation": "MDLM's forward process only ever masks monotonically; no remasking appears in training. It doesn't need to — masked is masked, however it got that way."},
        ],
    },
    {
        "question": "DPO eliminates the explicit reward model by...",
        "options": [
            {"text": "Inverting the closed-form optimal policy of the KL-regularized objective to rewrite the reward.",
             "correct": True,
             "explanation": "π*(y|x) ∝ π<sub>ref</sub>(y|x)·exp(r/β) inverts to r = β·log(π/π<sub>ref</sub>) + const; substituted into Bradley–Terry, the partition function cancels and a simple classification loss on preference pairs remains."},
            {"text": "Distilling PPO rollouts from a stronger teacher model into the policy.",
             "explanation": "No distillation and no PPO anywhere — avoiding that machinery is the paper's selling point."},
            {"text": "Scoring preferences with the raw perplexity gap between the two answers.",
             "explanation": "Close in flavor but not it: the implicit reward is the log-probability <em>ratio against a frozen reference policy</em>, scaled by β — not a raw perplexity difference — and it's derived, not assumed."},
            {"text": "Alternating between reward-model fitting and policy updates inside one loop.",
             "explanation": "That alternation is roughly what classic RLHF does; DPO's contribution is collapsing the two stages into a single supervised loss."},
        ],
    },
    {
        "question": "Which claim about FlashAttention is FALSE?",
        "options": [
            {"text": "It sparsifies the attention pattern to cut the number of floating-point operations.",
             "correct": True,
             "explanation": "FlashAttention is <em>exact</em> — every query still attends to every key, and FLOPs slightly increase (backward-pass recomputation). The speedup is entirely from IO: tiling into SRAM and never materializing the N×N matrix in HBM."},
            {"text": "It processes Q, K, V in tiles sized to fit in on-chip SRAM.",
             "explanation": "True — tiling with an online softmax is the core mechanism, keeping the working set in fast memory."},
            {"text": "It avoids materializing the N&times;N attention matrix in GPU main memory.",
             "explanation": "True — that avoided read/write traffic is where the wall-clock win comes from."},
            {"text": "It recomputes attention during the backward pass instead of storing it.",
             "explanation": "True — recomputation trades cheap FLOPs for expensive memory, the same IO-aware logic."},
        ],
    },
    {
        "question": "At inference time, a Decision Transformer produces good actions by...",
        "options": [
            {"text": "Conditioning on a high desired return-to-go and predicting the next action.",
             "correct": True,
             "explanation": "RL as conditional sequence modeling: prompt the model with the outcome you want and let it emit the actions consistent with achieving it. No value function, no planning — and a direct ancestor of d1's conviction that sequence models can absorb RL."},
            {"text": "Rolling out a learned world model and selecting the best imagined trajectory.",
             "explanation": "That's model-based planning (e.g. MuZero's family); DT never simulates the environment."},
            {"text": "Maximizing a Q-function that was trained on the same offline dataset.",
             "explanation": "Dispensing with value functions is the paper's point — conditioning replaces maximization."},
            {"text": "Sampling many candidate actions and reranking them with a reward model.",
             "explanation": "No reranking stage exists; the return-to-go conditioning does the selection implicitly in one pass."},
        ],
    },
]

CONCEPT_MAP = """
<p>Every idea on this page, one screen. Hover a node to trace what it connects to; click it for a
one-breath recap and a jump link back to the section that earns it. The three outlined nodes are
the load-bearing ones &mdash; if you can explain those three from memory, the rest hangs off them.</p>

<div class='widget' id='w-map'>
  <svg id='w-map-svg' viewBox='0 0 860 580' role='img' style='width:100%;height:auto'></svg>
  <div class='wstat' id='w-map-info'>hover to trace connections &middot; click a node for a recap and a jump link</div>
</div>

<p>A good self-test: pick any edge and say out loud <em>why</em> those two nodes are connected.
Every edge on this map is one sentence you should be able to produce.</p>
"""

KEEP_LEARNING = """
<p>Reading is the weakest form of studying and this page is still just reading with better
furniture. Here is the rest of the system &mdash; each piece targets a different failure mode of
&ldquo;I read it and it made sense.&rdquo;</p>

<h3>Spaced review (retention)</h3>
<p>The quizzes above fire once, at read time; the forgetting curve does not care. Two tools,
sharing the same 25-question bank:</p>
<ul>
<li><strong><a href='2026-08-02-diffusion-review.html'>The review deck</a></strong> &mdash; a
self-contained spaced-repetition app (Leitner boxes: 1&nbsp;&rarr;&nbsp;3&nbsp;&rarr;&nbsp;7&nbsp;&rarr;&nbsp;14&nbsp;&rarr;&nbsp;30 days).
Open it tomorrow, then whenever it says something is due. Progress lives in your browser; miss a
question and it comes back sooner. Finish a session and use <em>copy results for Claude</em> to
get targeted follow-up.</li>
<li><strong><a href='diffusion-dllm.apkg'>Anki deck</a></strong> &mdash; the same questions as a
standard <code>.apkg</code> if you already run Anki; import and it schedules itself.</li>
</ul>
<p>The section quizzes on this page also now remember your results (locally) &mdash; after
finishing any section&rsquo;s quiz, hit <em>copy results for Claude</em> and paste it into a
session to get re-quizzed on exactly what you missed.</p>

<h3>Teach it back (generation)</h3>
<p>Recognizing a correct option is the shallowest form of retrieval; reconstructing the argument
with no options in front of you is the deepest. Copy the prompt below into a fresh Claude session
and teach:</p>
<div class='callout' id='teachback-prompt'>
<p>I just studied diffusion language models (dLLMs). Play a curious student who wants to learn
this from me. Ask me to explain, one at a time: (1) why autoregressive decoding is slow (what is
actually the bottleneck, and how latency scales); (2) how masking replaces Gaussian noise for
text, and the forward process q(z_t|x); (3) the reveal probability (&alpha;_s&minus;&alpha;_t)/(1&minus;&alpha;_t)
and where mean-field error comes from when many tokens commit in one pass; (4) why DDIM can cut
sampling steps without retraining; (5) what Block Diffusion, ReMDM, discrete guidance, and d1
each fix. Probe every explanation with at least one &ldquo;why&rdquo; follow-up. Do not explain
anything yourself unless I am stuck after two attempts &mdash; then give a hint, not the answer.
At the end: grade me on mechanism, formulas, and honest caveats; list what I got wrong or fuzzy;
and write three new quiz questions targeting exactly my weak spots.</p>
</div>
<button class='wbtn' data-copy='teachback-prompt'>copy the teach-back prompt</button>

<h3>Build it yourself (transfer)</h3>
<p>The repo now contains a tutorial at <code>tutorials/masked-diffusion/</code>: a numpy skeleton of
everything Maths II derived &mdash; the forward masking process, the reveal-probability sampler
step, the NELBO weight, the weighted MLM loss, and a DDIM step &mdash; with the implementations
replaced by TODOs and a test suite that knows the right answers (including statistical checks
that your sampler preserves the marginals):</p>
<pre>cd tutorials/masked-diffusion
python3 -m pytest -q        # red until your implementations are right</pre>
<p>When the suite is green, you have not read masked diffusion &mdash; you have written it. Stuck
on one function? <code>solutions/</code> has a reference implementation; the honest move is to
peek at one function, not the file.</p>

<h3>The cheat sheet (consolidation)</h3>
<p>For review-at-a-glance later: <a href='2026-08-02-dllm-cheatsheet.html'>the one-page cheat
sheet</a> &mdash; every formula on this page with its one-sentence punchline, printable.</p>
"""

spec = {
    "title": "From Typewriter to Editor: Diffusion Language Models",
    "subtitle": "What diffusion models are, the mathematics underneath, and the nine papers behind Inception Labs' Mercury",
    "slug": "diffusion-language-models",
    "date": "2026-08-01",
    "multipage": True,
    "generator": {
        "skill": "learning-new-topic",
        "skill_url": "https://github.com/raghuramshankar/learning-with-llms/blob/main/skills/learning-new-topic/SKILL.md",
        "model": "Claude Fable 5",
    },
    "site_title": "\u2190 Diffusion Language Models",
    "nav": [["Review deck", "2026-08-02-diffusion-review.html"],
            ["Cheat sheet", "2026-08-02-dllm-cheatsheet.html"],
            ["Tutorials", "https://github.com/raghuramshankar/learning-with-llms/tree/main/tutorials/masked-diffusion"]],
    "intro": """
<p>This is a deep dive into diffusion language models, grounded in the nine papers listed on
Inception Labs&rsquo; about page. It is built to be worked through, not skimmed: each part ends
with a hard five-question quiz, the math parts carry faded derivations to attempt on paper, and
the simulations ask you to predict before they run.</p>
<p>Read the parts in order &mdash; each one owes its vocabulary to the one before. If you already
know how LLMs generate text, start at Part&nbsp;2; if you only want the story without the
equations, Parts 1, 2 and 8 stand on their own.</p>
""",
    "repo": "inceptionlabs.ai/about · Some of the technologies we've developed",
    "sections": [
        {"id": "background", "title": "Background", "html": BACKGROUND, "quiz": QUIZ_BACKGROUND},
        {"id": "intuition", "title": "Intuition", "html": INTUITION, "quiz": QUIZ_INTUITION},
        {"id": "math-continuous", "title": "The Maths I: Continuous Diffusion", "html": MATH1, "quiz": QUIZ_MATH1},
        {"id": "math-discrete", "title": "The Maths II: Masked Discrete Diffusion", "html": MATH2, "quiz": QUIZ_MATH2},
        {"id": "math-sota", "title": "The Maths III: Inside the SOTA Methods", "html": MATH_SOTA, "quiz": QUIZ_MATH_SOTA},
        {"id": "concept-map", "title": "The Concept Map", "html": CONCEPT_MAP},
        {"id": "keep-learning", "title": "Keep Learning", "html": KEEP_LEARNING},
        {"id": "papers", "title": "The Papers & Sources", "html": PAPERS, "quiz": QUIZ_PAPERS},
    ],
    "scripts": [(TOOLS / "widgets_lib.js").read_text(),
                (HERE / "widgets.js").read_text()],
}
# shared plotly.min.js next to the pages by default; --inline embeds it for a
# portable single-file export
if "--inline" in sys.argv:
    spec["head_scripts"] = [(DOCS / "plotly.min.js").read_text()]
else:
    spec["head_script_srcs"] = ["plotly.min.js"]

PLOTS = HERE / "plots"

# ---- token-strip rows and fraction snippets, spliced into the HTML ----
row_clean = toks(("the","ok"),("cat","ok"),("sat","ok"),("on","ok"),("the","ok"),("mat","ok"))
row_part  = toks(("the","ok"),("&#9634;","dim"),("sat","ok"),("on","ok"),("&#9634;","dim"),("mat","ok"))
row_full  = toks(*[("&#9634;","dim")]*6)

gen0 = row_full
gen1 = toks(("&#9634;","dim"),("cat","accent"),("&#9634;","dim"),("&#9634;","dim"),("&#9634;","dim"),("mat","accent"))
gen2 = toks(("the","accent"),("cat","ok"),("sat","accent"),("&#9634;","dim"),("&#9634;","dim"),("mat","ok"))
gen3 = toks(("the","ok"),("cat","ok"),("sat","ok"),("on","accent"),("the","accent"),("mat","ok"))

fix1 = toks(("a","ok"),("&#9634;","dim"),("fell","ok"),("from","ok"),("the","ok"),("tree","ok"))
fix2 = toks(("a","fail"),("apple","accent"),("fell","ok"),("from","ok"),("the","ok"),("tree","ok"))
fix3 = toks(("&#9634;","dim"),("apple","ok"),("fell","ok"),("from","ok"),("the","ok"),("tree","ok"))
fix4 = toks(("an","accent"),("apple","ok"),("fell","ok"),("from","ok"),("the","ok"),("tree","ok"))

subs = {
    "__ROW_CLEAN__": row_clean, "__ROW_PART__": row_part, "__ROW_FULL__": row_full,
    "__GEN_ROW0__": gen0, "__GEN_ROW1__": gen1, "__GEN_ROW2__": gen2, "__GEN_ROW3__": gen3,
    "__FIX_ROW1__": fix1, "__FIX_ROW2__": fix2, "__FIX_ROW3__": fix3, "__FIX_ROW4__": fix4,
    # Maths I: DDPM posterior mean/variance and noise-parameterized mean
    "__FRAC_MU1__": frac("&radic;&#8113;<sub>t&minus;1</sub>&middot;&beta;<sub>t</sub>", "1&minus;&#8113;<sub>t</sub>"),
    "__FRAC_MU2__": frac("&radic;&alpha;<sub>t</sub>&middot;(1&minus;&#8113;<sub>t&minus;1</sub>)", "1&minus;&#8113;<sub>t</sub>"),
    "__FRAC_BETA__": frac("1&minus;&#8113;<sub>t&minus;1</sub>", "1&minus;&#8113;<sub>t</sub>"),
    "__FRAC_MUTHETA1__": frac("1", "&radic;&alpha;<sub>t</sub>"),
    "__FRAC_MUTHETA2__": frac("&beta;<sub>t</sub>", "&radic;(1&minus;&#8113;<sub>t</sub>)"),
    "__FRAC_XHAT__": frac("x<sub>t</sub> &minus; &radic;(1&minus;&#8113;<sub>t</sub>)&middot;&epsilon;<sub>&theta;</sub>(x<sub>t</sub>, t)", "&radic;&#8113;<sub>t</sub>"),
    # Maths II: masked reverse posterior and NELBO weight
    "__FRAC_STAY__": frac("1&minus;&alpha;<sub>s</sub>", "1&minus;&alpha;<sub>t</sub>"),
    "__FRAC_REVEAL__": frac("&alpha;<sub>s</sub>&minus;&alpha;<sub>t</sub>", "1&minus;&alpha;<sub>t</sub>"),
    "__FRAC_WEIGHT__": frac("&minus;&alpha;&prime;<sub>t</sub>", "1&minus;&alpha;<sub>t</sub>"),
    # Plotly figures precomputed by make_plots.py
    "__FIG_LATENCY__": (PLOTS / "fig_latency.html").read_text(),
    "__FIG_DDIM_SWEEP__": (PLOTS / "fig_ddim_sweep.html").read_text(),
    "__FIG_QUALITY__": (PLOTS / "fig_quality.html").read_text(),
    "__FIG_SCHED__": (PLOTS / "fig_sched.html").read_text(),
    "__FIG_REVEAL__": (PLOTS / "fig_reveal.html").read_text(),
}
for s in spec["sections"]:
    for k, v in subs.items():
        s["html"] = s["html"].replace(k, v)

out = HERE / "spec.json"
out.write_text(json.dumps(spec, indent=1))
print(out)
