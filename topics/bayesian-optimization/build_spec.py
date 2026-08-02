#!/usr/bin/env python3
"""Build the JSON content spec for the Bayesian optimization explainer."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOLS = HERE.parents[1] / "tools"
DOCS = HERE.parents[1] / "docs"

def frac(num, den):
    return ("<span class='frac'><span class='num'>" + num +
            "</span><span class='den'>" + den + "</span></span>")

BACKGROUND = """
<p>Bayesian optimization is what you reach for when every single measurement hurts: finding the
best value of a function you cannot see inside, cannot differentiate, and can only afford to
evaluate a few dozen times. This page builds it from the ground up &mdash; the Gaussian process
machinery, the acquisition functions derived by hand, and the 2024&ndash;2026 research threads
&mdash; with every paper verified against arXiv. If you already know why gradient descent
doesn&rsquo;t apply here, skip to the second subtopic.</p>

<h3>Optimization when you can query, not differentiate</h3>
<p>Most optimization you meet in machine learning is <em>white-box</em>: you have a formula for
the objective, you differentiate it, and gradient descent takes millions of cheap, informed
steps. Now delete every one of those advantages. The function f you want to maximize is a
<strong>black box</strong>: you choose an input x, something slow and expensive happens &mdash; a
neural network trains for six hours, a chemist runs a reaction overnight, a wind-tunnel test
burns a day of scheduling &mdash; and a single number comes back. No formula, no gradient, often
some noise.</p>

<div class='diagram'>
  <div class='flow'>
    <div class='box accent'>x<small>hyperparameters &middot; reaction conditions &middot; design</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box'>black box f<small>hours or days per evaluation</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box ok'>f(x)<small>one number (maybe noisy)</small></div>
  </div>
  <div class='caption'>The setting: each query is precious, and nothing else about f is visible.
  The question is not &ldquo;how do I step downhill&rdquo; but &ldquo;where do I spend my next
  evaluation?&rdquo;</div>
</div>

<p>Classical answers exist. Grid search tiles the space with a lattice of evaluations. Random
search draws inputs uniformly. Evolutionary strategies like CMA-ES mutate populations of
candidates. All of them share one blind spot: <em>they never build a model of what they have
already learned</em>. Evaluation #37 is chosen exactly as if evaluations #1&ndash;36 had never
happened.</p>

<h3>The budget problem</h3>
<p>That blind spot is fatal when the budget is small. A classic result sharpens the point:
Bergstra &amp; Bengio (2012) showed random search beats grid search for hyperparameters &mdash;
not because randomness is smart, but because real objectives have <em>low effective
dimensionality</em>. A 5&times;5&times;5 grid spends 125 evaluations but tests only five distinct
values of each coordinate; if only one coordinate really matters, 120 evaluations were wasted
duplicates. Random search covers every 1-D projection densely. Neither, however, learns anything
from the values it observes.</p>
<p>Bayesian optimization&rsquo;s bet: with evaluations this expensive, it is worth spending real
computation <em>between</em> evaluations &mdash; fitting a probabilistic model of f and
optimizing a decision criterion on it &mdash; to make each evaluation count. Here is that bet,
measured rather than asserted:</p>

<div class='widget'>
__FIG_REGRET__
<div class='caption'>A real GP-EI Bayesian optimizer (implemented in numpy for this page) against
random and grid search on the negated Branin function: 30 evaluations, 20 repetitions, shaded
bands are &plusmn;1 sd. BO&rsquo;s curve climbs faster and plateaus higher &mdash; the entire
value proposition in one figure. (Grid&rsquo;s staircase is an artifact of its fixed visiting
order.)</div>
</div>

<h3>A brief history: 1964 &rarr; 1998 &rarr; 2012</h3>
<p>The idea is older than its fame. <strong>Kushner (1964)</strong> proposed maximizing the
probability of improvement over a random-process model of an unknown curve. <strong>Mockus
(1974&ndash;78)</strong> introduced expected improvement and gave the field its name. The modern
template arrived with <strong>Jones, Schonlau &amp; Welch (1998)</strong>: EGO &mdash;
&ldquo;Efficient Global Optimization&rdquo; &mdash; married a kriging (Gaussian process)
surrogate to the EI criterion and showed expensive engineering objectives falling in tens of
evaluations (~8k citations since). And the field went mainstream when <strong>Snoek, Larochelle
&amp; Adams (2012)</strong> pointed the machinery at machine-learning hyperparameters,
beat human experts at tuning convolutional networks, and released Spearmint (~10k citations).
Every modern AutoML system descends from that moment.</p>

<div class='callout'>
<p><strong>The tension the whole field resolves:</strong> with a handful of evaluations left,
should the next one go where the model <em>predicts the best value</em> (exploit), or where the
model <em>knows least</em> (explore)? Answering that question precisely requires two things: a
model that knows what it doesn&rsquo;t know, and a rule for converting belief plus ignorance into
a decision. Those two things &mdash; the Gaussian process posterior and the acquisition function
&mdash; are the next three sections.</p>
</div>
"""

QUIZ_BACKGROUND = [
    {
        "question": "What property of the problem justifies BO spending minutes of computation (fit a GP, optimize an acquisition function) just to choose ONE next input?",
        "options": [
            {"text": "Each function evaluation costs vastly more than the modeling computation.",
             "correct": True,
             "explanation": "The whole economics of BO: when one evaluation is six GPU-hours or an overnight reaction, seconds-to-minutes of surrogate math per decision is a rounding error. When evaluations are cheap, this overhead is a bad trade and random search wins on wall-clock."},
            {"text": "The surrogate model is guaranteed to contain the true function.",
             "explanation": "No such guarantee exists — GP model mismatch is a real, known failure mode (the honesty box in the next section). BO's bet is economic, not a correctness proof."},
            {"text": "Acquisition optimization finds the global optimum of f directly.",
             "explanation": "The acquisition is optimized over the *surrogate*, not f — it only chooses where to query next; f's optimum still has to be found through the evaluations themselves."},
            {"text": "Gradient information from the black box makes the model exact.",
             "explanation": "The setting assumes no gradients at all — that absence is what rules out ordinary gradient descent in the first place."},
        ],
    },
    {
        "question": "Bergstra & Bengio (2012): why does random search beat grid search for hyperparameter tuning?",
        "options": [
            {"text": "Grids duplicate values along unimportant dimensions; random covers each 1-D projection densely.",
             "correct": True,
             "explanation": "With low effective dimensionality, a grid tests only a few distinct values of the coordinate that actually matters; random search gives every coordinate as many distinct values as there are trials. Neither learns from observations — that's the gap BO fills."},
            {"text": "Random search adaptively concentrates samples near the best value found so far.",
             "explanation": "Pure random search is completely non-adaptive — that's exactly what it shares with grid search, and what separates both from BO."},
            {"text": "Grid search cannot be parallelized across machines, while random search can.",
             "explanation": "Both parallelize trivially — every point is chosen independently of all results. Parallelism isn't the differentiator."},
            {"text": "Random search evaluates fewer points to cover the same volume of space.",
             "explanation": "Volume coverage is equally cursed for both in high dimensions; the argument is about per-coordinate projections, not volume."},
        ],
    },
    {
        "question": "Which paper-to-contribution pairing is correct?",
        "options": [
            {"text": "Jones, Schonlau & Welch (1998) — the kriging + EI template that defined the modern BO loop.",
             "correct": True,
             "explanation": "EGO is the seminal crystallization: GP surrogate, closed-form EI, expensive engineering objectives. The other options each belong to a different paper: PI is Kushner (1964), regret bounds are GP-UCB (Srinivas et al.), the ML-hyperparameter era is Snoek et al. (2012)."},
            {"text": "Jones, Schonlau & Welch (1998) — the first proposal of probability of improvement.",
             "explanation": "PI is Kushner (1964), thirty-four years earlier. EGO's criterion was expected improvement."},
            {"text": "Jones, Schonlau & Welch (1998) — the first regret bounds for GP bandits.",
             "explanation": "That's GP-UCB (Srinivas, Krause, Kakade & Seeger, ICML 2010) — the theory pillar, not the EGO paper."},
            {"text": "Jones, Schonlau & Welch (1998) — Bayesian tuning of neural network hyperparameters.",
             "explanation": "That moment is Snoek, Larochelle & Adams (NeurIPS 2012), which pointed the 1998 machinery at ML and ignited AutoML."},
        ],
    },
    {
        "question": "Which cost does Bayesian optimization NOT reduce?",
        "options": [
            {"text": "The wall-clock cost of each individual function evaluation.",
             "correct": True,
             "explanation": "The experiment still takes its six hours — BO can't make a reaction run faster. What it cuts is the NUMBER of evaluations needed, which is the only lever available when each one has a fixed price."},
            {"text": "The number of evaluations needed to reach a given objective value.",
             "explanation": "This is precisely what BO reduces — the regret figure above shows the same quality reached in a fraction of the evaluations."},
            {"text": "The waste from testing duplicate values of unimportant coordinates.",
             "explanation": "The surrogate learns which directions matter (lengthscales per dimension), so BO stops spending evaluations along flat coordinates — grid search's signature waste."},
            {"text": "The chance of exhausting the budget before finding a good region.",
             "explanation": "Directed exploration via the acquisition function is exactly the mechanism that lowers this risk relative to blind sampling."},
        ],
    },
    {
        "question": "What made Snoek, Larochelle & Adams (2012) the field's breakout paper rather than just another application?",
        "options": [
            {"text": "It beat human experts at tuning real deep networks and shipped usable software.",
             "correct": True,
             "explanation": "Practical wins (state-of-the-art CIFAR results found automatically) plus Spearmint plus engineering guidance (MCMC over GP hyperparameters, cost-aware EI per second, parallelism) turned BO from a niche technique into ML infrastructure."},
            {"text": "It introduced the expected improvement acquisition function.",
             "explanation": "EI is Mockus (1970s), operationalized by EGO (1998) — Snoek et al. inherited it."},
            {"text": "It proved the first convergence guarantees for GP-based optimization.",
             "explanation": "Regret theory is GP-UCB's contribution (2010); the 2012 paper's force was empirical and practical."},
            {"text": "It replaced the Gaussian process surrogate with random forests.",
             "explanation": "That's SMAC (Hutter et al., 2011). Snoek et al. doubled down on GPs and made them work in practice."},
        ],
    },
]

INTUITION = """
<p>The core idea fits in one sentence: <strong>maintain a probabilistic belief about the whole
function, and let the next evaluation be chosen by a decision rule that weighs predicted value
against remaining ignorance.</strong> Everything else is machinery for making that sentence
precise.</p>

<h3>The loop: model, decide, evaluate, repeat</h3>

<div class='diagram'>
  <div class='flow'>
    <div class='box accent'>fit surrogate<small>GP posterior over f<br>from all data so far</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box accent'>optimize acquisition<small>thousands of cheap queries<br>on the model</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box fail'>evaluate f<small>the one expensive step</small></div>
    <span class='arr'>&rarr;</span>
    <div class='box ok'>augment data<small>and loop</small></div>
  </div>
  <div class='caption'>One iteration of Bayesian optimization. Note the asymmetry: the inner
  optimization (of the acquisition function) is hammered with thousands of evaluations precisely
  because they cost nothing — all the expense is quarantined in one red box.</div>
</div>

<p>Two functions are being optimized, and confusing them is the classic beginner error. The
<em>acquisition function</em> is optimized every iteration, cheaply, on the surrogate. The
<em>objective</em> f is only ever evaluated at the acquisition&rsquo;s single chosen point. BO is
a machine for converting cheap model computation into expensive-evaluation frugality.</p>

<div class='callout'>
<p><strong>Definition.</strong> <strong>Bayesian optimization</strong> is sequential optimization
of an expensive black-box function in which a Bayesian surrogate model (usually a Gaussian
process) supplies a posterior belief over f, and an <strong>acquisition function</strong> maps
that belief &mdash; both its predictions and its uncertainties &mdash; to the choice of the next
evaluation point.</p>
</div>

<h3>Uncertainty is the resource</h3>
<p>What does the posterior actually give you at every candidate x? Two numbers: a mean
&mu;(x) &mdash; the model&rsquo;s best guess of f(x) &mdash; and a standard deviation &sigma;(x)
&mdash; how wrong that guess could plausibly be. Near observed points, &sigma; collapses (with
noiseless observations, to exactly zero: the posterior interpolates). Far from data, &sigma;
swells back toward the prior. This &ldquo;map of ignorance&rdquo; is what separates BO from every
model-free method: <em>greedy on &mu; alone</em> re-samples the neighborhood of the best point
found and gets stuck; <em>greedy on &sigma; alone</em> is uniform space-filling that ignores
everything learned. Every acquisition function is some principled blend of the two.</p>

<h3>Drive the loop yourself</h3>
<p>Below is a live Gaussian process on a hidden function, with a budget of 12 evaluations.
Click anywhere on the top panel to spend an evaluation there yourself &mdash; or press
<em>BO step</em> and let expected improvement (the green curve underneath, red dot = its argmax)
choose. Try to beat the machine: spend your 12, reset, then let EI spend its 12, and compare
best-found values. Reveal the true function only at the end.</p>

<div class='widget' id='w-gp'>
  <div class='wctl'>
    <button class='wbtn' id='w-gp-bo'>BO step (EI picks)</button>
    <button class='wbtn' id='w-gp-reveal'>reveal / hide true f</button>
    <button class='wbtn' id='w-gp-reset'>reset</button>
  </div>
  <canvas id='w-gp-cv'></canvas>
  <div class='wstat' id='w-gp-stat'></div>
  <div class='caption'>A genuine GP posterior (RBF kernel, &#8467; = 0.12) and genuine expected
  improvement, computed live in your browser at every click. Watch how EI ignores regions the
  posterior already pins down, and how the band collapses at each observation.</div>
</div>

<h3>Does the model really help? Run the experiment</h3>
<p>The regret figure in the Background was one function. Here you can run fresh head-to-head
episodes yourself: each one draws a new random smooth function, gives EI-driven BO and uniform
random search 12 evaluations each, and records who found the higher value.</p>

<div class='widget' id='w-race-bo'>
  <div class='wctl' id='w-race-predict'>
    <label>Predict first &mdash; over many episodes, EI beats random in:</label>
    <button class='wbtn' data-pred='half'>~50% (no real edge)</button>
    <button class='wbtn' data-pred='most'>most, but not all</button>
    <button class='wbtn' data-pred='all'>every single one</button>
  </div>
  <div class='wstat' id='w-race-predfb' style='display:none'></div>
  <div class='wctl'>
    <button class='wbtn' id='w-race-1'>run 1 episode</button>
    <button class='wbtn' id='w-race-20'>run 20</button>
    <button class='wbtn' id='w-race-reset'>reset</button>
  </div>
  <div class='wstat' id='w-race-stat'>no episodes yet</div>
  <div class='wchips' id='w-race-chips'></div>
  <div class='caption'>Every episode is a real 12-evaluation contest on a fresh random function
  (both methods run live in your browser). The tally converges to a strong-but-imperfect edge:
  sample efficiency is a statistical advantage, not a guarantee.</div>
</div>

<h3>The catch: what can go wrong</h3>
<p>Three failure modes matter enough to name before the math:</p>
<ul>
<li><strong>Myopia.</strong> EI, PI and UCB are <em>one-step</em> criteria: they value the next
evaluation as if it were the last. A truly optimal policy would plan over the remaining budget
&mdash; that is a dynamic program nobody can afford, and one-step lookahead is the working
compromise (knowledge-gradient and entropy-search methods push one step further).</li>
<li><strong>Model mismatch.</strong> The posterior is only a map of ignorance if the prior was
roughly right. A lengthscale that is too long makes the GP confidently smooth over a spike it
has never seen; the acquisition then never looks there. Hyperparameter fitting (Maths I) is the
defense, not a cure.</li>
<li><strong>Dimensionality.</strong> Distances concentrate in high dimensions and the
&ldquo;far from data&rdquo; region is everywhere; vanilla GPs struggle past a few dozen
dimensions. What to do about it is a <em>live research fight</em> (the Papers section) —
2024&ndash;25 papers claim plain GPs do fine after all; a late-2025 rebuttal says we still
don&rsquo;t understand why or when.</li>
</ul>

<table>
<tr><th></th><th>Grid</th><th>Random</th><th>CMA-ES</th><th>BO (GP + EI)</th></tr>
<tr><td>Learns from observed values</td><td>No</td><td>No</td><td>Population statistics</td><td>Full posterior model</td></tr>
<tr><td>Uncertainty estimate</td><td>&mdash;</td><td>&mdash;</td><td>&mdash;</td><td>&sigma;(x) everywhere</td></tr>
<tr><td>Evaluations to be competitive</td><td>Hundreds+</td><td>Hundreds</td><td>Thousands</td><td><strong>Tens</strong></td></tr>
<tr><td>Per-decision overhead</td><td>None</td><td>None</td><td>Tiny</td><td>O(n&sup3;) GP fit + acquisition search</td></tr>
<tr><td>Sweet spot</td><td>&le;3 dims, cheap f</td><td>Cheap f, many cores</td><td>Cheap f, 10&sup2;&ndash;10&sup3; dims</td><td>Expensive f, &le;~20 dims</td></tr>
</table>

<div class='callout warn'>
<p><strong>Honesty box.</strong> BO is not magic sample efficiency; it is a bet that model-based
decisions beat blind ones <em>when the model is decent and the budget is small</em>. If
evaluations are cheap, random search plus more hardware is genuinely hard to beat (and is the
right baseline to report). If dimensions are many and structure is absent, the GP&rsquo;s map of
ignorance goes blank. And the entire loop inherits every pathology of its surrogate &mdash;
which is why the next section is about getting that surrogate right.</p>
</div>
"""

QUIZ_INTUITION = [
    {
        "question": "In one BO iteration, which function gets evaluated thousands of times — and which exactly once?",
        "options": [
            {"text": "The acquisition function thousands of times (on the surrogate); the objective f once.",
             "correct": True,
             "explanation": "The acquisition is a cheap function of the GP's μ(x) and σ(x), so its inner optimization can afford brute force. All expense is quarantined into the single evaluation of f at the acquisition's argmax."},
            {"text": "The objective f thousands of times (to build the surrogate); the acquisition once.",
             "explanation": "Backwards — if f could be evaluated thousands of times per iteration there would be no reason to run BO at all."},
            {"text": "Both are evaluated once per iteration to keep the loop balanced.",
             "explanation": "The acquisition's argmax can't be found from one query — it's located by dense search/multi-start optimization over the model, which is affordable precisely because the model is cheap."},
            {"text": "Both are evaluated thousands of times, but f's evaluations are parallelized.",
             "explanation": "Parallel/batch BO exists (qEI), but batches are of size 4–100, not thousands — and the sequential textbook loop evaluates f exactly once per round."},
        ],
    },
    {
        "question": "A purely exploitative policy — always evaluate at argmax of the posterior mean μ(x) — typically fails how?",
        "options": [
            {"text": "It re-samples around the best point found early and never visits high-σ regions.",
             "correct": True,
             "explanation": "With no reward for uncertainty, the policy locks onto the first decent basin: each new sample there further shrinks σ locally, μ stays highest there, and the true optimum elsewhere is never probed. This is exactly why every acquisition function pays something for σ."},
            {"text": "It spreads evaluations uniformly and learns nothing about the best region.",
             "explanation": "Uniform space-filling is the failure of the *opposite* policy — pure exploration on σ alone."},
            {"text": "It diverges because μ(x) is unbounded far away from the data.",
             "explanation": "With a zero-mean prior, μ(x) shrinks back toward 0 far from data — the mean is bounded and pulls toward the prior, not to infinity."},
            {"text": "It is identical to expected improvement when observations are noiseless.",
             "explanation": "EI at the current best point is ~0 (nothing left to improve there) while greedy-μ would happily resample it — the two disagree exactly where it matters."},
        ],
    },
    {
        "question": "With noiseless observations, what are μ(x) and σ(x) at an already-evaluated point — and what follows for the acquisition there?",
        "options": [
            {"text": "μ equals the observed value, σ = 0, so improvement-based acquisitions score it ~0.",
             "correct": True,
             "explanation": "A noiseless GP posterior interpolates: belief collapses onto the data. With σ=0 and no possible improvement, EI=0 — the loop automatically never wastes an evaluation repeating itself. (With noise, σ stays positive and re-evaluation can be rational.)"},
            {"text": "μ equals the observed value, σ = 1, so the point stays attractive to explore.",
             "explanation": "σ returning to the prior value happens far FROM data, not at it — at a noiseless observation the variance is exactly zero."},
            {"text": "Both μ and σ are undefined because the kernel matrix becomes singular.",
             "explanation": "Evaluating the posterior AT a data point is fine (jitter handles conditioning); singularity threatens only when two observations nearly coincide."},
            {"text": "μ reverts to the prior mean and σ to the prior variance immediately.",
             "explanation": "That's the behavior at distances ≫ lengthscale — the opposite end of the kernel's reach."},
        ],
    },
    {
        "question": "EI, PI and UCB are all called 'myopic.' What precisely is the criticism?",
        "options": [
            {"text": "They value the next evaluation as if it were the last one in the budget.",
             "correct": True,
             "explanation": "They're one-step decision rules: maximize the immediate (expected) payoff of a single query. The optimal policy would plan over all remaining evaluations — an intractable dynamic program — so one-step lookahead is the standard compromise; knowledge gradient and entropy search reach one level deeper."},
            {"text": "They only use observations from a local neighborhood of the best point.",
             "explanation": "All three are computed from the full GP posterior, which conditions on every observation everywhere. Myopia is about the planning horizon, not spatial locality."},
            {"text": "They cannot be evaluated without Monte-Carlo sampling of the posterior.",
             "explanation": "EI and PI have exact closed forms (that's much of their appeal); MC enters for batch versions and exotic acquisitions."},
            {"text": "They ignore the posterior variance and act on the mean alone.",
             "explanation": "σ appears explicitly in all three formulas — pricing uncertainty is the entire point of an acquisition function."},
        ],
    },
    {
        "question": "The GP's lengthscale is set far too LONG for the true function. What does the BO loop do?",
        "options": [
            {"text": "The posterior smooths over narrow peaks with high confidence, so they are never queried.",
             "correct": True,
             "explanation": "Too-long ℓ means the model believes f varies slowly, so between observations σ stays small — the map of ignorance reads 'nothing to see here' exactly where a spike hides. Confident model mismatch is the most dangerous BO failure because the acquisition trusts σ."},
            {"text": "The posterior bands widen everywhere and BO degrades gracefully into random search.",
             "explanation": "Instructively wrong: that's roughly what a too-SHORT lengthscale does (everything far from data looks unknown). Too-long ℓ fails in the dangerous direction — overconfidence, not caution."},
            {"text": "The kernel matrix becomes non-positive-definite and the GP fit crashes.",
             "explanation": "Long lengthscales worsen conditioning, but jitter handles that; the statistical failure, not a numerical crash, is the real problem."},
            {"text": "Nothing changes, because the acquisition function corrects for the prior.",
             "explanation": "The acquisition consumes μ and σ as truth — it has no independent access to f with which to correct a miscalibrated posterior. Garbage in, garbage out."},
        ],
    },
]

MATH_GP = """
<p>Everything BO does flows through two numbers per candidate point &mdash; &mu;<sub>n</sub>(x)
and &sigma;<sub>n</sub>(x) &mdash; so this section builds the machine that produces them.
Notation follows Frazier&rsquo;s tutorial (and Garnett&rsquo;s book): n observations
X = (x<sub>1</sub>, &hellip;, x<sub>n</sub>) with values y = (y<sub>1</sub>, &hellip;,
y<sub>n</sub>), maximization convention throughout.</p>

<h3>A prior over functions</h3>
<p>A <strong>Gaussian process</strong> is a probability distribution over functions with one
defining property: for <em>any</em> finite set of inputs, the vector of function values is
jointly Gaussian. It is fully specified by a mean function (we take m(x) = 0 after normalizing
the data) and a <strong>kernel</strong> k(x, x&prime;) giving the covariance between the values
at any two inputs:</p>
<div class='math'>f ~ GP(0, k),&nbsp;&nbsp;&nbsp;
Cov(f(x), f(x&prime;)) = k(x, x&prime;)</div>
<p>The workhorse kernel is the squared exponential (RBF), with signal variance fixed at 1 and a
<strong>lengthscale</strong> &#8467;:</p>
<div class='math'>k(x, x&prime;) = exp( &minus;""" + frac("&Vert;x &minus; x&prime;&Vert;&sup2;", "2&#8467;&sup2;") + """ )</div>
<p>Nearby inputs get correlation &asymp;1 (the function barely moves between them); inputs more
than a few lengthscales apart get correlation &asymp;0 (their values are essentially
independent). <em>The kernel is a hypothesis about how fast f varies</em> &mdash; and you can
see exactly what that hypothesis asserts by drawing random functions from the prior:</p>

<div class='widget'>
__FIG_LS_SAMPLES__
<div class='caption'>Slide the lengthscale. &#8467; = 0.03 hypothesizes a jittery function that
data can pin down only locally; &#8467; = 1 hypothesizes near-linear behavior across the whole
domain. Every posterior in this topic is this prior, conditioned on data.</div>
</div>

<h3>Conditioning: the posterior in closed form</h3>
<p>Observations (with Gaussian noise variance &sigma;<sub>n</sub>&sup2;, possibly &asymp;0) and
the prediction target f(x) are jointly Gaussian by definition &mdash; so conditioning is not an
approximation, it is an identity. Writing K for the n&times;n matrix K<sub>ij</sub> =
k(x<sub>i</sub>, x<sub>j</sub>) and k<sub>*</sub> for the vector k(x, x<sub>i</sub>):</p>
<div class='math'>&mu;<sub>n</sub>(x) = k<sub>*</sub><sup>T</sup> (K + &sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup> y</div>
<div class='math'>&sigma;<sub>n</sub>&sup2;(x) = k(x, x) &minus; k<sub>*</sub><sup>T</sup> (K + &sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup> k<sub>*</sub></div>
<p>Read them as prose. The mean is a data-weighted correction to the prior: y filtered through
the kernel&rsquo;s similarity geometry. The variance is <em>prior variance minus explained
variance</em> &mdash; what the data could not pin down. Note that &sigma;<sub>n</sub>&sup2;(x)
does not depend on y at all: with a fixed kernel, <em>where</em> you measured determines what
you still don&rsquo;t know, not <em>what</em> you measured.</p>

<div class='callout'>
<p><strong>Why the closed form is the whole game.</strong> One O(n&sup3;) factorization of
(K + &sigma;<sub>n</sub>&sup2;I), then every candidate x costs O(n) for &mu; and O(n&sup2;) for
&sigma; &mdash; microseconds at BO&rsquo;s n &le; a few hundred. That is what lets the
acquisition step hammer the posterior with thousands of queries per iteration. The same
O(n&sup3;) is also the ceiling that motivates sparse GPs and the pretrained-transformer
surrogates in the Papers section.</p>
</div>

<div class='deriv'>
  <div class='deriv-head'>
    <span class='deriv-title'>Faded derivation: the posterior from Gaussian conditioning</span>
    <button class='wbtn deriv-practice'>practice (hide all)</button>
    <button class='wbtn deriv-worked'>worked (show all)</button>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>1</span><span class='dstep-goal'>Write the joint distribution of the noisy observations y and the target value f(x).</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>""" + frac("y", "f(x)") + """ ~ &#119977;( 0, """ + frac("K + &sigma;<sub>n</sub>&sup2;I &nbsp;&nbsp; k<sub>*</sub>", "k<sub>*</sub><sup>T</sup> &nbsp;&nbsp; k(x,x)") + """ )</div><p>(Block notation: the stacked vector is Gaussian with the blocked covariance.) This step is free &mdash; it is the definition of a GP plus independent Gaussian noise on the diagonal.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>2</span><span class='dstep-goal'>State the Gaussian conditioning identity for a partitioned Gaussian (a | b).</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>a | b ~ &#119977;( &mu;<sub>a</sub> + &Sigma;<sub>ab</sub>&Sigma;<sub>bb</sub><sup>&minus;1</sup>(b &minus; &mu;<sub>b</sub>), &nbsp;&Sigma;<sub>aa</sub> &minus; &Sigma;<sub>ab</sub>&Sigma;<sub>bb</sub><sup>&minus;1</sup>&Sigma;<sub>ba</sub> )</div><p>The single most useful identity in GP land; derived by completing the square in the joint density&rsquo;s exponent.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>3</span><span class='dstep-goal'>Apply it with a = f(x), b = y to get the posterior mean.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&mu;<sub>n</sub>(x) = 0 + k<sub>*</sub><sup>T</sup>(K + &sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup>(y &minus; 0) = k<sub>*</sub><sup>T</sup>(K + &sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup> y</div><p>&Sigma;<sub>ab</sub> = k<sub>*</sub><sup>T</sup>, &Sigma;<sub>bb</sub> = K + &sigma;<sub>n</sub>&sup2;I, priors zero.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>4</span><span class='dstep-goal'>Same substitution for the posterior variance.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&sigma;<sub>n</sub>&sup2;(x) = k(x,x) &minus; k<sub>*</sub><sup>T</sup>(K + &sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup> k<sub>*</sub></div><p>Prior variance minus what the data explains. Independent of y &mdash; measurement <em>locations</em> alone determine remaining ignorance.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>5</span><span class='dstep-goal'>Sanity check: one noiseless observation (n=1, &sigma;<sub>n</sub>=0, k(x&#8321;,x&#8321;)=1). What happens at x = x&#8321;?</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&mu;(x) = k(x, x&#8321;)&middot;y&#8321;,&nbsp;&nbsp;&nbsp;&sigma;&sup2;(x) = 1 &minus; k(x, x&#8321;)&sup2;</div><p>At x = x&#8321;: k = 1, so &mu; = y&#8321; and &sigma;&sup2; = 0 &mdash; exact interpolation. Far away: k &rarr; 0, so &mu; &rarr; 0 and &sigma;&sup2; &rarr; 1 &mdash; the prior reasserts itself. Both limits are the picture from the Intuition widget.</p></div>
  </div>
  <div class='caption'>Attempt each step on paper first. Step 2 is worth memorizing outright — it
  is also the innovation-free heart of Kalman filters and linear-Gaussian everything.</div>
</div>

<h3>Choosing hyperparameters: the marginal likelihood</h3>
<p>Everything above assumed &#8467; and &sigma;<sub>n</sub> were known. In practice they are fit
by maximizing the <strong>log marginal likelihood</strong> &mdash; the probability the prior
assigns to the data, with the latent function integrated out (closed form, because everything is
Gaussian):</p>
<div class='math'>log p(y | X, &theta;) = &minus;&frac12; y<sup>T</sup>(K<sub>&theta;</sub> + &sigma;<sub>n</sub>&sup2;I)<sup>&minus;1</sup>y
&nbsp;&minus;&nbsp; &frac12; log det(K<sub>&theta;</sub> + &sigma;<sub>n</sub>&sup2;I)
&nbsp;&minus;&nbsp; """ + frac("n", "2") + """ log 2&pi;</div>
<p>The first term rewards fitting the data; the second &mdash; the log-determinant &mdash;
punishes models flexible enough to fit anything (short lengthscales inflate it). Maximizing
their sum is an automatic Occam&rsquo;s razor. Feel it yourself:</p>

<div class='widget' id='w-kernel'>
  <div class='wctl'>
    <label>lengthscale &#8467;</label>
    <input type='range' id='w-kernel-ls' min='-170' max='0' value='-80'>
    <label>noise &sigma;<sub>n</sub>&sup2;</label>
    <input type='range' id='w-kernel-n' min='-400' max='-30' value='-400'>
  </div>
  <canvas id='w-kernel-cv'></canvas>
  <div class='wstat' id='w-kernel-stat'></div>
  <div class='caption'>Five fixed observations, live posterior, live log marginal likelihood.
  Slide &#8467; to the extremes: very short fits the data but pays a huge complexity penalty;
  very long is simple but fits badly. The LML peak sits where the working posterior looks
  &ldquo;right&rdquo; &mdash; that agreement is the point.</div>
</div>

<p>One honest caveat: the LML surface can be multi-modal (a &ldquo;fits-the-noise&rdquo; mode and
a &ldquo;smooths-through&rdquo; mode), which is why careful implementations use restarts or, as
in Snoek et al. (2012), integrate over hyperparameters with MCMC rather than committing to one
point estimate.</p>
"""

QUIZ_MATH_GP = [
    {
        "question": "One noiseless observation y₁ = 2 at x₁ (unit-variance RBF kernel, zero prior mean). At a point x with k(x, x₁) = 0.6, the posterior is:",
        "options": [
            {"text": "μ = 1.2, σ² = 0.64",
             "correct": True,
             "explanation": "With n=1 and k(x₁,x₁)=1: μ(x) = k·y₁ = 0.6×2 = 1.2 and σ²(x) = 1 − k² = 1 − 0.36 = 0.64. The n=1 case is worth internalizing — it exposes the formulas' anatomy with no linear algebra in the way."},
            {"text": "μ = 1.2, σ² = 0.36",
             "explanation": "The variance is 1 − k², not k² — you subtract the explained variance from the prior's 1, and k² = 0.36 is the explained part."},
            {"text": "μ = 0.6, σ² = 0.64",
             "explanation": "The mean scales the OBSERVED VALUE by the correlation: k·y₁ = 1.2. Using k alone forgets that the observation was 2, not 1."},
            {"text": "μ = 2.0, σ² = 0.40",
             "explanation": "μ = 2 would be the answer at x = x₁ itself (k=1); at correlation 0.6 the prediction shrinks toward the prior mean 0."},
        ],
    },
    {
        "question": "Halving the RBF lengthscale ℓ (all else fixed) does what to the posterior?",
        "options": [
            {"text": "Correlations decay faster, so uncertainty between observations grows.",
             "correct": True,
             "explanation": "Shorter ℓ = a hypothesis of faster variation = data constrains only a smaller neighborhood. Bands between points widen and the mean wiggles more. (Too-short over-explores; too-long confidently smooths over hidden structure — the dangerous direction.)"},
            {"text": "Correlations decay more slowly, so the posterior becomes smoother overall.",
             "explanation": "Backwards — that describes DOUBLING ℓ. Halving it shrinks the kernel's reach."},
            {"text": "The posterior mean is unchanged, since ℓ appears only in the variance formula.",
             "explanation": "ℓ enters through K and k*, which both μ and σ² are built from — the mean changes too."},
            {"text": "The prior variance at unobserved points increases above 1.",
             "explanation": "The prior variance is fixed by the signal-variance parameter (1 here); ℓ redistributes correlation, it can't inflate the prior."},
        ],
    },
    {
        "question": "In the log marginal likelihood, what role does the −½ log det(K + σₙ²I) term play?",
        "options": [
            {"text": "It penalizes flexible kernels, making the objective an automatic Occam's razor.",
             "correct": True,
             "explanation": "Short lengthscales make K closer to the identity, inflating the determinant of the prior covariance over datasets — the model 'spreads its bets' over too many possible datasets and pays for it. Fit term + complexity term = automatic bias-variance trade, no validation set needed."},
            {"text": "It rewards kernels that interpolate the training data exactly.",
             "explanation": "Rewarding data fit is the FIRST term's job (−½ yᵀK⁻¹y); the log-det pushes in the opposite direction."},
            {"text": "It ensures the kernel matrix stays positive definite during optimization.",
             "explanation": "Positive-definiteness comes from the kernel's mathematical validity plus jitter — the log-det term measures volume, it doesn't enforce anything."},
            {"text": "It normalizes the likelihood so values are comparable across datasets.",
             "explanation": "The −(n/2)log 2π constant is the normalizer; the log-det is data- and hyperparameter-dependent, which is precisely why it can act as a complexity penalty."},
        ],
    },
    {
        "question": "Why is exact GP-based BO comfortable at n = 200 observations but painful at n = 50,000?",
        "options": [
            {"text": "Fitting requires factorizing an n×n kernel matrix — O(n³) time.",
             "correct": True,
             "explanation": "The Cholesky of K + σₙ²I dominates: ~10⁶ flops at n=200 (microseconds), ~10¹⁴ at n=50k (hours, plus O(n²) memory = 20 GB). BO's small-budget setting is exactly the regime where exact GPs are cheap — and this ceiling is what sparse GPs and pretrained transformer surrogates attack."},
            {"text": "The acquisition function has n local optima, one per observation.",
             "explanation": "Acquisition landscapes are multimodal, but that's handled with multi-start/dense-grid search and isn't what scales cubically."},
            {"text": "Posterior variance becomes negative when n exceeds a few thousand.",
             "explanation": "Numerical conditioning degrades with near-duplicate points at any n; jitter handles it. It's a precision issue, not the scaling wall."},
            {"text": "The marginal likelihood stops being differentiable at large n.",
             "explanation": "LML stays smooth in θ at any n — its gradient just costs the same O(n³) factorization."},
        ],
    },
    {
        "question": "Which fact about the GP posterior variance σₙ²(x) is TRUE — and slightly surprising?",
        "options": [
            {"text": "It depends on where you measured, but not on what values you observed.",
             "correct": True,
             "explanation": "y appears in the mean formula only; σₙ²(x) = k(x,x) − k*ᵀ(K+σₙ²I)⁻¹k* is pure geometry of the design points (given fixed hyperparameters). Corollary: with a fixed kernel you could plan ALL measurement locations in advance — it's hyperparameter re-fitting after each observation that makes BO genuinely adaptive."},
            {"text": "It can exceed the prior variance when observations disagree with each other.",
             "explanation": "Conditioning can only remove variance: σₙ²(x) ≤ k(x,x) always. Disagreeing data inflates the fitted noise σₙ², not the posterior variance above the prior."},
            {"text": "It is zero everywhere once n exceeds the input dimension.",
             "explanation": "That would be true for a linear model with n ≥ d — a GP with an RBF kernel is nonparametric, and uncertainty persists between and beyond observations at any n."},
            {"text": "It equals the prior variance exactly at each observed location.",
             "explanation": "The opposite: at a noiseless observation the posterior variance hits its minimum (zero). It's far from the data that the prior variance reasserts itself."},
        ],
    },
]

MATH_ACQ = """
<p>The posterior hands us &mu;<sub>n</sub>(x) and &sigma;<sub>n</sub>(x); an acquisition function
turns those two numbers into a decision. This section derives the classic by hand, tours the
zoo, and covers the two upgrades a modern practitioner actually ships: LogEI and batch
acquisition. Throughout, f<sup>*</sup><sub>n</sub> = max<sub>i</sub> y<sub>i</sub> is the best
value observed so far and z = (&mu;<sub>n</sub>(x) &minus; f<sup>*</sup><sub>n</sub>) /
&sigma;<sub>n</sub>(x).</p>

<h3>Improvement, and its expectation</h3>
<p>Define the <strong>improvement</strong> a new evaluation at x would deliver:
I(x) = max( f(x) &minus; f<sup>*</sup><sub>n</sub>, 0 ). Under the posterior, f(x) is Gaussian,
so I(x) is a censored Gaussian &mdash; and its expectation has a closed form:</p>
<div class='math'>EI<sub>n</sub>(x) = &#120124;[I(x)] =
(&mu;<sub>n</sub>(x) &minus; f<sup>*</sup><sub>n</sub>)&middot;&Phi;(z) + &sigma;<sub>n</sub>(x)&middot;&phi;(z)</div>
<p>with &Phi; and &phi; the standard normal CDF and PDF. The two terms are exploitation and
exploration wearing mathematical clothes: the first pays for a mean above the incumbent
(weighted by the probability the improvement is real), the second pays for spread &mdash; a
chance of a large upside even when the mean is unpromising. <strong>At z = 0 (mean exactly at
the incumbent), EI = &sigma;&middot;&phi;(0) &asymp; 0.40&sigma;: pure uncertainty value.</strong></p>

<div class='deriv'>
  <div class='deriv-head'>
    <span class='deriv-title'>Faded derivation: EI in closed form</span>
    <button class='wbtn deriv-practice'>practice (hide all)</button>
    <button class='wbtn deriv-worked'>worked (show all)</button>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>1</span><span class='dstep-goal'>Write EI as an integral over the posterior for f(x) (drop subscripts: mean &mu;, sd &sigma;, incumbent f<sup>*</sup>).</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>EI = &int;<sub>f*</sub><sup>&infin;</sup> (v &minus; f<sup>*</sup>) &middot; &#119977;(v; &mu;, &sigma;&sup2;) dv</div><p>Values below f<sup>*</sup> contribute zero improvement, so the integral starts at f<sup>*</sup>.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>2</span><span class='dstep-goal'>Standardize: substitute v = &mu; + &sigma;u and z = (&mu; &minus; f<sup>*</sup>)/&sigma;.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>EI = &sigma; &int;<sub>&minus;z</sub><sup>&infin;</sup> (u + z) &phi;(u) du</div><p>The lower limit v = f<sup>*</sup> maps to u = (f<sup>*</sup> &minus; &mu;)/&sigma; = &minus;z, and (v &minus; f<sup>*</sup>) = &sigma;(u + z).</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>3</span><span class='dstep-goal'>Evaluate &int;<sub>&minus;z</sub><sup>&infin;</sup> u&middot;&phi;(u) du using the identity &phi;&prime;(u) = &minus;u&phi;(u).</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>&int;<sub>&minus;z</sub><sup>&infin;</sup> u&phi;(u) du = [&minus;&phi;(u)]<sub>&minus;z</sub><sup>&infin;</sup> = &phi;(&minus;z) = &phi;(z)</div><p>The Gaussian PDF is its own (negated) antiderivative under multiplication by u, and it is symmetric.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>4</span><span class='dstep-goal'>Evaluate the remaining piece z&middot;&int;<sub>&minus;z</sub><sup>&infin;</sup> &phi;(u) du.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>z &int;<sub>&minus;z</sub><sup>&infin;</sup> &phi;(u) du = z (1 &minus; &Phi;(&minus;z)) = z&middot;&Phi;(z)</div><p>Tail probability plus symmetry of &Phi;.</p></div>
  </div>
  <div class='dstep'>
    <div class='dstep-label'><span class='tag'>5</span><span class='dstep-goal'>Assemble, un-standardize, and sanity-check z = 0.</span><button class='wbtn dstep-toggle'>reveal</button></div>
    <div class='dstep-body'><div class='math'>EI = &sigma;[ z&Phi;(z) + &phi;(z) ] = (&mu; &minus; f<sup>*</sup>)&Phi;(z) + &sigma;&phi;(z)</div><p>At z = 0: EI = &sigma;&phi;(0) = &sigma;/&radic;(2&pi;) &asymp; 0.399&sigma; &mdash; a point whose mean merely ties the incumbent is still worth two-fifths of its standard deviation. Uncertainty alone has cash value.</p></div>
  </div>
  <div class='caption'>Five steps, one substitution, two normal identities. This is the same
  derivation Jones et al. relied on in 1998 &mdash; try to reproduce it cold.</div>
</div>

<h3>The acquisition zoo</h3>
<p><strong>Probability of improvement</strong> (Kushner, 1964) keeps only the probability and
discards the magnitude: PI<sub>n</sub>(x) = &Phi;(z). Its pathology is instructive: a point
offering a near-certain microscopic gain outscores one offering a coin-flip at a large gain, so
PI hugs the incumbent unless you hand-tune a margin &xi;. EI fixed exactly this by weighting
improvements by their size.</p>
<p><strong>Upper confidence bound</strong> (GP-UCB; Srinivas et al., 2010) is optimism made
explicit:</p>
<div class='math'>UCB<sub>n</sub>(x) = &mu;<sub>n</sub>(x) + &beta;<sub>n</sub><sup>&frac12;</sup>&sigma;<sub>n</sub>(x)</div>
<p>&mdash; act as if every point is as good as its confidence interval allows. Its gift is
theory: with &beta;<sub>n</sub> growing logarithmically in n, cumulative regret is sublinear
&mdash; BO provably converges. Its price is that &beta; is a dial someone must set.</p>
<p><strong>Information-theoretic</strong> acquisitions (Entropy Search, PES, and the practical
favorite <strong>MES</strong> &mdash; Wang &amp; Jegelka, 2017) change the question from
&ldquo;where might I improve?&rdquo; to &ldquo;which measurement most reduces my uncertainty
about the optimum?&rdquo; &mdash; MES scores the expected drop in entropy of the maximum
<em>value</em> f<sup>*</sup>. Less myopic in spirit, costlier to compute, and the natural choice
when the goal is to <em>learn where the optimum is</em> rather than to rack up good evaluations.</p>
<p>Here are all three reading the same posterior &mdash; note how they disagree about where to
go next:</p>

<div class='widget'>
__FIG_ACQ_ANATOMY__
<div class='caption'>Same six observations, same GP posterior (top); switch the acquisition
(bottom) and watch the recommended next evaluation (red marker) move. EI and PI both spike in
the gap around x &asymp; 0.5 but weight it differently; UCB with &beta; = 2 is happy to chase
wide bands even where the mean is mediocre.</div>
</div>

<p>And here is the arithmetic itself &mdash; the quiz will ask you to do this by hand, so try a
few settings (in particular &Delta; = 0):</p>

<div class='widget' id='w-acq'>
  <div class='wctl'>
    <label>&Delta; = &mu; &minus; f<sup>*</sup></label>
    <input type='range' id='w-acq-d' min='-200' max='200' value='0'>
    <label>&sigma;</label>
    <input type='range' id='w-acq-s' min='5' max='200' value='100'>
    <label>&beta;</label>
    <input type='range' id='w-acq-b' min='0' max='400' value='200'>
  </div>
  <div class='wstat' id='w-acq-stat'></div>
  <div style='max-width:420px;margin:.6rem auto 0'>
    <div style='font-size:.78rem;color:var(--muted)'>EI</div>
    <div style='background:var(--box-bg);border:1px solid var(--line);border-radius:6px'><div id='w-acq-ei' style='height:10px;background:var(--ok);border-radius:6px;width:0'></div></div>
    <div style='font-size:.78rem;color:var(--muted);margin-top:.3rem'>PI</div>
    <div style='background:var(--box-bg);border:1px solid var(--line);border-radius:6px'><div id='w-acq-pi' style='height:10px;background:var(--accent);border-radius:6px;width:0'></div></div>
    <div style='font-size:.78rem;color:var(--muted);margin-top:.3rem'>UCB &minus; f<sup>*</sup></div>
    <div style='background:var(--box-bg);border:1px solid var(--line);border-radius:6px'><div id='w-acq-ucb' style='height:10px;background:var(--fail);border-radius:6px;width:0'></div></div>
  </div>
  <div class='caption'>Slide &Delta; negative and watch EI shrink but never quite die (the
  &sigma;&phi;(z) term), while PI collapses toward zero — and UCB march on linearly. The three
  disagree most exactly where decisions are hardest: promising-but-uncertain territory.</div>
</div>

<h3>Numerics and batches: LogEI and qEI</h3>
<p>A 2023 result embarrassed a quarter-century of default practice. When z is very negative
&mdash; the model is fairly sure x won&rsquo;t improve &mdash; EI underflows: the value AND its
gradient are numerically zero across wide swaths of the domain, so the (gradient-based) inner
optimization of the acquisition silently stalls at whatever flat region it started in.
<strong>Ament et al. (NeurIPS 2023)</strong> showed that simply computing and optimizing
<em>log</em> EI &mdash; with a numerically careful expansion of log(z&Phi;(z) + &phi;(z)) &mdash;
repairs the failure, and that a chunk of EI&rsquo;s reported underperformance versus fancier
acquisitions was this artifact all along. LogEI variants are now the BoTorch default.</p>
<p>Real experiments also run in parallel: you want q candidate points at once, not one. The
joint criterion is the expected improvement of the <em>best of the batch</em>:</p>
<div class='math'>qEI(x<sub>1..q</sub>) = &#120124;[ max( max<sub>j</sub> f(x<sub>j</sub>) &minus; f<sup>*</sup><sub>n</sub>, 0 ) ]</div>
<p>No closed form survives the inner max, so BoTorch estimates it by Monte Carlo: draw posterior
samples over the batch jointly (reparameterization trick, so gradients flow), average the
improvements, ascend. The joint expectation is what stops the optimizer from proposing q copies
of the same argmax &mdash; a batch&rsquo;s members are valued for <em>covering</em> each
other&rsquo;s failure modes.</p>
"""

QUIZ_MATH_ACQ = [
    {
        "question": "A candidate's posterior mean exactly ties the incumbent (μ = f*, so z = 0) with σ = 0.5. Its EI is:",
        "options": [
            {"text": "≈ 0.20 — half of φ(0)",
             "correct": True,
             "explanation": "EI = (μ−f*)Φ(z) + σφ(z) = 0 + 0.5·φ(0) = 0.5·0.3989 ≈ 0.20. At z = 0 the entire value is the exploration term — uncertainty alone is worth ~0.4σ of expected improvement."},
            {"text": "Exactly 0 — the mean promises no improvement",
             "explanation": "The classic error EI was designed to avoid: the mean is only the center of the belief. Half the posterior mass lies ABOVE f*, and those outcomes contribute positive improvement while the other half contributes zero, not negative."},
            {"text": "≈ 0.25 — half of Φ(0)",
             "explanation": "Φ(0) = 0.5 is PI's answer here (a coin-flip probability of improving); EI weights by magnitude, giving σφ(0) instead."},
            {"text": "≈ 0.5 — exactly σ",
             "explanation": "EI at z=0 is σ·φ(0) ≈ 0.4σ = 0.2, not σ itself — the censored-Gaussian expectation discounts the spread by φ(0)."},
        ],
    },
    {
        "question": "What is the instructive pathology of probability of improvement (PI) that EI repairs?",
        "options": [
            {"text": "PI prefers a near-certain microscopic gain over a coin-flip at a large one.",
             "correct": True,
             "explanation": "PI = Φ(z) counts all improvements equally, however small — so it hugs the incumbent's neighborhood where tiny certain gains live. EI multiplies probability by magnitude, restoring the appetite for big uncertain payoffs."},
            {"text": "PI ignores the posterior variance and reduces to greedy mean-maximization.",
             "explanation": "σ is inside z = (μ−f*)/σ, so PI definitely uses uncertainty — its flaw is ignoring the SIZE of improvements, not their probability."},
            {"text": "PI has no closed form and must be estimated by Monte Carlo.",
             "explanation": "PI is the simplest closed form of all: one Φ evaluation. The MC-only criteria are the batch and information-theoretic ones."},
            {"text": "PI is not defined when observations are noisy.",
             "explanation": "Noise complicates the definition of the incumbent for EVERY improvement-based acquisition equally (you use the best posterior mean instead); nothing singles out PI."},
        ],
    },
    {
        "question": "What does GP-UCB's theory require of β_n to guarantee no regret, and why grow it at all?",
        "options": [
            {"text": "β_n grows ~ logarithmically, so confidence bounds stay valid over ever more candidate decisions.",
             "correct": True,
             "explanation": "Srinivas et al. set β_n ≍ log n (times dimension/δ factors) so that, union-bounded over the growing sequence of decisions, the true f stays inside every used confidence interval w.h.p. — optimism then forces enough exploration for sublinear cumulative regret."},
            {"text": "β_n shrinks toward zero so the policy becomes purely exploitative in the limit.",
             "explanation": "Backwards: β must GROW (slowly). Shrinking β would let a single early overconfident posterior permanently hide the optimum."},
            {"text": "β_n is constant at 2, matching the 95% confidence interval convention.",
             "explanation": "β = 2 is the pragmatic default in practice (and in the figure above), but the no-regret theorem specifically needs the schedule, not a constant."},
            {"text": "β_n grows linearly with n to dominate the posterior mean asymptotically.",
             "explanation": "Linear growth would drown μ entirely and degrade to pure space-filling — the theorem's log schedule is precisely calibrated to avoid both extremes."},
        ],
    },
    {
        "question": "What problem does LogEI (Ament et al., NeurIPS 2023) actually solve?",
        "options": [
            {"text": "EI and its gradients underflow to zero where improvement is unlikely, stalling the acquisition's inner optimizer.",
             "correct": True,
             "explanation": "For strongly negative z, EI ∝ φ(z)/z² vanishes below float precision across most of the domain — gradient ascent sees a flat function and quits. Optimizing a carefully computed log EI restores signal; decades of 'EI underperforms' comparisons were partly measuring this bug."},
            {"text": "EI is myopic; LogEI adds multi-step lookahead over the remaining budget.",
             "explanation": "LogEI is the SAME one-step criterion under a monotone transform — myopia is untouched. Lookahead is the knowledge-gradient / entropy-search family's territory."},
            {"text": "EI cannot handle noisy observations; LogEI marginalizes the noise.",
             "explanation": "Noisy-EI variants exist (e.g. qNEI) but that's an orthogonal issue — LogEI's contribution is purely numerical robustness."},
            {"text": "EI's closed form is wrong; LogEI corrects the constant in the φ term.",
             "explanation": "The 1998-era closed form is exactly right in real arithmetic — the failure is floating-point, which is what makes it such a sneaky, instructive bug."},
        ],
    },
    {
        "question": "Why does batch BO optimize the JOINT criterion qEI = E[max over the batch] instead of just taking the top-q points ranked by single-point EI?",
        "options": [
            {"text": "Top-q ranking picks q near-duplicates around one peak; the joint expectation rewards batches that cover.",
             "correct": True,
             "explanation": "Single-point EI's top scores cluster on the same mode — q copies of almost the same experiment. Under E[max f(x_j) − f*], a second point near the first adds nearly nothing (their samples are correlated), so the optimizer spreads the batch to hedge. Estimated by Monte Carlo with reparameterized posterior samples so gradients flow."},
            {"text": "Single-point EI cannot be evaluated at more than one point per iteration.",
             "explanation": "EI is a function — you can score a million points with it. The issue is that its top-q argmaxes are redundant, not unavailable."},
            {"text": "The joint criterion has a closed form while single-point EI does not.",
             "explanation": "Exactly reversed: single-point EI is the closed-form one, and the joint max forces Monte-Carlo estimation."},
            {"text": "Evaluating q points at once changes the GP posterior update rule.",
             "explanation": "Conditioning on q new observations is the same Gaussian algebra regardless of how they were chosen — the batch problem is about selection, not updating."},
        ],
    },
]

MATH_SOTA = """
<p>The Papers &amp; Sources part tells the story of the modern methods; this section shows their
actual mathematics. Each of the five ideas below changes one specific piece of the machinery you
already know &mdash; a prior, a search region, an acquisition weight, or the surrogate itself.
Nothing here requires more than Maths I and II.</p>

<h3>The &radic;d fix: why vanilla GPs drown, and the prior that saves them</h3>
<p>Start with the failure. For two independent uniform points x, x&prime; in [0,1]<sup>d</sup>,
each coordinate contributes &#120124;[(x<sub>i</sub>&minus;x&prime;<sub>i</sub>)&sup2;] =
2&middot;Var(u) = 1/6, so:</p>
<div class='math'>&#120124;&Vert;x &minus; x&prime;&Vert;&sup2; = d/6</div>
<p>Feed that into the RBF kernel with a fixed lengthscale &#8467;: the typical correlation
between random points is exp(&minus;d/(12&#8467;&sup2;)), which crashes to zero as d grows.
Every point is &ldquo;far from the data,&rdquo; the posterior reverts to the prior everywhere,
and BO degenerates into uniform sampling. The 2024 &ldquo;vanilla BO&rdquo; result (Hvarfner et
al.) is, at heart, one observation: <strong>keep &Vert;x&minus;x&prime;&Vert;&sup2;/&#8467;&sup2;
stable by scaling the lengthscale like &radic;d</strong> &mdash; implemented as a dimension-scaled
prior:</p>
<div class='math'>&#8467; ~ LogNormal( &mu;<sub>0</sub> + &frac12; log d, &nbsp;&sigma;<sub>0</sub>&sup2; )</div>
<p>With that one change, plain GP-EI matches the specialized high-dimensional machinery on many
benchmarks &mdash; the claim the ICLR 2025 paper sharpened and the Nov 2025 rebuttal now
contests. The controversy is real, but the &radic;d arithmetic above is not in dispute; what is
disputed is whether it is the whole story.</p>

<h3>SAASBO: sparsity as a prior</h3>
<p>Give each dimension its own inverse squared lengthscale &rho;<sub>i</sub> (this is the ARD
kernel), then be Bayesian about relevance:</p>
<div class='math'>k(x, x&prime;) = exp( &minus;&frac12; &sum;<sub>i</sub> &rho;<sub>i</sub> (x<sub>i</sub> &minus; x&prime;<sub>i</sub>)&sup2; ),&nbsp;&nbsp;&nbsp;
&tau; ~ HalfCauchy(&alpha;),&nbsp;&nbsp;&rho;<sub>i</sub> ~ HalfCauchy(&tau;)</div>
<p>The half-Cauchy is the load-bearing choice: its mode at zero means every dimension is
switched <em>off</em> by default (&rho;<sub>i</sub> &asymp; 0 &rArr; that coordinate leaves the
kernel), while its heavy tail lets the handful of dimensions the data actually supports escape
to large &rho;<sub>i</sub>. The global scale &tau; adapts how many escape. Inference is fully
Bayesian (NUTS), with predictions averaged over posterior samples &mdash; expensive per
iteration, but in BO&rsquo;s regime the evaluations still dominate. <em>Sparsity is assumed
until the data earns density</em> &mdash; the exact opposite of fitting 100 lengthscales by
maximum likelihood to 30 points.</p>

<h3>TuRBO: the trust-region algebra</h3>
<p>TuRBO abandons global modeling: it keeps a hyperrectangle centered on the incumbent and only
models/acquires inside it. The region has base side L, shaped per-dimension by the fitted
lengthscales (so the box is wide where the function is flat):</p>
<div class='math'>&lambda;<sub>i</sub> = &#8467;<sub>i</sub> &middot; L / ( &prod;<sub>j</sub> &#8467;<sub>j</sub> )<sup>1/d</sup></div>
<p>The side length then <em>breathes</em> by simple counters: &tau;<sub>succ</sub> consecutive
improving evaluations double it (capped at L<sub>max</sub>); &tau;<sub>fail</sub> consecutive
failures halve it; and when L &lt; L<sub>min</sub> the region has collapsed onto a local
optimum &mdash; declare victory locally, restart elsewhere, and let an implicit multi-armed
bandit (Thompson sampling across regions, in the full algorithm) allocate the budget. Here is
that dynamic, computed live for this page:</p>

<div class='widget'>
__FIG_TURBO__
<div class='caption'>Our simplified single-region TuRBO (numpy, one run on negated Branin):
the side length halves through dry spells, and a dotted line marks a restart after collapse
&mdash; the algorithm concluding a local search and spending its remaining budget elsewhere.
Local + restart beats confidently-wrong global modeling in high dimensions; that is
TuRBO&rsquo;s entire bet.</div>
</div>

<h3>&pi;BO: folding expert beliefs into the acquisition</h3>
<p>When a practitioner already believes the optimum lives somewhere (&ldquo;learning rates
around 10<sup>&minus;3</sup>&rdquo;), &pi;BO injects that belief &pi;(x) as a decaying
multiplicative reweighting of any acquisition &alpha;:</p>
<div class='math'>&alpha;<sub>&pi;,n</sub>(x) = &alpha;<sub>n</sub>(x) &middot; &pi;(x)<sup>&beta;/n</sup></div>
<p>At n = &beta; the prior speaks with full voice; as n grows the exponent &beta;/n &rarr; 0 and
&pi;(x)<sup>&beta;/n</sup> &rarr; 1 &mdash; <em>the data always wins eventually</em>, and a wrong
prior costs only the early iterations (the paper proves EI&rsquo;s convergence rate survives).
Compare Maths I&rsquo;s marginal likelihood: that fits the <em>smoothness</em> prior from data;
&pi;BO adds a <em>location</em> prior from humans. The two compose.</p>

<h3>PFNs: surrogates without fitting</h3>
<p>The pretrained-surrogate line replaces GP fitting with in-context learning. Draw synthetic
tasks from a prior over functions; train a transformer q<sub>&theta;</sub> to predict a held-out
value from the context set D by straight cross-entropy:</p>
<div class='math'>min<sub>&theta;</sub> &nbsp;&#120124;<sub>D, x<sup>*</sup>, y<sup>*</sup></sub> [ &minus;log q<sub>&theta;</sub>( y<sup>*</sup> | x<sup>*</sup>, D ) ]</div>
<p>The identity that makes this Bayesian rather than a heuristic: that expectation decomposes as
&#120124;<sub>D,x*</sub>[ KL( p(&middot;|x<sup>*</sup>, D) &Vert; q<sub>&theta;</sub>(&middot;|x<sup>*</sup>, D) ) ] plus an
entropy constant &mdash; so the loss is minimized <em>exactly</em> when q<sub>&theta;</sub>
equals the true posterior predictive under the synthetic prior. <strong>A PFN is amortized
Bayesian inference: the prior is baked into the weights, and the O(n&sup3;) fit becomes one
forward pass.</strong> PFNs4BO plugs this predictive (a binned &ldquo;Riemann&rdquo;
distribution over y) straight into EI; GIT-BO rides the same idea on tabular foundation models
into hundreds of dimensions. The open question is the one you&rsquo;d expect: you no longer
choose the prior &mdash; the pretraining distribution did, and auditing what a network
believes is harder than reading a kernel.</p>
"""

QUIZ_MATH_SOTA = [
    {
        "question": "For x, x&prime; uniform in [0,1]<sup>d</sup>, &#120124;&Vert;x&minus;x&prime;&Vert;&sup2; = d/6. To keep typical RBF correlations from collapsing as d grows, the lengthscale must scale like:",
        "options": [
            {"text": "&#8467; &prop; &radic;d",
             "correct": True,
             "explanation": "The kernel sees ||x−x′||²/ℓ² — with the numerator growing linearly in d, ℓ² must grow linearly too, i.e. ℓ ∝ √d. This one line of arithmetic is the core of the 2024 'vanilla BO' result, implemented as the LogNormal(μ₀ + ½log d, σ₀²) prior."},
            {"text": "&#8467; &prop; d",
             "explanation": "Scaling ℓ linearly in d over-corrects: typical distances grow like √d (distances, not squared distances), so linear ℓ would make everything look correlated and the posterior overly smooth."},
            {"text": "&#8467; &prop; log d",
             "explanation": "Logarithmic growth is far too slow — the d/6 in the exponent would still crush correlations to zero for moderate d."},
            {"text": "&#8467; constant, with more observations instead",
             "explanation": "The budget is the one thing you can't scale in BO — and the required sample count for a fixed-ℓ GP grows exponentially in d. The prior fix costs nothing."},
        ],
    },
    {
        "question": "Why does SAASBO put a half-Cauchy prior on the inverse squared lengthscales &rho;<sub>i</sub>?",
        "options": [
            {"text": "Mass at zero turns dimensions off by default; heavy tails let the data turn a few on.",
             "correct": True,
             "explanation": "The half-Cauchy's mode at 0 encodes 'irrelevant until proven otherwise' (ρᵢ ≈ 0 removes coordinate i from the kernel), while its fat tail permits the few truly relevant ρᵢ to grow large without penalty. Global scale τ adapts how many escape."},
            {"text": "It is conjugate to the Gaussian likelihood, giving closed-form posteriors.",
             "explanation": "Nothing about half-Cauchy is conjugate here — that's exactly why SAASBO needs NUTS sampling rather than closed-form updates."},
            {"text": "It forces all dimensions to share one common lengthscale.",
             "explanation": "That's the opposite design — a single shared ℓ is the non-ARD kernel; SAASBO's point is per-dimension relevance."},
            {"text": "It guarantees the marginal likelihood surface becomes unimodal.",
             "explanation": "No prior rescues LML unimodality; SAASBO sidesteps the issue by integrating over the posterior instead of optimizing to a point."},
        ],
    },
    {
        "question": "In TuRBO, what happens after &tau;<sub>fail</sub> consecutive non-improving evaluations, and after L falls below L<sub>min</sub>?",
        "options": [
            {"text": "The side length halves; below L<sub>min</sub> the region restarts elsewhere.",
             "correct": True,
             "explanation": "Failures shrink the box (L ← L/2) to concentrate near the incumbent; collapse below L_min is read as 'local optimum found', triggering a fresh region — with Thompson sampling allocating budget across regions in the full algorithm."},
            {"text": "The side length halves; below L<sub>min</sub> the region freezes at the incumbent.",
             "explanation": "Freezing would waste the remaining budget polishing a solved local problem — the restart is the mechanism that converts local searches into global coverage."},
            {"text": "The side length doubles to escape the plateau; below L<sub>min</sub> it resets to L<sub>max</sub>.",
             "explanation": "Doubling is the SUCCESS response (after τ_succ improvements). Expanding on failure would abandon the locality that makes the local GP trustworthy."},
            {"text": "The lengthscales are refit; L never changes, only its shape λ<sub>i</sub>.",
             "explanation": "The per-dimension shaping λᵢ ∝ ℓᵢ is real, but it modulates a base L that very much changes — the breathing L is the whole control loop."},
        ],
    },
    {
        "question": "The PFN training loss &#120124;[&minus;log q<sub>&theta;</sub>(y*|x*, D)] over synthetic tasks is minimized exactly when q<sub>&theta;</sub> equals:",
        "options": [
            {"text": "The posterior predictive p(y*|x*, D) under the synthetic prior.",
             "correct": True,
             "explanation": "The expected cross-entropy decomposes into E[KL(p‖q_θ)] plus a constant entropy term, and KL vanishes iff q_θ matches p(y*|x*,D). That identity is what makes a PFN amortized Bayesian inference rather than curve-fitting — with the prior baked into the weights."},
            {"text": "The MAP estimate of the function under the synthetic prior.",
             "explanation": "Cross-entropy against sampled y* penalizes point-mass answers — the optimum is the full predictive distribution, not its mode."},
            {"text": "The marginal likelihood of the context set D.",
             "explanation": "The LML scores hyperparameters given data (Maths I); the PFN objective targets prediction of held-out values, a different conditional."},
            {"text": "The true black-box function f itself.",
             "explanation": "No optimizer output can converge to f from finite noisy context — and the loss explicitly rewards honest uncertainty about it."},
        ],
    },
    {
        "question": "In &pi;BO's reweighting &alpha;(x)&middot;&pi;(x)<sup>&beta;/n</sup>, what happens to a WRONG expert prior as evaluations accumulate?",
        "options": [
            {"text": "Its influence decays to nothing — it only taxes the early iterations.",
             "correct": True,
             "explanation": "The exponent β/n → 0, so π(x)^(β/n) → 1 for every x and the acquisition returns to its data-driven form; the paper shows EI's convergence rate survives a misleading prior. Cheap insurance: strong when you know something, harmless when you don't."},
            {"text": "It permanently biases the search away from the true optimum.",
             "explanation": "That's the failure mode of hard constraints or fixed prior means — the decaying exponent is πBO's specific device for avoiding it."},
            {"text": "It is overwritten by refitting the GP lengthscales each iteration.",
             "explanation": "Lengthscale fitting (Maths I) adapts the smoothness model; πBO's π(x) lives in the acquisition and decays by schedule, not by refitting."},
            {"text": "The method detects the conflict and switches to pure UCB.",
             "explanation": "No detection logic exists or is needed — the vanishing exponent handles wrong priors continuously, without a mode switch."},
        ],
    },
]

PAPERS = """
<p>One section for the whole literature: the narrative arc first (what each work fixed), then
the complete reference list. Methodology: this survey was run on 2026-08-02; every arXiv-hosted
paper was verified against the arXiv API (exact title, ID, date, first author) and
recent-submission sweeps ground the 2025&ndash;26 claims. Citation magnitudes are approximate
(&sim;) reputation figures, as arXiv carries none. The math sections follow Frazier&rsquo;s and
Garnett&rsquo;s notation. Errors of interpretation are this page&rsquo;s, not the
sources&rsquo;.</p>

<h3>The foundations</h3>
<p><strong>Kushner (1964)</strong> proposed maximizing the probability of improvement over a
stochastic-process model; <strong>Mockus (1974&ndash;78)</strong> introduced expected
improvement and named the field. The modern template is <strong>Jones, Schonlau &amp; Welch
(1998)</strong> &mdash; EGO: kriging surrogate + closed-form EI, expensive engineering
objectives falling in tens of evaluations (~8k citations); its worked examples and its honesty
about kriging&rsquo;s failure modes still read well. <strong>GP-UCB</strong>
(<a href='https://arxiv.org/abs/0912.3995'>arXiv:0912.3995</a>, ICML 2010, ~4k) recast BO as a
bandit problem and proved sublinear regret &mdash; the reason &ldquo;BO provably
converges&rdquo; can be said with a straight face. And <strong>Snoek, Larochelle &amp; Adams
(2012)</strong> (<a href='https://arxiv.org/abs/1206.2944'>arXiv:1206.2944</a>, NeurIPS 2012,
~10k) took the machinery to ML hyperparameters, beat human experts, shipped Spearmint, and
started AutoML &mdash; quietly also a great engineering paper (MCMC over GP hyperparameters,
EI-per-second for cost-aware search, asynchronous parallelism).</p>

<h3>The modern toolbox</h3>
<p><strong>Beyond-GP surrogates:</strong> TPE (Bergstra et al., NeurIPS 2011, ~5k) models
p(x|good)/p(x|bad) densities &mdash; the engine of Optuna and Hyperopt, natural for
conditional, tree-shaped search spaces; SMAC (Hutter et al., 2011) uses random forests for the
same reason. <strong>Scaling</strong> (the equations are in
<a href='#math-sota'>Maths III</a>): TuRBO
(<a href='https://arxiv.org/abs/1910.01739'>1910.01739</a>, NeurIPS 2019) with trust regions;
SAASBO (<a href='https://arxiv.org/abs/2103.00349'>2103.00349</a>, UAI 2021) with sparsity
priors; Bounce (<a href='https://arxiv.org/abs/2307.00618'>2307.00618</a>, NeurIPS 2023) for
combinatorial and mixed spaces; qEHVI
(<a href='https://arxiv.org/abs/2006.05078'>2006.05078</a>, NeurIPS 2020) for parallel
multi-objective; &pi;BO (<a href='https://arxiv.org/abs/2204.11051'>2204.11051</a>, ICLR 2022)
for expert beliefs. <strong>Numerics:</strong> LogEI
(<a href='https://arxiv.org/abs/2310.20708'>2310.20708</a>, NeurIPS 2023) fixed EI&rsquo;s
silent underflow and is the modern default. <strong>Software:</strong> BoTorch
(<a href='https://arxiv.org/abs/1910.06403'>1910.06403</a>, NeurIPS 2020) rebuilt acquisitions
on Monte Carlo + autograd; Ax wraps it; Optuna (KDD 2019) ships TPE to everyone; HEBO won the
NeurIPS 2020 black-box competition and was modernized in
<a href='https://arxiv.org/abs/2607.10669'>2607.10669</a> (July 2026). <strong>The
application:</strong> Shields et al. (<em>Nature</em> 2021) &mdash; BO matching-to-beating
expert chemists, the template for self-driving laboratories.</p>

<h3>Three live SOTA threads (2024 &rarr; mid-2026)</h3>
<p><strong>1. The high-dimensional controversy.</strong> <em>Vanilla BO Performs Great in High
Dimensions</em> (<a href='https://arxiv.org/abs/2402.02229'>2402.02229</a>, ICML 2024) and
<em>Standard GP is All You Need</em>
(<a href='https://arxiv.org/abs/2402.02746'>2402.02746</a>, ICLR 2025) argue the specialized
machinery was never necessary &mdash; the &radic;d prior from Maths III does the work. <em>We
Still Don&rsquo;t Understand High-Dimensional BO</em>
(<a href='https://arxiv.org/abs/2512.00170'>2512.00170</a>, Nov 2025) pushes back: wins are
benchmark-dependent, explanations don&rsquo;t hold. Follow-ups continue
(<a href='https://arxiv.org/abs/2605.20249'>2605.20249</a>, May 2026). Genuinely unresolved.
<strong>2. Pretrained surrogates.</strong> OptFormer
(<a href='https://arxiv.org/abs/2205.13320'>2205.13320</a>, NeurIPS 2022), PFNs4BO
(<a href='https://arxiv.org/abs/2305.17535'>2305.17535</a>, ICML 2023), GIT-BO
(<a href='https://arxiv.org/abs/2505.20685'>2505.20685</a>, 2025): amortized Bayesian inference
in a transformer forward pass (the objective is derived in Maths III).
<strong>3. LLMs in the loop.</strong> LLAMBO
(<a href='https://arxiv.org/abs/2402.03921'>2402.03921</a>, ICLR 2024), evidence-gated LLM
priors (<a href='https://arxiv.org/abs/2606.01730'>2606.01730</a>, 2026), LLM-evolved MOBO
algorithms (<a href='https://arxiv.org/abs/2607.08791'>2607.08791</a>, 2026) &mdash; against
budget-matched skepticism (<a href='https://arxiv.org/abs/2606.21641'>2606.21641</a>;
<a href='https://arxiv.org/abs/2509.21403'>2509.21403</a>). Current verdict: strong for warm
starts and priors, not yet a surrogate replacement.</p>

<h3>The complete reference list</h3>
<ol>
<li>H. J. Kushner (1964). A New Method of Locating the Maximum Point of an Arbitrary Multipeak
Curve in the Presence of Noise. <em>J. Basic Engineering</em> 86(1). <em>Probability of
improvement.</em></li>
<li>J. Mockus (1974/1978). On Bayesian Methods for Seeking the Extremum. <em>Optimization
Techniques / Towards Global Optimization 2</em>. <em>Expected improvement; the name.</em></li>
<li>Donald R. Jones, Matthias Schonlau, William J. Welch (1998). Efficient Global Optimization
of Expensive Black-Box Functions. <em>J. Global Optimization</em> 13:455&ndash;492. <em>The
seminal EGO template.</em></li>
<li>Niranjan Srinivas, Andreas Krause, Sham Kakade, Matthias Seeger (2010).
<a href='https://arxiv.org/abs/0912.3995'>Gaussian Process Optimization in the Bandit Setting:
No Regret and Experimental Design</a>. ICML 2010. arXiv:0912.3995.</li>
<li>James Bergstra, R&eacute;mi Bardenet, Yoshua Bengio, Bal&aacute;zs K&eacute;gl (2011).
Algorithms for Hyper-Parameter Optimization (TPE). NeurIPS 2011.</li>
<li>Frank Hutter, Holger H. Hoos, Kevin Leyton-Brown (2011). Sequential Model-Based
Optimization for General Algorithm Configuration (SMAC). LION 5.</li>
<li>Jasper Snoek, Hugo Larochelle, Ryan P. Adams (2012).
<a href='https://arxiv.org/abs/1206.2944'>Practical Bayesian Optimization of Machine Learning
Algorithms</a>. NeurIPS 2012. arXiv:1206.2944.</li>
<li>Philipp Hennig, Christian J. Schuler (2012). <a href='https://arxiv.org/abs/1112.1217'>
Entropy Search for Information-Efficient Global Optimization</a>. arXiv:1112.1217.</li>
<li>Jos&eacute; Miguel Hern&aacute;ndez-Lobato, Matthew W. Hoffman, Zoubin Ghahramani (2014).
<a href='https://arxiv.org/abs/1406.2541'>Predictive Entropy Search</a>. NeurIPS 2014.
arXiv:1406.2541.</li>
<li>Bobak Shahriari, Kevin Swersky, Ziyu Wang, Ryan P. Adams, Nando de Freitas (2016). Taking
the Human Out of the Loop: A Review of Bayesian Optimization. <em>Proc. IEEE</em> 104(1).</li>
<li>Zi Wang, Stefanie Jegelka (2017). <a href='https://arxiv.org/abs/1703.01968'>Max-value
Entropy Search for Efficient Bayesian Optimization</a>. ICML 2017. arXiv:1703.01968.</li>
<li>Peter I. Frazier (2018). <a href='https://arxiv.org/abs/1807.02811'>A Tutorial on Bayesian
Optimization</a>. arXiv:1807.02811. <em>This page&rsquo;s notation.</em></li>
<li>David Eriksson, Michael Pearce, Jacob R. Gardner, Ryan Turner, Matthias Poloczek (2019).
<a href='https://arxiv.org/abs/1910.01739'>Scalable Global Optimization via Local Bayesian
Optimization (TuRBO)</a>. NeurIPS 2019. arXiv:1910.01739.</li>
<li>Maximilian Balandat, Brian Karrer, Daniel R. Jiang, Samuel Daulton, Benjamin Letham,
Andrew Gordon Wilson, Eytan Bakshy (2020). <a href='https://arxiv.org/abs/1910.06403'>BoTorch:
A Framework for Efficient Monte-Carlo Bayesian Optimization</a>. NeurIPS 2020.
arXiv:1910.06403.</li>
<li>Samuel Daulton, Maximilian Balandat, Eytan Bakshy (2020).
<a href='https://arxiv.org/abs/2006.05078'>Differentiable Expected Hypervolume Improvement for
Parallel Multi-Objective Bayesian Optimization (qEHVI)</a>. NeurIPS 2020. arXiv:2006.05078.</li>
<li>David Eriksson, Martin Jankowiak (2021). <a href='https://arxiv.org/abs/2103.00349'>
High-Dimensional Bayesian Optimization with Sparse Axis-Aligned Subspaces (SAASBO)</a>. UAI
2021. arXiv:2103.00349.</li>
<li>Benjamin J. Shields, Jason Stevens, Jun Li, et al. (2021). Bayesian reaction optimization
as a tool for chemical synthesis. <em>Nature</em> 590:89&ndash;96.</li>
<li>Carl Hvarfner, Danny Stoll, Artur Souza, Marius Lindauer, Frank Hutter, Luigi Nardi (2022).
<a href='https://arxiv.org/abs/2204.11051'>&pi;BO: Augmenting Acquisition Functions with User
Beliefs</a>. ICLR 2022. arXiv:2204.11051.</li>
<li>Yutian Chen et al. (2022). <a href='https://arxiv.org/abs/2205.13320'>Towards Learning
Universal Hyperparameter Optimizers with Transformers (OptFormer)</a>. NeurIPS 2022.
arXiv:2205.13320.</li>
<li>Alexander I. Cowen-Rivers et al. (2022). HEBO: Pushing the Limits of Sample-Efficient
Hyper-parameter Optimisation. <em>JAIR</em> 74; modernized in
<a href='https://arxiv.org/abs/2607.10669'>arXiv:2607.10669</a> (2026).</li>
<li>Samuel M&uuml;ller, Matthias Feurer, Noah Hollmann, Frank Hutter (2023).
<a href='https://arxiv.org/abs/2305.17535'>PFNs4BO: In-Context Learning for Bayesian
Optimization</a>. ICML 2023. arXiv:2305.17535.</li>
<li>Leonard Papenmeier, Luigi Nardi, Matthias Poloczek (2023).
<a href='https://arxiv.org/abs/2307.00618'>Bounce: Reliable High-Dimensional Bayesian
Optimization for Combinatorial and Mixed Spaces</a>. NeurIPS 2023. arXiv:2307.00618.</li>
<li>Sebastian Ament, Samuel Daulton, David Eriksson, Maximilian Balandat, Eytan Bakshy (2023).
<a href='https://arxiv.org/abs/2310.20708'>Unexpected Improvements to Expected Improvement for
Bayesian Optimization (LogEI)</a>. NeurIPS 2023. arXiv:2310.20708.</li>
<li>Carl Hvarfner, Erik Orm Hellsten, Luigi Nardi (2024).
<a href='https://arxiv.org/abs/2402.02229'>Vanilla Bayesian Optimization Performs Great in High
Dimensions</a>. ICML 2024. arXiv:2402.02229.</li>
<li>Zhitong Xu, Shandian Zhe (2024). <a href='https://arxiv.org/abs/2402.02746'>Standard
Gaussian Process is All You Need for High-Dimensional Bayesian Optimization</a>. ICLR 2025.
arXiv:2402.02746.</li>
<li>Tennison Liu, Nicol&aacute;s Astorga, Nabeel Seedat, Mihaela van der Schaar (2024).
<a href='https://arxiv.org/abs/2402.03921'>Large Language Models to Enhance Bayesian
Optimization (LLAMBO)</a>. ICLR 2024. arXiv:2402.03921.</li>
<li>Willie Neiswanger et al. (2025). <a href='https://arxiv.org/abs/2502.06789'>
Information-theoretic Bayesian Optimization: Survey and Tutorial</a>. arXiv:2502.06789.</li>
<li>(2025). <a href='https://arxiv.org/abs/2505.20685'>GIT-BO: High-Dimensional Bayesian
Optimization with Tabular Foundation Models</a>. arXiv:2505.20685.</li>
<li>(2025). <a href='https://arxiv.org/abs/2509.21403'>LLMs for Bayesian Optimization in
Scientific Domains: Are We There Yet?</a>. arXiv:2509.21403.</li>
<li>Leonard Papenmeier et al. (2025). <a href='https://arxiv.org/abs/2512.00170'>We Still
Don&rsquo;t Understand High-Dimensional Bayesian Optimization</a>. arXiv:2512.00170.</li>
<li>(2026). <a href='https://arxiv.org/abs/2605.20249'>Automated Kernel Discovery Towards
Understanding High-dimensional Bayesian Optimization</a>. arXiv:2605.20249.</li>
<li>(2026). <a href='https://arxiv.org/abs/2606.01730'>Evidence-Gated LLM Priors for
Multi-Objective Bayesian Optimization</a>. arXiv:2606.01730.</li>
<li>(2026). <a href='https://arxiv.org/abs/2606.21641'>When Is an LLM Worth It for
Hyperparameter Optimization? A Budget-Matched Study</a>. arXiv:2606.21641.</li>
<li>(2026). <a href='https://arxiv.org/abs/2607.08791'>LLM-Driven Evolutionary Generation of
Multi-Objective Bayesian Optimization Algorithms</a>. arXiv:2607.08791.</li>
<li>Takuya Akiba, Shotaro Sano, Toshihiko Yanase, Takeru Ohta, Masanori Koyama (2019). Optuna:
A Next-generation Hyperparameter Optimization Framework. KDD 2019.</li>
<li>Roman Garnett (2023). <em>Bayesian Optimization</em>. Cambridge University Press. Free at
<a href='https://bayesoptbook.com/'>bayesoptbook.com</a> (verified live).</li>
<li>Carl E. Rasmussen, Christopher K. I. Williams (2006). <em>Gaussian Processes for Machine
Learning</em>. MIT Press. Free at
<a href='https://gaussianprocess.org/gpml/'>gaussianprocess.org/gpml</a>.</li>
<li>Robert B. Gramacy (2020). <em>Surrogates</em>. CRC. Free at
<a href='https://bobby.gramacy.com/surrogates/'>bobby.gramacy.com/surrogates</a>.</li>
<li>(2023). <a href='https://arxiv.org/abs/2311.13050'>Multi-fidelity Bayesian Optimization: A
Review</a>. arXiv:2311.13050. <em>The fidelity axis this page leaves aside.</em></li>
<li>James Bergstra, Yoshua Bengio (2012). Random Search for Hyper-Parameter Optimization.
<em>JMLR</em> 13. <em>The random-beats-grid argument in the Background.</em></li>
<li>Nikolaus Hansen (2016). <a href='https://arxiv.org/abs/1604.00772'>The CMA Evolution
Strategy: A Tutorial</a>. arXiv:1604.00772. <em>The evolutionary baseline in tables and quiz
distractors.</em></li>
<li>Peter I. Frazier, Warren B. Powell, Savas Dayanik (2008). A Knowledge-Gradient Policy for
Sequential Information Collection. <em>SIAM J. Control Optim.</em> 47(5). <em>The
one-step-deeper lookahead mentioned under myopia.</em></li>
<li>Jacob R. Gardner et al. (2018). <a href='https://arxiv.org/abs/1809.11165'>GPyTorch</a>.
NeurIPS 2018. <em>The GP engine under BoTorch.</em></li>
</ol>

<p>Suggested reading order: Frazier&rsquo;s tutorial first (this page with more proofs), EGO
for the origin, Snoek et al. for the practice, LogEI to update your defaults, then the three
high-dimensional papers as a set &mdash; a live scientific argument in motion.
Garnett&rsquo;s book whenever a chapter-length treatment is wanted.</p>
"""

QUIZ_PAPERS = [
    {
        "question": "Which limitation-to-fix pairing from the modern toolbox is correct?",
        "options": [
            {"text": "Global GPs mislead in high dimensions → TuRBO's shrinking/growing trust regions.",
             "correct": True,
             "explanation": "TuRBO gives up on modeling the whole space and runs local searches with trust-region dynamics. The other pairings are shuffled: sparse-dimension priors are SAASBO, user beliefs are πBO, combinatorial/mixed spaces are Bounce."},
            {"text": "Global GPs mislead in high dimensions → πBO's user-belief priors.",
             "explanation": "πBO injects expert opinion about WHERE the optimum sits into the acquisition — orthogonal to dimensionality. The trust-region answer is TuRBO."},
            {"text": "Few dimensions actually matter → Bounce's combinatorial embeddings.",
             "explanation": "Sparse effective dimensionality is SAASBO's premise (sparse axis-aligned lengthscale priors); Bounce targets combinatorial and mixed variable types."},
            {"text": "Experiments run in parallel batches → SAASBO's sparse priors.",
             "explanation": "Parallelism is the q-family's territory (qEI/qEHVI via joint Monte-Carlo); SAASBO is about high-dimensional sparsity."},
        ],
    },
    {
        "question": "TPE — the engine inside Optuna — differs from GP-based BO how?",
        "options": [
            {"text": "It models densities p(x|good) and p(x|bad) and ranks candidates by their ratio.",
             "correct": True,
             "explanation": "TPE flips the modeling: instead of a posterior over f's VALUES, it splits observations at a quantile and models where good vs bad configurations LIVE — which handles conditional, tree-shaped hyperparameter spaces naturally and scales linearly. The ratio is monotone in EI under TPE's assumptions."},
            {"text": "It fits the same GP but replaces EI with a UCB acquisition.",
             "explanation": "TPE has no GP at all — the surrogate itself (Parzen density estimators) is the departure, not the acquisition."},
            {"text": "It uses a random forest to predict function values with uncertainty.",
             "explanation": "Forest-based value regression is SMAC's approach — the other 2011 non-GP surrogate, easy to conflate with TPE."},
            {"text": "It is a pure evolutionary strategy with no probabilistic model.",
             "explanation": "Model-free evolution is CMA-ES territory; TPE is thoroughly probabilistic — just generative over inputs instead of predictive over outputs."},
        ],
    },
    {
        "question": "What is the actual state of the high-dimensional BO controversy as of early 2026?",
        "options": [
            {"text": "Contested: 2024–25 papers said plain GPs suffice; a late-2025 rebuttal disputes the evidence.",
             "correct": True,
             "explanation": "Vanilla-BO (ICML 2024) and Standard-GP (ICLR 2025) claimed properly-scaled priors let vanilla BO match TuRBO/SAASBO; 'We Still Don't Understand High-Dimensional BO' (Nov 2025) argues the wins are benchmark-dependent and the proposed explanations fail. Genuinely unresolved — cite accordingly."},
            {"text": "Settled: subspace methods like SAASBO are proven necessary above ~20 dimensions.",
             "explanation": "That was the pre-2024 conventional wisdom — exactly what the vanilla-BO papers attacked. Nothing about it is 'proven' today."},
            {"text": "Settled: vanilla GPs won and the specialized high-dim methods are deprecated.",
             "explanation": "The 2025 rebuttal specifically contests this reading; treating the vanilla-BO result as final is premature by the field's own current papers."},
            {"text": "Abandoned: the field moved entirely to LLM-based optimizers in 2025.",
             "explanation": "LLM-BO is a separate (and itself contested) thread; the high-dim GP argument is being actively litigated in 2025–26 papers."},
        ],
    },
    {
        "question": "Pretrained surrogates (OptFormer, PFNs4BO, GIT-BO) attack which TWO weaknesses of classic GP-BO at once?",
        "options": [
            {"text": "The per-iteration refitting cost and the need to hand-choose a prior.",
             "correct": True,
             "explanation": "A transformer pre-trained on millions of synthetic functions does surrogate inference in one forward pass (no O(n³) refit, no LML optimization) and has effectively absorbed a rich prior from its training distribution — trading both problems for a new one: what did it learn to assume?"},
            {"text": "The myopia of one-step acquisitions and the lack of regret guarantees.",
             "explanation": "Amortized surrogates change the MODEL, not the decision rule — most still run EI-style acquisitions on top, exactly as myopic and less analyzable than before."},
            {"text": "The cost of function evaluations and the noise in observations.",
             "explanation": "No surrogate can make the chemistry run faster — evaluation cost is the problem's constant, and noise handling is standard in GPs already."},
            {"text": "The curse of dimensionality and the explore-exploit trade-off itself.",
             "explanation": "GIT-BO does target higher dimensions, but no method dissolves the explore-exploit trade-off — it's the problem statement, not an implementation flaw."},
        ],
    },
    {
        "question": "Per the 2026 budget-matched studies, where do LLMs currently EARN their keep in the BO loop?",
        "options": [
            {"text": "Warm-starting and prior-shaping from problem descriptions — not replacing the surrogate.",
             "correct": True,
             "explanation": "The consistent 2025–26 finding (LLAMBO's strongest results, evidence-gated priors, the budget-matched HPO study): LLM world-knowledge helps most before/around the loop — initialization and priors — while calibrated uncertainty from a statistical surrogate remains hard to beat once real observations accumulate."},
            {"text": "Replacing the GP posterior end-to-end with prompted value predictions.",
             "explanation": "Tried, and it's exactly what the skeptical studies target: prompted point predictions lack calibrated uncertainty, and budget-matched comparisons show statistical surrogates holding the line."},
            {"text": "Providing regret guarantees that GP methods lack.",
             "explanation": "Inverted: the GP line HAS regret theory (GP-UCB); LLM components currently have none at all."},
            {"text": "Reducing the wall-clock cost of each function evaluation.",
             "explanation": "The evaluation is a physical experiment or a training run — no optimizer component, LLM or otherwise, touches its cost."},
        ],
    },
]

CONCEPT_MAP = """
<p>Every idea on this page, one screen. Hover a node to trace its connections; click for a
recap and a jump link. The three outlined hubs &mdash; the loop, the posterior, and EI &mdash;
carry everything else: if you can rebuild those three from memory, the rest reattaches
naturally.</p>

<div class='widget' id='w-map'>
  <svg id='w-map-svg' viewBox='0 0 860 580' role='img' style='width:100%;height:auto'></svg>
  <div class='wstat' id='w-map-info'>hover to trace connections &middot; click a node for a recap and a jump link</div>
</div>

<p>Self-test: pick any edge and say out loud <em>why</em> those two nodes are connected. Every
edge is one sentence you should be able to produce.</p>
"""

KEEP_LEARNING = """
<p>This page taught you to recognize the machinery. The pieces below make it stick and make it
yours &mdash; each targets a different failure mode of &ldquo;I read it and it made
sense.&rdquo;</p>

<h3>Spaced review (retention)</h3>
<ul>
<li><strong><a href='2026-08-02-bayesian-optimization-review.html'>The review deck</a></strong>
&mdash; spaced repetition over this page&rsquo;s 30 questions (Leitner boxes,
1&rarr;3&rarr;7&rarr;14&rarr;30 days). Open it tomorrow, then whenever it says cards are due;
finish a session and use <em>copy results for Claude</em> for targeted follow-up.</li>
<li><strong><a href='bayesian-optimization.apkg'>Anki deck</a></strong> &mdash; the same
questions as a standard .apkg if you already run Anki.</li>
</ul>
<p>The section quizzes here also store your results locally &mdash; after any section, <em>copy
results for Claude</em> and paste into a session to get re-quizzed on your misses.</p>

<h3>Teach it back (generation)</h3>
<p>Copy the prompt below into a fresh Claude session and teach the topic to a curious
student:</p>
<div class='callout' id='teachback-prompt'>
<p>I just studied Bayesian optimization. Play a curious student who wants to learn it from me.
Ask me to explain, one at a time: (1) why expensive black-box problems need a different kind of
optimizer, and what the surrogate/acquisition loop is; (2) the GP posterior &mdash; what
&mu;(x) and &sigma;(x) mean, their closed forms, and why &sigma; doesn&rsquo;t depend on the
observed values; (3) the derivation of expected improvement, including the value of EI when
&mu; = f* (and why it isn&rsquo;t zero); (4) UCB and what GP-UCB&rsquo;s regret result buys,
plus what &ldquo;myopic&rdquo; means for all one-step acquisitions; (5) the current research
picture: the high-dimensional controversy, pretrained surrogates, and what LLMs are and
aren&rsquo;t good for in the loop. Probe every explanation with at least one &ldquo;why&rdquo;
follow-up. Do not explain anything yourself unless I am stuck after two attempts &mdash; then
give a hint, not the answer. At the end: grade me on mechanism, formulas, and honest caveats;
list what I got wrong or fuzzy; and write three new quiz questions targeting exactly my weak
spots.</p>
</div>
<button class='wbtn' data-copy='teachback-prompt'>copy the teach-back prompt</button>

<h3>Build it yourself (transfer)</h3>
<p>The repo contains a tutorial at <code>tutorials/bayesian-optimization/</code>: a numpy
skeleton of everything the maths sections derived &mdash; the RBF kernel, the GP posterior,
closed-form EI, UCB, and the full BO loop &mdash; with implementations replaced by TODOs and a
pytest suite that knows the right answers (exact values from the quizzes, plus an integration
test where your loop must beat random search):</p>
<pre>cd tutorials/bayesian-optimization
python3 -m pytest -q        # red until your implementations are right</pre>
<p>When the suite is green you have not read Bayesian optimization &mdash; you have written it.
Stuck on one function? <code>solutions/</code> has a reference; peek at one function, not the
file.</p>

<h3>Going deeper (the books)</h3>
<p><a href='https://bayesoptbook.com/'>Garnett, <em>Bayesian Optimization</em></a> (Cambridge
2023, free PDF) when you want chapter-length treatments of everything here; Rasmussen &amp;
Williams, <a href='https://gaussianprocess.org/gpml/'><em>Gaussian Processes for Machine
Learning</em></a> (MIT 2006, free) for the surrogate in full depth; Gramacy,
<a href='https://bobby.gramacy.com/surrogates/'><em>Surrogates</em></a> (2020, free) for the
statistician&rsquo;s view with code.</p>

<h3>The cheat sheet (consolidation)</h3>
<p><a href='2026-08-02-bo-cheatsheet.html'>One printable page</a>: every formula above with its
one-sentence punchline.</p>
"""

spec = {
    "title": "Fifty Evaluations to Find the Best: Bayesian Optimization",
    "subtitle": "Gaussian process surrogates, acquisition functions derived by hand, and the 2024–2026 state of the art — every paper verified on arXiv",
    "slug": "bayesian-optimization",
    "date": "2026-08-02",
    "multipage": True,
    "site_title": "← Bayesian Optimization",
    "nav": [["Review deck", "2026-08-02-bayesian-optimization-review.html"],
            ["Cheat sheet", "2026-08-02-bo-cheatsheet.html"],
            ["Tutorials", "https://github.com/raghuramshankar/learning-with-llms/tree/main/tutorials/bayesian-optimization"]],
    "generator": {
        "skill": "learning-new-topic",
        "skill_url": "https://github.com/raghuramshankar/learning-with-llms/blob/main/skills/learning-new-topic/SKILL.md",
        "model": "Claude Fable 5",
    },
    "intro": """
<p>This is a deep dive into Bayesian optimization &mdash; how to find the best value of an
expensive black-box function in a few dozen evaluations. It was built from a fresh literature
survey (every arXiv paper cited is API-verified), and it is meant to be worked through, not
skimmed: each part ends with a hard five-question quiz, the math parts carry faded derivations
to attempt on paper, and the simulations ask you to predict outcomes before running them.</p>
<p>Read the parts in order. If you already know why gradient descent doesn&rsquo;t apply to
black boxes, start at Part&nbsp;2; if you want only the story without equations, Parts 1, 2 and
8 stand on their own. The interactive GP playground in Part&nbsp;2 is the heart of the page
&mdash; everything before it motivates it, everything after it formalizes it.</p>
""",
    "sections": [
        {"id": "background", "title": "Background", "html": BACKGROUND, "quiz": QUIZ_BACKGROUND},
        {"id": "intuition", "title": "Intuition", "html": INTUITION, "quiz": QUIZ_INTUITION},
        {"id": "math-gp", "title": "The Maths I: Gaussian Process Surrogates", "html": MATH_GP, "quiz": QUIZ_MATH_GP},
        {"id": "math-acq", "title": "The Maths II: Acquisition Functions", "html": MATH_ACQ, "quiz": QUIZ_MATH_ACQ},
        {"id": "math-sota", "title": "The Maths III: Inside the SOTA Methods", "html": MATH_SOTA, "quiz": QUIZ_MATH_SOTA},
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
    "__FIG_REGRET__": (PLOTS / "fig_regret.html").read_text(),
    "__FIG_LS_SAMPLES__": (PLOTS / "fig_ls_samples.html").read_text(),
    "__FIG_ACQ_ANATOMY__": (PLOTS / "fig_acq_anatomy.html").read_text(),
    "__FIG_TURBO__": (PLOTS / "fig_turbo.html").read_text(),
}
for s in spec["sections"]:
    for k, v in subs.items():
        s["html"] = s["html"].replace(k, v)

out = HERE / "spec.json"
out.write_text(json.dumps(spec, indent=1))
print(out)
