# Tutorial: Bayesian optimization, from scratch

You read the explainer. Now write it. Six numpy functions — the RBF kernel,
the GP posterior, closed-form expected improvement, UCB, the acquisition step,
and the full BO loop — with a test suite that knows the right answers,
including the exact quiz numbers (EI at z=0 is 0.399σ; the one-observation
posterior) and an integration test where your loop must find a hidden peak and
beat random search at an equal budget.

## How to work

```bash
cd tutorials/bayesian-optimization
python3 -m pytest -q          # red: 13 failing tests
```

Open `bayes_opt.py` and replace each `raise NotImplementedError`, re-running
tests as you go. Suggested order (easiest → hardest):

1. `rbf_kernel` — one vectorized line
2. `ucb` — one line
3. `expected_improvement` — the closed form you derived (math.erf gives Φ)
4. `gp_posterior` — the two conditioning formulas
5. `propose_next` — argmax of EI over candidates
6. `run_bo` — glue the loop together

When the suite is green, your own sampler has located a hidden optimum in 12
evaluations and out-searched random sampling — i.e. you have implemented
working Bayesian optimization.

## Rules of engagement

- The tests are the spec: read a failing test before asking what a function
  should do.
- `solutions/bayes_opt.py` exists. Peeking at **one function** after a real
  attempt is studying; reading the file is not.
- Everything needed is in the explainer:
  `docs/2026-08-02-bayesian-optimization.html` (Maths I and Maths II).
