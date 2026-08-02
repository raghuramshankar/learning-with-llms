# Literature survey: Bayesian Optimization & its SOTA methods

Surveyed 2026-08-02 (arXiv API title-verification + recent-submission sweeps; web search for
the LLM-era thread). Raw material for the explainer build per the learning-new-topic skill.
Citation magnitudes are approximate (Semantic Scholar rate-limited; verify at build time if
exact counts are wanted).

## 1. Seminal papers (the origin story)

- Kushner (1964), *A New Method of Locating the Maximum Point of an Arbitrary Multipeak Curve
  in the Presence of Noise*, J. Basic Engineering — probability of improvement; the first
  "optimize a surrogate's belief" idea.
- Mockus (1974/1978), *On Bayesian Methods for Seeking the Extremum* — expected improvement,
  and the name "Bayesian optimization".
- **Jones, Schonlau, Welch (1998), *Efficient Global Optimization of Expensive Black-Box
  Functions*, J. Global Optimization** — EGO: kriging surrogate + EI, the paper that
  crystallized the modern loop (~8k citations). THE seminal reference for the page.
- Srinivas, Krause, Kakade, Seeger (2009), *GP Optimization in the Bandit Setting: No Regret
  and Experimental Design* — GP-UCB, first regret bounds. arXiv:0912.3995.
- **Snoek, Larochelle, Adams (2012), *Practical Bayesian Optimization of Machine Learning
  Algorithms*** — ignited the ML-hyperparameter era; Spearmint; (~10k citations).
  arXiv:1206.2944.

## 2. Most-cited / most important papers (the main line)

Surveys/tutorials (read first):
- Shahriari et al. (2016), *Taking the Human Out of the Loop: A Review of BO*, Proc. IEEE
  (~5k citations) — the canonical survey.
- Frazier (2018), *A Tutorial on Bayesian Optimization* — cleanest math walkthrough;
  arXiv:1807.02811. Math sections should follow ITS notation.

Acquisition functions:
- Bergstra et al. (2011), *Algorithms for Hyper-Parameter Optimization* (TPE; powers
  Hyperopt/Optuna; ~5k citations).
- Hernández-Lobato et al. (2014), Predictive Entropy Search, arXiv:1406.2541; Wang & Jegelka
  (2017), Max-value Entropy Search, arXiv:1703.01968; info-theoretic family.
- Frazier et al. (2008), Knowledge Gradient (SIAM J. Control Optim.).
- Ament et al. (2023), *Unexpected Improvements to Expected Improvement* (LogEI),
  arXiv:2310.20708 — fixed EI's numerical pathologies; the modern default in BoTorch.

Scaling & structure:
- Eriksson et al. (2019), TuRBO — trust-region local BO, arXiv:1910.01739.
- Eriksson & Jankowiak (2021), SAASBO — sparse axis-aligned subspaces, arXiv:2103.00349.
- Papenmeier et al. (2023), Bounce — combinatorial/mixed spaces, arXiv:2307.00618.
- Daulton et al. (2020), qEHVI — parallel multi-objective, arXiv:2006.05078.
- Astudillo & Frazier (2021), BO of Function Networks, arXiv:2112.15311.
- Hvarfner et al. (2022), πBO — user-belief priors in acquisition, arXiv:2204.11051.

Software/infrastructure:
- Balandat et al. (2020), BoTorch, arXiv:1910.06403 (+ Ax); GPyTorch (Gardner 2018);
  SMAC (Hutter 2011); Optuna (Akiba 2019, TPE-based); HEBO (Cowen-Rivers 2020, NeurIPS
  2020 black-box comp winner; "Modernizing HEBO" arXiv:2607.10669, Jul 2026).

Killer application (for motivation):
- Shields et al. (2021), *Bayesian reaction optimization as a tool for chemical synthesis*,
  Nature — BO beats expert chemists; the self-driving-lab thread.

## 3. Books

- **Garnett (2023), *Bayesian Optimization*, Cambridge UP** — THE textbook; free at
  bayesoptbook.com (verified live). GP modeling → decision theory → policies.
- Rasmussen & Williams (2006), *Gaussian Processes for ML*, MIT Press — the surrogate's
  bible; free at gaussianprocess.org/gpml.
- Gramacy (2020), *Surrogates*, CRC — GP modeling + design + optimization, R-flavored,
  free at bobby.gramacy.com/surrogates.

## 4. Current SOTA (2024 → mid-2026)

High-dimensional BO — an active controversy (honesty-box gold):
- Hvarfner et al. (2024), *Vanilla BO Performs Great in High Dimensions*, arXiv:2402.02229.
- Xu & Zhe (ICLR 2025), *Standard GP is All You Need for High-Dim BO*, arXiv:2402.02746.
- **Papenmeier et al. (Nov 2025), *We Still Don't Understand High-Dimensional Bayesian
  Optimization*, arXiv:2512.00170** — pushback; the debate is live.
- Also: GIT-BO (tabular foundation models for high-dim, arXiv:2505.20685), automated kernel
  discovery (arXiv:2605.20249, May 2026), Lasso variable selection (2504.01743).

Pretrained/amortized surrogates ("BO meets foundation models"):
- OptFormer (Chen et al. 2022), arXiv:2205.13320 — transformer hyperparameter optimizer.
- PFNs4BO (Müller et al. 2023), arXiv:2305.17535 — prior-fitted networks: BO surrogate as a
  single transformer forward pass; TabPFN-v2 lineage continues (GIT-BO above).

LLM-era BO (2024–2026):
- LLAMBO (Liu et al., ICLR 2024), arXiv:2402.03921 — LLM as warm-starter/surrogate/sampler.
- Evidence-Gated LLM Priors for MOBO (Jun 2026), arXiv:2606.01730.
- LLM-Driven Evolutionary Generation of MOBO Algorithms (Jul 2026), arXiv:2607.08791.
- Skeptical/benchmark counterweight: *LLMs for BO in Scientific Domains: Are We There Yet?*
  arXiv:2509.21403; *When Is an LLM Worth It for HPO? A Budget-Matched Study* (Jun 2026),
  arXiv:2606.21641.

Other active threads worth one line each: information-theoretic survey+tutorial
(arXiv:2502.06789, 2025); multi-fidelity review (arXiv:2311.13050); heteroscedastic/risk-aware
BO for AutoRL (arXiv:2607.26680, Jul 2026); safe BO (2607.05620); "How Many Initial Points
Does BO Need?" (arXiv:2607.04356, Jul 2026); applications everywhere in the Jul-2026 sweep
(photonic lasers, clinical trial design, tumor models, 6G antennas, agricultural IoT).

## Proposed page structure (for the build)

1. Background — expensive black-box functions; why grid/random search wastes evaluations;
   the puzzle: explore vs exploit with a budget of ~50 evaluations.
2. Intuition — surrogate + acquisition loop; uncertainty as the resource; toy 1-D GP.
3. Maths I: Gaussian process surrogates — priors over functions, posterior closed form,
   kernels, hyperparameters (Frazier/Garnett notation).
4. Maths II: Acquisition functions — EI derivation (closed form!), PI, UCB with regret
   flavor, one info-theoretic (MES), LogEI numerics, batch/parallel via MC.
5. The papers — foundations (Kushner→Mockus→EGO→GP-UCB→Snoek) then the modern line
   (TuRBO/SAASBO/LogEI/PFNs4BO/LLAMBO + the high-dim controversy).
6. Concept map, Keep Learning, Sources — per skill.

Widget ideas: live 1-D GP posterior scrubber (click to add observations); acquisition-function
race (EI vs UCB vs random on a hidden function, regret curves); interactive EI computed from
posterior mean/σ sliders; BO-vs-random Monte Carlo tally; DDIM-lab-equivalent: full BO loop on
a 2-D test function with selectable acquisition (exact GP math in JS is feasible for small n);
Plotly: regret-vs-iterations precomputed in numpy, kernel-family explorer, high-dim
controversy figure. Tutorial: implement GP posterior + EI + the loop in numpy against pytest
(closed forms make exact tests easy). Cheat sheet + Anki + review deck as usual.
