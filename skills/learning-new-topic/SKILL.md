---
name: learning-new-topic
description: "Use when the user wants to learn or deeply understand a new topic — a concept, paper, technology, or research area. Produces a rich, self-contained interactive HTML explainer with intuition, math deep dives, interactive widgets, and hard per-section quizzes."
---

# Learning New Topic

Teach the specified topic in the most intuitive and rigorous way possible, as a single
self-contained interactive HTML page.

## Step 1: Ground it in real sources

**If the user points at a source** (a URL, a company page, a paper list, a repo, a PDF): fetch it
first and extract the concrete anchors: the actual papers, links, names, claims. Get exact
titles, authors, years, and working links (pull hrefs from the page DOM, don't guess URLs). If the
sources have a hidden narrative (e.g. "these nine papers are the founders' research trail"), find
it and make it the spine of the page.

**If the user only names a topic**: run a literature survey FIRST — it is the raw material for the
whole page. Do not write from memory alone. **Primary tool: the arXiv API**
(`http://export.arxiv.org/api/query`): verify every paper you plan to cite with a
`ti:"<exact title>"` query (returns the exact title, arXiv id, submission date, first author);
find the current SOTA with `sortBy=submittedDate&sortOrder=descending` sweeps on
`ti:"<topic>"` and thematic sub-queries; find recent surveys with `ti:<topic> AND (ti:survey OR
ti:review OR ti:tutorial)`. Respect the API etiquette (~1 request / 3 s — batch queries in one
python script with time.sleep, not shell sleep). arXiv has no citation counts, so state citation
magnitudes as approximate (~) from reputation, or omit them; classics that predate arXiv
(pre-~1992 or journal-only) are cited by venue directly. WebSearch/WebFetch supplement for
non-arXiv sources (books, Nature/Science papers, software docs). Recent survey papers are gold —
they've done the curation for you. Assemble four lists:

1. **The seminal paper(s)** — the work that founded the topic (and its key precursor if the
   origin story needs it). Read it, or at minimum its abstract, intro, and main theorem/figure:
   the Background section's tension/puzzle usually IS this paper's motivation.
2. **The most-cited / most important papers** — the 5–10 works with the highest citations or
   clearest influence that define the topic's main line. Verify citation standing via a scholarly
   index, not vibes. These become the Source Walkthrough's narrative arc (each = which obstacle
   it removed) and the anchors for the math sections.
3. **The most important books** — the 1–3 standard textbooks or monographs practitioners actually
   recommend (check syllabi, course pages, "further reading" of surveys). These go in Keep
   Learning as the going-deeper path, each with one line on when to reach for it.
4. **The current SOTA** — what is state of the art RIGHT NOW: latest benchmark leaders, the
   strongest recent papers (last ~12 months), active leaderboards, open problems. Search with the
   current year in the query; knowledge cutoffs make this the part memory gets wrong. SOTA feeds
   the honesty box (what's settled vs an open bet) and dates the page's claims.

Actually read/understand the top items (fetch the papers or their pages; skim structure, punchline
results, and notation) — the explainer's math sections should use the seminal/most-cited papers'
own notation so the user can graduate to the sources. Every item surveyed gets exact title,
authors, year, and a verified link; the full list lands in the Source Walkthrough's reference
table, ordered for reading. If the survey changes your mental model of the topic (it often will),
restructure the page around what the literature says, not around what you assumed.

## Step 2: Structure

Sections, in this order (adapt names to the topic):

1. **Background** — deep background for complete beginners (explicitly marked skippable), then the
   narrow background the topic needs. End with the tension/puzzle the topic resolves, as a callout.
2. **Intuition** — the core idea via toy examples with small concrete data, using a few reusable
   diagram families. Name every key trade-off precisely *in words* here (e.g. "parallel commits
   are independent commits"), so the math sections can formalize rather than surprise. Include an
   "honesty box" callout with the topic's real caveats and open questions.
3. **Maths deep dives** — one or more sections, as technical as needed; never dumb it down. Real
   derivations with display math, notation defined at first use, each key formula followed by a
   "why this matters" callout. Bold the punchline sentence of each derivation. Carry ONE running
   example/notation through all math sections. **The LAST maths section must cover the SOTA
   methods from the literature survey** — the actual equations each modern method changes (its
   prior, update rule, search region, or training objective), one h3 per method, cross-referenced
   from the papers narrative; not just prose about what the papers claim.
4. **Concept map** — a final interactive SVG map of every idea on the page: hover a node to
   highlight its connections (dim the rest), click for a one-breath recap + jump link to its
   section. Mark the 2–3 load-bearing hub nodes. Build it from a nodes/edges data array in the
   widgets JS (`.cmap-*` classes in the renderer); the self-test framing: "every edge is one
   sentence you should be able to produce."
5. **Keep Learning** — closes the loop past reading (see "Beyond the page" below): links to the
   review deck, Anki deck, and cheat sheet; the copyable teach-back prompt; the tutorial instructions.
6. **Papers & Sources** — ONE combined section, the LAST part of the topic. Never create two
   separate citing sections (no standalone "Sources"): the narrative arc (foundations → modern
   toolbox → live SOTA threads; each entry = which obstacle it removes, cross-referencing the
   maths sections that derive its formulas) followed, in the same section, by THE COMPLETE
   REFERENCE LIST: every work discussed plus the supporting literature behind background claims
   INCLUDING the real papers used as quiz distractors — complete author lists, year, venue (only
   when confident — else "arXiv preprint"), verified links, and one italic purpose note each for
   the supporting entries. Open the section with the survey-methodology note (how papers were
   verified, access date, citation-magnitude caveat, whose notation the maths follows, "errors of
   interpretation are this page's"). Close with a suggested reading order.
7. **Quiz after every section** — 5 questions each (rules below). The concept-map and
   keep-learning sections don't need quizzes; the combined Papers & Sources section DOES get one.

## Quiz rules (learned from user feedback — do not regress)

- 5 questions per section, placed at the end of that section, each block scored separately.
- Hard, not trivia: computation questions (plug in numbers), formula-recognition with
  coefficient-swapped decoys, and distractors that are *real adjacent concepts* (a different actual
  paper/method/assumption), so wrong answers are instructive.
- **Balance option lengths.** The correct option must NOT be wordier or more hedged than the
  distractors — the user can guess by length. Put nuance in the explanation, not the option.
  Occasionally make the correct answer the shortest.
- Every option (right and wrong) gets an explanation; a wrong option's explanation says what that
  option actually describes and why it's not this.
- After the user answers, the UI reveals the explanations for ALL options — not just the one they
  clicked (the shared renderer's quiz script does this; don't regress it). Write each explanation
  to stand alone under its option.
- The renderer also persists results to localStorage and adds a "copy results for Claude" button
  after each completed section — the copied summary lists score, missed questions, and repeat
  offenders, ending with a request to re-quiz weak areas. When a user pastes one in, honor it:
  re-teach the missed concepts from first principles, then quiz harder on exactly those.

## Step 3: Interactivity (widgets and animations)

Add interactive widgets wherever they genuinely teach. Proven widget types:

- **Scrubber sliders** for continuous processes (noise level, corruption ratio) — precompute one
  seeded random draw so scrubbing is repeatable, and show live values (α, expected vs actual).
- **Play/step race animations** for sequential-vs-parallel claims — two rows, tick = one unit of
  cost for both, visible pass counters.
- **Monte-Carlo demos** for statistical claims — draw-1 / draw-50 buttons, chips for outcomes,
  running tally that converges to the theoretical number stated in prose.
- **Real algorithm demos** — where the math permits, implement the *actual* algorithm on a toy
  problem (e.g. a real DDIM sampler on a 2-D Gaussian mixture whose exact score is closed-form,
  standing in for a perfectly-trained network). Prefer honest simulations; when a visualization is
  NOT the real algorithm (e.g. scrubbing a blend backwards instead of denoising), say so in the
  caption ("honesty note").
- **Formula calculators** — sliders for a formula's inputs, live bar/number output, with the
  formula printed and evaluated numerically. Tie one to a quiz question ("this is the quantity the
  quiz asks about").
- **Parameter-family plots** — plot with buttons switching between family members
  (schedules, regimes), showing what changes and what is invariant.

Choose the right tool per visual:

- **Python + Plotly (preferred for quantitative plots and precomputed animations).** Generate with
  a `make_plots.py` using numpy for the actual computation (run the real algorithm, sweep the real
  parameter) and plotly.graph_objects for the figure. Good fits: scaling-law line charts with
  hover, precomputed sweeps as Plotly frames with a slider + play button (e.g. sampler output vs
  step count), quality-vs-parameter curves computed from the simulation, heatmaps of a formula's
  full landscape, curve families switched via updatemenus buttons. Embed via
  `plotly.offline.plot(fig, include_plotlyjs=False, output_type='div')` snippets spliced into
  section html, and put `plotly.offline.get_plotlyjs()` ONCE in the spec's `head_scripts` so the
  library loads before the figure divs (keeps the page offline-capable; never use the CDN).
  Theme-neutral styling: transparent paper/plot backgrounds, mid-gray font (#8b95a5), translucent
  gridlines, `displayModeBar: False`, `responsive: True`.
- **Hand-written canvas/DOM JS (for simulations tightly coupled to page state).** Live races,
  Monte-Carlo tally demos, scrubbing a seeded corruption process over DOM token boxes, demos whose
  controls change an algorithm that then animates step by step. Container HTML lives in the
  section `html` with unique ids; ALL JS goes in one self-contained IIFE passed via the spec's
  `scripts` field. No external libraries — the page must work offline.
- Use a seeded PRNG (mulberry32 + Box-Muller) for anything random that should be reproducible.
- Read colors from the CSS variables at draw time (`getComputedStyle`) and register redraws on
  `prefers-color-scheme` change, so canvases follow light/dark theme.
- Each widget guards on `if (document.getElementById('w-...'))` so the script is reusable.

## Step 3.5: Beyond the page (retention, generation, transfer)

A page teaches once; these artifacts make it stick. Build ALL of them for a full explainer:

**Retention**
- **Spaced-repetition review deck** (`<date>-<topic>-review.html`, via a `build_review.py` that
  reads the quiz bank out of spec.json): a standalone Leitner-box app — intervals 1/3/7/14/30
  days by box, state in localStorage, correct-first-try promotes a box, a miss demotes to box 0
  AND requeues the card later in the same session, all options' explanations revealed on answer,
  end-of-session "copy results for Claude" summary (first-try score, misses, chronic weak spots
  with 2+ lapses), and a reset button. Link it from Keep Learning.
- **Anki deck** (`make_anki.py` using the `genanki` package, pip-installable): one note per quiz
  question — front = question + lettered options, back = correct letter + every option's ✓/✗
  explanation. Ship the `.apkg` next to the pages.

**Generation**
- **Teach-it-back prompt**: a copyable prompt block (callout + `data-copy` button) that makes a
  fresh Claude session play a curious student: it asks the user to explain the 4–6 core ideas one
  at a time, probes each with a "why" follow-up, never explains unless the user is stuck twice
  (then hints), and ends by grading mechanism/formulas/caveats, listing weak spots, and writing 3
  new questions targeting them. When YOU receive this prompt, play that student faithfully.
- **Faded derivations** (renderer `.deriv`/`.dstep` component): the 1–2 key derivations as
  numbered steps where each step's GOAL is visible but its content is hidden behind a reveal
  button; "practice (hide all)" / "worked (show all)" controls. Steps start hidden — the user
  should attempt each on paper first. Place them at the end of the math subsection they drill.
- **Predict-before-run**: the flagship simulation widgets ask for a prediction (3 options)
  BEFORE the first run; lock the pick, then resolve with feedback when the relevant run/tally
  completes. Committing to a prediction before seeing the outcome is what makes a demo stick.

**Transfer**
- **Build-it-yourself tutorial** (`tutorials/<topic>/` in the project): a small skeleton module (numpy or
  the domain's tool) implementing the page's core formulas, with every body replaced by
  `raise NotImplementedError` + a docstring hint referencing the section that derives it; a pytest
  suite that encodes the right answers (exact values from the quiz, statistical checks for
  samplers, an integration test proving the whole pipeline works — e.g. "oracle model + your
  sampler reconstructs the input"); a `solutions/` reference; a README with a suggested
  easiest→hardest order. VERIFY both states: tests all fail on the stubs, all pass on the
  solution.

**Consolidation**
- **One-page cheat sheet** (`<date>-<topic>-cheatsheet.html`, rendered via the same render.py
  using `.grid2`/`.cheat-card`): every key formula with its one-sentence punchline, plus the
  source links. Printable, no widgets, no plotly.

Cross-link everything from the Keep Learning section, and keep filenames stable: pin the spec's
`date` field on rebuilds so links to the main page never break.

## Repo layout (the learning-with-llms repo)

The user's explainers live in a git repo (github.com/raghuramshankar/learning-with-llms, local:
~/Developer/learning-using-claude). Work inside it:

- `tools/` — shared infra: `render.py` (CANONICAL here; `~/.claude/tools/explain-diff/render.py`
  is a symlink to it — improve it in place, never fork), `plot_style.py` (Plotly theme helpers:
  style/save_div/write_plotly_lib + colors), `widgets_lib.js` (window.WLib: $, cssVar,
  mulberry32, gauss, onTheme — pass it as the FIRST entry in the spec's `scripts`).
- `topics/<slug>/` — one folder per topic holding ALL sources: build_spec.py, widgets.js,
  make_plots.py, build_review.py, make_anki.py, make_cheatsheet.py, plots/ (generated figure
  snippets, committed), and a `build.sh` that rebuilds everything into docs/. NEVER leave these
  only in a scratchpad — they are the reproducibility story.
- `docs/` — rendered outputs. `plotly.min.js` is committed ONCE here; pages reference it via the
  spec's `head_script_srcs: ["plotly.min.js"]` (default, ~100KB pages). `--inline` on build_spec
  embeds it instead for a portable single-file export. Pin the spec's `date` so filenames and
  cross-links stay stable across rebuilds.
- `tutorials/<topic>/` — the build-it-yourself tutorials.
- `skills/` — versioned copies of the skills; after editing the live skill in `~/.claude/skills/`,
  re-copy it here so the repo tracks it.
- `site/` — the project website: `topics.json` (catalog manifest) + `build_site.py` →
  `docs/index.html`. After finishing a topic, ADD ITS ENTRY to topics.json (slug, title, blurb,
  date, tags, links to explainer/review/cheatsheet/anki/tutorial) and run `python3 site/build_site.py`
  so it appears on the homepage.
- The site is live via GitHub Pages (main branch, /docs folder) at
  https://raghuramshankar.github.io/learning-with-llms/ — a push deploys it.
- Commit when a topic build is done (the user has authorized commits to this repo); outputs in
  docs/ are committed alongside sources.

## Step 4: Rendering

- **Use the shared renderer `~/.claude/tools/explain-diff/render.py`** (shared with the
  explain-diff-html skill — improve it in place rather than forking). It handles CSS/JS
  scaffolding, TOC, per-section quizzes (`quiz` on a section), display math (`.math`, `.m`,
  `.frac` with `.num`/`.den`), diagram classes (`.diagram`/`.flow`/`.box ok|fail|dim|accent`),
  widget chrome (`.widget`/`.wctl`/`.wbtn`/`.wstat`/`.wchips`), and a `scripts` field for raw JS.
  Run `python render.py --help` for the exact schema.
- **Multipage is the default for full explainers** (`"multipage": true` in the spec, plus an
  `"intro"` html block for the overview page and a `"nav"` list for the masthead). The renderer
  splits every section at its `<h3>` subtopics: overview page (un-suffixed filename, so older
  links keep working) → part landing pages (section intro + subtopic list) → ONE PAGE PER h3
  SUBTOPIC → a quiz page per part. Every page gets a top nav row (← previous · ⌂ topic overview ·
  next →), bottom prev/next, and a sticky sidebar listing all parts with the current part's
  subtopic pages nested and highlighted. Reading order is linear across all pages. Cross-section
  `href="#<sid>"` anchors are rewritten to part-landing links, and `window.__PAGES__` =
  {sectionId: href} is defined on every page — widget JS that links to sections (e.g. the concept
  map) must resolve targets through it. Because sections split at `<h3>`, keep h3s top-level in
  section html (never inside an unclosed div) and keep each widget + its predict panel within one
  h3 chunk. Single-page mode remains for the cheat sheet.
- **Theme**: academic style modeled on adaptive-bayesian.ai at the user's request — Merriweather
  serif (Google Fonts link with Georgia fallback), deep-blue `#063c92` section headings over a
  thin blue rule, green `#168551` h3s/eyebrows, navy underlined links, light-gray masthead bar.
  Light theme is PURE `#ffffff` background with `#000000` text; dark theme background is PURE `#000000` (user requirements). Every page has
  a masthead light/dark toggle: persists to localStorage('theme') via a `data-theme` attribute on
  <html> (set in <head> before paint to avoid flashing), overrides the `prefers-color-scheme`
  default, and dispatches a 'themechange' event that widgets_lib listens to so canvases repaint.
  The review deck and site homepage carry the same palette + toggle — keep all three in sync.
- **Site chrome conventions** (user-set; apply to every new topic):
  - Topic-page masthead: brand is "← <Topic Title>" linking back to index.html (all topics);
    nav is exactly Review deck / Cheat sheet / Tutorials (+ the theme toggle). No "All topics"
    or GitHub links in the masthead. The cheat sheet swaps in an "Explainer" link for its own.
  - The REVIEW DECK carries the full site chrome, not just the palette: same masthead, the
    generator footer, and 950px content width. Its build_review.py stamps the footer timestamp
    itself (UTC).
  - The site homepage opens directly with the Topics list — no hero, no methodology section, no
    under-the-hood section. A topic card's title links to the explainer; its link row is exactly
    Review deck / Cheat sheet / Tutorials (no explainer duplicate, no Anki link — the Anki deck
    is still built and linked from the topic's Keep Learning section).
  - Footer everywhere: "Generated by the <skill link> skill using <model> on
    YYYY-MM-DD HH:MM:SS UTC".
- Write a `build_spec.py` that constructs the JSON spec in Python (triple-quoted HTML strings stay
  readable; splice repeated snippets like token rows and fractions via placeholder substitution)
  and reads the widget JS from a sibling `widgets.js`.
- Math typesetting is HTML entities + `<sub>/<sup>` + `.frac` — check codepoints carefully
  (ᾱ is `&#8113;`, 𝒩 is `&#119977;`, 𝔼 is `&#120124;`; `&#7745;` is ṁ, a classic wrong guess).
  Verify no stray characters render by grepping the output.
- **Footer attribution**: set the spec's `generator` field on every rendered spec (explainer AND
  cheat sheet): `{"skill": "learning-new-topic", "skill_url": <link to this SKILL.md on GitHub>,
  "model": <the ACTUAL model building the page, e.g. "Claude Fable 5">}`. The renderer emits
  "Generated by the <skill> skill using <model> on <YYYY-MM-DD HH:MM:SS UTC>". Fill in the real
  current model name — don't copy a stale one from an earlier topic.

## Step 5: Writing style

- Kleppmann-ish classic style: concrete before abstract, one idea per paragraph, smooth transitions
  that state what the next section owes the reader. Engaging, never breathless.
- The intuition section promises; the math sections pay. Cross-reference them both ways.
- Be honest about what's an open bet vs established fact.

## Step 6: Verify, store, deliver

- After rendering, copy the HTML into the `docs/` folder of the current project (create it if
  needed); keep the date-prefixed filename.
- Verify in the browser BEFORE delivering: counts of sections/quiz cards/widgets via JS; click a
  correct and a wrong quiz option; drive each widget programmatically (dispatch `input` events,
  click Run) and assert on its status text and canvas pixel variety; check a Monte-Carlo widget
  converges to the theoretical value. If the browser pane is hidden, DOM checks via JS still work.
- Deliver with SendUserFile (`display: render`) and summarize what's on the page and what changed.
