# learning-with-llms

Interactive HTML explainers for learning new topics deeply — built with Claude Code and the
`learning-new-topic` skill. Each topic produces a full learning system: an explainer page
(intuition → math deep dives → source walkthrough, with interactive widgets, Plotly figures,
faded derivations, per-section quizzes, and a concept map), a spaced-repetition review deck,
an Anki export, a one-page cheat sheet, and a build-it-yourself lab with tests.

## Layout

```
tools/      shared infrastructure
              render.py       spec-JSON → HTML renderer (quizzes, math CSS, widgets, TOC)
              plot_style.py   theme-neutral Plotly styling helpers
              widgets_lib.js  shared widget JS (seeded PRNG, theme redraw, ...)
topics/     one folder per topic: the SOURCES that generate the pages
              <slug>/build_spec.py    content (sections, quizzes, widget HTML)
              <slug>/widgets.js       topic-specific interactive widgets
              <slug>/make_plots.py    numpy + Plotly figure generation → plots/
              <slug>/build_review.py  spaced-repetition deck generator
              <slug>/make_anki.py     Anki .apkg export
              <slug>/make_cheatsheet.py
              <slug>/build.sh         rebuild everything into docs/
docs/       rendered OUTPUTS (open in a browser; GitHub Pages-ready)
              plotly.min.js   committed once, referenced by every page
labs/       build-it-yourself labs (stubs + pytest suites + solutions/)
skills/     versioned copies of the Claude Code skills driving this workflow
```

## Building a topic

```bash
cd topics/diffusion-language-models
./build.sh              # pages share docs/plotly.min.js (small pages)
./build.sh --inline     # embed plotly into the page (portable single file)
```

## Notes

- The canonical `render.py` lives here; `~/.claude/tools/explain-diff/render.py` is a symlink to
  it so the `explain-diff-html` and `learning-new-topic` skills keep working from any project.
  After cloning on a new machine: `ln -sf "$PWD/tools/render.py" ~/.claude/tools/explain-diff/render.py`.
- `skills/` holds versioned copies; the live skills are in `~/.claude/skills/`.
- Quiz scores and review-deck progress live in browser localStorage, keyed per origin — moving
  from `file://` to hosted pages resets progress once.

## Topics

- **diffusion-language-models** — diffusion models and diffusion LLMs, grounded in the nine
  papers on Inception Labs' about page. Start at
  `docs/2026-08-01-diffusion-language-models.html`.
