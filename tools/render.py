#!/usr/bin/env python3
"""Render an interactive "explain this diff" HTML page from a JSON content spec.

Usage:
    python render.py <spec.json> [-o OUTDIR]

Writes <OUTDIR>/<date>-<slug>.html (OUTDIR defaults to the spec's directory)
and prints the output path.

JSON spec schema
----------------
{
  "title":    "Page title",                  (required)
  "subtitle": "One-line subtitle",           (optional)
  "slug":     "kebab-case-name",             (required; used in the filename)
  "date":     "YYYY-MM-DD",                  (optional; defaults to today)
  "repo":     "project name / branch note",  (optional; shown under subtitle)
  "sections": [                              (required)
    { "id":    "background",                 (optional; derived from title)
      "title": "Background",
      "html":  "<p>raw HTML body...</p>",
      "quiz":  [ ...questions... ] }         (optional; rendered as a quiz
  ],                                          block at the end of the section)
  "quiz": [                                  (optional; rendered as its own
    { "question": "What ... ?",               final section "Quiz". Question
      "options": [                            and option text may contain
        { "text": "An answer",                inline HTML, e.g. <sub>/<sup>.)
          "correct": true,                   (exactly one per question)
          "explanation": "Shown after the user answers." }
      ] }
  ],
  "scripts": [ "raw JS ..." ],               (optional; each string is emitted
                                              in a <script> tag at the end of
                                              <body>, after the quiz script —
                                              use for interactive widgets whose
                                              container HTML lives in sections)
  "head_scripts": [ "raw JS ..." ],          (optional; emitted in <head>, so
                                              libraries load before section
                                              content runs — e.g. an inlined
                                              plotly.min.js for figures whose
                                              divs+scripts sit in section html)
  "head_script_srcs": [ "plotly.min.js" ]    (optional; emitted in <head> as
}                                             <script src=...> references —
                                              point at a shared library file
                                              committed next to the output
                                              pages instead of inlining
                                              megabytes into every page; the
                                              page then needs that file beside
                                              it and is no longer single-file)

HTML vocabulary available inside section "html" bodies
------------------------------------------------------
  <pre>...</pre>                 code block (white-space: pre-wrap, scrolls)
  <div class="diagram">          centred figure container with padding
    <div class="flow">           horizontal flex row (wraps on small screens)
      <div class="box">A</div>   node; variants: box ok / box fail / box dim
      <span class="arr">→</span> arrow between boxes
      <div class="box fail">B</div>
    </div>
    <div class="flow vertical">  column variant; use <span class="arr">↓</span>
    <div class="caption">...</div>
  </div>
  <div class="callout">...</div>           key definition / edge case
  <div class="callout warn">...</div>      warning variant
  <div class="math">...</div>              display math (serif, centred);
                                           write with entities, <sub>, <sup>
  <span class="m">...</span>               inline math (serif, no-wrap)
  <span class="frac"><span class="num">a</span><span class="den">b</span></span>
                                           stacked fraction a/b
  <table>...</table>                       styled automatically
  <span class="tag">label</span>           small inline pill
  <div class="formbox"><span class="boxno">24</span> ... </div>
                                           a mock tax-form / UI field row
  <div class="widget">...</div>            interactive-widget container; put
    <div class="wctl">                     controls in a wctl row:
      <button class="wbtn">Play</button>   (wbtn.active for selected state)
      <input type="range" ...>
    <canvas ...>                           canvases centre themselves
    <div class="wstat">...</div>           monospace status/counter line
    <div class="wchips">                   sample-outcome chips:
      <span class="wchip good">ok</span>   good = green, bad = red
Everything else (h3, p, ul, code, strong, em) is styled sensibly. Quizzes
shuffle option order per page load and give per-option feedback on click;
each quiz block shows its own score once fully answered.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import json
import sys
from pathlib import Path

CSS = """
:root {
  --bg: #ffffff; --fg: #1a1f27; --muted: #5b6472; --line: #e3e7ee;
  --accent: #2458c5; --accent-soft: #eaf0fc;
  --ok: #1e7e46; --ok-soft: #e8f5ec; --fail: #b33a3a; --fail-soft: #faecec;
  --warn-soft: #fdf4e3; --warn-line: #e2b458;
  --code-bg: #f6f8fa; --box-bg: #f9fafc;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #14171c; --fg: #e6e9ee; --muted: #9aa3b0; --line: #2b313b;
    --accent: #7aa5f0; --accent-soft: #1d2942;
    --ok: #62c088; --ok-soft: #17301f; --fail: #e08585; --fail-soft: #3a2020;
    --warn-soft: #33290f; --warn-line: #a8842e;
    --code-bg: #1c2129; --box-bg: #1a1f27;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 17px/1.65 -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
        "Helvetica Neue", Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
}
.wrap { max-width: 880px; margin: 0 auto; padding: 2.5rem 1.4rem 5rem; }
header.masthead { border-bottom: 1px solid var(--line); padding-bottom: 1.4rem;
  margin-bottom: 1.6rem; }
h1 { font-size: 1.9rem; line-height: 1.25; margin: 0 0 .4rem; }
.subtitle { color: var(--muted); font-size: 1.08rem; margin: 0; }
.repo { color: var(--muted); font-size: .85rem; margin-top: .6rem; }
nav.toc { background: var(--box-bg); border: 1px solid var(--line);
  border-radius: 10px; padding: 1rem 1.3rem; margin-bottom: 2.2rem; }
nav.toc .toc-title { font-weight: 600; font-size: .8rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
nav.toc ol { margin: 0; padding-left: 1.2rem; }
nav.toc a { color: var(--accent); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
section { margin-bottom: 3rem; }
h2 { font-size: 1.45rem; border-bottom: 1px solid var(--line);
  padding-bottom: .35rem; margin: 2.4rem 0 1rem; }
h3 { font-size: 1.12rem; margin: 1.8rem 0 .5rem; }
h4 { font-size: 1rem; margin: 1.4rem 0 .4rem; }
p, ul, ol { margin: 0 0 1rem; }
li { margin-bottom: .35rem; }
a { color: var(--accent); }
code { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 4px; padding: .08em .35em; font-size: .86em;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 8px; padding: .9rem 1.1rem; overflow-x: auto;
  white-space: pre-wrap; font-size: .84rem; line-height: 1.5;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre code { background: none; border: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; margin: 1rem 0; width: 100%;
  font-size: .92rem; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: .45rem .7rem;
  text-align: left; vertical-align: top; }
th { background: var(--box-bg); font-weight: 600; }
.callout { background: var(--accent-soft); border-left: 4px solid var(--accent);
  border-radius: 0 8px 8px 0; padding: .8rem 1.1rem; margin: 1.2rem 0; }
.callout.warn { background: var(--warn-soft); border-left-color: var(--warn-line); }
.callout p:last-child, .callout ul:last-child { margin-bottom: 0; }
.diagram { background: var(--box-bg); border: 1px solid var(--line);
  border-radius: 10px; padding: 1.3rem 1.1rem; margin: 1.4rem 0;
  overflow-x: auto; }
.flow { display: flex; align-items: center; justify-content: center;
  gap: .55rem; flex-wrap: wrap; margin: .4rem 0; }
.flow.vertical { flex-direction: column; }
.flow.left { justify-content: flex-start; }
.box { background: var(--bg); border: 1.5px solid var(--fg); border-radius: 8px;
  padding: .45rem .85rem; font-size: .88rem; text-align: center;
  line-height: 1.4; }
.box small { display: block; color: var(--muted); font-size: .78rem; }
.box.ok { border-color: var(--ok); background: var(--ok-soft); }
.box.fail { border-color: var(--fail); background: var(--fail-soft); }
.box.dim { border-color: var(--line); color: var(--muted); }
.box.accent { border-color: var(--accent); background: var(--accent-soft); }
.arr { color: var(--muted); font-size: 1.1rem; flex-shrink: 0; }
.caption { text-align: center; color: var(--muted); font-size: .84rem;
  margin-top: .8rem; }
.tag { display: inline-block; background: var(--accent-soft);
  color: var(--accent); border-radius: 999px; padding: .05em .6em;
  font-size: .78em; font-weight: 600; }
.formbox { display: flex; align-items: baseline; gap: .8rem;
  border: 1px solid var(--line); border-radius: 6px; background: var(--bg);
  padding: .5rem .8rem; margin: .4rem 0; font-size: .92rem; }
.formbox .boxno { background: var(--fg); color: var(--bg); border-radius: 4px;
  padding: .05em .5em; font-size: .8em; font-weight: 700; flex-shrink: 0;
  font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }
.formbox .val { margin-left: auto; font-family: ui-monospace, Menlo, monospace;
  font-weight: 600; white-space: nowrap; }
/* math */
.math { font-family: Georgia, 'Times New Roman', serif; text-align: center;
  margin: 1.2rem 0; padding: .2rem 0; overflow-x: auto; font-size: 1.05rem;
  line-height: 2.1; }
.m { font-family: Georgia, 'Times New Roman', serif; white-space: nowrap; }
.frac { display: inline-block; vertical-align: middle; text-align: center;
  margin: 0 .15em; line-height: 1.25; }
.frac .num, .frac .den { display: block; padding: 0 .3em; }
.frac .den { border-top: 1px solid currentColor; }
/* interactive widgets */
.widget { background: var(--box-bg); border: 1px solid var(--line);
  border-radius: 10px; padding: 1.1rem; margin: 1.4rem 0; }
.widget canvas { display: block; margin: .6rem auto; max-width: 100%;
  border-radius: 8px; }
.wctl { display: flex; align-items: center; justify-content: center;
  gap: .7rem; flex-wrap: wrap; margin: .5rem 0; }
.wctl label { font-size: .88rem; color: var(--muted); }
.wctl input[type=range] { width: min(240px, 55vw); accent-color: var(--accent); }
.wbtn { font: inherit; font-size: .88rem; padding: .35rem .9rem;
  border-radius: 8px; border: 1px solid var(--accent);
  background: var(--accent-soft); color: var(--accent); cursor: pointer;
  font-weight: 600; }
.wbtn:hover { filter: brightness(1.08); }
.wbtn.active { background: var(--accent); color: var(--bg); }
.wstat { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .85rem; color: var(--muted); text-align: center; margin-top: .5rem; }
.wchips { display: flex; flex-wrap: wrap; gap: .3rem; justify-content: center;
  margin-top: .5rem; max-height: 130px; overflow-y: auto; }
.wchip { font-size: .78rem; border: 1px solid var(--line); border-radius: 999px;
  padding: .05em .55em; }
.wchip.good { border-color: var(--ok); background: var(--ok-soft); }
.wchip.bad { border-color: var(--fail); background: var(--fail-soft); }
/* faded derivations */
.deriv { border: 1px solid var(--line); border-radius: 10px;
  background: var(--box-bg); padding: 1rem 1.2rem; margin: 1.4rem 0; }
.deriv-head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap;
  margin-bottom: .5rem; }
.deriv-title { font-weight: 600; margin-right: auto; }
.dstep { border-top: 1px solid var(--line); padding: .55rem 0; }
.dstep-label { display: flex; align-items: baseline; gap: .6rem; }
.dstep-goal { flex: 1; font-size: .95rem; }
.dstep-toggle { flex-shrink: 0; font-size: .78rem; padding: .2rem .6rem; }
.dstep-body { display: none; margin-top: .55rem; }
.dstep.open .dstep-body { display: block; }
.dstep.open .dstep-toggle { opacity: .55; }
/* concept map */
.cmap-edge { stroke: var(--muted); stroke-width: 1.2; opacity: .4; fill: none; }
.cmap-edge.hl { stroke: var(--accent); opacity: 1; stroke-width: 2.4; }
.cmap-node { cursor: pointer; }
.cmap-node rect { fill: var(--bg); stroke: var(--line); stroke-width: 1.4; }
.cmap-node text { fill: var(--fg); font-size: 13px; }
.cmap-node.hub rect { stroke: var(--accent); stroke-width: 2; }
.cmap-node.hl rect { stroke: var(--accent); stroke-width: 2.2; }
.cmap-node.dim, .cmap-edge.dim { opacity: .15; }
/* cheat-sheet grid */
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem; margin: 1.2rem 0; }
.cheat-card { border: 1px solid var(--line); border-radius: 10px;
  background: var(--box-bg); padding: .8rem 1rem; }
.cheat-card h4 { margin: 0 0 .4rem; }
.cheat-card .math { margin: .6rem 0; font-size: .98rem; }
.cheat-card p { font-size: .9rem; margin-bottom: .4rem; }
/* quiz */
.quiz-note { color: var(--muted); font-size: .88rem; }
.quiz-q { border: 1px solid var(--line); border-radius: 10px;
  padding: 1.1rem 1.3rem; margin-bottom: 1.4rem; background: var(--box-bg); }
.quiz-q .qtext { font-weight: 600; margin-bottom: .8rem; }
.quiz-opt { display: block; width: 100%; text-align: left; margin: .4rem 0;
  padding: .6rem .9rem; border: 1px solid var(--line); border-radius: 8px;
  background: var(--bg); color: var(--fg); font: inherit; font-size: .95rem;
  cursor: pointer; transition: border-color .12s; }
.quiz-opt:hover:not(:disabled) { border-color: var(--accent); }
.quiz-opt:disabled { cursor: default; opacity: .75; }
.quiz-opt.chosen-right { border-color: var(--ok); background: var(--ok-soft);
  opacity: 1; }
.quiz-opt.chosen-wrong { border-color: var(--fail); background: var(--fail-soft);
  opacity: 1; }
.quiz-opt.reveal-right { border-color: var(--ok); opacity: 1; }
.quiz-expl { font-size: .85rem; line-height: 1.5; margin: -.15rem 0 .55rem;
  padding: .35rem .9rem; border-left: 3px solid var(--line);
  color: var(--muted); }
.quiz-expl.right { border-left-color: var(--ok); color: var(--fg);
  background: var(--ok-soft); border-radius: 0 8px 8px 0; }
.quiz-expl.wrong { border-left-color: var(--fail); }
.quiz-fb { margin-top: .7rem; padding: .6rem .9rem; border-radius: 8px;
  font-size: .92rem; display: none; }
.quiz-fb.right { display: block; background: var(--ok-soft); color: var(--fg); }
.quiz-fb.wrong { display: block; background: var(--fail-soft); color: var(--fg); }
.quiz-score { font-weight: 600; margin-top: 1rem; display: none; }
footer { color: var(--muted); font-size: .82rem; border-top: 1px solid var(--line);
  padding-top: 1rem; margin-top: 3rem; }
"""

QUIZ_JS = """
window.copyTextFallback = function (text) {
  function legacy() {
    var ta = document.createElement('textarea');
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) {}
    ta.remove();
  }
  if (navigator.clipboard && navigator.clipboard.writeText) {
    navigator.clipboard.writeText(text).catch(legacy);
  } else legacy();
};
(function () {
  const groups = window.__QUIZZES__ || [];
  const LOGKEY = 'quizlog:' + location.pathname.split('/').pop();
  const loadLog = () => { try { return JSON.parse(localStorage.getItem(LOGKEY)) || {}; } catch (e) { return {}; } };
  const saveLog = (l) => { try { localStorage.setItem(LOGKEY, JSON.stringify(l)); } catch (e) {} };
  const sectionTitle = (host) => {
    const sec = document.getElementById(host.replace(/^qh-/, ''));
    const h = sec && sec.querySelector('h2');
    return h ? h.textContent : host;
  };
  const buildSummary = () => {
    const log = loadLog();
    const lines = ['Quiz results — ' + document.title + ' — ' + new Date().toISOString().slice(0, 10)];
    let tot = 0, got = 0;
    groups.forEach(g => {
      const e = log[g.host];
      if (!e) { lines.push(sectionTitle(g.host) + ': not attempted'); return; }
      tot += e.total; got += e.score;
      lines.push(sectionTitle(g.host) + ': ' + e.score + '/' + e.total +
        (e.wrong.length ? '. Missed: ' + e.wrong.map(w => '"' + w + '"').join('; ') : ''));
    });
    lines.push('Total: ' + got + '/' + tot);
    const miss = (loadLog().__miss) || {};
    const rep = Object.entries(miss).filter(([, n]) => n > 1).map(([q, n]) => '"' + q + '" (x' + n + ')');
    if (rep.length) lines.push('Repeat offenders: ' + rep.join('; '));
    lines.push('Please quiz me again on my weak areas, harder.');
    return lines.join('\\n');
  };
  groups.forEach(function (group) {
    const host = document.getElementById(group.host);
    const data = group.questions || [];
    if (!host || !data.length) return;
    let answered = 0, right = 0;
    const results = [];
    const score = document.createElement('div');
    score.className = 'quiz-score';
    data.forEach((q, qi) => {
      const card = document.createElement('div');
      card.className = 'quiz-q';
      const qt = document.createElement('div');
      qt.className = 'qtext';
      qt.innerHTML = (qi + 1) + '. ' + q.question;
      card.appendChild(qt);
      const fb = document.createElement('div');
      fb.className = 'quiz-fb';
      const opts = q.options.map((o, i) => ({ o, i }));
      for (let i = opts.length - 1; i > 0; i--) {          // shuffle
        const j = Math.floor(Math.random() * (i + 1));
        [opts[i], opts[j]] = [opts[j], opts[i]];
      }
      const buttons = [];
      opts.forEach(({ o }) => {
        const b = document.createElement('button');
        b.className = 'quiz-opt';
        b.innerHTML = o.text;
        b.addEventListener('click', () => {
          answered++;
          // reveal every option's explanation, marked right/wrong
          buttons.forEach(x => {
            x.disabled = true;
            if (x.__correct) x.classList.add('reveal-right');
            const ex = document.createElement('div');
            ex.className = 'quiz-expl ' + (x.__correct ? 'right' : 'wrong');
            ex.innerHTML = (x.__correct ? '<strong>✓ Correct answer.</strong> '
                                        : '<strong>✗</strong> ') +
                           (x.__opt.explanation || '');
            x.insertAdjacentElement('afterend', ex);
          });
          if (o.correct) {
            right++;
            b.classList.add('chosen-right');
            fb.className = 'quiz-fb right';
            fb.innerHTML = '<strong>Correct.</strong> Explanations for every option are shown above.';
          } else {
            b.classList.add('chosen-wrong');
            fb.className = 'quiz-fb wrong';
            fb.innerHTML = '<strong>Not quite.</strong> The correct answer is outlined above, with explanations for every option.';
          }
          results.push({ q: q.question.replace(/<[^>]*>/g, ''), ok: !!o.correct });
          if (answered === data.length) {
            score.textContent = 'Section score: ' + right + ' / ' + data.length;
            score.style.display = 'block';
            const log = loadLog();
            log[group.host] = { ts: Date.now(), score: right, total: data.length,
              wrong: results.filter(r => !r.ok).map(r => r.q) };
            log.__miss = log.__miss || {};
            results.filter(r => !r.ok).forEach(r => { log.__miss[r.q] = (log.__miss[r.q] || 0) + 1; });
            saveLog(log);
            const cp = document.createElement('button');
            cp.className = 'wbtn'; cp.style.marginTop = '.6rem';
            cp.textContent = 'copy results for Claude';
            cp.addEventListener('click', () => {
              window.copyTextFallback(buildSummary());
              cp.textContent = 'copied \\u2713';
              setTimeout(() => { cp.textContent = 'copy results for Claude'; }, 1500);
            });
            score.insertAdjacentElement('afterend', cp);
          }
        });
        b.__correct = !!o.correct;
        b.__opt = o;
        buttons.push(b);
        card.appendChild(b);
      });
      card.appendChild(fb);
      host.appendChild(card);
    });
    host.appendChild(score);
  });
})();
"""

PAGE_JS = """
(function () {
  document.querySelectorAll('.dstep-toggle').forEach(b =>
    b.addEventListener('click', () => b.closest('.dstep').classList.toggle('open')));
  document.querySelectorAll('.deriv').forEach(d => {
    const setAll = open => d.querySelectorAll('.dstep').forEach(s => s.classList.toggle('open', open));
    const w = d.querySelector('.deriv-worked'), p = d.querySelector('.deriv-practice');
    if (w) w.addEventListener('click', () => setAll(true));
    if (p) p.addEventListener('click', () => setAll(false));
  });
  document.querySelectorAll('[data-copy]').forEach(b =>
    b.addEventListener('click', () => {
      const el = document.getElementById(b.getAttribute('data-copy'));
      if (!el) return;
      window.copyTextFallback(el.innerText);
      const old = b.textContent;
      b.textContent = 'copied \\u2713';
      setTimeout(() => { b.textContent = old; }, 1500);
    }));
})();
"""


def _slugify(text: str) -> str:
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def _check_quiz(quiz: list) -> None:
    for q in quiz:
        n_correct = sum(1 for o in q.get("options", []) if o.get("correct"))
        if n_correct != 1:
            sys.exit(f"quiz question needs exactly one correct option: {q.get('question')!r}")


def render(spec: dict) -> str:
    for key in ("title", "slug", "sections"):
        if key not in spec:
            sys.exit(f"spec is missing required key: {key!r}")

    sections = list(spec["sections"])
    global_quiz = spec.get("quiz") or []
    quizzes = []

    toc_items, body_parts = [], []
    for s in sections:
        sid = s.get("id") or _slugify(s["title"])
        toc_items.append((sid, s["title"]))
        quiz_html = ""
        if s.get("quiz"):
            _check_quiz(s["quiz"])
            hid = f"qh-{sid}"
            quizzes.append({"host": hid, "questions": s["quiz"]})
            quiz_html = (
                f"<h3 class='quiz-heading'>Quiz: {_html.escape(s['title'])}</h3>"
                "<p class='quiz-note'>Option order is shuffled on every load; "
                "answers lock after one click.</p>"
                f"<div id='{hid}'></div>"
            )
        body_parts.append(
            f'<section id="{_html.escape(sid)}">'
            f'<h2>{_html.escape(s["title"])}</h2>{s["html"]}{quiz_html}</section>'
        )
    if global_quiz:
        _check_quiz(global_quiz)
        quizzes.append({"host": "qh-quiz", "questions": global_quiz})
        toc_items.append(("quiz", "Quiz"))
        body_parts.append(
            '<section id="quiz"><h2>Quiz</h2>'
            "<p class='quiz-note'>Option order is shuffled on every load; "
            "answers lock after one click.</p>"
            '<div id="qh-quiz"></div></section>'
        )

    toc = "".join(
        f'<li><a href="#{_html.escape(sid)}">{_html.escape(title)}</a></li>'
        for sid, title in toc_items
    )
    subtitle = spec.get("subtitle", "")
    repo = spec.get("repo", "")
    date = spec.get("date") or _dt.date.today().isoformat()

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_html.escape(spec["title"])}</title>
<style>{CSS}</style>
{"".join(f'<script src="{_html.escape(s)}"></script>' for s in spec.get("head_script_srcs", []))}
{"".join(f'<script>{s}</script>' for s in spec.get("head_scripts", []))}
</head>
<body>
<div class="wrap">
<header class="masthead">
<h1>{_html.escape(spec["title"])}</h1>
{f'<p class="subtitle">{_html.escape(subtitle)}</p>' if subtitle else ''}
{f'<div class="repo">{_html.escape(repo)} · {date}</div>' if repo else f'<div class="repo">{date}</div>'}
</header>
<nav class="toc"><div class="toc-title">Contents</div><ol>{toc}</ol></nav>
{"".join(body_parts)}
<footer>Generated by explain-diff · {date}</footer>
</div>
<script>window.__QUIZZES__ = {json.dumps(quizzes)};</script>
<script>{QUIZ_JS}</script>
<script>{PAGE_JS}</script>
{"".join(f'<script>{s}</script>' for s in spec.get("scripts", []))}
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("spec", help="path to the JSON content spec")
    ap.add_argument("-o", "--outdir", default=None,
                    help="output directory (default: alongside the spec)")
    args = ap.parse_args()

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text())
    date = spec.get("date") or _dt.date.today().isoformat()
    outdir = Path(args.outdir) if args.outdir else spec_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{date}-{spec['slug']}.html"
    out.write_text(render(spec))
    print(out)


if __name__ == "__main__":
    main()
