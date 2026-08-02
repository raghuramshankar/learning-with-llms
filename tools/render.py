#!/usr/bin/env python3
"""Render interactive explainer HTML from a JSON content spec.

Usage:
    python render.py <spec.json> [-o OUTDIR]

Single-page mode (default): writes <OUTDIR>/<date>-<slug>.html.
Multipage mode ("multipage": true): writes one page per section named
<date>-<slug>-<section-id>.html plus an overview page <date>-<slug>.html,
each with a sticky sidebar table of contents, the current page's h3
subsections nested under it, and prev/next navigation. Cross-section anchors
(href="#<section-id>") in section html are rewritten to page links
automatically, and every page defines window.__PAGES__ = {sectionId: href}
so widget JS can link across pages (falls back to #anchors in single-page
mode).

Theme: academic light theme modeled on adaptive-bayesian.ai — Merriweather
serif, deep-blue section headings over a thin blue rule, green sub-headings,
navy underlined links, light-gray masthead. Dark-scheme variant preserves the
hue system. Merriweather loads from Google Fonts and degrades to Georgia
offline.

JSON spec schema
----------------
{
  "title":    "Page title",                  (required)
  "subtitle": "One-line subtitle",           (optional)
  "slug":     "kebab-case-name",             (required; used in the filename)
  "date":     "YYYY-MM-DD",                  (optional; defaults to today —
                                              PIN THIS on rebuilds so links
                                              stay stable)
  "repo":     "project name / branch note",  (optional; shown under subtitle)
  "multipage": true,                         (optional; one page per section)
  "intro":    "<p>overview html...</p>",     (optional; multipage overview
                                              body above the parts list)
  "site_title": "Learning with LLMs",        (optional; masthead brand, links
                                              to index.html)
  "nav": [["All topics", "index.html"],      (optional; masthead links)
          ["GitHub", "https://..."]],
  "sections": [                              (required)
    { "id":    "background",                 (optional; derived from title)
      "title": "Background",
      "html":  "<p>raw HTML body...</p>",
      "quiz":  [ ...questions... ] }         (optional; rendered as a quiz
  ],                                          block at the end of the section)
  "quiz": [ ... ],                           (optional; single-page mode only:
                                              a final "Quiz" section. Question
                                              and option text may contain
                                              inline HTML, e.g. <sub>/<sup>.
                                              Exactly one option correct.)
  "scripts": [ "raw JS ..." ],               (optional; emitted at the end of
                                              <body> on EVERY page — widget JS
                                              guards on its container ids)
  "head_scripts": [ "raw JS ..." ],          (optional; inlined in <head>)
  "head_script_srcs": [ "plotly.min.js" ]    (optional; <script src=...> in
}                                             <head> — share one library file
                                              next to the pages instead of
                                              inlining megabytes per page)

HTML vocabulary available inside section "html" bodies
------------------------------------------------------
  <pre>...</pre>                 code block (white-space: pre-wrap, scrolls)
  <div class="diagram">          centred figure container with padding
    <div class="flow">           horizontal flex row (wraps on small screens)
      <div class="box">A</div>   node; variants: box ok / box fail / box dim
      <span class="arr">→</span> arrow between boxes
    <div class="flow vertical">  column variant; use <span class="arr">↓</span>
    <div class="caption">...</div>
  </div>
  <div class="callout">...</div>           key definition / edge case
  <div class="callout warn">...</div>      warning variant
  <div class="math">...</div>              display math (serif, centred)
  <span class="m">...</span>               inline math (serif, no-wrap)
  <span class="frac"><span class="num">a</span><span class="den">b</span></span>
  <table>...</table>                       styled automatically
  <span class="tag">label</span>           small inline pill
  <div class="widget">...</div>            interactive-widget container with
    <div class="wctl">controls</div>       .wbtn buttons, range inputs,
    <canvas>, .wstat, .wchips              status lines, outcome chips
  <div class="deriv">...</div>             faded derivation: .deriv-head with
    .deriv-practice/.deriv-worked buttons, .dstep rows with .dstep-label
    (.tag number + .dstep-goal + .dstep-toggle button) and hidden .dstep-body
  <div class="grid2"> + .cheat-card        cheat-sheet card grid
Everything else (h3, p, ul, code, strong, em) is styled sensibly. Quizzes
shuffle options per load, reveal every option's explanation on answer, track
results in localStorage, and offer a "copy results for Claude" button.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import html as _html
import json
import re
import sys
from pathlib import Path

FONTS = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,400;1,700&display=swap" rel="stylesheet">"""

CSS = """
:root {
  --bg: #ffffff; --fg: #111418; --muted: #5c6670; --line: #e1e5ea;
  --accent: #063c92; --link: #053075; --accent-soft: #eef2fb;
  --ok: #168551; --ok-soft: #e9f5ee; --fail: #b03434; --fail-soft: #faeceb;
  --warn-soft: #fdf6e3; --warn-line: #c9a227;
  --code-bg: #f6f8fa; --box-bg: #f8f9fb; --mast: #f5f5f5;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #13161b; --fg: #e5e8ee; --muted: #9aa3b0; --line: #2a313b;
    --accent: #7d9fdd; --link: #8fb0e8; --accent-soft: #1b2540;
    --ok: #57b98a; --ok-soft: #16301f; --fail: #e08585; --fail-soft: #3a2020;
    --warn-soft: #33290f; --warn-line: #a8842e;
    --code-bg: #1b2027; --box-bg: #181c23; --mast: #1a1e25;
  }
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--bg); color: var(--fg);
  font: 16px/1.75 Merriweather, Georgia, 'Palatino Linotype', serif;
  -webkit-font-smoothing: antialiased;
}
/* masthead */
.masthead { background: var(--mast); border-bottom: 1px solid var(--line); }
.mast-inner { max-width: 1160px; margin: 0 auto; padding: .55rem 1.4rem;
  display: flex; align-items: baseline; gap: 1.4rem; flex-wrap: wrap; }
.mast-title { font-weight: 700; font-size: .98rem; }
.mast-title a { color: var(--fg); text-decoration: none; }
.mast-nav { margin-left: auto; display: flex; gap: 1.1rem; flex-wrap: wrap; }
.mast-nav a { color: var(--muted); text-decoration: none; font-size: .85rem; }
.mast-nav a:hover { color: var(--link); text-decoration: underline; }
/* layout */
.wrap { max-width: 950px; margin: 0 auto; padding: 2.2rem 1.4rem 5rem; }
.layout { max-width: 1160px; margin: 0 auto; padding: 0 1.4rem 4rem;
  display: grid; grid-template-columns: 235px minmax(0, 1fr); gap: 2.6rem; }
main.content { min-width: 0; padding-top: 1.8rem; }
/* sidebar */
aside.sidebar { position: sticky; top: 0; align-self: start;
  max-height: 100vh; overflow-y: auto; padding: 1.8rem 0 2rem;
  font-size: .84rem; line-height: 1.55; }
.side-topic { font-weight: 700; margin-bottom: .7rem; font-size: .9rem; }
.side-topic a { color: var(--accent); text-decoration: none; }
.side-topic a:hover { text-decoration: underline; }
aside.sidebar ol { list-style: none; margin: 0; padding: 0;
  border-left: 2px solid var(--line); }
aside.sidebar ol > li { padding: .22rem 0 .22rem .8rem; margin-left: -2px;
  border-left: 2px solid transparent; }
aside.sidebar ol > li.cur { border-left-color: var(--accent); }
aside.sidebar ol > li > a { color: var(--muted); text-decoration: none; }
aside.sidebar ol > li.cur > a { color: var(--accent); font-weight: 700; }
aside.sidebar ol > li > a:hover { color: var(--link); }
aside.sidebar ul { list-style: none; margin: .3rem 0 .1rem; padding: 0 0 0 .9rem; }
aside.sidebar ul li { padding: .12rem 0; }
aside.sidebar ul a { color: var(--muted); text-decoration: none; font-size: .8rem; }
aside.sidebar ul a:hover { color: var(--link); }
.side-foot { margin-top: 1rem; }
.side-foot a { color: var(--muted); text-decoration: none; font-size: .8rem; }
.side-foot a:hover { color: var(--link); }
@media (max-width: 920px) {
  .layout { grid-template-columns: 1fr; gap: 0; }
  aside.sidebar { position: static; max-height: none; margin-top: 1.4rem;
    padding: .9rem 1.1rem; border: 1px solid var(--line); border-radius: 10px;
    background: var(--box-bg); }
}
/* headings, in the adaptive-bayesian style */
header.pagehead { padding-bottom: 1rem; margin-bottom: 1.2rem; }
.part-eyebrow { color: var(--ok); font-weight: 700; font-size: .82rem;
  margin: 0 0 .35rem; letter-spacing: .02em; }
h1 { font-size: 1.55rem; line-height: 1.3; margin: 0 0 .5rem; font-weight: 700; }
.subtitle { color: var(--muted); font-size: 1.02rem; margin: 0; }
.repo { color: var(--muted); font-size: .8rem; margin-top: .55rem; }
h2 { font-size: 1.12rem; color: var(--accent); font-weight: 700;
  border-bottom: 1px solid var(--accent); padding-bottom: .3rem;
  margin: 2.4rem 0 1rem; }
h3 { font-size: 1rem; color: var(--ok); font-weight: 700; margin: 1.9rem 0 .5rem; }
h4 { font-size: .95rem; margin: 1.5rem 0 .4rem; color: var(--fg); }
p, ul, ol { margin: 0 0 1rem; }
li { margin-bottom: .35rem; }
a { color: var(--link); }
/* toc box (single-page mode) */
nav.toc { background: var(--box-bg); border: 1px solid var(--line);
  border-radius: 10px; padding: 1rem 1.3rem; margin-bottom: 2.2rem; }
nav.toc .toc-title { font-weight: 700; font-size: .78rem; letter-spacing: .08em;
  text-transform: uppercase; color: var(--muted); margin-bottom: .4rem; }
nav.toc ol { margin: 0; padding-left: 1.2rem; }
nav.toc a { color: var(--link); text-decoration: none; }
nav.toc a:hover { text-decoration: underline; }
section { margin-bottom: 3rem; }
code { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 4px; padding: .08em .35em; font-size: .84em;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre { background: var(--code-bg); border: 1px solid var(--line);
  border-radius: 8px; padding: .9rem 1.1rem; overflow-x: auto;
  white-space: pre-wrap; font-size: .82rem; line-height: 1.5;
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
pre code { background: none; border: none; padding: 0; font-size: inherit; }
table { border-collapse: collapse; margin: 1rem 0; width: 100%;
  font-size: .88rem; display: block; overflow-x: auto; }
th, td { border: 1px solid var(--line); padding: .45rem .7rem;
  text-align: left; vertical-align: top; }
th { background: var(--box-bg); font-weight: 700; }
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
  padding: .45rem .85rem; font-size: .85rem; text-align: center;
  line-height: 1.4; }
.box small { display: block; color: var(--muted); font-size: .76rem; }
.box.ok { border-color: var(--ok); background: var(--ok-soft); }
.box.fail { border-color: var(--fail); background: var(--fail-soft); }
.box.dim { border-color: var(--line); color: var(--muted); }
.box.accent { border-color: var(--accent); background: var(--accent-soft); }
.arr { color: var(--muted); font-size: 1.05rem; flex-shrink: 0; }
.caption { text-align: center; color: var(--muted); font-size: .82rem;
  margin-top: .8rem; }
.tag { display: inline-block; background: var(--accent-soft);
  color: var(--accent); border-radius: 999px; padding: .05em .6em;
  font-size: .76em; font-weight: 700; }
/* math */
.math { font-family: Georgia, 'Times New Roman', serif; text-align: center;
  margin: 1.2rem 0; padding: .2rem 0; overflow-x: auto; font-size: 1.02rem;
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
.wctl label { font-size: .84rem; color: var(--muted); }
.wctl input[type=range] { width: min(240px, 55vw); accent-color: var(--accent); }
.wbtn { font: inherit; font-size: .82rem; padding: .35rem .9rem;
  border-radius: 8px; border: 1px solid var(--accent);
  background: var(--accent-soft); color: var(--accent); cursor: pointer;
  font-weight: 700; }
.wbtn:hover { filter: brightness(1.05); }
.wbtn.active { background: var(--accent); color: var(--bg); }
.wstat { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: .82rem; color: var(--muted); text-align: center; margin-top: .5rem; }
.wchips { display: flex; flex-wrap: wrap; gap: .3rem; justify-content: center;
  margin-top: .5rem; max-height: 130px; overflow-y: auto; }
.wchip { font-size: .76rem; border: 1px solid var(--line); border-radius: 999px;
  padding: .05em .55em; }
.wchip.good { border-color: var(--ok); background: var(--ok-soft); }
.wchip.bad { border-color: var(--fail); background: var(--fail-soft); }
/* faded derivations */
.deriv { border: 1px solid var(--line); border-radius: 10px;
  background: var(--box-bg); padding: 1rem 1.2rem; margin: 1.4rem 0; }
.deriv-head { display: flex; align-items: center; gap: .6rem; flex-wrap: wrap;
  margin-bottom: .5rem; }
.deriv-title { font-weight: 700; margin-right: auto; color: var(--accent); }
.dstep { border-top: 1px solid var(--line); padding: .55rem 0; }
.dstep-label { display: flex; align-items: baseline; gap: .6rem; }
.dstep-goal { flex: 1; font-size: .92rem; }
.dstep-toggle { flex-shrink: 0; font-size: .74rem; padding: .2rem .6rem; }
.dstep-body { display: none; margin-top: .55rem; }
.dstep.open .dstep-body { display: block; }
.dstep.open .dstep-toggle { opacity: .55; }
/* concept map */
.cmap-edge { stroke: var(--muted); stroke-width: 1.2; opacity: .4; fill: none; }
.cmap-edge.hl { stroke: var(--accent); opacity: 1; stroke-width: 2.4; }
.cmap-node { cursor: pointer; }
.cmap-node rect { fill: var(--bg); stroke: var(--line); stroke-width: 1.4; }
.cmap-node text { fill: var(--fg); font-size: 13px;
  font-family: -apple-system, 'Segoe UI', Roboto, sans-serif; }
.cmap-node.hub rect { stroke: var(--accent); stroke-width: 2; }
.cmap-node.hl rect { stroke: var(--accent); stroke-width: 2.2; }
.cmap-node.dim, .cmap-edge.dim { opacity: .15; }
/* cheat-sheet grid */
.grid2 { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 1rem; margin: 1.2rem 0; }
.cheat-card { border: 1px solid var(--line); border-radius: 10px;
  background: var(--box-bg); padding: .8rem 1rem; }
.cheat-card h4 { margin: 0 0 .4rem; color: var(--accent); }
.cheat-card .math { margin: .6rem 0; font-size: .95rem; }
.cheat-card p { font-size: .86rem; margin-bottom: .4rem; }
/* quiz */
.quiz-heading { color: var(--accent); }
.quiz-note { color: var(--muted); font-size: .84rem; }
.quiz-q { border: 1px solid var(--line); border-radius: 10px;
  padding: 1.1rem 1.3rem; margin-bottom: 1.4rem; background: var(--box-bg); }
.quiz-q .qtext { font-weight: 700; margin-bottom: .8rem; }
.quiz-opt { display: block; width: 100%; text-align: left; margin: .4rem 0;
  padding: .6rem .9rem; border: 1px solid var(--line); border-radius: 8px;
  background: var(--bg); color: var(--fg); font: inherit; font-size: .9rem;
  cursor: pointer; transition: border-color .12s; }
.quiz-opt:hover:not(:disabled) { border-color: var(--accent); }
.quiz-opt:disabled { cursor: default; opacity: .75; }
.quiz-opt.chosen-right { border-color: var(--ok); background: var(--ok-soft);
  opacity: 1; }
.quiz-opt.chosen-wrong { border-color: var(--fail); background: var(--fail-soft);
  opacity: 1; }
.quiz-opt.reveal-right { border-color: var(--ok); opacity: 1; }
.quiz-expl { font-size: .82rem; line-height: 1.55; margin: -.15rem 0 .55rem;
  padding: .35rem .9rem; border-left: 3px solid var(--line);
  color: var(--muted); }
.quiz-expl.right { border-left-color: var(--ok); color: var(--fg);
  background: var(--ok-soft); border-radius: 0 8px 8px 0; }
.quiz-expl.wrong { border-left-color: var(--fail); }
.quiz-fb { margin-top: .7rem; padding: .6rem .9rem; border-radius: 8px;
  font-size: .88rem; display: none; }
.quiz-fb.right { display: block; background: var(--ok-soft); color: var(--fg); }
.quiz-fb.wrong { display: block; background: var(--fail-soft); color: var(--fg); }
.quiz-score { font-weight: 700; margin-top: 1rem; display: none; }
/* prev/next + overview parts */
.pn { display: flex; justify-content: space-between; gap: 1rem;
  border-top: 1px solid var(--line); margin-top: 3rem; padding-top: 1.2rem;
  flex-wrap: wrap; }
.pn a { text-decoration: none; font-size: .92rem; max-width: 46%; }
.pn a:hover { text-decoration: underline; }
.pn .pn-next { margin-left: auto; text-align: right; }
.parts { list-style: none; padding: 0; margin: 1.4rem 0; counter-reset: part; }
.parts li { border: 1px solid var(--line); border-radius: 10px;
  background: var(--box-bg); margin: .6rem 0; counter-increment: part; }
.parts li a { display: block; padding: .8rem 1.1rem; text-decoration: none;
  color: var(--fg); }
.parts li a:hover { border-color: var(--accent); color: var(--link); }
.parts li a::before { content: counter(part) ". "; color: var(--ok);
  font-weight: 700; }
.startbtn { display: inline-block; background: var(--accent); color: var(--bg);
  padding: .55rem 1.2rem; border-radius: 9px; font-weight: 700;
  text-decoration: none; }
.startbtn:hover { filter: brightness(1.1); }
footer.pagefoot { color: var(--muted); font-size: .8rem;
  border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 3rem; }
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
    return h ? h.textContent : document.title;
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


def _strip_tags(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s).strip()


def _inject_h3_ids(html_str: str, sid: str):
    """Give plain <h3> headings ids and return (html, [(id, label), ...])."""
    subs = []

    def repl(m):
        hid = f"{sid}-h{len(subs) + 1}"
        subs.append((hid, _strip_tags(m.group(1))))
        return f'<h3 id="{hid}">{m.group(1)}</h3>'

    return re.sub(r"<h3>(.*?)</h3>", repl, html_str, flags=re.S), subs


def _quiz_block(s: dict, quizzes: list) -> str:
    if not s.get("quiz"):
        return ""
    _check_quiz(s["quiz"])
    hid = f"qh-{s['_sid']}"
    quizzes.append({"host": hid, "questions": s["quiz"]})
    return (
        f"<h3 class='quiz-heading' id='{hid}-title'>Quiz: {_html.escape(s['title'])}</h3>"
        "<p class='quiz-note'>Option order is shuffled on every load; "
        "answers lock after one click.</p>"
        f"<div id='{hid}'></div>"
    )


def _shell(spec, *, title, body, quizzes, pages_map, date) -> str:
    site_title = spec.get("site_title", "Learning with LLMs")
    nav = spec.get("nav", [["All topics", "index.html"]])
    nav_html = "".join(
        f'<a href="{_html.escape(h)}">{_html.escape(l)}</a>' for l, h in nav)
    head_srcs = "".join(
        f'<script src="{_html.escape(s)}"></script>' for s in spec.get("head_script_srcs", []))
    head_inline = "".join(f"<script>{s}</script>" for s in spec.get("head_scripts", []))
    body_scripts = "".join(f"<script>{s}</script>" for s in spec.get("scripts", []))
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>{_html.escape(title)}</title>
{FONTS}
<style>{CSS}</style>
{head_srcs}
{head_inline}
</head>
<body>
<div class="masthead"><div class="mast-inner">
<span class="mast-title"><a href="index.html">{_html.escape(site_title)}</a></span>
<nav class="mast-nav">{nav_html}</nav>
</div></div>
{body}
<script>window.__PAGES__ = {json.dumps(pages_map)};</script>
<script>window.__QUIZZES__ = {json.dumps(quizzes)};</script>
<script>{QUIZ_JS}</script>
<script>{PAGE_JS}</script>
{body_scripts}
</body>
</html>
"""


def render(spec: dict) -> list:
    """Return [(filename, html), ...]."""
    for key in ("title", "slug", "sections"):
        if key not in spec:
            sys.exit(f"spec is missing required key: {key!r}")

    date = spec.get("date") or _dt.date.today().isoformat()
    prefix = f"{date}-{spec['slug']}"
    sections = list(spec["sections"])
    for s in sections:
        s["_sid"] = s.get("id") or _slugify(s["title"])

    subtitle = spec.get("subtitle", "")
    repo = spec.get("repo", "")
    head_block = (
        '<header class="pagehead">'
        f'<h1>{_html.escape(spec["title"])}</h1>'
        + (f'<p class="subtitle">{_html.escape(subtitle)}</p>' if subtitle else "")
        + (f'<div class="repo">{_html.escape(repo)} &middot; {date}</div>'
           if repo else f'<div class="repo">{date}</div>')
        + "</header>"
    )
    foot = f'<footer class="pagefoot">Generated by explain-diff &middot; {date}</footer>'

    if not spec.get("multipage"):
        # ---- single page -------------------------------------------------
        pages_map = {s["_sid"]: f"#{s['_sid']}" for s in sections}
        quizzes, toc_items, body_parts = [], [], []
        for s in sections:
            toc_items.append((s["_sid"], s["title"]))
            body_parts.append(
                f'<section id="{_html.escape(s["_sid"])}">'
                f'<h2>{_html.escape(s["title"])}</h2>{s["html"]}{_quiz_block(s, quizzes)}</section>'
            )
        if spec.get("quiz"):
            _check_quiz(spec["quiz"])
            quizzes.append({"host": "qh-quiz", "questions": spec["quiz"]})
            toc_items.append(("quiz", "Quiz"))
            body_parts.append(
                '<section id="quiz"><h2>Quiz</h2>'
                "<p class='quiz-note'>Option order is shuffled on every load; "
                "answers lock after one click.</p>"
                '<div id="qh-quiz"></div></section>'
            )
        toc = "".join(
            f'<li><a href="#{_html.escape(sid)}">{_html.escape(t)}</a></li>'
            for sid, t in toc_items)
        body = (f'<div class="wrap">{head_block}'
                f'<nav class="toc"><div class="toc-title">Contents</div><ol>{toc}</ol></nav>'
                f'{"".join(body_parts)}{foot}</div>')
        return [(f"{prefix}.html",
                 _shell(spec, title=spec["title"], body=body, quizzes=quizzes,
                        pages_map=pages_map, date=date))]

    # ---- multipage -------------------------------------------------------
    if spec.get("quiz"):
        sys.exit("multipage mode: attach quizzes to sections, not a global quiz")
    files = []
    page_files = {s["_sid"]: f"{prefix}-{s['_sid']}.html" for s in sections}
    overview_file = f"{prefix}.html"
    pages_map = dict(page_files)

    def rewrite_anchors(html_str: str) -> str:
        for osid, fname in page_files.items():
            html_str = (html_str
                        .replace(f"href='#{osid}'", f"href='{fname}'")
                        .replace(f'href="#{osid}"', f'href="{fname}"'))
        return html_str

    def sidebar(cur_sid, cur_subs) -> str:
        items = [f'<div class="side-topic"><a href="{overview_file}">'
                 f'{_html.escape(spec["title"])}</a></div><ol>']
        for s in sections:
            cur = s["_sid"] == cur_sid
            cls = ' class="cur"' if cur else ''
            items.append(f'<li{cls}>'
                         f'<a href="{page_files[s["_sid"]]}">{_html.escape(s["title"])}</a>')
            if cur and cur_subs:
                items.append("<ul>")
                for hid, label in cur_subs:
                    items.append(f'<li><a href="#{hid}">{_html.escape(label)}</a></li>')
                if s.get("quiz"):
                    items.append(f'<li><a href="#qh-{cur_sid}-title">Quiz</a></li>')
                items.append("</ul>")
            items.append("</li>")
        items.append("</ol>")
        items.append('<div class="side-foot"><a href="index.html">&larr; all topics</a></div>')
        return f'<aside class="sidebar"><nav>{"".join(items)}</nav></aside>'

    n = len(sections)
    for i, s in enumerate(sections):
        quizzes = []
        body_html, subs = _inject_h3_ids(rewrite_anchors(s["html"]), s["_sid"])
        quiz_html = _quiz_block(s, quizzes)
        pn = ['<div class="pn">']
        if i > 0:
            p = sections[i - 1]
            pn.append(f'<a class="pn-prev" href="{page_files[p["_sid"]]}">&larr; '
                      f'{_html.escape(p["title"])}</a>')
        else:
            pn.append(f'<a class="pn-prev" href="{overview_file}">&larr; Overview</a>')
        if i < n - 1:
            nx = sections[i + 1]
            pn.append(f'<a class="pn-next" href="{page_files[nx["_sid"]]}">'
                      f'{_html.escape(nx["title"])} &rarr;</a>')
        pn.append("</div>")
        content = (
            '<main class="content">'
            f'<p class="part-eyebrow">Part {i + 1} of {n} &middot; '
            f'{_html.escape(spec["title"])}</p>'
            f'<h1>{_html.escape(s["title"])}</h1>'
            f'{body_html}{quiz_html}{"".join(pn)}{foot}</main>'
        )
        body = f'<div class="layout">{sidebar(s["_sid"], subs)}{content}</div>'
        files.append((page_files[s["_sid"]],
                      _shell(spec, title=f'{s["title"]} — {spec["title"]}',
                             body=body, quizzes=quizzes, pages_map=pages_map,
                             date=date)))

    # overview page
    parts = "".join(
        f'<li><a href="{page_files[s["_sid"]]}">{_html.escape(s["title"])}</a></li>'
        for s in sections)
    intro = spec.get("intro", "")
    ov_content = (
        f'<main class="content">{head_block}{intro}'
        f'<ol class="parts">{parts}</ol>'
        f'<p><a class="startbtn" href="{page_files[sections[0]["_sid"]]}">'
        f'Start reading &rarr;</a></p>{foot}</main>'
    )
    ov_body = f'<div class="layout">{sidebar(None, [])}{ov_content}</div>'
    files.append((overview_file,
                  _shell(spec, title=spec["title"], body=ov_body, quizzes=[],
                         pages_map=pages_map, date=date)))
    return files


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
    outdir = Path(args.outdir) if args.outdir else spec_path.parent
    outdir.mkdir(parents=True, exist_ok=True)
    for fname, html in render(spec):
        (outdir / fname).write_text(html)
        print(outdir / fname)


if __name__ == "__main__":
    main()
