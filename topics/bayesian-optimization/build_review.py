#!/usr/bin/env python3
"""Build the spaced-repetition review page for the Bayesian optimization explainer."""
import json
from pathlib import Path

spec = json.loads((Path(__file__).parent / "spec.json").read_text())
bank = []
for si, s in enumerate(spec["sections"]):
    for qi, q in enumerate(s.get("quiz") or []):
        bank.append({
            "id": f"{s['id']}-{qi}",
            "section": s["title"],
            "question": q["question"],
            "options": q["options"],
        })

CSS = """
:root { --bg:#ffffff; --fg:#000000; --muted:#555e66; --line:#e1e5ea;
  --accent:#b509ac; --accent-soft:#f9e9f8; --ok:#168551; --ok-soft:#e9f5ee;
  --fail:#b03434; --fail-soft:#faeceb; --box-bg:#f8f9fb; --mast:#f5f5f5; }
:root[data-theme="dark"] { --bg:#000000; --fg:#e5e8ee;
  --muted:#9aa3b0; --line:#2a313b; --accent:#2698ba; --accent-soft:#0d2e38;
  --ok:#57b98a; --ok-soft:#16301f; --fail:#e08585; --fail-soft:#3a2020;
  --box-bg:#181c23; --mast:#1a1e25; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#000000; --fg:#e5e8ee;
  --muted:#9aa3b0; --line:#2a313b; --accent:#2698ba; --accent-soft:#0d2e38;
  --ok:#57b98a; --ok-soft:#16301f; --fail:#e08585; --fail-soft:#3a2020;
  --box-bg:#181c23; --mast:#1a1e25; } }
.masthead { position:fixed; top:0; left:0; right:0; z-index:100;
  background:var(--bg); border-bottom:1px solid var(--line); }
.mast-inner { max-width:1160px; margin:0 auto; padding:.85rem 1.4rem;
  display:flex; align-items:baseline; gap:1.35rem; flex-wrap:wrap;
  justify-content:flex-end; }
.mast-brand { margin-right:auto; color:var(--fg); text-decoration:none;
  font-size:1rem; font-weight:300; }
.mast-brand strong { font-weight:700; }
.mast-brand:hover { color:var(--accent); }
.gnav-link { color:var(--accent); text-decoration:none; font-size:1rem; }
.gnav-link:hover { text-decoration:underline; }
.theme-toggle { border:none; background:none; color:var(--accent);
  cursor:pointer; font-size:1rem; padding:0; font-family:inherit; }
.theme-toggle:hover { text-decoration:underline; }
body { padding-top:56px; }
.topicbar { background:var(--mast); border-bottom:1px solid var(--line); }
.topicbar-inner { max-width:1160px; margin:0 auto; padding:.5rem 1.4rem;
  display:flex; align-items:baseline; gap:1.1rem; flex-wrap:wrap; }
.topic-title a { color:var(--fg); font-weight:700; font-size:.95rem;
  text-decoration:none; }
.topic-title a:hover { color:var(--accent); }
.topicbar-nav { margin-left:auto; display:flex; gap:1.1rem; flex-wrap:wrap; }
.topicbar-nav a { color:var(--muted); text-decoration:none; font-size:.85rem; }
.topicbar-nav a:hover { color:var(--accent); text-decoration:underline; }
@media (max-width:920px) { .masthead { position:static; } body { padding-top:0; } }
footer.pagefoot { color:var(--muted); font-size:.8rem; border-top:1px solid var(--line);
  padding-top:1rem; margin-top:3rem; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.7 Merriweather, Georgia, serif; }
.wrap { max-width:950px; margin:0 auto; padding:2.2rem 1.3rem 4rem; }
h1 { font-size:1.5rem; margin:0 0 .3rem; color:var(--accent); }
.sub { color:var(--muted); margin:0 0 1.4rem; font-size:.95rem; }
.stats { display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1.2rem; }
.stat { background:var(--box-bg); border:1px solid var(--line); border-radius:10px;
  padding:.6rem 1rem; font-size:.9rem; }
.stat b { display:block; font-size:1.3rem; }
button.btn { font:inherit; font-size:.92rem; padding:.45rem 1rem; border-radius:8px;
  border:1px solid var(--accent); background:var(--accent-soft); color:var(--accent);
  cursor:pointer; font-weight:600; margin:.2rem .3rem .2rem 0; }
button.btn.subtle { border-color:var(--line); background:var(--box-bg); color:var(--muted); }
.card { border:1px solid var(--line); border-radius:12px; background:var(--box-bg);
  padding:1.2rem 1.4rem; margin:1rem 0; }
.qsec { color:var(--muted); font-size:.8rem; text-transform:uppercase;
  letter-spacing:.06em; margin-bottom:.5rem; }
.qtext { font-weight:600; margin-bottom:.9rem; }
.opt { display:block; width:100%; text-align:left; margin:.4rem 0; padding:.6rem .9rem;
  border:1px solid var(--line); border-radius:8px; background:var(--bg); color:var(--fg);
  font:inherit; font-size:.95rem; cursor:pointer; }
.opt:hover:not(:disabled) { border-color:var(--accent); }
.opt:disabled { cursor:default; opacity:.8; }
.opt.right { border-color:var(--ok); background:var(--ok-soft); opacity:1; }
.opt.wrong { border-color:var(--fail); background:var(--fail-soft); opacity:1; }
.expl { font-size:.85rem; line-height:1.5; margin:-.1rem 0 .55rem; padding:.35rem .9rem;
  border-left:3px solid var(--line); color:var(--muted); }
.expl.right { border-left-color:var(--ok); color:var(--fg); background:var(--ok-soft);
  border-radius:0 8px 8px 0; }
.expl.wrong { border-left-color:var(--fail); }
.meta { color:var(--muted); font-size:.82rem; margin-top:.8rem;
  font-family:ui-monospace, Menlo, monospace; }
.done { text-align:center; padding:2rem 0; }
.hidden { display:none; }
a { color:var(--accent); }
"""

JS = """
const BANK = window.__BANK__;
const KEY = 'sr:bayesian-optimization';
const DAY = 86400000;
const INT = [0, 1, 3, 7, 14, 30];                    // days per box
const $ = id => document.getElementById(id);
const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || {}; } catch (e) { return {}; } };
const save = s => { try { localStorage.setItem(KEY, JSON.stringify(s)); } catch (e) {} };
function copyText(t) {
  const go = () => { const ta = document.createElement('textarea'); ta.value = t;
    document.body.appendChild(ta); ta.select();
    try { document.execCommand('copy'); } catch (e) {} ta.remove(); };
  if (navigator.clipboard && navigator.clipboard.writeText)
    navigator.clipboard.writeText(t).catch(go); else go();
}
let state = load(), queue = [], idx = 0, session = { seen: 0, firstTryRight: 0, wrong: [] };

function dueList() {
  const now = Date.now();
  return BANK.filter(q => { const e = state[q.id]; return !e || e.due <= now; });
}
function refreshStats() {
  const now = Date.now();
  let neu = 0, due = 0, learned = 0;
  BANK.forEach(q => {
    const e = state[q.id];
    if (!e) neu++;
    else if (e.due <= now) due++;
    if (e && e.box >= 4) learned++;
  });
  $('st-due').textContent = due; $('st-new').textContent = neu;
  $('st-learned').textContent = learned; $('st-total').textContent = BANK.length;
  $('start').disabled = (due + neu) === 0;
  $('start').textContent = (due + neu) ? ('start review (' + (due + neu) + ' cards)') : 'nothing due — come back later';
}
function startSession() {
  queue = dueList(); idx = 0;
  session = { seen: 0, firstTryRight: 0, wrong: [] };
  if (!queue.length) return;
  $('home').classList.add('hidden'); $('play').classList.remove('hidden');
  showCard();
}
function schedule(q, correct, firstTry) {
  const e = state[q.id] || { box: 0, reps: 0, lapses: 0 };
  if (correct) {
    e.box = firstTry ? Math.min(e.box + 1, 5) : 1;
    e.due = Date.now() + INT[e.box] * DAY;
  } else {
    e.lapses++; e.box = 0; e.due = Date.now();      // retried within session
  }
  e.reps++; state[q.id] = e; save(state);
}
function showCard() {
  if (idx >= queue.length) return endSession();
  const q = queue[idx];
  const host = $('card'); host.innerHTML = '';
  const sec = document.createElement('div'); sec.className = 'qsec';
  sec.textContent = q.section + ' · card ' + (idx + 1) + '/' + queue.length;
  const qt = document.createElement('div'); qt.className = 'qtext'; qt.innerHTML = q.question;
  host.appendChild(sec); host.appendChild(qt);
  const opts = q.options.map((o, i) => ({ o, i }));
  for (let i = opts.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1)); [opts[i], opts[j]] = [opts[j], opts[i]];
  }
  const btns = [];
  opts.forEach(({ o }) => {
    const b = document.createElement('button'); b.className = 'opt'; b.innerHTML = o.text;
    b.__o = o;
    b.addEventListener('click', () => {
      btns.forEach(x => {
        x.disabled = true;
        if (x.__o.correct) x.classList.add('right');
        const ex = document.createElement('div');
        ex.className = 'expl ' + (x.__o.correct ? 'right' : 'wrong');
        ex.innerHTML = (x.__o.correct ? '<strong>✓</strong> ' : '<strong>✗</strong> ') + (x.__o.explanation || '');
        x.insertAdjacentElement('afterend', ex);
      });
      const firstTry = !q.__retried;
      if (!o.correct) b.classList.add('wrong');
      if (firstTry) { session.seen++; if (o.correct) session.firstTryRight++; else session.wrong.push(q.question.replace(/<[^>]*>/g, '')); }
      schedule(q, !!o.correct, firstTry);
      const nxt = document.createElement('button'); nxt.className = 'btn'; nxt.style.marginTop = '.8rem';
      if (o.correct) {
        const e = state[q.id];
        nxt.textContent = 'next — back in ' + (INT[e.box] || '<1') + ' day(s)';
        nxt.addEventListener('click', () => { idx++; showCard(); });
      } else {
        nxt.textContent = 'got it — retry this card later in the session';
        nxt.addEventListener('click', () => { q.__retried = true; queue.push(q); idx++; showCard(); });
      }
      host.appendChild(nxt);
    });
    btns.push(b); host.appendChild(b);
  });
}
function endSession() {
  $('play').classList.add('hidden'); $('doneBox').classList.remove('hidden');
  $('doneStats').textContent = session.firstTryRight + '/' + session.seen + ' correct on first try';
  refreshStats();
}
function summary() {
  const lines = ['Spaced-review session — Bayesian optimization — ' + new Date().toISOString().slice(0, 10)];
  lines.push('First-try score: ' + session.firstTryRight + '/' + session.seen);
  if (session.wrong.length) lines.push('Missed: ' + session.wrong.map(w => '"' + w + '"').join('; '));
  const weak = BANK.filter(q => (state[q.id] || {}).lapses > 1).map(q => '"' + q.question.replace(/<[^>]*>/g, '') + '"');
  if (weak.length) lines.push('Chronic weak spots (2+ lapses): ' + weak.join('; '));
  lines.push('Please re-teach my weak areas from first principles, then quiz me harder on them.');
  return lines.join('\\n');
}
$('start').addEventListener('click', startSession);
$('copyRes').addEventListener('click', function () {
  copyText(summary()); this.textContent = 'copied ✓';
  setTimeout(() => { this.textContent = 'copy results for Claude'; }, 1500);
});
$('again').addEventListener('click', () => {
  $('doneBox').classList.add('hidden'); $('home').classList.remove('hidden'); refreshStats();
});
$('reset').addEventListener('click', () => {
  if (confirm('Erase all review progress?')) { state = {}; save(state); refreshStats(); }
});
refreshStats();

(function(){
  var btn=document.getElementById('theme-toggle'); if(!btn) return;
  function mode(){var a=document.documentElement.getAttribute('data-theme');
    if(a) return a;
    return (window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light';}
  function paint(){btn.textContent = mode()==='dark' ? '\\u2600' : '\\u263E';}
  btn.addEventListener('click',function(){
    var next=mode()==='dark'?'light':'dark';
    document.documentElement.setAttribute('data-theme',next);
    try{localStorage.setItem('theme',next);}catch(e){}
    paint(); window.dispatchEvent(new Event('themechange'));});
  paint();
})();
"""

html = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<title>Review Deck: Bayesian Optimization</title>
<script>(function(){try{var t=localStorage.getItem('theme');
if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>""" + CSS + """</style>
</head>
<body>
<div class="masthead"><div class="mast-inner">
<a class="mast-brand" href="https://raghuramshankar.github.io/"><strong>Raghuram</strong> Shankar</a>
<a class="gnav-link" href="https://raghuramshankar.github.io/">about</a>
<a class="gnav-link" href="https://raghuramshankar.github.io/blog/">blog</a>
<button id="theme-toggle" class="theme-toggle" type="button" title="Toggle light/dark theme">&#9789;</button>
</div></div>
<div class="topicbar"><div class="topicbar-inner">
<span class="topic-title"><a href="2026-08-02-bayesian-optimization.html">Bayesian Optimization</a></span>
<nav class="topicbar-nav"><a href="2026-08-02-bayesian-optimization.html">explainer</a>
<a href="2026-08-02-bo-cheatsheet.html">cheat sheet</a>
<a href="https://github.com/raghuramshankar/learning-with-llms/tree/main/tutorials/bayesian-optimization">tutorials</a></nav>
</div></div>
<div class="wrap">
<h1>Review deck: Bayesian optimization</h1>
<p class="sub">Spaced repetition over the explainer's 30 questions (Leitner boxes:
1 → 3 → 7 → 14 → 30 days). Progress is stored in this browser. Companion to
<a href="2026-08-02-bayesian-optimization.html">the full explainer</a>.</p>

<div id="home">
  <div class="stats">
    <div class="stat"><b id="st-due">–</b>due now</div>
    <div class="stat"><b id="st-new">–</b>new</div>
    <div class="stat"><b id="st-learned">–</b>in long boxes</div>
    <div class="stat"><b id="st-total">–</b>total</div>
  </div>
  <button class="btn" id="start">start review</button>
  <button class="btn subtle" id="reset">reset progress</button>
</div>

<div id="play" class="hidden"><div class="card" id="card"></div></div>

<div id="doneBox" class="hidden done">
  <h2>Session complete</h2>
  <p id="doneStats"></p>
  <button class="btn" id="copyRes">copy results for Claude</button>
  <button class="btn subtle" id="again">back to deck</button>
</div>
<footer class="pagefoot">Generated by the <a href="https://github.com/raghuramshankar/learning-with-llms/blob/main/skills/learning-new-topic/SKILL.md">learning-new-topic</a> skill using Claude Fable 5 on __FOOTER_TS__</footer>
</div>
<script>window.__BANK__ = """ + json.dumps(bank) + """;</script>
<script>""" + JS + """</script>
</body>
</html>
"""

import datetime
html = html.replace("__FOOTER_TS__",
                    datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
out = Path(__file__).resolve().parents[2] / "docs" / "2026-08-02-bayesian-optimization-review.html"
out.write_text(html)
print(out, len(bank), "cards")
