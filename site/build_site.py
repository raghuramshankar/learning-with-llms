#!/usr/bin/env python3
"""Generate the project website (docs/index.html) from site/topics.json.

Add a topic: append an entry to topics.json, run this script, commit docs/.
"""
import html
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOCS = HERE.parent / "docs"
REPO_URL = "https://github.com/raghuramshankar/learning-with-llms"

topics = json.loads((HERE / "topics.json").read_text())

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<meta name="description" content="Interactive deep dives into technical topics: intuition, full mathematics, live simulations, spaced repetition, and build-it-yourself labs.">
<title>Learning with LLMs</title>
<script>(function(){try{var t=localStorage.getItem('theme');
if(t==='light'||t==='dark')document.documentElement.setAttribute('data-theme',t);}catch(e){}})();</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Merriweather:ital,wght@0,300;0,400;0,700;1,400&display=swap" rel="stylesheet">
<style>
:root {
  --bg:#ffffff; --fg:#000000; --muted:#555e66; --line:#e1e5ea;
  --accent:#063c92; --accent-soft:#eef2fb; --ok:#168551; --ok-soft:#e9f5ee;
  --card:#f8f9fb; --hero-glow:rgba(6,60,146,0.07);
}
:root[data-theme="dark"] {
  --bg:#000000; --fg:#e5e8ee; --muted:#9aa3b0; --line:#2a313b;
  --accent:#7d9fdd; --accent-soft:#1b2540; --ok:#57b98a; --ok-soft:#16301f;
  --card:#181c23; --hero-glow:rgba(125,159,221,0.08);
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg:#000000; --fg:#e5e8ee; --muted:#9aa3b0; --line:#2a313b;
  --accent:#7d9fdd; --accent-soft:#1b2540; --ok:#57b98a; --ok-soft:#16301f;
  --card:#181c23; --hero-glow:rgba(125,159,221,0.08);
} }
.theme-toggle { position:fixed; top:.9rem; right:1rem; z-index:50;
  border:1px solid var(--line); background:var(--bg); color:var(--fg);
  border-radius:8px; cursor:pointer; padding:.1rem .55rem; font:inherit; font-size:.8rem; }
.theme-toggle:hover { border-color:var(--accent); }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg);
  font:16px/1.7 Merriweather, Georgia, serif;
  -webkit-font-smoothing:antialiased; }
.wrap { max-width:960px; margin:0 auto; padding:0 1.4rem; }
a { color:var(--accent); }

/* hero */
header { position:relative; overflow:hidden; border-bottom:1px solid var(--line);
  background:radial-gradient(900px 420px at 25% -10%, var(--hero-glow), transparent); }
.hero { padding:4.5rem 0 3.5rem; }
.eyebrow { font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size:.82rem; color:var(--accent); letter-spacing:.06em; margin-bottom:1rem; }
h1 { font-family:Merriweather, Georgia, serif; font-weight:700; color:var(--accent);
  font-size:clamp(2rem, 5vw, 3rem); line-height:1.15; margin:0 0 1rem; max-width:21ch; }
.lede { color:var(--muted); font-size:1.12rem; max-width:56ch; margin:0 0 1.8rem; }
.cta { display:inline-block; padding:.6rem 1.2rem; border-radius:10px;
  font-weight:600; text-decoration:none; margin:0 .6rem .6rem 0; }
.cta.primary { background:var(--accent); color:var(--bg); }
.cta.ghost { color:var(--accent); border:1.5px solid var(--accent); }
.cta:hover { filter:brightness(1.1); }

/* sections */
section { padding:3rem 0 1rem; }
h2 { font-family:Merriweather, Georgia, serif; font-weight:700;
  font-size:1.3rem; margin:0 0 .4rem; color:var(--accent);
  border-bottom:1px solid var(--accent); padding-bottom:.3rem; }
.secsub { color:var(--muted); margin:0 0 1.6rem; max-width:62ch; }

/* system cards */
.sys { display:grid; grid-template-columns:repeat(auto-fit, minmax(250px, 1fr));
  gap:1rem; }
.sys-card { background:var(--card); border:1px solid var(--line);
  border-radius:14px; padding:1.1rem 1.2rem; }
.sys-card .ico { font-size:1.4rem; }
.sys-card h3 { margin:.4rem 0 .3rem; font-size:1.02rem; }
.sys-card p { margin:0; color:var(--muted); font-size:.92rem; }

/* topic cards */
.topic { background:var(--card); border:1px solid var(--line); border-radius:16px;
  padding:1.5rem 1.6rem; margin:1.1rem 0; transition:border-color .12s; }
.topic:hover { border-color:var(--accent); }
.topic-head { display:flex; align-items:baseline; gap:.8rem; flex-wrap:wrap; }
.topic h3 { font-family:Merriweather, Georgia, serif; font-size:1.25rem; margin:0; }
.topic h3 a { color:var(--fg); text-decoration:none; }
.topic h3 a:hover { color:var(--accent); }
.tdate { color:var(--muted); font-size:.85rem;
  font-family:ui-monospace, Menlo, monospace; }
.tags { margin:.3rem 0 .6rem; }
.tag { display:inline-block; background:var(--accent-soft); color:var(--accent);
  border-radius:999px; padding:.08em .7em; font-size:.76rem; font-weight:600;
  margin-right:.35rem; }
.topic p { color:var(--muted); margin:.2rem 0 1rem; }
.tlinks a { display:inline-block; margin:0 1.1rem .4rem 0; font-size:.95rem;
  font-weight:600; text-decoration:none; }
.tlinks a:hover { text-decoration:underline; }
.tlinks .go { color:var(--ok); }

/* under the hood */
.hood { background:var(--card); border:1px solid var(--line); border-radius:14px;
  padding:1.2rem 1.4rem; font-size:.95rem; color:var(--muted); }
.hood code { background:var(--accent-soft); color:var(--fg);
  border-radius:5px; padding:.06em .4em; font-size:.85em;
  font-family:ui-monospace, Menlo, monospace; }

footer { border-top:1px solid var(--line); margin-top:3rem;
  padding:1.4rem 0 2.5rem; color:var(--muted); font-size:.88rem; }
</style>
</head>
<body>
<button id="theme-toggle" class="theme-toggle" type="button">&#9789; dark</button>
<main class="wrap">

<section id="topics" style="padding-top:2.2rem">
  <h2>Topics</h2>
  <p class="secsub">Newest first. Each one is self-contained.</p>
__TOPIC_CARDS__
</section>

</main>

<footer><div class="wrap">
  Built with <a href="https://claude.com/claude-code">Claude Code</a> &middot;
  <a href="__REPO_URL__">github.com/raghuramshankar/learning-with-llms</a>
</div></footer>

<script>
(function(){
  var btn=document.getElementById('theme-toggle'); if(!btn) return;
  function mode(){var a=document.documentElement.getAttribute('data-theme');
    if(a) return a;
    return (window.matchMedia&&matchMedia('(prefers-color-scheme: dark)').matches)?'dark':'light';}
  function paint(){btn.textContent = mode()==='dark' ? '\\u2600 light' : '\\u263E dark';}
  btn.addEventListener('click',function(){
    var next=mode()==='dark'?'light':'dark';
    document.documentElement.setAttribute('data-theme',next);
    try{localStorage.setItem('theme',next);}catch(e){}
    paint(); window.dispatchEvent(new Event('themechange'));});
  paint();
})();</script>
</body>
</html>
"""

CARD = """  <div class="topic">
    <div class="topic-head">
      <h3><a href="{explainer}">{title}</a></h3>
      <span class="tdate">{date}</span>
    </div>
    <div class="tags">{tags}</div>
    <p>{blurb}</p>
    <div class="tlinks">
      <a href="{review}">Review deck</a>
      <a href="{cheatsheet}">Cheat sheet</a>
      <a href="{lab}">The lab</a>
    </div>
  </div>
"""

cards = ""
for t in sorted(topics, key=lambda x: x["date"], reverse=True):
    cards += CARD.format(
        title=html.escape(t["title"]), date=t["date"],
        blurb=html.escape(t["blurb"]),
        tags="".join(f"<span class='tag'>{html.escape(tag)}</span>" for tag in t.get("tags", [])),
        explainer=t["explainer"], review=t["review"],
        cheatsheet=t["cheatsheet"], anki=t["anki"], lab=t["lab"],
    )

newest = sorted(topics, key=lambda x: x["date"], reverse=True)[0]
out = (TEMPLATE
       .replace("__TOPIC_CARDS__", cards)
       .replace("__FIRST_EXPLAINER__", newest["explainer"])
       .replace("__FIRST_TITLE__", html.escape(newest["title"]))
       .replace("__REPO_URL__", REPO_URL))
(DOCS / "index.html").write_text(out)
print(DOCS / "index.html", "—", len(topics), "topic(s)")
