#!/usr/bin/env python3
"""Verify cited papers against arXiv using BATCHED `OR` queries.

One request covers ~8 titles, so the whole bibliography costs ~7 requests
instead of ~53 — far friendlier to the rate limiter. Results merge into the
same survey/verified.json cache.
"""
import json, sys, time, urllib.parse, urllib.request, xml.etree.ElementTree as ET
from pathlib import Path

API = "http://export.arxiv.org/api/query?"
NS = {"a": "http://www.w3.org/2005/Atom"}
HERE = Path(__file__).resolve().parent
CACHE = HERE / "verified.json"
BATCH, DELAY = 8, 9.0

sys.path.insert(0, str(HERE))
from verify_arxiv import TITLES            # reuse the one canonical list

db = json.loads(CACHE.read_text()) if CACHE.exists() else {}
todo = [t for t in TITLES if not db.get(t)]
print(f"{len(TITLES)-len(todo)} cached, {len(todo)} to fetch "
      f"in {(len(todo)+BATCH-1)//BATCH} batches\n", flush=True)

def norm(s):
    return " ".join(s.lower().replace("-", " ").split())

def fetch(q, tries=6):
    url = API + urllib.parse.urlencode(
        {"search_query": q, "max_results": 60, "sortBy": "relevance"})
    wait = DELAY
    for a in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                return ET.fromstring(r.read())
        except Exception as e:
            if a == tries - 1: raise
            wait *= 1.7
            print(f"   retry {a+1} ({e}) in {wait:.0f}s", flush=True)
            time.sleep(wait)

for b in range(0, len(todo), BATCH):
    chunk = todo[b:b + BATCH]
    q = " OR ".join('ti:"%s"' % t for t in chunk)
    print(f"batch {b//BATCH+1}: {len(chunk)} titles", flush=True)
    try:
        root = fetch(q)
    except Exception as e:
        print("  FAILED:", e, flush=True); continue
    entries = []
    for e in root.findall("a:entry", NS):
        entries.append({
            "title": " ".join(e.find("a:title", NS).text.split()),
            "id": e.find("a:id", NS).text.rsplit("/", 1)[-1],
            "published": e.find("a:published", NS).text[:10],
            "first_author": e.find("a:author/a:name", NS).text})
    for t in chunk:
        hit = next((x for x in entries if norm(x["title"]) == norm(t)), None)
        if hit is None:
            hit = next((x for x in entries
                        if norm(t)[:45] in norm(x["title"])), None)
            if hit: hit = dict(hit, match="FUZZY")
        else:
            hit = dict(hit, match="EXACT")
        db[t] = hit
        print("   %-5s %-13s %-10s %s" %
              (hit["match"] if hit else "NONE", hit["id"] if hit else "-",
               hit["published"] if hit else "-", t[:58]), flush=True)
    CACHE.write_text(json.dumps(db, indent=1))
    time.sleep(DELAY)

ok = [t for t in TITLES if db.get(t)]
print(f"\nresolved {len(ok)}/{len(TITLES)}")
for t in TITLES:
    if not db.get(t): print("  MISSING:", t)
