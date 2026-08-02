#!/usr/bin/env python3
"""Export the explainer's quiz bank as an Anki .apkg deck."""
import json
import random
from pathlib import Path

import genanki

spec = json.loads((Path(__file__).parent / "spec.json").read_text())

model = genanki.Model(
    1607392320,
    "Bayesian Optimization MCQ",
    fields=[{"name": "Question"}, {"name": "Options"}, {"name": "Answer"}, {"name": "Why"}],
    templates=[{
        "name": "mcq",
        "qfmt": "<div style='font-size:18px'>{{Question}}</div><hr>{{Options}}",
        "afmt": "{{FrontSide}}<hr id=answer><b>{{Answer}}</b><br><br>"
                "<div style='text-align:left;font-size:14px'>{{Why}}</div>",
    }],
    css=".card { font-family: -apple-system, sans-serif; font-size: 16px; "
        "text-align: center; color: black; background-color: white; }",
)

deck = genanki.Deck(2059400111, "Bayesian Optimization (from the explainer)")
rng = random.Random(7)

for s in spec["sections"]:
    for q in s.get("quiz") or []:
        opts = list(q["options"])
        rng.shuffle(opts)
        letters = "ABCD"
        listing = "<br>".join(
            f"<b>{letters[i]}.</b> {o['text']}" for i, o in enumerate(opts))
        ans_i = next(i for i, o in enumerate(opts) if o.get("correct"))
        answer = f"{letters[ans_i]}. {opts[ans_i]['text']}"
        why = "<br><br>".join(
            f"<b>{letters[i]}. {'✓' if o.get('correct') else '✗'}</b> {o.get('explanation', '')}"
            for i, o in enumerate(opts))
        deck.add_note(genanki.Note(
            model=model,
            fields=[f"[{s['title']}] {q['question']}", listing, answer, why],
        ))

out = Path(__file__).resolve().parents[2] / "docs" / "bayesian-optimization.apkg"
genanki.Package(deck).write_to_file(str(out))
print(out, len(deck.notes), "notes")
