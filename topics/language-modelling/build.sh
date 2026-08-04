#!/bin/bash
# Rebuild every artifact for this topic into ../../docs.
# Usage: ./build.sh            (pages reference the shared docs/plotly.min.js)
#        ./build.sh --inline   (embed plotly for a portable single-file page)
# Run `python3 make_plots.py` first only when the figures themselves change —
# it recomputes the numpy sweeps and refreshes plots/.
set -e
cd "$(dirname "$0")"
python3 build_spec.py "$@"
python3 ../../tools/render.py spec.json -o ../../docs
python3 build_review.py
python3 make_anki.py
python3 make_cheatsheet.py
echo "done → docs/"
