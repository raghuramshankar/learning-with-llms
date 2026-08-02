"""Shared Plotly styling for explainer figures.

Theme-neutral: transparent backgrounds + mid-gray text render legibly in both
the light and dark page themes. Import from a topic's make_plots.py via:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    from plot_style import style, save_div, BLUE, GREEN, RED, GREY, FONT, GRID
"""
import plotly.offline

FONT = "#8b95a5"          # readable on light and dark
GRID = "rgba(128,140,160,0.28)"
BLUE = "#5b8def"
GREEN = "#4aa96c"
RED = "#d9705f"
GREY = "rgba(128,140,160,0.55)"


def style(fig, height=360, **kw):
    """Apply the shared theme-neutral layout to a figure."""
    fig.update_layout(
        template=None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=FONT, family="-apple-system, Segoe UI, Roboto, sans-serif", size=13),
        margin=dict(l=55, r=20, t=40, b=45),
        height=height,
        **kw,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID)
    return fig


def save_div(fig, path, div_id):
    """Write a figure as a library-free <div> snippet for splicing into a page.

    The page must load plotly.min.js itself (spec head_script_srcs for the
    shared file, or head_scripts to inline it for a portable single file).
    """
    html = plotly.offline.plot(
        fig, include_plotlyjs=False, output_type="div",
        config={"displayModeBar": False, "responsive": True},
    )
    path.write_text("<div id='" + div_id + "'>" + html + "</div>")
    print("wrote", path.name)


def write_plotly_lib(path):
    """Write plotly.min.js to `path` (commit ONE copy next to the output pages)."""
    path.write_text(plotly.offline.get_plotlyjs())
    print("wrote", path, path.stat().st_size // 1024, "KB")
