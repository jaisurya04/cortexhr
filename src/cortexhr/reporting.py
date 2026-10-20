"""Small helpers for tables, markdown, figures and run metadata."""
from __future__ import annotations
import json, platform, sys, textwrap
from importlib import metadata
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from . import config as C


def md_table(df: pd.DataFrame, floatfmt="{:.3f}", max_rows=None) -> str:
    d = df.head(max_rows) if max_rows else df
    cols = list(d.columns)
    def fmt(v):
        if isinstance(v, (float, np.floating)): return "" if np.isnan(v) else floatfmt.format(v)
        return str(v).replace("|", "/").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, cols)) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    rows += ["| " + " | ".join(fmt(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)


def save_table(df: pd.DataFrame, name: str) -> str:
    p = C.reports_dir() / "tables" / f"{name}.csv"; df.to_csv(p, index=False); return str(p)


def save_fig(fig, name: str) -> str:
    p = C.reports_dir() / "figures" / f"{name}.png"; fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close(fig); return str(p)


def table_as_image(df: pd.DataFrame, name: str, col_widths=None, wrap=None, fontsize=7, title=None, width_in=20) -> str:
    """Screenshot-ready PNG of a table; text is wrapped per column and row heights follow the wrapped text."""
    n = len(df.columns); col_widths = col_widths or [1 / n] * n
    cpi = 72 / (fontsize * 0.58)                       # characters per inch at this font size
    wraps = [max(8, int(w * width_in * cpi * 0.92)) for w in col_widths]
    cells = [[textwrap.fill(str(v), wraps[j]) for j, v in enumerate(r)] for r in df.itertuples(index=False)]
    heads = [textwrap.fill(str(c).replace("_", " "), wraps[j]) for j, c in enumerate(df.columns)]
    line_in = fontsize * 1.45 / 72
    row_h = [max(x.count("\n") + 1 for x in r) * line_in + 0.12 for r in cells]
    head_h = max(h.count("\n") + 1 for h in heads) * line_in + 0.12
    total = sum(row_h) + head_h + (0.5 if title else 0.1)
    fig = plt.figure(figsize=(width_in, total)); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    t = ax.table(cellText=cells, colLabels=heads, loc="upper left", cellLoc="left", colWidths=col_widths, bbox=[0, 0, 1, 1 - (0.5 if title else 0.1) / total])
    t.auto_set_font_size(False); t.set_fontsize(fontsize)
    avail = 1 - (0.5 if title else 0.1) / total
    for (r, c), cell in t.get_celld().items():
        h = head_h if r == 0 else row_h[r - 1]
        cell.set_height(h / total); cell.set_edgecolor("#999"); cell.PAD = 0.02
        if r == 0: cell.set_facecolor("#dbe5f1"); cell.set_text_props(fontweight="bold")
    if title: fig.text(0.005, 1 - 0.28 / total, title, fontsize=11, fontweight="bold", va="center")
    return save_fig(fig, name)


def package_versions() -> dict:
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for p in ["pandas", "numpy", "scikit-learn", "scipy", "matplotlib", "joblib", "openpyxl"]:
        try: out[p] = metadata.version(p)
        except metadata.PackageNotFoundError: out[p] = "n/a"
    return out


def save_json(obj, name: str) -> str:
    p = C.project_root() / "artifacts" / "metadata" / f"{name}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, default=str)); return str(p)
