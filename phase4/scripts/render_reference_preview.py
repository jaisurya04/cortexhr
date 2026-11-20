"""Render the dashboard with the DESIGN-REFERENCE sample values (the numbers printed in the approved screenshot) and screenshot it at
1365x768 so it can be compared with the reference image. These are NOT real company numbers; the live console never uses them.

    python scripts/render_reference_preview.py --screenshot [--hour 9] [--no-welcome]
Needs only Playwright + Chromium (no Phase 1-3 data, no FastAPI).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cortexhr_phase4.console import dashboard  # noqa: E402
from cortexhr_phase4.console.design_reference import REFERENCE_OVERVIEW, REFERENCE_USER  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", action="store_true")
    ap.add_argument("--width", type=int, default=1365)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--hour", type=int, default=9, help="IST hour to preview (test hook ?cx_hour=)")
    ap.add_argument("--no-welcome", action="store_true")
    ap.add_argument("--name", default="reference_preview")
    ap.add_argument("--outdir", default=str(ROOT / "outputs" / "console"))
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.name}.html"
    f.write_text(dashboard.build(REFERENCE_OVERVIEW, [], REFERENCE_USER, mode="full", show_welcome=not a.no_welcome), encoding="utf-8")
    print("wrote", f)
    if a.screenshot:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": a.width, "height": a.height})
            errs = []
            pg.on("pageerror", lambda e: errs.append(str(e)))
            pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
            pg.goto(f.as_uri() + f"?cx_hour={a.hour}")
            pg.wait_for_selector("html[data-ready='1']")
            pg.wait_for_timeout(2200)
            png = out / f"{a.name}_{a.width}x{a.height}.png"
            pg.screenshot(path=str(png))
            b.close()
        print("screenshot", png, "console/page errors:", errs or "none")


if __name__ == "__main__":
    main()
