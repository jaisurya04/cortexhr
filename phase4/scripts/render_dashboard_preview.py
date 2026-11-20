"""Render the dashboard from the real handlers (no web server) to outputs/console/dashboard_preview.html, and optionally screenshot it.

    python phase4/scripts/render_dashboard_preview.py --screenshot   # needs: pip install playwright && playwright install chromium
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cortexhr_phase4.console import dashboard  # noqa: E402
from cortexhr_phase4.core.settings import load_settings  # noqa: E402
from cortexhr_phase4.services import handlers  # noqa: E402
from cortexhr_phase4.services.registry import Registry  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot", action="store_true")
    ap.add_argument("--width", type=int, default=1365)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--no-welcome", action="store_true")
    ap.add_argument("--name", default="dashboard_preview")
    a = ap.parse_args()
    s = load_settings()
    reg = Registry(s).load_all()
    doc = dashboard.build(handlers.overview(reg), handlers.attrition_list(reg, limit=500)["employees"], mode="full", show_welcome=not a.no_welcome)
    out = s.output_dir / "console"
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{a.name}.html"
    f.write_text(doc, encoding="utf-8")
    print("wrote", f)
    if a.screenshot:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            b = p.chromium.launch()
            pg = b.new_page(viewport={"width": a.width, "height": a.height})
            pg.goto(f.as_uri())
            pg.wait_for_selector("html[data-ready='1']")
            pg.wait_for_timeout(1900)
            png = out / f"{a.name}_{a.width}x{a.height}.png"
            pg.screenshot(path=str(png))
            b.close()
        print("screenshot", png)


if __name__ == "__main__":
    main()
