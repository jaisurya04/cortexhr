"""Day 19: the dashboard canvas, rendered in a real browser (Chromium via Playwright) with real data. Skipped, and reported as skipped, if Playwright is missing."""
import json
import tempfile
from pathlib import Path

from _common import registry, skip

from cortexhr_phase4.console import charts, dashboard
from cortexhr_phase4.services import handlers as h

_CACHE = {}


def _html(mode="full", welcome=True):
    r = registry()
    return dashboard.build(h.overview(r), h.attrition_list(r, limit=500)["employees"], mode=mode, show_welcome=welcome)


def _browser():
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        skip("playwright is not installed in this environment")
    pw = sync_playwright().start()
    try:
        return pw, pw.chromium.launch()
    except Exception as exc:  # no browser binary
        pw.stop()
        skip(f"chromium could not be launched: {type(exc).__name__}")


def _page(b, html, w=1365, h_=768, name="t"):
    f = Path(tempfile.mkdtemp()) / f"{name}.html"
    f.write_text(html, encoding="utf-8")
    pg = b.new_page(viewport={"width": w, "height": h_})
    errors = []
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    pg.goto(f.as_uri())
    pg.wait_for_selector("html[data-ready='1']")
    return pg, errors


def test_no_demo_values_in_the_generated_page():
    page = _html()
    data = page[page.index("window.CX=") + 10: page.index(";</script>")]
    payload = json.loads(data)
    flat = json.dumps(payload["overview"])
    for demo in ("12480", "12,480", "8.92"):
        assert demo not in flat, f"reference-screenshot demo value {demo} must not appear"
    assert payload["overview"]["kpis"]["total_employees"]["value"] == 441.0


def test_desktop_layout_kpis_and_no_console_errors():
    pw, b = _browser()
    try:
        pg, errors = _page(b, _html(welcome=False))
        assert pg.evaluate("document.documentElement.scrollWidth") <= 1365
        box = pg.evaluate("(()=>{const r=document.getElementById('cx-stage').getBoundingClientRect();return [r.left,r.width,r.height]})()")
        assert abs(box[0] - 167) < 1 and abs(box[1] - 1198) < 1 and abs(box[2] - 768) < 1, box
        assert pg.inner_text("#v-emp") == "441" and pg.inner_text("#v-open") == "29" and pg.inner_text("#v-risk").endswith("%") and pg.inner_text("#v-sal").startswith("\u20b9")
        assert "53,925" in pg.inner_text("#cx-mp") and "0.983" in pg.inner_text("#cx-mp") and "1,15,094" in pg.inner_text("#cx-mp")
        assert pg.inner_text("#lg-high") == "10.0%" and pg.inner_text("#lg-low") == "70.0%"
        assert pg.inner_text(".cx-nav a.on").strip() == "Dashboard" and pg.locator(".cx-nav a").count() == 8
        assert not errors, errors
        pg.screenshot(path=str(Path(__file__).resolve().parents[1] / "outputs" / "console" / "test_desktop_1365x768.png"))
    finally:
        b.close(); pw.stop()


def test_welcome_popup_opens_closes_and_is_remembered():
    pw, b = _browser()
    try:
        pg, errors = _page(b, _html(welcome=True))
        assert pg.is_visible("#cx-welcome-bg") and "Jaisurya" in pg.inner_text("#cx-wtitle")
        assert pg.inner_text("#cx-wtitle").split(",")[0] in ("Good Morning", "Good Afternoon", "Good Evening", "Good Night")
        pg.wait_for_timeout(900)   # let the 0.6 s entrance animation (translateY/scale) finish before measuring
        box = pg.evaluate("(()=>{const r=document.querySelector('.cx-welcome').getBoundingClientRect();return [r.left+r.width/2,r.top+r.height/2]})()")
        assert abs(box[0] - 682.5) < 8 and abs(box[1] - 392) < 8, f"popup must be centred on the 1365x768 screen: {box}"
        pg.click("#cx-welcome-x")
        assert not pg.is_visible("#cx-welcome-bg")
        pg.reload(); pg.wait_for_selector("html[data-ready='1']")
        assert not pg.is_visible("#cx-welcome-bg"), "dismissed greeting must not reappear in the same session"
        pg.click("#cx-btn-welcome")
        assert pg.is_visible("#cx-welcome-bg")
        pg.keyboard.press("Escape")
        assert not pg.is_visible("#cx-welcome-bg")
        assert not errors, errors
    finally:
        b.close(); pw.stop()


def test_period_search_menus_and_modals_work():
    pw, b = _browser()
    try:
        pg, errors = _page(b, _html(welcome=False))
        six = pg.evaluate("document.querySelectorAll('#cx-wf circle').length")
        pg.select_option("#cx-period", "12")
        assert pg.evaluate("document.querySelectorAll('#cx-wf circle').length") == 12 and six == 6
        pg.select_option("#cx-period", "3")
        assert pg.evaluate("document.querySelectorAll('#cx-wf circle').length") == 3
        pg.fill("#cx-q", "EMP-0002")
        assert pg.is_visible("#cx-results") and "EMP-0002" in pg.inner_text("#cx-results")
        pg.click("#cx-results .cx-res >> nth=0")
        assert pg.is_visible("#cx-modal-bg") and "12-month risk score" in pg.inner_text("#cx-modal-body")
        pg.keyboard.press("Escape")
        pg.fill("#cx-q", "finance")
        assert pg.locator("#cx-results .cx-res").count() >= 1
        pg.keyboard.press("Escape")
        pg.click("#cx-bell")
        assert pg.is_visible("#cx-bell-menu") and "PSI" in pg.inner_text("#cx-bell-menu")
        pg.click("#lnk-att")
        assert "Highest attrition risk" in pg.inner_text("#cx-modal-body") and pg.locator("#cx-modal-body tbody tr").count() == 12
        pg.keyboard.press("Escape")
        pg.click("#lnk-perf")
        assert "Ridge" in pg.inner_text("#cx-modal-body")
        pg.keyboard.press("Escape")
        pg.click("#btn-feat")
        assert pg.locator("#cx-modal-body tbody tr").count() == 10
        assert not errors, errors
    finally:
        b.close(); pw.stop()


def test_mobile_layout_has_no_horizontal_overflow_and_popup_fits():
    pw, b = _browser()
    try:
        pg, errors = _page(b, _html(welcome=True), 390, 844)
        assert pg.evaluate("document.documentElement.scrollWidth") <= 390, "horizontal scroll on a phone"
        r = pg.evaluate("(()=>{const r=document.querySelector('.cx-welcome').getBoundingClientRect();return [r.left,r.right,r.top,r.bottom]})()")
        assert r[0] >= 0 and r[1] <= 391 and r[2] >= 0 and r[3] <= 844, r
        pg.click("#cx-welcome-x")
        assert pg.is_visible("#v-emp") and pg.inner_text("#v-emp") == "441"
        pg.screenshot(path=str(Path(__file__).resolve().parents[1] / "outputs" / "console" / "test_mobile_390x844.png"))
        assert not errors, errors
    finally:
        b.close(); pw.stop()


def test_charts_render_and_hover_tooltip_works():
    pw, b = _browser()
    try:
        r = registry()
        f = h.forecast(r)
        pg, errors = _page(b, charts.forecast_chart_html("t", f["history"], f["forecast"], f["events"]), 1000, 440)
        assert pg.locator("#c path").count() >= 3 and pg.locator("#c circle").count() == 6
        pg.mouse.move(700, 200)
        assert pg.is_visible("#tip") and "Actual" in pg.inner_text("#tip") or "Forecast" in pg.inner_text("#tip")
        pg.uncheck("#band")
        n_off = pg.locator("#c path").count()
        pg.check("#band")
        assert pg.locator("#c path").count() == n_off + 1, "the 80% range is a toggleable band"
        m = h.monitoring(r)
        pg2, e2 = _page(b, charts.drift_chart_html(m["report"]["components"]["drift"]["table"], m["history"]), 1000, 400)
        assert pg2.locator("#c rect").count() == 8
        assert not errors and not e2, errors + e2
    finally:
        b.close(); pw.stop()


def test_greeting_and_account_chip_use_the_signed_in_name_not_a_fixed_one():
    """Whoever signs in is greeted by name. A dashboard built for Priya never contains the default name, and each person gets their own popup."""
    r = registry()
    page = dashboard.build(h.overview(r), h.attrition_list(r, limit=500)["employees"], {"name": "Priya Sharma", "role": "Admin"}, mode="embed", show_welcome=True)
    assert "Good Morning, Priya!" in page and '"first": "Priya"' in page.replace('"first":"Priya"', '"first": "Priya"')
    assert "Priya Sharma" in page and "Jaisurya" not in page
    js = (Path(dashboard.__file__).parent / "static" / "dashboard.js").read_text(encoding="utf-8")
    assert 'cx_welcome_seen:" + D.user.first' in js                      # the "already shown" memory is per person
    app = (Path(dashboard.__file__).parent / "app.py").read_text(encoding="utf-8")
    assert "_sign_in()" in app and '"Jaisurya"' not in app                # no hard-coded name left in the console


def test_streamlit_embed_mode_initialises_without_errors_and_shows_the_hiring_map():
    """Regression: the embed page (what Streamlit shows) has no sidebar promo button; a missing element must not stop init() and leave a blank page."""
    pw, b = _browser()
    try:
        pg, errors = _page(b, _html(mode="embed", welcome=False))
        assert not errors, errors
        assert pg.locator(".cx-hirep").count() == 1 and pg.locator("#lnk-hire").count() == 1
        label = pg.inner_text("#cx-hires")
        assert "Hires, last 6 months" in label and "Global Hires" not in label, label
        hires = h.overview(registry())["hiring"]["hires_last_6m"]
        assert str(hires) in label.replace(",", ""), (hires, label)
        assert pg.inner_text("#cx-regions").startswith("Top locations")
        assert pg.inner_text("#v-emp") == "441"
    finally:
        b.close(); pw.stop()
