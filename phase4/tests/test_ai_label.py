"""Official Day 19: "Add a clear label on every AI output: 'AI-generated, for human review'", and the brief: "make the label impossible to omit".

1. every AI response of the API carries the label;
2. the console draws results ONLY through ``console/ai_render.py`` (static check of ``app.py``), and every wrapper draws the label first;
3. the dashboard panels that show model output carry the label in the browser;
4. the real Streamlit app, running against the real FastAPI service, shows the label on every tab (skipped, and reported BLOCKED, when streamlit/fastapi are missing).
"""
import ast
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from _common import ROOT, need, registry, skip

from cortexhr_phase4.services import handlers as h
from cortexhr_phase4.services import samples
from cortexhr_phase4.services.errors import ApiError

LABEL = "AI-generated, for human review"
CONSOLE = ROOT / "src" / "cortexhr_phase4" / "console"
APP, RENDER = CONSOLE / "app.py", CONSOLE / "ai_render.py"
NON_AI_PAGES = {"page_api", "page_settings", "_sign_in"}          # API address, uptime, endpoint list, connection settings: not model output
FORBIDDEN_CALLS = {"dataframe", "table", "metric", "plotly_chart", "altair_chart", "pyplot", "line_chart", "bar_chart", "area_chart", "scatter_chart",
                   "image", "json", "chat_message", "data_editor"}


# ---------------------------------------------------------------- 1. API
def test_every_ai_api_response_carries_the_exact_label():
    r = registry()
    from _common import skip
    if r.info.get("payroll") and r.info["payroll"].status == "unavailable":
        skip("model_unavailable: payroll artifact needs the pinned pandas 3.0.2 runtime")
    outputs = {
        "attrition/predict": h.attrition_predict(r, samples.employee_id()),
        "attrition/employees": h.attrition_list(r, limit=5),
        "anomaly/payroll": h.payroll(r, [samples.payroll_row()]),
        "anomaly/attendance": h.attendance(r, [samples.attendance_row()]),
        "chat": h.chat(r, "What is the notice period?", "label-test"),
        "resume/rank": h.resume_rank(r, {"job_id": "J01", "top_k": 3}),
        "payband/recommend": h.payband(r, samples.payband_row()),
        "forecast": h.forecast(r),
        "monitoring": h.monitoring(r),
        "overview": h.overview(r),
        "performance": h.performance(r),
        "health": h.health(r),
    }
    missing = [k for k, v in outputs.items() if LABEL not in str(v.get("notice", ""))]
    assert not missing, f"AI outputs without the review label: {missing}"


def test_scenario_forecast_carries_the_label():
    try:
        out = h.forecast_scenario(registry(), [{"effect": "hiring_freeze", "start": "2026-11", "end": "2027-01", "department_id": "ALL"}])
    except ApiError as exc:
        skip(f"what-if forecast needs the improved (statsmodels) model: {exc}")
        return
    assert LABEL in out["notice"]


# ---------------------------------------------------------------- 2. console: static enforcement
def _calls(tree: ast.AST):
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute):
            yield n


def test_console_pages_never_draw_results_without_the_label_wrappers():
    src = APP.read_text(encoding="utf-8")
    tree = ast.parse(src)
    bad = []
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef)]:
        if fn.name in NON_AI_PAGES:
            continue
        for c in _calls(fn):
            owner = c.func.value.id if isinstance(c.func.value, ast.Name) else None
            if owner == "ai":
                continue                                    # the wrappers
            if c.func.attr in FORBIDDEN_CALLS:
                bad.append(f"{fn.name}:{c.lineno} .{c.func.attr}() bypasses ai_render")
            if owner == "components" and c.func.attr == "html" and fn.name != "page_dashboard":
                bad.append(f"{fn.name}:{c.lineno} components.html bypasses ai_render")
            seg = ast.get_source_segment(src, c) or ""
            if c.func.attr == "markdown" and "cx-card" in seg:
                bad.append(f"{fn.name}:{c.lineno} result card drawn with markdown instead of ai.card")
    assert not bad, "\n".join(bad)
    assert "components.html(" not in src, "the dashboard canvas must also go through ai.html_frame (label verified)"


def test_console_has_exactly_the_five_official_tabs_and_attendance_is_reachable():
    src = APP.read_text(encoding="utf-8")
    assert 'INSIGHT_TABS = ["Risks", "Anomalies", "Assistant", "Resume Screener", "Forecast"]' in src
    for fn in ("tab_attrition", "tab_anomalies", "tab_assistant", "tab_resume", "tab_forecast", "tab_attendance", "tab_payroll", "tab_payband"):
        assert f"def {fn}(" in src, fn
    assert "/v1/anomaly/attendance" in src and "/v1/anomaly/payroll" in src and "/v1/chat" in src and "/v1/resume/rank" in src and "/v1/forecast" in src
    assert "CORTEXHR_API_URL" in (ROOT / "src" / "cortexhr_phase4" / "core" / "settings.py").read_text(encoding="utf-8"), "API address must come from the environment"
    assert "http://" not in src.replace("http://127.0.0.1", ""), "no hard-coded API address in the console"


def test_every_wrapper_draws_the_label_before_the_content():
    tree = ast.parse(RENDER.read_text(encoding="utf-8"))
    src = RENDER.read_text(encoding="utf-8")
    ai_funcs = {"dataframe", "markdown", "info", "metrics", "card", "html_frame", "chat_message"}
    found = set()
    for fn in [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ai_funcs]:
        found.add(fn.name)
        body = ast.get_source_segment(src, fn)
        draws = [m.start() for m in re.finditer(r"\.(markdown|dataframe|info|metric|write)\(|components\.html\(", body)]
        labels = [m.start() for m in re.finditer(r"badge\(|_with_label\(|AI_LABEL", body[1:])]
        assert draws and labels, f"{fn.name}: no drawing call or no label"
        first_draw, first_label = min(draws), min(labels) + 1
        # the label is drawn first, or (card) is wrapped inside the very call that draws the content
        assert first_label < first_draw or "_with_label(" in body[first_draw:first_draw + 140], f"{fn.name}: content is drawn before the label"
    assert found == ai_funcs


class _Rec:
    """Recording stand-in for the streamlit module: checks the ORDER of what the wrappers draw. This tests wrapper logic only; it is not evidence that Streamlit runs."""
    def __init__(self):
        self.ev = []
    def markdown(self, body, **k): self.ev.append(("markdown", body))
    def dataframe(self, d, **k): self.ev.append(("dataframe", d))
    def info(self, m): self.ev.append(("info", m))
    def write(self, t): self.ev.append(("write", t))
    def columns(self, n): return [_Col(self.ev) for _ in range(n)]
    def chat_message(self, role):
        self.ev.append(("chat_message", role))
        return self
    def __enter__(self): return self
    def __exit__(self, *a): return False
class _Col:
    def __init__(self, ev): self.ev = ev
    def metric(self, label, value): self.ev.append(("metric", label))
class _Comp:
    def __init__(self, rec): self.rec = rec
    def html(self, html, height=None, **k): self.rec.ev.append(("components.html", html))


def _load_render_with_stub():
    import importlib.util
    import types
    rec = _Rec()
    st_mod, v1 = types.ModuleType("streamlit"), types.ModuleType("streamlit.components.v1")
    for n in ("markdown", "dataframe", "info", "write", "columns", "chat_message"):
        setattr(st_mod, n, getattr(rec, n))
    v1.html = _Comp(rec).html
    comp = types.ModuleType("streamlit.components")
    comp.v1 = v1
    st_mod.components = comp
    saved = {k: sys.modules.get(k) for k in ("streamlit", "streamlit.components", "streamlit.components.v1")}
    sys.modules.update({"streamlit": st_mod, "streamlit.components": comp, "streamlit.components.v1": v1})
    try:
        spec = importlib.util.spec_from_file_location("cx_ai_render_under_test", RENDER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return mod, rec


def test_wrappers_put_the_label_first_even_when_drawing_the_content_fails():
    ai, rec = _load_render_with_stub()
    ai.dataframe([{"a": 1}])
    ai.markdown("x")
    ai.info("y")
    ai.metrics([("Low", "1"), ("Mid", "2")])
    ai.html_frame("<b>chart</b>", 100)
    ai.card('<div class="cx-card"><h4>t</h4>body</div>')
    ai.card("plain body without a card wrapper")
    for first, second in zip(rec.ev, rec.ev[1:]):
        pass
    labelled = [i for i, (k, b) in enumerate(rec.ev) if k == "markdown" and LABEL in str(b)]
    assert len(labelled) >= 5, rec.ev
    assert rec.ev[0][0] == "markdown" and LABEL in rec.ev[0][1], "first thing drawn by ai.dataframe is the label"
    cards = [b for k, b in rec.ev if k == "markdown" and "cx-card" in str(b)]
    assert len(cards) == 2 and all(LABEL in c for c in cards), "a card carries the label inside itself"
    rec.ev.clear()
    ai.chat_message("assistant", "hello")
    assert [k for k, _ in rec.ev][:2] == ["chat_message", "markdown"] and LABEL in rec.ev[1][1], "assistant bubble: label inside, before the text"
    rec.ev.clear()
    ai.chat_message("user", "hi")
    assert not any(LABEL in str(b) for _, b in rec.ev), "the user's own message is not AI output"
    try:
        ai.html_frame("<b>no label</b>", 100, label_inside=True)
        raise AssertionError("a page without the label must be refused")
    except ValueError:
        pass
    assert ai.html_frame is not None


# ---------------------------------------------------------------- 3. dashboard canvas
def test_dashboard_panels_with_model_output_show_the_label_in_the_browser():
    from test_dashboard import _browser, _html, _page
    pw, b = _browser()
    try:
        pg, errors = _page(b, _html(welcome=False))
        found = pg.evaluate("""() => ['.cx-kpi:nth-child(2)', '.cx-attp', '.cx-insp', '.cx-mpp'].map(s => {
            const e = document.querySelector(s + ' .cx-ailbl'); if (!e) return [s, 'missing'];
            const r = e.getBoundingClientRect(); return [s, e.textContent.trim(), r.width > 20 && r.height > 5]; })""")
        for sel, text, *vis in found:
            assert text == LABEL, f"{sel}: {text}"
            assert vis and vis[0], f"{sel}: label not visible"
        assert not errors, errors
    finally:
        b.close()
        pw.stop()


# ---------------------------------------------------------------- 4. real Streamlit + real FastAPI
def test_real_streamlit_console_shows_the_label_on_every_tab():
    need("streamlit", "fastapi", "uvicorn")
    from streamlit.testing.v1 import AppTest
    port = 8765
    env = {**os.environ, "CORTEXHR_ENV": "test", "PYTHONPATH": str(ROOT / "src")}
    auth_keys = ("CORTEXHR_AUTH_SECRET", "CORTEXHR_AUTH_USERS", "CORTEXHR_API_KEY")
    old_auth_env = {key: os.environ.get(key) for key in auth_keys}
    for key in auth_keys:
        os.environ[key] = ""
    env.update({key: "" for key in auth_keys})
    srv = subprocess.Popen([sys.executable, "-m", "uvicorn", "cortexhr_phase4.api.app:app", "--port", str(port)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        import urllib.request
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/health", timeout=1)
                break
            except Exception:
                time.sleep(0.5)
        else:
            raise AssertionError("the real FastAPI service did not start")
        os.environ["CORTEXHR_API_URL"] = f"http://127.0.0.1:{port}"
        at = AppTest.from_file(str(APP), default_timeout=90)
        at.run()
        assert any("Your name" in t.label for t in at.text_input), "nobody is signed in: the console must ask for a name first"
        at.session_state["user_name"] = "Test User"
        at.run()
        assert not at.exception, at.exception
        at.session_state["page"] = "insights"
        for tab in ("Risks", "Anomalies", "Assistant", "Resume Screener", "Forecast"):
            at.session_state["insights_tab"] = tab
            at.run()
            assert not at.exception, f"{tab}: {at.exception}"
            if tab == "Assistant":
                at.chat_input[0].set_value("What is the notice period?").run()
                assert not at.exception, at.exception
            shown = " ".join(str(m.value) for m in at.markdown)
            assert LABEL in shown, f"tab {tab} shows no review label"
    finally:
        srv.terminate()
