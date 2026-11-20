"""Builds the CortexHR dashboard (the approved reference design) as one self-contained HTML document.

``mode="embed"`` is the 1198x768 area to the right of the sidebar (used inside Streamlit); ``mode="full"`` adds the sidebar so the
whole 1365x768 composition can be previewed and screenshotted. All numbers arrive in ``overview`` (from ``GET /v1/overview``).
"""
from __future__ import annotations

import base64
import html
import json
import math
from pathlib import Path

from ..core.settings import PHASE4_ROOT

STATIC = Path(__file__).parent / "static"
ASSETS = PHASE4_ROOT / "assets"

NAV = [("dashboard", "Dashboard", '<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>'),
       ("data", "Data Foundation", '<path d="M4 5h16M4 12h16M4 19h16"/><circle cx="7" cy="5" r="1"/><circle cx="7" cy="12" r="1"/><circle cx="7" cy="19" r="1"/>'),
       ("insights", "Insights", '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>'),
       ("fairness", "Fairness & Audit", '<path d="M12 3v18M5 7h14M5 7l-3 7a3.5 3.5 0 0 0 6 0zM19 7l-3 7a3.5 3.5 0 0 0 6 0z"/>'),
       ("models", "Models", '<path d="M12 2 3 7v10l9 5 9-5V7zM3 7l9 5 9-5M12 12v10"/>'),
       ("resume", "Resume Screener", '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>'),
       ("api", "API & Deploy", '<path d="M17.5 19a4.5 4.5 0 1 0-1.4-8.8A6 6 0 1 0 5 14.5 3.5 3.5 0 0 0 6.5 19z"/>'),
       ("settings", "Settings", '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3M4.2 19.8l2.1-2.1M17.7 6.3l2.1-2.1"/>')]

FEATURES = [("Workforce Overview", "Headcount, hires, exits and open roles on the Dashboard."), ("Attrition Risk", "A 12-month risk score and the three main reasons for every active employee."),
            ("Payroll Anomalies", "Unusual payslips flagged for a person to review; never proof of an error."), ("HR Assistant", "Answers HR policy questions and routes to the right intent."),
            ("Resume Screener", "Ranks resumes against a job without seeing name, gender, age, e-mail, city or college tier."), ("Pay-Band Advisor", "A low / mid / high pay range for a role, from job-related inputs only."),
            ("Six-Month Forecast", "Headcount and exits with an 80% range, tested on past data."), ("Fairness and Drift Monitoring", "PSI drift per feature and fairness re-checks, shown as OK / WATCH / ALERT."),
            ("Model Performance", "Honest metrics for every model, with limits written down."), ("API Status", "Live health of the CortexHR API and each model.")]


def _b64(name: str) -> str:
    return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()


def _ribbon(i: int, n: int, x0: int, x1: int, y0: float, amp: float, ph: float, step: int = 14) -> list[tuple[float, float]]:
    return [(x, y0 + amp * math.sin(x / (170.0 + i * 9) + ph + i * .5) * math.cos(x / (390.0 + i * 20) + i * .35) + i * 3.2) for x in range(x0, x1 + 1, step)]


def _waves() -> str:
    """Glowing network-wave background: flowing teal ribbons with bright nodes (top-right and bottom band). Deterministic, no randomness."""
    out, nodes = [], []
    for i in range(18):                                   # top-right ribbons that sweep diagonally down toward the right edge (reference composition)
        pts = [(x, y0 + (x - 760) * (.16 + i * .006) + 22 * math.sin(x / (120.0 + i * 7) + i * .5)) for x, y0 in [(x, -30 + i * 9) for x in range(760, 1201, 12)]]
        d = " ".join(f"{'M' if k == 0 else 'L'}{x} {y:.1f}" for k, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="url(#cxw)" stroke-width="{1.5 - i * .045:.2f}" opacity="{.62 - i * .02:.2f}"/>')
        for k in range(3):
            x, y = pts[(i * 5 + k * 23) % len(pts)]
            nodes.append(f'<circle cx="{x}" cy="{y:.1f}" r="{1.2 + (i + k) % 3 * .5:.1f}" fill="#9ff3dc" opacity=".8"/>')
    for i in range(14):                                   # bottom band, brightest near the middle
        pts = _ribbon(i, 14, 0, 1200, 712 + i * 3.6, 24 + i * 1.6, 1.4 + i * .27, 16)
        d = " ".join(f"{'M' if k == 0 else 'L'}{x} {y:.1f}" for k, (x, y) in enumerate(pts))
        out.append(f'<path d="{d}" fill="none" stroke="url(#cxw2)" stroke-width="{1.35 - i * .05:.2f}" opacity="{.7 - i * .025:.2f}"/>')
        for k in range(2):
            x, y = pts[(i * 7 + k * 31) % len(pts)]
            nodes.append(f'<circle cx="{x}" cy="{y:.1f}" r="{1.1 + (i + k) % 3 * .45:.1f}" fill="#a8f5e0" opacity=".75"/>')
    dust = "".join(f'<circle cx="{(i * 197) % 880 + 300}" cy="{(i * 131) % 120 + 8}" r="1.1" fill="rgba(150,240,215,.55)"/>' for i in range(26))
    defs = ('<defs><linearGradient id="cxw" x1="0" x2="1"><stop offset="0" stop-color="#4fd8bb" stop-opacity="0"/><stop offset=".45" stop-color="#6ef0d2" stop-opacity=".9"/><stop offset="1" stop-color="#2fb89c" stop-opacity=".25"/></linearGradient>'
            '<linearGradient id="cxw2" x1="0" x2="1"><stop offset="0" stop-color="#3cc8ab" stop-opacity=".1"/><stop offset=".55" stop-color="#8ff5de" stop-opacity=".95"/><stop offset="1" stop-color="#3cc8ab" stop-opacity=".2"/></linearGradient></defs>')
    return f'<svg class="cx-waves" viewBox="0 0 1198 768" preserveAspectRatio="none" aria-hidden="true">{defs}<g class="glow">{"".join(out)}{"".join(nodes)}</g>{dust}</svg>'


def _side_waves() -> str:
    out = []
    for i in range(9):
        d = " ".join(f"{'M' if k == 0 else 'L'}{x} {170 + i * 9 + 22 * math.sin(x / 38.0 + i * .6) + (x * .35):.1f}" for k, x in enumerate(range(0, 168, 8)))
        out.append(f'<path d="{d}" fill="none" stroke="rgba(110,235,205,{.5 - i * .04:.2f})" stroke-width="1"/>')
    return f'<svg class="sw" viewBox="0 0 167 300" aria-hidden="true">{"".join(out)}</svg>'


def _defs() -> str:
    return ('<svg class="cx-defs" aria-hidden="true" focusable="false"><defs>'
            '<linearGradient id="cxg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f6dc9c"/><stop offset=".55" stop-color="#d4a95f"/><stop offset="1" stop-color="#b9894a"/></linearGradient>'
            '<linearGradient id="cxc" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#e4a574"/><stop offset=".6" stop-color="#b9774a"/><stop offset="1" stop-color="#8a5232"/></linearGradient></defs></svg>')


def _sun() -> str:
    nodes = "".join(
        f'<g transform="translate({116 + 66 * math.cos(math.radians(a)):.1f} {76 + 66 * math.sin(math.radians(a)):.1f})"><circle r="11" fill="#0e4a43" stroke="#d6b262" stroke-width="1.6"/>'
        f'<circle cy="-3" r="3.4" fill="#e3b86f"/><path d="M-6 7a6 5 0 0 1 12 0z" fill="#e3b86f"/></g>' for a in (-150, -90, -30, 30, 90, 150))
    rays = "".join(f'<path d="M116 {76 - 52} l5 -12 l5 12z" fill="#ffd05a" transform="rotate({k * 30} 116 76) translate(-5 0)"/>' for k in range(12))
    stars = "".join(f'<path class="tw" transform="translate({x} {y}) scale({sc})" d="M0-5 1.4-1.4 5 0 1.4 1.4 0 5-1.4 1.4-5 0-1.4-1.4z" fill="#fff6c8"/>' for x, y, sc in ((26, 26, 1), (206, 18, .8), (236, 66, 1.1), (34, 108, .7), (214, 112, .9), (150, 10, .7), (70, 12, .9), (250, 100, .6), (20, 66, .6)))
    spark = "".join(f'<circle class="tw" cx="{x}" cy="{y}" r="1.7" fill="#f3d27a"/>' for x, y in ((24, 28), (206, 20), (232, 70), (30, 112), (214, 118), (150, 8), (72, 14), (250, 104)))
    return f'''<svg class="cx-sun" data-scene="morning" viewBox="0 0 268 150" aria-hidden="true"><defs>
      <radialGradient id="sg" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#fff0a0"/><stop offset=".6" stop-color="#ffd24d"/><stop offset="1" stop-color="#f3a62b"/></radialGradient>
      <radialGradient id="gl"><stop offset="0" stop-color="#ffe68a" stop-opacity=".55"/><stop offset="1" stop-color="#ffe68a" stop-opacity="0"/></radialGradient>
      <linearGradient id="rg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f1d28a"/><stop offset=".5" stop-color="#b98b3c"/><stop offset="1" stop-color="#f1d28a"/></linearGradient>
      <radialGradient id="sgs" cx="50%" cy="40%" r="60%"><stop offset="0" stop-color="#ffe2a8"/><stop offset=".55" stop-color="#ff9d4f"/><stop offset="1" stop-color="#e8582f"/></radialGradient>
      <radialGradient id="mb" cx="40%" cy="35%" r="70%"><stop offset="0" stop-color="#fffbe6"/><stop offset="1" stop-color="#e3dcb4"/></radialGradient><radialGradient id="mg"><stop offset="0" stop-color="#e9f0ff" stop-opacity=".45"/><stop offset="1" stop-color="#e9f0ff" stop-opacity="0"/></radialGradient>
      <linearGradient id="hz" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ff8a4a" stop-opacity="0"/><stop offset="1" stop-color="#ff8a4a" stop-opacity=".55"/></linearGradient>
      <linearGradient id="cl" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#ffffff"/><stop offset="1" stop-color="#cfd8ec"/></linearGradient></defs>
      <circle class="glow" cx="116" cy="76" r="76" fill="url(#gl)"/>
      <g class="ring"><circle cx="116" cy="76" r="66" fill="none" stroke="url(#rg)" stroke-width="4"/><circle cx="116" cy="76" r="59" fill="none" stroke="rgba(214,178,98,.45)" stroke-width="1"/>{nodes}</g>
      <g class="rays">{rays}</g>
      <g class="sunbody"><circle cx="116" cy="76" r="35" fill="url(#sg)"/><path d="M103 72q5-7 10 0" fill="none" stroke="#9a5a14" stroke-width="2.4" stroke-linecap="round"/><path class="wink" d="M121 72q5-7 10 0" fill="none" stroke="#9a5a14" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M105 85q11 11 22 0" fill="none" stroke="#9a5a14" stroke-width="2.6" stroke-linecap="round"/><circle cx="98" cy="84" r="5" fill="#ff9a8a" opacity=".5"/><circle cx="134" cy="84" r="5" fill="#ff9a8a" opacity=".5"/></g>
      <g class="moon"><circle class="mglow" cx="116" cy="76" r="62" fill="url(#mg)"/><circle cx="116" cy="76" r="35" fill="url(#mb)"/><circle cx="130" cy="62" r="5" fill="#d9d4b0" opacity=".55"/><circle cx="100" cy="92" r="6.5" fill="#d9d4b0" opacity=".45"/><circle cx="132" cy="94" r="3.5" fill="#d9d4b0" opacity=".5"/>
        <path d="M103 72q5-7 10 0" fill="none" stroke="#6a6a52" stroke-width="2.4" stroke-linecap="round"/><path class="wink" d="M121 72q5-7 10 0" fill="none" stroke="#6a6a52" stroke-width="2.4" stroke-linecap="round"/>
        <path d="M105 85q11 11 22 0" fill="none" stroke="#6a6a52" stroke-width="2.6" stroke-linecap="round"/><circle cx="98" cy="84" r="5" fill="#f0a0a0" opacity=".4"/><circle cx="134" cy="84" r="5" fill="#f0a0a0" opacity=".4"/></g>
      <g class="stars">{stars}</g><g class="horizon"><ellipse cx="116" cy="124" rx="80" ry="13" fill="url(#hz)"/></g>
      <g class="c1"><ellipse cx="62" cy="122" rx="30" ry="14" fill="url(#cl)"/><ellipse cx="50" cy="113" rx="14" ry="11" fill="url(#cl)"/><ellipse cx="72" cy="111" rx="16" ry="13" fill="url(#cl)"/></g>
      <g class="c2"><ellipse cx="150" cy="124" rx="42" ry="16" fill="url(#cl)"/><ellipse cx="132" cy="112" rx="18" ry="14" fill="url(#cl)"/><ellipse cx="160" cy="108" rx="22" ry="17" fill="url(#cl)"/></g>{spark}</svg>'''


def _ai_btn(flow: bool, href: str) -> str:
    """The standalone preview opens its feature dialog; Streamlit uses a native session-aware launcher."""
    return ('<span class="cx-ai-hint">Open Mind / AI Empower from the sidebar</span>' if flow
            else '<button class="cx-btn" id="btn-ai">Explore AI &rarr;</button>')


def sidebar_html(active: str = "dashboard", flow: bool = False, href: str = "?page={key}", version: str = "1.0.0", part: str = "all") -> str:
    """Logo, navigation, promo card and version label exactly as in the reference."""
    items = "".join(f'<a href="{html.escape(href.format(key=k))}" target="_self" class="{"on" if k == active else ""}" {"aria-current=page" if k == active else ""}>'
                    f'<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{p}</svg>{html.escape(lbl)}</a>' for k, lbl, p in NAV)
    cls = "cx-side" if not flow else "cx-side-flow"
    logo = f'<img class="logo" src="{_b64("cortexhr_logo_sidebar_clear.png")}" alt="CortexHR - Smarter Workforce. Brighter Future.">'
    promo = (f'<div class="cx-panel cx-sidecard"><b>Build. Predict.<br>Empower.</b><img src="{_b64("brain_promo_clear.png")}" alt="">{_ai_btn(flow, href)}</div>'
             f'<div class="cx-ver">v{html.escape(version)} &bull; Production</div>')
    if part == "top":                        # Streamlit draws the menu itself (state-preserving buttons) between the two halves
        return f'<aside class="{cls}">{logo}</aside>'
    if part == "bottom":
        return f'<aside class="{cls}">{promo}</aside>'
    return (f'<aside class="{cls}">{_side_waves() if not flow else ""}<img class="logo" src="{_b64("cortexhr_logo_sidebar_clear.png")}" alt="CortexHR - Smarter Workforce. Brighter Future.">'
            f'<nav class="cx-nav" aria-label="Main">{items}</nav>'
            f'<div class="cx-panel cx-sidecard"><b>Build. Predict.<br>Empower.</b><img src="{_b64("brain_promo_clear.png")}" alt="">{_ai_btn(flow, href)}</div>'
            f'<div class="cx-ver">v{html.escape(version)} &bull; Production</div></aside>')


def build(overview: dict, employees: list[dict], user: dict | None = None, mode: str = "embed", active: str = "dashboard", env: str = "production",
          show_welcome: bool = True, nav_href: str = "?page={key}") -> str:
    """Return the complete HTML document."""
    user = user or {"name": "Jaisurya", "role": "Admin"}
    first = user["name"].split()[0]
    data = {"overview": overview, "employees": employees, "departments": overview.get("departments", []), "notifications": overview.get("notifications", []),
            "features": FEATURES, "user": {**user, "first": first}, "env": env, "showWelcome": show_welcome}
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    css = (STATIC / "dashboard.css").read_text(encoding="utf-8") + (STATIC / "reference_skin.css").read_text(encoding="utf-8")
    js = (STATIC / "dashboard.js").read_text(encoding="utf-8") + "\n" + (STATIC / "reference_skin.js").read_text(encoding="utf-8")
    full = mode == "full"
    side = sidebar_html(active, False, nav_href, overview.get("version", "1.0.0")) if full else ""
    extra_css = ".cx-fit{margin-left:167px;width:calc(100% - 167px)}@media (max-width:760px){.cx-fit{margin-left:0;width:100%}}" if full else ""
    kpis = [("emp", "Total Employees", "users", 21, 271), ("risk", "Attrition Risk", "alert", 302, 268), ("open", "Open Positions", "brief", 580, 295), ("sal", "Avg. Salary", "coin", 886, 293)]
    icon = {"users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
            "alert": '<path d="m10.3 3.9-8.5 14.7A2 2 0 0 0 3.5 21.6h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
            "brief": '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>', "coin": '<circle cx="12" cy="12" r="9"/><path d="M8 8h8M8 12h8M10 8c4 0 4 5 0 5l4 4"/>'}
    kpi_html = "".join(f'<div class="cx-panel cx-kpi" style="left:{x}px;width:{w}px"><span class="ic"><svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">{icon[ic]}</svg></span>'
                       f'<span class="lbl">{lbl}</span>' + ('<span class="cx-ailbl">AI-generated, for human review</span>' if k == "risk" else '') + f'<span class="vr"><span class="val" id="v-{k}">&ndash;</span><span class="chg na" id="c-{k}"></span></span><svg class="sp" id="s-{k}"></svg></div>' for k, lbl, ic, x, w in kpis)
    pen = '<path d="M3 17c4-1 5-9 9-9s5 8 9 8"/>'
    body = f'''{_defs()}<div class="cx-fit" id="cx-fit"><div class="cx-stage" id="cx-stage">{_waves()}
  <div class="cx-top">
    <div class="cx-panel cx-search" role="search"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/></svg><input id="cx-q" type="search" placeholder="Search employees, roles, departments..." aria-label="Search employees, roles, departments" autocomplete="off"><span class="cx-kbd" id="cx-kbd">Ctrl K</span></div>
    <div class="cx-panel cx-results" id="cx-results" role="listbox"></div>
    <button class="cx-iconbtn" id="cx-btn-welcome" style="left:952px" title="Show the welcome greeting" aria-label="Show the welcome greeting"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/></svg></button>
    <button class="cx-iconbtn" id="cx-bell" style="left:1006px" title="Notifications" aria-label="Notifications"><svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9M10.3 21a1.9 1.9 0 0 0 3.4 0"/></svg><span class="cx-dot" id="cx-bell-dot" style="display:none"></span></button>
    <div class="cx-panel cx-menu" id="cx-bell-menu" style="left:760px;top:64px;width:330px"></div>
    <button class="cx-user" id="cx-user" aria-label="Account menu"><span class="cx-avatar" role="img" aria-label="{html.escape(first)}" style="background-image:url({_b64("avatar_reference.png")})">{html.escape(first[:1])}</span><span><b>{html.escape(user["name"])}</b><small>{html.escape(user["role"])}</small></span><svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#9db8ae" stroke-width="2.2" stroke-linecap="round"><path d="m6 9 6 6 6-6"/></svg></button>
    <div class="cx-panel cx-menu" id="cx-user-menu" style="left:940px;top:64px;width:240px"></div>
  </div>
  <div class="cx-hello"><h1>Welcome back, {html.escape(first)}! &#128075;</h1><p>Here&rsquo;s your workforce intelligence at a glance.</p></div>
  <div class="cx-tag">Smarter Insights.<i>Stronger Decisions.</i><svg width="150" height="16" viewBox="0 0 150 16" aria-hidden="true"><path d="M4 11c40 4 100 2 142-6" fill="none" stroke="#d6b262" stroke-width="2.4" stroke-linecap="round"/></svg></div>
  <label class="cx-panel cx-period"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#9db8ae" stroke-width="1.9"><rect x="3" y="5" width="18" height="16" rx="2"/><path d="M3 10h18M8 3v4M16 3v4"/></svg>
    <select id="cx-period" aria-label="Period"><option value="3">Last 3 Months</option><option value="6" selected>Last 6 Months</option><option value="12">Last 12 Months</option></select></label>
  <div class="cx-kpis">{kpi_html}</div>
  <section class="cx-panel cx-wfp" style="left:21px;top:273px;width:330px;height:244px"><div class="cx-h"><span class="hi">{_icon("bars")}</span>Workforce Overview</div>
    <div class="cx-legend"><span><i style="background:#c9a24d"></i>Employees</span><span><i style="background:#e0b982"></i>Exits per month</span></div><svg id="cx-wf" role="img" aria-label="Employees and exits per month"></svg></section>
  <section class="cx-panel cx-hirep" style="left:361px;top:273px;width:426px;height:244px"><div class="cx-h"><span class="hi">{_icon("pin")}</span>Workforce Locations</div><button class="cx-link" id="lnk-hire">View Details &rarr;</button>
    <svg class="cx-map" id="cx-map" viewBox="0 0 420 170" role="img" aria-label="Schematic India map with clickable approximate city-centre workforce markers"></svg><div class="cx-hires" id="cx-hires" role="status"></div><div class="cx-regions" id="cx-regions"></div></section>
  <section class="cx-panel cx-attp" style="left:797px;top:273px;width:382px;height:244px"><div class="cx-h"><span class="hi">{_icon("alert")}</span>Attrition Risk Analysis</div><span class="cx-ailbl">AI-generated, for human review</span><button class="cx-link" id="lnk-att">View Details &rarr;</button>
    <svg class="cx-donut" id="cx-donut" width="160" height="160" viewBox="0 0 160 160" role="img" aria-label="Share of employees by risk band"></svg>
    <div class="cx-leg2"><div><span><i style="background:#b9774a"></i>High Risk</span><b id="lg-high"></b></div><div><span><i style="background:#d4a95f"></i>Medium Risk</span><b id="lg-med"></b></div><div><span><i style="background:#efe1b8"></i>Low Risk</span><b id="lg-low"></b></div></div>
    <div class="cx-note">{_icon("info", 18)}<span id="att-note"></span><svg class="sp" id="s-att"></svg></div></section>
  <section class="cx-panel cx-insp" style="left:21px;top:530px;width:291px;height:176px"><div class="cx-h"><span class="hi">{_icon("info")}</span>Key Insights</div><span class="cx-ailbl">AI-generated, for human review</span><button class="cx-link" id="lnk-ins">View All &rarr;</button><div class="cx-ins" id="cx-ins"></div></section>
  <section class="cx-panel cx-mpp" style="left:322px;top:530px;width:392px;height:176px"><div class="cx-h"><span class="hi">{_icon("gear")}</span>Model Performance</div><span class="cx-ailbl">AI-generated, for human review</span><button class="cx-link" id="lnk-perf">View Report &rarr;</button><div class="cx-mp" id="cx-mp"></div><div class="cx-foot" id="cx-mp-foot"></div></section>
  <section class="cx-panel cx-actp" style="left:724px;top:530px;width:239px;height:176px"><div class="cx-h"><span class="hi">{_icon("bolt")}</span>Recent Activity</div><button class="cx-link" id="lnk-act">View All &rarr;</button><div class="cx-act" id="cx-act"></div></section>
  <section class="cx-panel cx-promo" style="left:975px;top:530px;width:204px;height:176px"><b>CortexHR v{html.escape(overview.get("version", "1.0.0"))}</b><p>New with advanced AI insights &amp; real-time monitoring.</p><img src="{_b64("brain_promo_clear.png")}" alt=""><button class="cx-btn" id="btn-feat">Explore Features &rarr;</button></section>
  <div class="cx-modal-bg" id="cx-modal-bg"><div class="cx-panel cx-modal" role="dialog" aria-modal="true"><button class="cx-x" id="cx-modal-x" aria-label="Close">&times;</button><div id="cx-modal-body"></div></div></div>
  <div class="cx-welcome-bg" id="cx-welcome-bg"><div class="cx-wwrap" id="cx-wwrap"><div class="cx-welcome" role="dialog" aria-modal="true" aria-labelledby="cx-wtitle"><img class="mark" src="{_b64("cortexhr_logo_mark_clear.png")}" alt="CortexHR logo"><div class="sep"></div>{_sun()}
    <h2 id="cx-wtitle">Good Morning, {html.escape(first)}!</h2><p class="l1">Welcome back to <b>CortexHR</b></p><p class="l2">Your workforce intelligence starts here.</p><hr><button class="x" id="cx-welcome-x" aria-label="Close welcome message">&times;</button></div></div></div>
</div></div>'''
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CortexHR Dashboard</title>'
            f'<style>{css}{_flow_css()}{extra_css}</style></head><body class="{"cx-full" if full else ""}">{side}{body}<script>window.CX={payload};</script><script>{js}</script></body></html>')


def _flow_css() -> str:
    return (".cx-side-flow{position:relative}.cx-side-flow img.logo{display:block;margin:0 auto 10px;width:144px}.cx-side-flow .cx-nav{position:static}"
            ".cx-side-flow .cx-sidecard{position:relative;left:auto;top:auto;margin:260px 0 0}")


def _icon(name: str, size: int = 17) -> str:
    paths = {"bars": '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>', "alert": '<path d="m10.3 3.9-8.5 14.7A2 2 0 0 0 3.5 21.6h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
             "info": '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>', "gear": '<circle cx="12" cy="12" r="3"/><path d="M12 2v3M12 19v3M4.2 4.2l2.1 2.1M17.7 17.7l2.1 2.1M2 12h3M19 12h3"/>', "bolt": '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>', "pin": '<path d="M12 21s7-6.2 7-11a7 7 0 1 0-14 0c0 4.8 7 11 7 11z"/><circle cx="12" cy="10" r="2.5"/>'}
    return f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">{paths[name]}</svg>'
