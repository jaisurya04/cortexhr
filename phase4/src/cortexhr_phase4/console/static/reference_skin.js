/* Reference skin: swaps the outline icons for the filled gold / copper glyphs of the approved screenshot. Pure presentation; runs after dashboard.js. */
(function () {
  "use strict";
  const G = "url(#cxg)", C = "url(#cxc)", K = "#082a27";
  const S = (inner, vb = "0 0 24 24") => `<svg viewBox="${vb}" aria-hidden="true">${inner}</svg>`;
  const st = (w = 1.9) => `fill="none" stroke="${K}" stroke-width="${w}" stroke-linecap="round" stroke-linejoin="round"`;
  const I = {
    dashboard: S(`<path fill="${G}" d="M12 2.6 2.4 11.2h2.9V20a.9.9 0 0 0 .9.9h3.9v-5.5h3.8v5.5h3.9a.9.9 0 0 0 .9-.9v-8.8h2.9z"/>`),
    insights: S(`<g fill="${G}"><rect x="2.8" y="11" width="5" height="10" rx="1.5"/><rect x="9.5" y="3" width="5" height="18" rx="1.5"/><rect x="16.2" y="8" width="5" height="13" rx="1.5"/></g>`),
    forecast: S(`<rect x="2.5" y="2.5" width="19" height="19" rx="4.2" fill="${G}"/><path d="M6 15.6 10 11.2l3 3 5-6.4" ${st(2)}/><path d="M14.8 7.8h3.5v3.5" ${st(2)}/>`),
    fairness: S(`<circle cx="12" cy="12" r="10" fill="${G}"/><path d="M12 5.8v11.4M7.6 8.4h8.8M7.6 8.4l-2 4.6h4zM16.4 8.4l-2 4.6h4z" ${st(1.3)}/>`),
    models: S(`<path fill="${G}" d="M12 2 21 7v10l-9 5-9-5V7z"/><path d="M3.6 7.2 12 12l8.4-4.8M12 12v9.6" ${st(1.4)}/>`),
    resume: S(`<rect x="3.5" y="2.5" width="17" height="19" rx="3.6" fill="${G}"/><path d="M8.4 8h6.4M8.4 12h6.4M8.4 16h4.2" ${st(2)}/>`),
    api: S(`<circle cx="12" cy="12" r="10" fill="${G}"/><path d="M7.4 10.4 12 15l4.6-4.6" ${st(2.2)}/>`),
    settings: S(`<circle cx="12" cy="12" r="6.6" fill="${G}"/><circle cx="12" cy="12" r="8.9" fill="none" stroke="${G}" stroke-width="3.2" stroke-dasharray="2.6 4.14"/><circle cx="12" cy="12" r="2.5" fill="${K}"/>`),
    users: S(`<g fill="${G}"><circle cx="12" cy="7.6" r="3.5"/><circle cx="5" cy="9.4" r="2.5"/><circle cx="19" cy="9.4" r="2.5"/><path d="M6.2 20v-2.4c0-2.9 2.6-4.9 5.8-4.9s5.8 2 5.8 4.9V20z"/><path d="M.9 18v-1.4c0-1.9 1.6-3.3 3.8-3.4-1 1-1.5 2.2-1.5 3.6V18zM23.1 18v-1.4c0-1.9-1.6-3.3-3.8-3.4 1 1 1.5 2.2 1.5 3.6V18z"/></g>`),
    risk: S(`<path fill="${C}" d="M12 1.8 20.7 6.9v10.2L12 22.2 3.3 17.1V6.9z"/><path d="M12 7.4 17.2 16.4H6.8z" fill="${K}"/><path d="M12 10.4v2.9M12 14.9v.1" stroke="#e7b27c" stroke-width="1.5" stroke-linecap="round"/>`),
    brief: S(`<rect x="2.4" y="7" width="19.2" height="13.4" rx="2.6" fill="${G}"/><path d="M8.6 7V5.4a1.6 1.6 0 0 1 1.6-1.6h3.6a1.6 1.6 0 0 1 1.6 1.6V7" fill="none" stroke="${G}" stroke-width="1.8"/><path d="M2.6 12.4h18.8" stroke="${K}" stroke-width="1.5"/><rect x="10.3" y="11" width="3.4" height="3" rx=".8" fill="${K}"/>`),
    coin: S(`<circle cx="12" cy="12" r="10.4" fill="${C}"/><circle cx="12" cy="12" r="8.2" fill="none" stroke="#f0c99c" stroke-width=".8" opacity=".8"/><text x="12" y="16.7" text-anchor="middle" font-size="12.5" font-weight="700" font-family="Inter,Arial,sans-serif" fill="${K}">\u20b9</text>`),
    bars: S(`<g fill="${G}"><rect x="3" y="11" width="4.6" height="10" rx="1.2"/><rect x="9.7" y="3" width="4.6" height="18" rx="1.2"/><rect x="16.4" y="8" width="4.6" height="13" rx="1.2"/></g>`),
    bulb: S(`<path fill="${G}" d="M12 2.4a6.6 6.6 0 0 0-3.6 12.1c.5.4.8 1 .8 1.6v.8h5.6v-.8c0-.6.3-1.2.8-1.6A6.6 6.6 0 0 0 12 2.4z"/><rect x="9.2" y="18.2" width="5.6" height="1.9" rx=".9" fill="${G}"/><rect x="10.2" y="20.6" width="3.6" height="1.5" rx=".7" fill="${G}"/>`),
    gear: S(`<circle cx="12" cy="12" r="6.2" fill="${C}"/><circle cx="12" cy="12" r="8.8" fill="none" stroke="${C}" stroke-width="3.4" stroke-dasharray="2.6 4.1"/><circle cx="12" cy="12" r="2.3" fill="${K}"/>`),
    pin: S(`<path fill="${G}" d="M12 22s7.6-6.6 7.6-12.2A7.6 7.6 0 0 0 4.4 9.8C4.4 15.4 12 22 12 22z"/><circle cx="12" cy="9.8" r="2.8" fill="${K}"/>`),
    bolt: S(`<path fill="${G}" d="M13.4 1.8 4 13.6h6.6l-1.2 8.6 9.8-12.4h-6.8z"/>`),
    trend: S(`<path d="M3 17 9.4 10.6l3.6 3.6L21 6" fill="none" stroke="#f2e6cf" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/><path d="M15 6h6v6" fill="none" stroke="#f2e6cf" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/>`),
    warn: S(`<path fill="${G}" d="M12 2.8 22 20H2z" stroke="${G}" stroke-width="1.6" stroke-linejoin="round"/><path d="M12 9.2v5M12 16.6v.1" ${st(2)}/>`),
    info: S(`<circle cx="12" cy="12" r="10" fill="${G}"/><path d="M12 11v5.6M12 7.4v.1" ${st(2.3)}/>`),
    chart: S(`<path d="M5 4v15h15" fill="none" stroke="#f2e6cf" stroke-width="1.8" stroke-linecap="round"/><path d="m8 15 3.2-4 2.8 2.4 4-5.4" fill="none" stroke="#f2e6cf" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`),
    fair: S(`<circle cx="12" cy="12" r="9.4" fill="${G}"/><path d="M12 6v12M8 8.6h8M8 8.6l-1.7 4h3.4zM16 8.6l-1.7 4h3.4z" ${st(1.2)}/>`),
    rocket: S(`<path d="M12 2.6c3 1.8 4.4 5.2 4 9.4l-1.8 2.6H9.8L8 12c-.4-4.2 1-7.6 4-9.4z" fill="#f2e6cf"/><circle cx="12" cy="9.4" r="1.7" fill="#4a3226"/><path d="M8.2 12.6 5.4 15l.4 3 3-1.4zM15.8 12.6l2.8 2.4-.4 3-3-1.4z" fill="#d8b878"/><path d="M10.4 17.4 12 21.4l1.6-4z" fill="#ff9d4f"/>`),
    check: S(`<circle cx="12" cy="12" r="10.5" fill="${G}"/><path d="m7 12.4 3.4 3.4L17.2 8.6" ${st(2.4)}/>`)
  };
  const put = (el, name) => { if (el && I[name]) el.innerHTML = I[name]; };
  function apply() {
    document.querySelectorAll(".cx-nav a").forEach((a, i) => { const n = ["dashboard", "insights", "forecast", "fairness", "models", "resume", "api", "settings"][i]; const s = a.querySelector("svg"); if (s && n) s.outerHTML = I[n]; });
    [["emp", "users", 0], ["risk", "risk", 1], ["open", "brief", 0], ["sal", "coin", 1]].forEach(([k, n, cu]) => { const v = document.getElementById("v-" + k); const ic = v && v.closest(".cx-kpi").querySelector(".ic"); if (ic) { put(ic, n); if (cu) ic.classList.add("cu"); } });
    [[".cx-wfp", "bars"], [".cx-attp", "risk"], [".cx-insp", "bulb"], [".cx-mpp", "gear"], [".cx-actp", "bolt"], [".cx-hirep", "pin"]].forEach(([sel, n]) => put(document.querySelector(sel + " .hi"), n));
    document.querySelectorAll("#cx-ins .ico").forEach((e, i) => put(e, ["trend", "warn", "info"][i]));
    document.querySelectorAll("#cx-act .ai").forEach((e, i) => put(e, ["chart", "fair", "rocket"][i % 3]));
    const note = document.querySelector(".cx-note > svg:first-child"); if (note) note.outerHTML = `<span class="ni" style="width:20px;height:20px;display:inline-grid">${I.bulb}</span>`;
    const foot = document.querySelector("#cx-mp-foot svg"); if (foot) foot.outerHTML = I.check;
  }
  apply();
  new MutationObserver((_, o) => { o.disconnect(); apply(); o.observe(document.getElementById("cx-stage"), { childList: true, subtree: true }); })
    .observe(document.getElementById("cx-stage"), { childList: true, subtree: true });
})();
