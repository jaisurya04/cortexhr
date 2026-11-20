/* CortexHR dashboard behaviour. Reads window.CX (real data from the API) and draws every chart client-side. */
(function () {
  "use strict";
  const D = window.CX, $ = (id) => document.getElementById(id);
  const NS = "http://www.w3.org/2000/svg";
  const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const inr = (v) => "\u20B9" + Math.round(v).toLocaleString("en-IN");
  const lakh = (v) => "\u20B9" + (v / 1e5).toFixed(2) + "L";
  const pct = (v, d = 1) => (v === null || v === undefined ? "n/a" : v.toFixed(d) + "%");
  const ICON = {
    users: '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.9M16 3.1a4 4 0 0 1 0 7.8"/>',
    alert: '<path d="m10.3 3.9-8.5 14.7A2 2 0 0 0 3.5 21.6h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    brief: '<rect x="2" y="7" width="20" height="14" rx="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/>',
    coin: '<circle cx="12" cy="12" r="9"/><path d="M8 8h8M8 12h8M10 8c4 0 4 5 0 5l4 4"/>',
    trend: '<path d="m22 7-8.5 8.5-5-5L2 17"/><path d="M16 7h6v6"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/>',
    bars: '<path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/>', gear: '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.7 1.7 0 0 0 .3 1.8l.1.1a2 2 0 1 1-2.8 2.8l-.1-.1a1.7 1.7 0 0 0-1.8-.3 1.7 1.7 0 0 0-1 1.5V21a2 2 0 0 1-4 0v-.1a1.7 1.7 0 0 0-1.1-1.5 1.7 1.7 0 0 0-1.8.3l-.1.1a2 2 0 1 1-2.8-2.8l.1-.1a1.7 1.7 0 0 0 .3-1.8 1.7 1.7 0 0 0-1.5-1H3a2 2 0 0 1 0-4h.1a1.7 1.7 0 0 0 1.5-1.1 1.7 1.7 0 0 0-.3-1.8l-.1-.1a2 2 0 1 1 2.8-2.8l.1.1a1.7 1.7 0 0 0 1.8.3H9a1.7 1.7 0 0 0 1-1.5V3a2 2 0 0 1 4 0v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.8-.3l.1-.1a2 2 0 1 1 2.8 2.8l-.1.1a1.7 1.7 0 0 0-.3 1.8V9a1.7 1.7 0 0 0 1.5 1H21a2 2 0 0 1 0 4h-.1a1.7 1.7 0 0 0-1.5 1z"/>',
    bolt: '<path d="M13 2 3 14h9l-1 8 10-12h-9z"/>', chart: '<path d="M3 3v18h18"/><path d="m7 15 4-5 3 3 5-7"/>', check: '<path d="m5 13 4 4L19 7"/>', rocket: '<path d="M5 15c-1.5 1.3-2 5-2 5s3.7-.5 5-2M12 15l-3-3a22 22 0 0 1 2-4 13 13 0 0 1 11-5c0 3-1 8-5 11a22 22 0 0 1-4 2z"/><path d="M9 12H4s.5-3 2-4c1.6-1.1 5 0 5 0M12 15v5s3-.5 4-2c1.1-1.6 0-5 0-5"/>'
  };
  const svg = (name, s = 20) => `<svg width="${s}" height="${s}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">${ICON[name]}</svg>`;

  /* ------------------------------------------------------------------ helpers */
  function spark(el, arr, w = 250, h = 34) {
    if (!arr || arr.length < 2) { el.innerHTML = ""; return; }
    const mn = Math.min(...arr), mx = Math.max(...arr), rng = mx - mn || 1;
    const pts = arr.map((v, i) => [(i / (arr.length - 1)) * w, h - 4 - ((v - mn) / rng) * (h - 10)]);
    const line = pts.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join("");
    const id = "g" + Math.random().toString(36).slice(2, 7);
    el.setAttribute("viewBox", `0 0 ${w} ${h}`); el.setAttribute("preserveAspectRatio", "none");
    el.innerHTML = `<defs><linearGradient id="${id}" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c9a24d" stop-opacity=".38"/><stop offset="1" stop-color="#c9a24d" stop-opacity="0"/></linearGradient></defs>
      <path d="${line}L${w} ${h}L0 ${h}Z" fill="url(#${id})"/><path d="${line}" fill="none" stroke="#e3c183" stroke-width="1.5" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>${w > 200 ? pts.map((p, i) => (i % 2 === 0 || i === pts.length - 1) ? `<circle cx="${p[0].toFixed(1)}" cy="${p[1].toFixed(1)}" r="1.8" fill="#f0d9a0"/>` : "").join("") : ""}`;
  }
  function modal(title, html) {
    $("cx-modal-body").innerHTML = `<h2>${esc(title)}</h2>${html}`; $("cx-modal-bg").classList.add("open"); $("cx-modal-x").focus();
  }
  function closeModal() { $("cx-modal-bg").classList.remove("open"); }
  function rel(iso, upS) {
    if (upS !== undefined) return upS < 90 ? "just now" : Math.round(upS / 60) + " min ago";
    const s = (Date.now() - new Date(iso).getTime()) / 1000;
    if (!isFinite(s)) return ""; if (s < 90) return "just now"; if (s < 5400) return Math.round(s / 60) + " min ago";
    if (s < 129600) return Math.round(s / 3600) + " h ago"; return Math.round(s / 86400) + " days ago";
  }
  const table = (head, rows) => `<table><thead><tr>${head.map((h) => `<th>${esc(h)}</th>`).join("")}</tr></thead><tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

  /* ------------------------------------------------------------------ KPIs */
  let period = 6;
  const REF = !!D.overview.reference;           // design-reference preview only (sample numbers from the approved screenshot)
  function renderKpis() {
    const k = D.overview.kpis, s = D.overview.workforce.series, n = s.length;
    if (REF) {
      $("v-emp").textContent = Math.round(k.total_employees.value).toLocaleString("en-IN"); setChg("c-emp", k.total_employees.change_pct, "design reference");
      $("v-risk").textContent = pct(k.attrition_risk.value); setChg("c-risk", k.attrition_risk.change_pct, "design reference");
      $("v-open").textContent = k.open_positions.value; setChg("c-open", k.open_positions.change_pct, "design reference");
      $("v-sal").textContent = lakh(k.avg_salary.value); setChg("c-sal", k.avg_salary.change_pct, "design reference");
      spark($("s-emp"), k.total_employees.spark); spark($("s-risk"), k.attrition_risk.spark); spark($("s-open"), k.open_positions.spark); spark($("s-sal"), k.avg_salary.spark);
      return;
    }
    const back = s[Math.max(0, n - 1 - period)].employees, now = s[n - 1].employees, chg = (now / back - 1) * 100;
    $("v-emp").textContent = Math.round(now).toLocaleString("en-IN");
    setChg("c-emp", chg, `vs ${period} months ago`);
    $("v-risk").textContent = pct(k.attrition_risk.value);
    setChg("c-risk", null, "no earlier scores to compare");
    $("v-open").textContent = k.open_positions.value;
    setChg("c-open", null, `${k.open_positions.jobs} job posts`);
    $("v-sal").textContent = lakh(k.avg_salary.value);
    setChg("c-sal", null, "per year, active staff");
    spark($("s-emp"), s.slice(-1 - Math.max(period, 5)).map((x) => x.employees));
    spark($("s-risk"), k.attrition_risk.spark); spark($("s-open"), k.open_positions.spark); spark($("s-sal"), k.avg_salary.spark);
  }
  function setChg(id, v, tip) {
    const e = $(id); e.title = tip;
    if (v === null) { e.className = "chg na"; e.textContent = "\u2014"; return; }
    e.className = "chg"; e.textContent = (v >= 0 ? "\u2191 +" : "\u2193 ") + v.toFixed(1) + "%";
  }

  /* ------------------------------------------------------------------ workforce overview (two lines) */
  function renderWorkforceRef() {                 // reference layout: two smooth lines (Employees copper, Attrition tan) on one 0-10K axis, Jan/Feb/Mar 79px apart
    const s = D.overview.workforce.series, ax = D.overview.workforce.axis, W = 322, H = 190, L = 34, T = 35, B = 22, el = $("cx-wf"), ph = H - T - B;
    const X0 = 45, DX = (W - X0 - 14) / Math.max(1, s.length - 1), X = (i) => X0 + i * DX, Y = (v) => T + (1 - (v - ax.min) / (ax.max - ax.min)) * ph, XR = Math.min(W - 6, X(s.length - 1) + 40);
    let g = "";
    for (let v = ax.min; v <= ax.max; v += ax.step) { const y = Y(v); g += `<line x1="${L}" x2="${XR}" y1="${y}" y2="${y}" stroke="rgba(160,215,195,.10)"/><text x="${L - 5}" y="${y + 4}" text-anchor="end" fill="#e1ece6" font-size="11.5">${v / 1000}K</text>`; }
    s.forEach((p, i) => { g += `<line x1="${X(i)}" x2="${X(i)}" y1="${T}" y2="${Y(0)}" stroke="rgba(160,215,195,.10)"/><text x="${X(i)}" y="${H - 4}" text-anchor="middle" fill="#e1ece6" font-size="11.5">${p.label}</text>`; });
    const smooth = (a) => a.map((v, i) => { const x = X(i), y = Y(v); if (!i) return `M${x} ${y}`; const px = X(i - 1), py = Y(a[i - 1]), cx = (px + x) / 2; return `C${cx} ${py} ${cx} ${y} ${x} ${y}`; }).join("");
    const pe = smooth(s.map((p) => p.employees)), pa = smooth(s.map((p) => p.exits));
    let dots = "";
    const mid = (a, i) => { const x = (X(i - 1) + X(i)) / 2, y = (Y(a[i - 1]) + Y(a[i])) / 2; return [x, y]; };
    const E = s.map((p) => p.employees), A = s.map((p) => p.exits);
    s.forEach((p, i) => {
      dots += `<circle cx="${X(i)}" cy="${Y(p.employees)}" r="5.4" fill="#b98a62" stroke="#e3c9a8" stroke-width="1.2"><title>${p.month}: ${p.employees} employees</title></circle><circle cx="${X(i)}" cy="${Y(p.exits)}" r="5.4" fill="#d9bb7e" stroke="#f0e1b8" stroke-width="1.2"><title>${p.month}: ${p.exits} attrition</title></circle>`;
      if (i) { const [a1, b1] = mid(E, i), [a2, b2] = mid(A, i); dots += `<circle cx="${a1}" cy="${b1}" r="2.1" fill="#d3a98a"/><circle cx="${a2}" cy="${b2}" r="2.1" fill="#ead7a4"/>`; }
    });
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.innerHTML = `<defs><linearGradient id="wfa" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#b98a62" stop-opacity=".34"/><stop offset="1" stop-color="#b98a62" stop-opacity=".02"/></linearGradient><linearGradient id="wfb" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#d9bb7e" stop-opacity=".30"/><stop offset="1" stop-color="#d9bb7e" stop-opacity=".02"/></linearGradient></defs>${g}
      <path d="${pe}L${X(s.length - 1)} ${Y(0)}L${X(0)} ${Y(0)}Z" fill="url(#wfa)"/><path d="${pa}L${X(s.length - 1)} ${Y(0)}L${X(0)} ${Y(0)}Z" fill="url(#wfb)"/>
      <path d="${pe}" fill="none" stroke="#b98a62" stroke-width="2.2"/><path d="${pa}" fill="none" stroke="#d9bb7e" stroke-width="2.2"/>${dots}`;
  }
  function renderHiring() {
    const hr = D.overview.hiring || {}, fmt = (n) => (n == null ? "n/a" : Number(n).toLocaleString("en-IN"));
    const points = hr.location_counts || [];
    $("cx-hires").innerHTML = `<span>Hires, last 6 months</span><b>${fmt(hr.hires_last_6m)}</b><small>${fmt(points.length)} mapped locations</small>`;
    const svg = $("cx-map");
    if (!points.length) {
      svg.innerHTML = `<text x="210" y="84" text-anchor="middle" fill="#cfe0d8" font-size="13">Location data unavailable</text>`;
      $("cx-regions").textContent = "No mapped work locations in the current dataset.";
    } else {
      // Equirectangular city-centre plot; markers are approximate city centroids, not office addresses.
      const project = (lat, lon) => [74 + (lon - 68) * 13.2, 8 + (36 - lat) * 5.0];
      const max = Math.max(...points.map(p => Number(p.employees) || 0), 1);
      const outline = `M116 8 L152 15 L173 30 L188 39 L202 55 L223 66 L231 82 L218 99 L208 117 L196 133 L183 151 L168 163 L156 151 L148 132 L134 116 L124 95 L107 82 L95 63 L102 44 Z`;
      const markers = points.map((p, i) => {
        const [x, y] = project(Number(p.lat), Number(p.lon));
        const r = 4 + Math.sqrt(Number(p.employees) / max) * 7;
        return `<g class="cx-map-point" tabindex="0" role="button" aria-label="${esc(p.name)}: ${fmt(p.employees)} active employees" data-city="${esc(p.name)}" data-count="${fmt(p.employees)}"><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${(r + 4).toFixed(1)}" fill="#d6b262" opacity=".12"/><circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(1)}" fill="#d6b262" stroke="#fff0c7" stroke-width="1.2"><title>${esc(p.name)} · ${fmt(p.employees)} active employees</title></circle></g>`;
      }).join("");
      svg.innerHTML = `<defs><linearGradient id="cx-india" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#477c68" stop-opacity=".58"/><stop offset="1" stop-color="#173d35" stop-opacity=".22"/></linearGradient></defs><path d="${outline}" fill="url(#cx-india)" stroke="#7fae98" stroke-width="1.3" opacity=".9"/><path d="M108 53 Q154 75 211 88 M125 91 Q160 107 198 121 M149 29 Q163 73 174 146" fill="none" stroke="#a6c7b5" stroke-opacity=".18" stroke-dasharray="3 4"/>${markers}`;
      const openCity = (el) => modal("Workforce location", `<p><b>${esc(el.dataset.city)}</b></p><p>${esc(el.dataset.count)} active employees in the supplied workforce scoring dataset.</p><p><small>Marker is an approximate city-centre location, not an office address. Source: ${esc(hr.source || "work_location data")}</small></p>`);
      svg.querySelectorAll(".cx-map-point").forEach(el => {
        el.addEventListener("click", () => openCity(el));
        el.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); openCity(el); } });
      });
      $("cx-regions").innerHTML = `<span>Top locations:</span>` + points.slice(0, 3).map(p => `<span class="rg"><i></i>${esc(p.name)} · ${fmt(p.employees)}</span>`).join("");
    }
    $("lnk-hire").onclick = () => modal("Workforce locations", `<p><b>${fmt(hr.active_employees)}</b> active employees across <b>${fmt(hr.locations)}</b> work locations.</p><p>${points.map(p => `${esc(p.name)} (${fmt(p.employees)})`).join(" · ") || "No location data available."}</p><p><small>${esc(hr.source || "Source unavailable")}. City-centre markers are approximate and do not represent individual employee locations.</small></p>`);
  }
  function renderWorkforce() {
    if (REF) return renderWorkforceRef();
    const s = D.overview.workforce.series.slice(-period), W = 322, H = 190, L = 46, R = 14, T = 14, B = 26, el = $("cx-wf"), ph = H - T - B;
    const emp = s.map((x) => x.employees), ex = s.map((x) => x.exits);
    const step = Math.max(...emp) - Math.min(...emp) > 120 ? 50 : 25;
    const eMin = Math.floor(Math.min(...emp) / step) * step - step, eMax = Math.ceil(Math.max(...emp) / step) * step + step, xMax = Math.max(8, Math.ceil(Math.max(...ex) / 2) * 2);
    const X = (i) => L + 18 + (i / (s.length - 1)) * (W - L - R - 36), YE = (v) => T + (1 - (v - eMin) / (eMax - eMin)) * ph, bw = Math.min(26, (W - L - R) / s.length / 2.4);
    let g = "";
    for (let v = eMin; v <= eMax + 0.1; v += step) {
      const y = YE(v);
      g += `<line x1="${L}" x2="${W - R}" y1="${y}" y2="${y}" stroke="rgba(255,255,255,.07)"/><text x="${L - 8}" y="${y + 4}" text-anchor="end" fill="#9db8ae" font-size="11">${v}</text>`;
    }
    const path = emp.map((v, i) => (i ? "L" : "M") + X(i).toFixed(1) + " " + YE(v).toFixed(1)).join("");
    const area = path + `L${X(s.length - 1)} ${H - B}L${X(0)} ${H - B}Z`;
    let bars = "", dots = "", lab = "";
    s.forEach((p, i) => {
      const h = (p.exits / xMax) * ph * 0.36;
      bars += `<rect x="${X(i) - bw / 2}" y="${H - B - h}" width="${bw}" height="${h}" rx="3" fill="rgba(224,185,130,.5)" stroke="rgba(224,185,130,.9)" stroke-width="1"><title>${p.month}: ${p.exits} exits</title></rect>`;
      dots += `<circle cx="${X(i)}" cy="${YE(p.employees)}" r="4.2" fill="#c9a24d" stroke="#0a3a37" stroke-width="1.5"><title>${p.month}: ${p.employees} employees</title></circle>`;
      lab += `<text x="${X(i)}" y="${H - 7}" text-anchor="middle" fill="#9db8ae" font-size="11">${p.label}</text>`;
    });
    el.setAttribute("viewBox", `0 0 ${W} ${H}`);
    el.innerHTML = `<defs><linearGradient id="wfa" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#c9a24d" stop-opacity=".30"/><stop offset="1" stop-color="#c9a24d" stop-opacity="0"/></linearGradient></defs>${g}
      <path d="${area}" fill="url(#wfa)"/>${bars}<path d="${path}" fill="none" stroke="#c9a24d" stroke-width="2.4" stroke-linejoin="round"/>${dots}${lab}`;
  }

  /* ------------------------------------------------------------------ attrition donut + panels */
  function renderAttrition() {
    const a = D.overview.attrition, tot = a.high + a.medium + a.low || 1, R = 62, C = 2 * Math.PI * R;
    const seg = [["#b9774a", a.high], ["#d4a95f", a.medium], ["#efe1b8", a.low]];
    let off = 0, arcs = "";
    seg.forEach(([c, n]) => { const len = (n / tot) * C; arcs += `<circle cx="80" cy="80" r="${R}" fill="none" stroke="${c}" stroke-width="${REF ? 20 : 22}" stroke-dasharray="${Math.max(0, len - 2)} ${C}" stroke-dashoffset="${-off}" transform="rotate(${REF ? -40 : -90} 80 80)"><title>${n}${REF ? "%" : " employees"}</title></circle>`; off += len; });
    $("cx-donut").innerHTML = `<circle cx="80" cy="80" r="${R}" fill="none" stroke="rgba(255,255,255,.05)" stroke-width="22"/>${arcs}` + (REF
      ? `<text x="80" y="76" text-anchor="middle" fill="#ead9ae" font-size="25" font-weight="600">${pct(a.mean_risk_pct)}</text><text x="80" y="94" text-anchor="middle" fill="#e3ede7" font-size="13.5" font-weight="500">At Risk</text><text x="80" y="110" text-anchor="middle" fill="#cfe0d8" font-size="10.5">↓ ${Math.abs(a.change_pct).toFixed(1)}%</text>`
      : `<text x="80" y="76" text-anchor="middle" fill="#ead9ae" font-size="24" font-weight="600">${pct(a.mean_risk_pct)}</text><text x="80" y="95" text-anchor="middle" fill="#cfe0d8" font-size="12">Avg. risk</text><text x="80" y="111" text-anchor="middle" fill="#9db8ae" font-size="10.5">next 12 months</text>`);
    const sh = (n) => (REF ? n : (n / tot) * 100); $("lg-high").textContent = pct(sh(a.high)); $("lg-med").textContent = pct(sh(a.medium)); $("lg-low").textContent = pct(sh(a.low));
    $("att-note").textContent = REF ? a.note : `${a.high} of ${a.population} employees are in the High band (top 10% of scores).`;
    if (REF) { spark($("s-att"), [5, 5.4, 5.2, 5.9, 5.5, 6.1, 5.8, 6.4, 6.2, 7, 7.6, 8.6], 120, 26); return; }
    spark($("s-att"), D.overview.kpis.attrition_risk.spark, 120, 26);
  }
  function renderInsights() {
    const i = D.overview.insights, g = i.headcount_growth_pct_6m;
    if (REF) { $("cx-ins").innerHTML = i.lines.map((t, k) => `<div class="it"><span class="ico">${svg(["trend", "alert", "info"][k], 20)}</span><span>${esc(t)}</span></div>`).join(""); return; }
    const rows = [
      ["trend", g === null || g === undefined ? "Headcount forecast not available yet." : `Headcount is expected to ${g >= 0 ? "grow" : "shrink"} by ${Math.abs(g).toFixed(1)}% in the next 6 months.`],
      ["alert", `${i.high_risk_count} employees are at high attrition risk. Consider retention actions.`],
      ["info", i.monitoring_headline || "Monitoring has not run yet. Run phase4/scripts/monitor.py."]
    ];
    $("cx-ins").innerHTML = rows.map(([ic, t]) => `<div class="it"><span class="ico">${svg(ic, 18)}</span><span>${esc(t)}</span></div>`).join("");
  }
  function renderPerf() {
    const m = D.overview.model_performance, name = String(m.model).split(" (")[0];
    if (REF) {
      const ring = `<svg class="ring" width="24" height="24" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9" fill="none" stroke="rgba(255,255,255,.12)" stroke-width="3"/><circle cx="12" cy="12" r="9" fill="none" stroke="#c9a24d" stroke-width="3" stroke-linecap="round" stroke-dasharray="${(56.5 * m.coverage).toFixed(1)} 60" transform="rotate(-90 12 12)"/></svg>`;
      $("cx-mp").innerHTML = `<div class="m"><small>MAE</small><b>${inr(m.mae)}</b><em class="g">\u2193 ${m.mae_change_pct.toFixed(1)}%</em></div><div class="m"><small>R\u00B2</small><b>${m.r2.toFixed(3)}</b><em class="g">\u2191 +${m.r2_change.toFixed(2)}</em></div>`
        + `<div class="m"><small>Coverage</small><b>${pct(m.coverage * 100)}</b>${ring}</div><div class="m"><small>Baseline MAE</small><b>${inr(m.baseline_mae)}</b></div>`;
      $("cx-mp-foot").innerHTML = `${svg("check", 18)}<span>${esc(name)} selected as final model</span>`; return;
    }
    $("cx-mp").innerHTML = [["MAE", inr(m.mae)], ["R\u00B2", m.r2.toFixed(3)], ["Coverage", pct(m.coverage * 100)], ["Baseline MAE", inr(m.baseline_mae)]].map(([l, v]) => `<div class="m"><small>${l}</small><b>${v}</b></div>`).join("");
    $("cx-mp-foot").innerHTML = `${svg("check", 18)}<span>${esc(name)} selected as final model</span>`;
  }
  function renderActivity() {
    const ic = ["chart", "check", "rocket"];
    $("cx-act").innerHTML = D.overview.activity.map((a, i) => `<div class="a"><span class="ai">${svg(ic[i % 3], 16)}</span><span>${esc(a.text)}<small>${a.ago ? esc(a.ago) : a.uptime_s !== undefined ? rel(null, a.uptime_s) : rel(a.created_utc)}</small></span></div>`).join("") || "<div class='a'>No activity recorded yet.</div>";
  }

  /* ------------------------------------------------------------------ modals / search / menus */
  const lvlOf = (b) => (b === "High" ? "ALERT" : b === "Medium" ? "WATCH" : "OK");
  function empRow(e) { return [esc(e.employee_id), esc(e.department_id), `<span class="cx-lvl ${lvlOf(e.risk_band)}">${e.risk_band}</span> ${(e.risk_score_12m * 100).toFixed(0)}%`, esc(e.top_reason_1)]; }
  function openHighRisk() {
    const hi = D.employees.filter((e) => e.risk_band === "High").slice(0, 12);
    modal("Highest attrition risk", `<p style="font-size:12.5px;color:#cfe0d8;margin:0 0 8px">Top ${hi.length} of ${D.overview.attrition.high} High-band employees. AI-generated, for human review.</p>` + table(["Employee", "Dept", "Risk", "Main reason"], hi.map(empRow)));
  }
  function openInsights() { modal("Key insights", `<div class="cx-ins" style="position:static">${$("cx-ins").innerHTML}</div><p style="font-size:12.5px;color:#cfe0d8">${esc(D.overview.insights.monitoring_summary || "")}</p><p style="font-size:12px;color:#9db8ae">${esc(D.overview.notice)}</p>`); }
  function openPerf() {
    const m = D.overview.model_performance;
    modal("Pay-band model performance", table(["Measure", "Value"], [["Model", esc(m.model)], ["MAE (test hold-out)", inr(m.mae)], ["R\u00B2", m.r2.toFixed(4)], ["Band coverage", pct(m.coverage * 100, 2)], ["Baseline MAE (role average)", inr(m.baseline_mae)]]) + `<p style="font-size:12px;color:#9db8ae">Source: Phase 3 Task 15 test set. Coverage target is 80%.</p>`);
  }
  function openActivity() { modal("Recent activity", D.overview.activity.map((a) => `<p>${esc(a.text)} <small style="color:#9db8ae">${a.uptime_s !== undefined ? rel(null, a.uptime_s) : esc(a.created_utc || "")}</small></p>`).join("")); }
  function openFeatures() {
    modal("CortexHR v" + D.overview.version, table(["Area", "What it does"], D.features.map((f) => [`<b>${esc(f[0])}</b>`, esc(f[1])])) + `<p style="font-size:12px;color:#9db8ae">${esc(D.overview.notice)}</p>`);
  }
  function openEmployee(e) {
    modal(e.employee_id, table(["Field", "Value"], [["Department", esc(e.department_id)], ["12-month risk score", (e.risk_score_12m * 100).toFixed(1) + "%"], ["Band", `<span class="cx-lvl ${lvlOf(e.risk_band)}">${e.risk_band}</span>`], ["Reason 1", esc(e.top_reason_1)], ["Reason 2", esc(e.top_reason_2)], ["Reason 3", esc(e.top_reason_3)]]) + `<p style="font-size:12px;color:#9db8ae">${esc(D.overview.notice)}</p>`);
  }
  function openDept(d) { modal(d.name, table(["Field", "Value"], [["Department id", esc(d.id)], ["Headcount (" + d.as_of + ")", d.headcount]]) + `<p style="font-size:12px;color:#9db8ae">Open Forecast in the menu for this department's 6-month outlook.</p>`); }

  function wireSearch() {
    const q = $("cx-q"), box = $("cx-results");
    const hits = (t) => {
      t = t.trim().toLowerCase(); if (t.length < 2) return [];
      const e = D.employees.filter((x) => (x.employee_id + " " + x.department_id + " " + x.risk_band).toLowerCase().includes(t)).slice(0, 5).map((x) => ({ k: "e", x }));
      const d = D.departments.filter((x) => (x.id + " " + x.name).toLowerCase().includes(t)).slice(0, 4).map((x) => ({ k: "d", x }));
      return d.concat(e);
    };
    q.addEventListener("input", () => {
      const h = hits(q.value);
      box.innerHTML = h.length ? h.map((r, i) => r.k === "e" ? `<div class="cx-res" tabindex="0" data-i="${i}"><b>${esc(r.x.employee_id)}</b> ${esc(r.x.department_id)}<small>${r.x.risk_band} risk, ${(r.x.risk_score_12m * 100).toFixed(0)}%</small></div>` : `<div class="cx-res" tabindex="0" data-i="${i}"><b>${esc(r.x.name)}</b><small>Department ${esc(r.x.id)}, headcount ${r.x.headcount}</small></div>`).join("") : (q.value.trim().length > 1 ? '<div class="cx-res">No match. Try an employee id (EMP-0002) or a department name.</div>' : "");
      box.classList.toggle("open", !!box.innerHTML);
      box.onclick = (ev) => { const n = ev.target.closest(".cx-res"); if (!n || n.dataset.i === undefined) return; const r = h[+n.dataset.i]; box.classList.remove("open"); r.k === "e" ? openEmployee(r.x) : openDept(r.x); };
    });
    document.addEventListener("keydown", (ev) => { if ((ev.ctrlKey || ev.metaKey) && ev.key.toLowerCase() === "k") { ev.preventDefault(); q.focus(); } if (ev.key === "Escape") { closeModal(); closeWelcome(); box.classList.remove("open"); menus(); } });
  }
  function menus(open) { ["cx-bell-menu", "cx-user-menu"].forEach((id) => $(id).classList.toggle("open", id === open && !$(id).classList.contains("open"))); }
  function wireMenus() {
    const n = D.notifications;
    $("cx-bell-menu").innerHTML = n.length ? n.map((x) => `<div class="row"><span class="cx-lvl ${x.level}">${x.level}</span>${esc(x.text)}</div>`).join("") : '<div class="row">Nothing needs attention.</div>';
    if (n.some((x) => x.level === "ALERT" || x.level === "WATCH")) $("cx-bell-dot").style.display = "block";
    $("cx-bell").onclick = (e) => { e.stopPropagation(); menus("cx-bell-menu"); };
    $("cx-user").onclick = (e) => { e.stopPropagation(); menus("cx-user-menu"); };
    $("cx-user-menu").innerHTML = `<div class="row"><b>${esc(D.user.name)}</b><br><small>${esc(D.user.role)}</small></div><div class="row"><small>Environment</small><br>${esc(D.env)}</div><div class="row"><small>API</small><br>v${esc(D.overview.version)}</div>`;
    document.addEventListener("click", () => menus());
  }

  /* ------------------------------------------------------------------ welcome popup */
  /* Greeting follows India time (Asia/Kolkata) whatever the viewer's own clock says: 05:00-11:59 Morning, 12:00-16:59 Afternoon, 17:00-20:59 Evening, 21:00-04:59 Night. */
  function istHour() {
    const o = new URLSearchParams(location.search).get("cx_hour");          // test hook only: ?cx_hour=22 previews the 22:00 greeting
    if (o !== null && /^\d{1,2}$/.test(o) && +o < 24) return +o;
    try { return +new Intl.DateTimeFormat("en-GB", { timeZone: "Asia/Kolkata", hour: "2-digit", hourCycle: "h23" }).format(new Date()); }
    catch (e) { return (new Date().getUTCHours() + 5 + (new Date().getUTCMinutes() + 30 >= 60 ? 1 : 0)) % 24; }
  }
  function greetingFor(h) { return h >= 5 && h < 12 ? ["Good Morning", "morning"] : h >= 12 && h < 17 ? ["Good Afternoon", "afternoon"] : h >= 17 && h < 21 ? ["Good Evening", "evening"] : ["Good Night", "night"]; }
  function openWelcome() {
    const [text, scene] = greetingFor(istHour());
    $("cx-wtitle").textContent = `${text}, ${D.user.first}!`; document.querySelector(".cx-sun").setAttribute("data-scene", scene);
    $("cx-welcome-bg").classList.add("open"); const dlg = document.querySelector(".cx-welcome"); dlg.setAttribute("tabindex", "-1"); dlg.focus({ preventScroll: true });   // focus the dialog, not the X, so no focus ring is drawn on open; Tab still reaches the close button
  }
  function closeWelcome() { $("cx-welcome-bg").classList.remove("open"); try { sessionStorage.setItem("cx_welcome_seen:" + D.user.first, "1"); } catch (e) { /* storage may be blocked */ } }
  function wireWelcome() {
    $("cx-welcome-x").onclick = closeWelcome; $("cx-welcome-bg").addEventListener("click", (e) => { if (e.target === $("cx-welcome-bg")) closeWelcome(); });
    $("cx-btn-welcome").onclick = openWelcome;
    let seen = false; try { seen = sessionStorage.getItem("cx_welcome_seen:" + D.user.first) === "1"; } catch (e) { /* ignore */ }
    if (D.showWelcome && !seen) openWelcome();
  }

  /* ------------------------------------------------------------------ fit the 1198x768 stage to its container */
  function fit() {
    const f = $("cx-fit"), st = $("cx-stage"), w = f.clientWidth || window.innerWidth;
    const ww = $("cx-wwrap");
    if (w <= 760) {                                   // phone / narrow iframe: stacked layout, popup scaled to fit
      f.style.height = ""; st.style.transform = "";
      const k = Math.min(1, (w - 24) / 465);
      ww.style.transform = `scale(${k})`; ww.style.left = (w - 465 * k) / 2 + "px"; ww.style.top = Math.max(24, (window.innerHeight - 255 * k) / 2 - 40) + "px"; return;
    }
    ww.style.transform = ww.style.left = ww.style.top = "";
    const k = w / 1198; st.style.transform = `scale(${k})`; f.style.height = 768 * k + "px";
  }

  function init() {
    renderKpis(); renderWorkforce(); renderHiring(); renderAttrition(); renderInsights(); renderPerf(); renderActivity(); wireSearch(); wireMenus(); wireWelcome();
    if (REF) { $("cx-kbd").textContent = "38 K"; const lg = document.querySelector(".cx-legend span:nth-child(2)"); if (lg) lg.lastChild.textContent = "Attrition"; }
    $("cx-period").onchange = (e) => { period = +e.target.value; renderKpis(); renderWorkforce(); };
    $("lnk-att").onclick = openHighRisk; $("lnk-ins").onclick = openInsights; $("lnk-perf").onclick = openPerf; $("lnk-act").onclick = openActivity;
    // the sidebar promo button exists only in the full page, not in the Streamlit embed: never let a missing element stop init()
    if ($("btn-ai")) $("btn-ai").onclick = openFeatures;
    if ($("btn-feat")) $("btn-feat").onclick = openFeatures;
    if ($("cx-modal-x")) $("cx-modal-x").onclick = closeModal;
    $("cx-modal-bg").addEventListener("click", (e) => { if (e.target === $("cx-modal-bg")) closeModal(); });
    window.addEventListener("resize", fit); fit();
    if (window.ResizeObserver) new ResizeObserver(fit).observe($("cx-fit"));
    document.documentElement.setAttribute("data-ready", "1");
  }
  init();
})();
