"""Self-contained interactive charts (vanilla JS + SVG, no external libraries) in the CortexHR theme.

Each function returns a complete HTML document for ``streamlit.components.v1.html``; the same documents are rendered with Playwright
in the tests, so they are verified outside Streamlit.
"""
from __future__ import annotations

import json

_BASE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><style>
:root{--gold:#c9a24d;--ink:#e9efe9;--muted:#9db8ae;--tan:#d4a95f}*{box-sizing:border-box}html,body{margin:0;background:transparent;color:var(--ink);font-family:Inter,"Segoe UI",system-ui,sans-serif}
.box{border:1px solid rgba(201,162,77,.46);border-radius:18px;padding:12px 14px 10px;background:linear-gradient(150deg,rgba(10,72,63,.92),rgba(3,34,32,.96));position:relative}
.bar{display:flex;gap:16px;flex-wrap:wrap;align-items:center;font-size:12.5px;color:#cfe0d8;margin-bottom:4px}.bar label{display:flex;gap:6px;align-items:center;cursor:pointer}.bar input{accent-color:#c9a24d}
.k{display:inline-block;width:18px;height:3px;margin-right:6px;vertical-align:middle;border-radius:2px}svg{width:100%;height:auto;display:block}
#tip{position:absolute;pointer-events:none;background:#04201e;border:1px solid rgba(201,162,77,.6);border-radius:10px;padding:7px 10px;font-size:12px;line-height:1.4;display:none;z-index:5;white-space:nowrap}
.note{font-size:11.5px;color:var(--muted);margin-top:4px}</style></head><body>"""


def forecast_chart_html(title: str, history: list[dict], forecast: list[dict], events: list[dict], unit: str = "", note: str = "") -> str:
    """History line, forecast line, shaded 80% interval, and workforce-event bands, with hover values and toggles."""
    data = json.dumps({"title": title, "history": history, "forecast": forecast, "events": events, "unit": unit}).replace("</", "<\\/")
    return _BASE + f'''<div class="box" id="box"><div class="bar"><b style="color:#ead9ae;font-size:14px" id="t"></b>
<span><i class="k" style="background:#7fb5a6"></i>History</span><span><i class="k" style="background:#d9b36c"></i>Forecast</span><span><i class="k" style="background:rgba(217,179,108,.35);height:9px"></i>80% range</span>
<label><input type="checkbox" id="band" checked>Show 80% range</label><label><input type="checkbox" id="ev" checked>Show workforce events</label></div>
<svg id="c" viewBox="0 0 900 340" role="img"></svg><div id="tip"></div><div class="note">{note}</div></div>
<script>(function(){{const D={data},$=i=>document.getElementById(i);$("t").textContent=D.title;const NS="http://www.w3.org/2000/svg";
const W=900,H=340,L=54,R=18,T=14,B=34,h=D.history,f=D.forecast,all=h.map(x=>({{m:x.month,v:x.value}})).concat(f.map(x=>({{m:x.forecast_month,v:x.forecast_value}})));
const t0=new Date(all[0].m).getTime(),t1=new Date(all[all.length-1].m).getTime();let lo=Math.min(...h.map(x=>x.value),...f.map(x=>x.lower_80)),hi=Math.max(...h.map(x=>x.value),...f.map(x=>x.upper_80));
const pad=(hi-lo)*.08||1;lo=Math.max(0,lo-pad);hi+=pad;const X=m=>L+(new Date(m).getTime()-t0)/(t1-t0)*(W-L-R),Y=v=>T+(1-(v-lo)/(hi-lo))*(H-T-B),fmt=v=>(Math.abs(v)>=100?v.toFixed(0):v.toFixed(2))+D.unit;
function draw(){{const sb=$("band").checked,se=$("ev").checked;let s="";for(let i=0;i<=4;i++){{const v=lo+(hi-lo)*i/4,y=Y(v);s+=`<line x1="${{L}}" x2="${{W-R}}" y1="${{y}}" y2="${{y}}" stroke="rgba(255,255,255,.07)"/><text x="${{L-8}}" y="${{y+4}}" text-anchor="end" fill="#9db8ae" font-size="11">${{v>=100?v.toFixed(0):v.toFixed(1)}}</text>`}}
const yrs=new Set();all.forEach(p=>{{const d=new Date(p.m);if(d.getMonth()===0&&!yrs.has(d.getFullYear())){{yrs.add(d.getFullYear());s+=`<text x="${{X(p.m)}}" y="${{H-10}}" fill="#9db8ae" font-size="11" text-anchor="middle">${{d.getFullYear()}}</text><line x1="${{X(p.m)}}" x2="${{X(p.m)}}" y1="${{T}}" y2="${{H-B}}" stroke="rgba(255,255,255,.05)"/>`}}}});
if(se)D.events.forEach(e=>{{const a=Math.max(X(e.start_date),L),b=Math.min(X(e.end_date),W-R);if(b<L||a>W-R)return;const c=e.effect==="hiring_freeze"?"rgba(127,138,160,.30)":e.effect==="hiring_boost"?"rgba(127,176,105,.22)":"rgba(217,197,138,.20)";s+=`<rect x="${{a}}" y="${{T}}" width="${{Math.max(3,b-a)}}" height="${{H-T-B}}" fill="${{c}}"><title>${{e.event_id}}: ${{e.event_name}} (${{e.effect}})</title></rect>`}});
const split=X(f[0].forecast_month);s+=`<line x1="${{split}}" x2="${{split}}" y1="${{T}}" y2="${{H-B}}" stroke="#d9b36c" stroke-dasharray="4 4" opacity=".6"/><text x="${{split+5}}" y="${{T+11}}" fill="#d9b36c" font-size="10.5">forecast</text>`;
if(sb){{const up=f.map(p=>X(p.forecast_month)+" "+Y(p.upper_80)).join("L"),dn=f.slice().reverse().map(p=>X(p.forecast_month)+" "+Y(p.lower_80)).join("L");s+=`<path d="M${{up}}L${{dn}}Z" fill="rgba(217,179,108,.30)" stroke="rgba(217,179,108,.5)"/>`}}
const hl=h.map((p,i)=>(i?"L":"M")+X(p.month)+" "+Y(p.value)).join("");s+=`<path d="${{hl}}" fill="none" stroke="#7fb5a6" stroke-width="2.2" stroke-linejoin="round"/>`;
const fl="M"+X(h[h.length-1].month)+" "+Y(h[h.length-1].value)+f.map(p=>"L"+X(p.forecast_month)+" "+Y(p.forecast_value)).join("");s+=`<path d="${{fl}}" fill="none" stroke="#d9b36c" stroke-width="2.4" stroke-dasharray="6 4" stroke-linejoin="round"/>`;
f.forEach(p=>{{s+=`<circle cx="${{X(p.forecast_month)}}" cy="${{Y(p.forecast_value)}}" r="4.2" fill="#d9b36c" stroke="#0a3a37" stroke-width="1.5"/>`}});
s+=`<line id="cur" y1="${{T}}" y2="${{H-B}}" stroke="#fff" opacity="0" stroke-width="1"/><rect id="hit" x="${{L}}" y="${{T}}" width="${{W-L-R}}" height="${{H-T-B}}" fill="transparent"/>`;$("c").innerHTML=s;
const hit=$("hit"),cur=$("cur"),tip=$("tip"),box=$("box");hit.onmousemove=ev=>{{const r=$("c").getBoundingClientRect(),x=(ev.clientX-r.left)/r.width*W;let best=null,bd=1e18;all.forEach(p=>{{const d=Math.abs(X(p.m)-x);if(d<bd){{bd=d;best=p}}}});
const fp=f.find(p=>p.forecast_month===best.m),mo=new Date(best.m).toLocaleDateString("en-GB",{{month:"short",year:"numeric"}});tip.innerHTML=`<b>${{mo}}</b><br>${{fp?"Forecast":"Actual"}}: <b>${{fmt(best.v)}}</b>`+(fp?`<br>80% range: ${{fmt(fp.lower_80)}} to ${{fmt(fp.upper_80)}}`:"");
tip.style.display="block";const bx=box.getBoundingClientRect();tip.style.left=Math.min(ev.clientX-bx.left+12,bx.width-190)+"px";tip.style.top=(ev.clientY-bx.top-50)+"px";cur.setAttribute("x1",X(best.m));cur.setAttribute("x2",X(best.m));cur.setAttribute("opacity",".35")}};
hit.onmouseleave=()=>{{tip.style.display="none";cur.setAttribute("opacity","0")}}}}
$("band").onchange=draw;$("ev").onchange=draw;draw();document.documentElement.setAttribute("data-ready","1")}})();</script></body></html>'''


def drift_chart_html(psi_rows: list[dict], history: list[dict], watch: float = 0.10, alert: float = 0.25) -> str:
    """PSI per feature as bars against the WATCH/ALERT lines, plus the monitoring status over time."""
    data = json.dumps({"psi": psi_rows, "history": history, "watch": watch, "alert": alert}).replace("</", "<\\/")
    return _BASE + f'''<div class="box"><div class="bar"><b style="color:#ead9ae;font-size:14px">Population Stability Index (PSI) by feature</b>
<span><i class="k" style="background:#6fcf97"></i>OK</span><span><i class="k" style="background:#e0b24f"></i>WATCH</span><span><i class="k" style="background:#e0735a"></i>ALERT</span></div>
<svg id="c" viewBox="0 0 900 330" role="img"></svg><div class="note">Bars are capped at PSI 1.0 for readability; the number shows the true value. Lines mark 0.10 (WATCH) and 0.25 (ALERT).</div></div>
<script>(function(){{const D={data},$=i=>document.getElementById(i),W=900,L=250,R=60,T=10,rowH=34,rows=D.psi;let s="";const X=v=>L+Math.min(v,1)/1*(W-L-R);
rows.forEach((r,i)=>{{const y=T+i*rowH,c=r.status==="ALERT"?"#e0735a":r.status==="WATCH"?"#e0b24f":"#6fcf97";s+=`<text x="${{L-10}}" y="${{y+20}}" text-anchor="end" fill="#dbe8e1" font-size="13">${{r.feature}}</text><rect x="${{L}}" y="${{y+6}}" width="${{Math.max(2,X(r.psi)-L)}}" height="20" rx="5" fill="${{c}}" opacity=".85"><title>${{r.feature}}: PSI ${{r.psi.toFixed(3)}} (${{r.status}})</title></rect><text x="${{X(r.psi)+8}}" y="${{y+21}}" fill="#ead9ae" font-size="12.5">${{r.psi.toFixed(3)}}</text>`}});
const bottom=T+rows.length*rowH;[D.watch,D.alert].forEach((v,i)=>{{s+=`<line x1="${{X(v)}}" x2="${{X(v)}}" y1="${{T}}" y2="${{bottom}}" stroke="${{i?"#e0735a":"#e0b24f"}}" stroke-dasharray="5 4"/><text x="${{X(v)}}" y="${{bottom+14}}" fill="${{i?"#e0735a":"#e0b24f"}}" font-size="11" text-anchor="middle">${{v}}</text>`}});
const c=$("c");c.setAttribute("viewBox",`0 0 ${{W}} ${{bottom+24}}`);c.innerHTML=s;document.documentElement.setAttribute("data-ready","1")}})();</script></body></html>'''
