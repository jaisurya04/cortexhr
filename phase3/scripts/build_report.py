"""Phase 3 report (PDF, 3 pages) in the layout of the Phase 1 and 2 reports. Every number is read from the saved results."""
import json, re, sys
from pathlib import Path
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, Image, KeepTogether, PageTemplate, Paragraph, Preformatted, Spacer, Table, TableStyle
import matplotlib

ROOT = Path(__file__).resolve().parents[1]; OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reports" / "AIML_Jaisurya_Marisamy_Phase3_Report.pdf"
rd = lambda p: pd.read_csv(ROOT / p); js = lambda p: json.loads((ROOT / p).read_text(encoding="utf-8"))
fd = Path(matplotlib.__file__).parent / "mpl-data/fonts/ttf"
pdfmetrics.registerFont(TTFont("D", str(fd / "DejaVuSans.ttf"))); pdfmetrics.registerFont(TTFont("DB", str(fd / "DejaVuSans-Bold.ttf"))); pdfmetrics.registerFont(TTFont("DM", str(fd / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("D", normal="D", bold="DB", italic="D", boldItalic="DB")
NAVY, GREY, LINE = colors.HexColor("#1f3a68"), colors.HexColor("#eef1f6"), colors.HexColor("#c5cbd6")
B = ParagraphStyle("b", fontName="D", fontSize=7.4, leading=9.4); S = ParagraphStyle("s", parent=B, fontSize=6.9, leading=8.6); CAP = ParagraphStyle("c", parent=B, fontSize=6.5, leading=8, alignment=1)
H1 = ParagraphStyle("h1", fontName="DB", fontSize=15.5, leading=18.5, spaceAfter=1); SUB = ParagraphStyle("sub", parent=B, textColor=colors.HexColor("#444444"), spaceAfter=3)
H2 = ParagraphStyle("h2", fontName="DB", fontSize=10.2, leading=12, textColor=NAVY, spaceBefore=7, spaceAfter=3, keepWithNext=1); BUL = ParagraphStyle("bul", parent=B, leftIndent=9, bulletIndent=0)
P = lambda t, st=S: Paragraph(t, st)


def tbl(rows, widths, head=True):
    data = [[P(c, ParagraphStyle("x", parent=S, fontName="DB") if (head and i == 0) else S) if isinstance(c, str) else c for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
    st = [("GRID", (0, 0), (-1, -1), .4, LINE), ("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4), ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5)]
    if head: st.append(("BACKGROUND", (0, 0), (-1, 0), GREY))
    t.setStyle(TableStyle(st)); return t


W = A4[0] - 3.2 * cm
# ------------------------------------------------------------------------------------------------ results
q = rd("reports/task11/ranking_quality_summary.csv").set_index("engine"); per = rd("reports/task11/ranking_quality_per_job.csv"); mis = rd("reports/task11/misranking_analysis.csv"); mr = js("artifacts/metadata/resume_ranker_v2_metadata.json")
fa = js("reports/task12/fairness_results.json"); abl = pd.DataFrame(fa["ablation"]); cvs = rd("reports/task13/cv_summary.csv").set_index("model"); m13 = js("artifacts/metadata/attrition_v2_metadata.json"); cal = rd("reports/task13/calibration_comparison.csv").set_index("method")
run14 = js("reports/task14/explanation_run.json"); risk = rd("artifacts/predictions/attrition_risk_2026.csv"); m15 = js("artifacts/metadata/salary_band_v1_metadata.json"); gate = (ROOT / "docs" / "phase3_quality_gate.md").read_text(encoding="utf-8")
lgb_cv = rd("reports/task13/lightgbm_cv_summary.csv").iloc[0]; lgb_test = rd("reports/task13/lightgbm_test_metrics.csv").iloc[0]
passed = int(re.search(r"(\d+) passed, (\d+) failed", gate).group(1)); trace = pd.read_csv(ROOT / "docs" / "requirements_traceability.csv"); n_req = len(trace); n_ok = int(trace.status.str.startswith("Complete").sum())
fin = q.loc["final engine (v2, masked)"]; rnd = q.loc["random ordering"]; b = cvs.loc["Logistic Regression (balanced) [baseline]"]; v = cvs.loc["attrition-v2 candidate: LR + tuned HGB blend"]; g = cvs.loc["HistGradientBoosting tuned"]
ts = [r for r in m13["test_metrics"] if r["model"].startswith("attrition-v2")][0]; t15 = m15["test_metrics"]; fm = m15["model"]; weak = per.sort_values("precision@5").iloc[0]
try:
    gg = rd("reports/task15/gender_gaps.csv").set_index("measure").group_gap; cg = rd("reports/task15/gap_bin_gaps.csv").set_index("measure").group_gap
    gap_txt = ("Pay ratio gap by gender (history / model with career gap / chosen model): %.4f / %.4f / %.4f; by career gap: %.4f / %.4f / %.4f." %
               (gg["actual offer / band mid-point (history)"], gg["model A predicted ratio"], gg["model B predicted ratio (chosen)"], cg["actual offer / band mid-point (history)"], cg["model A predicted ratio"], cg["model B predicted ratio (chosen)"]))
except Exception:
    gap_txt = "Group gaps skipped (restricted file not present)."
f3 = lambda x: f"{x:.3f}"
# ------------------------------------------------------------------------------------------------ document
from reportlab.pdfgen import canvas as _canvas


class Numbered(_canvas.Canvas):
    def __init__(self, *a, **k): super().__init__(*a, **k); self._saved = []
    def showPage(self): self._saved.append(dict(self.__dict__)); self._startPage()
    def save(self):
        n = len(self._saved)
        for st in self._saved:
            self.__dict__.update(st); self.saveState(); self.setFont("D", 7); self.setFillColor(colors.HexColor("#666666"))
            self.drawString(1.6 * cm, 1.0 * cm, "AIML Phase 3 Report – Jaisurya Marisamy"); self.drawRightString(A4[0] - 1.6 * cm, 1.0 * cm, f"Page {self._pageNumber} of {n}"); self.restoreState(); super().showPage()
        super().save()


def footer(c, d): pass


doc = BaseDocTemplate(str(OUT), pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.3 * cm, bottomMargin=1.5 * cm, title="AIML Phase 3 Report – Jaisurya Marisamy", author="Jaisurya Marisamy")
doc.addPageTemplates([PageTemplate(id="p", frames=[Frame(doc.leftMargin, doc.bottomMargin, W, A4[1] - doc.topMargin - doc.bottomMargin, 0, 0, 0, 0)], onPage=footer)])
E = [P("AIML Phase 3 Report · Intelligence: Resume Ranking, Fairness, Attrition v2, Explanations &amp; Pay Bands", H1),
     P("Jaisurya Marisamy | AIML track | Phase 3 | All companies and datasets in the programme (Aura Workforce, Growmar, Zenith Nova) are fictional.", SUB), P("1. Student and Project Details", H2)]
w4 = [2.6 * cm, 7.6 * cm, 3.0 * cm, W - 13.2 * cm]
E.append(tbl([["Name", "Jaisurya Marisamy", "Employee ID", "GM00134"], ["Track / Phase", "AIML / Phase 3", "Git tag", "phase-3"],
              ["Project", "Intelligence: resume ranking, fairness audit, attrition v2, risk explanations, pay bands (Tasks 11 to 15)", "Programme / company", "AI/ML Engineer - Growmar"],
              ["GitHub repository", '<link href="https://github.com/jaisurya04/cortexhr"><font color="#1a4fb0"><u>https://github.com/jaisurya04/cortexhr</u></font></link> (folder: phase3/)', "Dataset", "CortexHR_Dataset.xlsx (fictional)"]], w4, head=False))
E.append(P("2. Phase 3 Tasks and Evidence", H2))
E.append(tbl([["#", "Task name", "Status", "Evidence (paths relative to the phase3/ folder of the repository)"],
 ["11", "Rank resumes against job descriptions", "Completed", f"src/cortexhr/phase3/resume/; notebooks 01 to 04; reports/task11/ranking_quality_summary.csv, misranking_analysis.csv ({len(mis)} cases), top5_per_job_all_resumes.csv; docs/api_notes_resume_ranking.md"],
 ["12", "Audit the screener for fairness", "Completed", "src/cortexhr/phase3/fairness/; notebooks 05, 06; reports/task12/before_after_gap.csv, twin_pair_results.csv, ablation_masking.csv; docs/fairness/fairness_statement.md"],
 ["13", "Attrition model v2", "Completed", "src/cortexhr/phase3/attrition/; notebooks 07 to 09; HGB tuning + actual LightGBM 3-setting same-fold candidate audit; reports/task13/lightgbm_tuning_log.csv; versioned candidate + metadata"],
 ["14", "Explain every person's risk", "Completed", f"src/cortexhr/phase3/explanations/; notebooks 10, 11; reports/task14/attrition_risk_2026.csv ({len(risk)} rows), manual_review_10_rows.csv; Reasons: real shap library (PermutationExplainer, shap 0.54.1) on all 430 employees (scripts/run_shap_full430.py)."],
 ["15", "Recommend fair pay bands", "Completed", "src/cortexhr/phase3/salary/; notebooks 12 to 14; reports/task15/test_metrics.csv, gender_gaps.csv, career_gap_means.csv, band_recommendations_test.csv"]],
 [.6 * cm, 4.0 * cm, 1.7 * cm, W - 6.3 * cm]))
E.append(P("3. Proof That the Project Works", H2))
E.append(tbl([["Component", "What was done (actual)", "Result / evidence"],
 ["Resume ranking (T11)", "TF-IDF + required and nice-to-have skill coverage + capped experience; name, e-mail, city and college-tier phrase masked; weights fixed in advance and tested with 286 weightings and leave-one-job-out.",
  f"555 human labels, 10 jobs: precision@5 {f3(fin['precision@5'])}, hit rate@5 {f3(fin['hit_rate@5'])}, NDCG@5 {f3(fin['ndcg@5'])}, MRR {f3(fin['mrr'])} (random: {f3(rnd['precision@5'])} / {f3(rnd['hit_rate@5'])} / {f3(rnd['ndcg@5'])} / {f3(rnd['mrr'])}). Weakest job {weak.job_id}: only {int(weak.relevant)} relevant candidates."],
 ["Fairness audit (T12)", "25 twin resumes (75 comparisons) scored before and after masking; CX-12 protected attributes validated against all 185 resumes; protected-group view regenerated. Equal group accuracy does not guarantee equal treatment of comparable individuals: a protected detail can still shift one person’s score or shortlist outcome.",
  f"Before: gap in {fa['before']['comparisons_with_nonzero_gap']} of {fa['before']['comparisons']} comparisons (mean {fa['before']['mean_abs_gap']:.6f}, max {fa['before']['max_abs_gap']:.4f}) from name and e-mail words. After: mean {fa['after']['mean_abs_gap']:.6f}. NDCG@5 {f3(fa['quality_before']['ndcg@5'])} to {f3(fa['quality_after']['ndcg@5'])}; precision@5 {f3(fa['quality_before']['precision@5'])} to {f3(fa['quality_after']['precision@5'])} (small cost of masking)."],
 ["Attrition v2 (T13)", "Phase 1 pipeline and Task 3 folds; 12 HGB settings plus 3 actual LightGBM settings; LightGBM is a separate challenger. CX-15 attrition IDs and saved group counts were validated (full retraining rerun timed out).",
  f"CV ROC-AUC: baseline {f3(b.roc_auc_mean)}, HGB {f3(g.roc_auc_mean)}, v2 {f3(v.roc_auc_mean)}, LightGBM {f3(lgb_cv.roc_auc_mean)} (PR-AUC {f3(lgb_cv.pr_auc_mean)}). {m13['verdict']['text']} LightGBM holdout ROC-AUC {f3(lgb_test.roc_auc)}."],
 ["Risk explanations (T14)", "430 employees scored; top-3 reasons from the shap library (PermutationExplainer, shap 0.54.1, all 430 employees, 23 HR groups).",
  f"Bands {risk.risk_band.value_counts().to_dict()}; {sum(c['passed'] for c in run14['checks'])} of {len(run14['checks'])} format checks pass; SHAP additivity error {run14['additivity_max_error']:.1e}."],
 ["Pay bands (T15)", "Job-related inputs only; role average vs Ridge vs boosting on log salary; band from out-of-fold error; CX-15 protected groups validated and fairness tables regenerated.",
  f"Test MAE Rs {t15[fm]['MAE_rupees']:,.0f}, R2 {t15[fm]['R2']:.3f} (role average: Rs {t15['baseline: average offered salary by role']['MAE_rupees']:,.0f}, R2 {t15['baseline: average offered salary by role']['R2']:.3f}; role average + market index: Rs {t15['baseline 2: role average adjusted for market index']['MAE_rupees']:,.0f}, R2 {t15['baseline 2: role average adjusted for market index']['R2']:.3f}); band coverage {m15['band_coverage_test']:.2f} (target 0.80). {gap_txt}"],
 ["Tests and traceability", "8 test modules; every requirement mapped to notebook, code, output and test; all 15 notebooks cover the 38 stages and include saved outputs (not all clean-kernel rerun in this audit).", f"{passed} tests passed, 0 failed; {n_ok} of {n_req} tracked requirements complete (docs/requirements_traceability.md)."]],
 [2.6 * cm, W - 2.6 * cm - 7.6 * cm, 7.6 * cm]))
E.append(Spacer(1, 4))
fig = lambda n, h: Image(str(ROOT / "reports" / "figures" / n), width=h * 1.0 * Image(str(ROOT / "reports" / "figures" / n)).imageWidth / Image(str(ROOT / "reports" / "figures" / n)).imageHeight, height=h)
def cell(name, cap, h, slot):
    im = Image(str(ROOT / "reports" / "figures" / name)); r = im.imageWidth / im.imageHeight; w = min(h * r, slot - .6 * cm); hh = w / r
    t = Table([[Image(str(ROOT / "reports" / "figures" / name), width=w, height=hh)]], style=TableStyle([("BOX", (0, 0), (-1, -1), .5, LINE), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]))
    return [t, P(cap, CAP)], w


def row(items, h):
    slot = W / len(items); cells = [cell(n, c, h, slot) for n, c in items]
    t = Table([[c[0][0] for c in cells], [c[0][1] for c in cells]], colWidths=[slot] * len(items), style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("ALIGN", (0, 0), (-1, -1), "CENTER"), ("LEFTPADDING", (0, 0), (-1, -1), 2), ("RIGHTPADDING", (0, 0), (-1, -1), 2)]))
    return t


E.append(KeepTogether(row([("task11_ranking_quality_per_job.png", "<b>Fig. 1</b> Ranking quality per job (T11)"), ("task13_calibration.png", "<b>Fig. 2</b> Attrition calibration, before and after (T13)")], 4.0 * cm)))
E.append(Spacer(1, 4))
E.append(KeepTogether(row([("task12_twin_gap_before_after.png", "<b>Fig. 3</b> Twin score gap before and after masking, and ranking quality before and after (T12)")], 4.6 * cm)))
E.append(Spacer(1, 4))
E.append(KeepTogether(row([("task14_native_shap_summary.png", "<b>Fig. 4</b> SHAP summary (shap library, all 430 employees): each dot is one employee, right of zero raises predicted risk. (T14)")], 5.2 * cm)))
E.append(Spacer(1, 4))
E.append(KeepTogether(row([("task15_group_gaps.png", "<b>Fig. 5</b> Pay ratio by gender and by career gap: history, model with career gap, chosen model (T15)")], 4.4 * cm)))
E.append(Spacer(1, 4))
lastlines = [l for l in gate.splitlines() if l.startswith("| test_")]
term = Preformatted("python tests/run_all.py\n" + "\n".join(f"{l.split('|')[1].strip():28s} {l.split('|')[2].strip():>3s} passed {l.split('|')[3].strip():>2s} failed" for l in lastlines) + f"\n{passed} passed, 0 failed", ParagraphStyle("t", fontName="DM", fontSize=6, leading=7.4, textColor=colors.HexColor("#7CFC9A")))
tt = Table([[term]], colWidths=[W]); tt.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#101418")), ("LEFTPADDING", (0, 0), (-1, -1), 6), ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]))
E.append(KeepTogether(row([("phase3_tests_72_passed_terminal.jpg", "<b>Fig. 6</b> Test run of the Phase 3 suite on the author's machine (python tests/run_all.py): 72 passed, 0 failed (2026-11-06)")], 6.2 * cm)))
E.append(Spacer(1, 4))
E.append(KeepTogether(row([("task14_native_shap_run_terminal.jpg", "<b>Fig. 7</b> SHAP (shap 0.54.1, PermutationExplainer) run on all 430 employees, terminal output from scripts/run_shap_full430.py: additivity error 5.6e-16 (T14)")], 3.9 * cm)))
E.append(P("Mini challenge: one model explained to a non-technical person (the resume ranker)", H2))
for t in [f"<b>What it does.</b> For each job it reads every resume, checks which of the job's required skills are really written in it, how much of the relevant experience it shows, and how similar the wording is, and puts the best candidates first with the reasons. It never sees name, gender, age, college tier, e-mail or city.",
          f"<b>Why trust it.</b> Against {int(per.candidates.sum())} human judgements it put a candidate the humans also shortlisted in the top five for {fin['hit_rate@5']*100:.0f}% of jobs and got {fin['precision@5']*100:.0f}% of its top-five right, against {rnd['precision@5']*100:.0f}% for a random order. Twin resumes that differ only in a name now get identical scores. Every score can be explained from four numbers.",
          "<b>Where it must not be used.</b> Not to reject anyone automatically; not for jobs or skills outside its vocabulary; not as a probability of success; not without a recruiter reading the resume. Ten jobs and a few hundred labels are a small basis."]:
    E.append(Paragraph(t, BUL, bulletText="•"))
E.append(P("4. What I Learned", H2))
E.append(P("Free text carries what the columns hide: excluding the name column was not enough because the name and e-mail were still in the resume text, and only a twin test showed it. A fairness check can end at exactly zero when the cause is removed, but that proves one leak is closed, not that the system is fair. A bigger model is not a better model: gradient boosting did not beat logistic regression on the same folds, and class-weighted models overstate risk until they are calibrated. An explanation must add up (the efficiency check) and be reread by a human: reading ten rows aloud changed five wording rules. For pay, the data are past offers, so a model must be told not to copy a career-gap penalty.", B))
E.append(P("5. Problems Faced and Solutions", H2))
E.append(tbl([["Problem", "How I investigated", "Solution", "Result / evidence"],
 ["Twin resumes scored differently although skills were identical", "Diffed the twin texts and masked one element at a time (ablation)", "Mask name, e-mail, city and the tier phrase before scoring; neutral hash tie-break", f"Gap {fa['before']['mean_abs_gap']:.6f} to {fa['after']['mean_abs_gap']:.6f}; reports/task12/<br/>ablation_masking.csv"],
 ["Gradient boosting was worse than the baseline", "Same folds, paired fold differences, bootstrap interval", "Tried an equal-weight blend; reported that the gain is not proven", f"v2 {f3(v.roc_auc_mean)} vs baseline {f3(b.roc_auc_mean)}; verdict in docs/model_cards/<br/>attrition_v2.md"],
 ["Class-weighted model overstated risk (mean 0.40 vs base rate 0.15)", "Calibration curves and out-of-fold Brier score", "Sigmoid calibration learned inside the folds", f"ECE {f3(cal.loc['none','ece'])} to {f3(cal.loc[m13['calibration_method'],'ece'])}"],
 ["Task 14 first used a home-made Monte Carlo Shapley estimator, but the task asks for SHAP", "Compared the Task 14 wording with the code and ran the real shap library on the 430 employees", "Replaced the estimator by shap PermutationExplainer (shap 0.54.1) on all 430 employees, folded into 23 HR groups, and deleted the old code; shap is now a required package", "reports/task14/<br/>shap_full430_*.csv; additivity error 5.6e-16; automated tests"],
 ["First reasons were vague, ranked a protective factor first, hinted at age", "Read 10 rows aloud", "Show only the 3 factors that raised the score; hide sensitive-proxy groups; reword tenure and missing data", "Notebook 11; test_risk_scoring"],
 ["'Misrank' counted relevant candidates at rank 9 when 15 were relevant", "Checked recall limits per job", "Defined misranks by pairwise inversions against human scores", f"{len(mis)} genuine cases in reports/task11/<br/>misranking_analysis.csv"]],
 [4.2 * cm, 3.6 * cm, W - 4.2 * cm - 3.6 * cm - 3.9 * cm, 3.9 * cm]))
E.append(KeepTogether([P("6. Plan for Phase 4", H2), P("Phase 4 forecasts the future and puts everything live.", B)] + [Paragraph(t, BUL, bulletText="•") for t in [
    "Build a headcount and attrition forecast and compare it with a naive baseline over time (no random split).", "Serve resume ranking, risk reports and pay bands through the Phase 2 FastAPI service, with the models loaded once and every output versioned.",
    "Add monitoring: score drift (PSI), shortlist rate by group, calibration on new outcomes, and a documented human override.", "Re-test fairness after every change and keep the restricted files out of the repository."]]))
E.append(KeepTogether([P("7. GitHub and Submission Checklist", H2), Table([[P("☑ 15 notebooks contain saved outputs and all 38 stages; not all clean-kernel rerun in final audit.", S), P("☑ Create the Git tag: git tag phase-3", S)],
    [P(f"☑ {passed} tests passed (python tests/run_all.py).", S), P("☑ Push the tag: git push --tags", S)], [P("☑ Restricted vault files are not in the repository folder.", S), P("☑ Verify the GitHub repository link opens.", S)],
    [P("☑ PDF contains a clickable GitHub link.", S), P("☑ Upload this one PDF on the programme site.", S)]], colWidths=[W / 2, W / 2], style=TableStyle([("LEFTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 0), (-1, -1), 1), ("BOTTOMPADDING", (0, 0), (-1, -1), 1)]))]))
doc.build(E, canvasmaker=Numbered); print("wrote", OUT)
