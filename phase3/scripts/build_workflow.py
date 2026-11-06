"""Builds notebook 15: the end-to-end Phase 3 workflow (reads the artifacts written by notebooks 01 to 14)."""
from nbkit import NB


def nb15():
    n = NB("Phase 3, Notebook 15: Complete workflow, evidence dashboard and walkthroughs", "11 to 15 (all five tasks)", "Show that every Phase 3 deliverable exists, agrees with the executed notebooks, and works end to end for one concrete case per task.", [])
    n.md("This notebook **recomputes nothing heavy**. It reads the artifacts the task notebooks wrote (models, tables, figures), checks them, and walks one example through each task. Run notebooks 01 to 14 first (`python scripts/execute_notebooks.py`).")
    n.md("## A. Environment and reproducibility")
    n.code("""import json, platform, sklearn, scipy, importlib, joblib
from cortexhr.phase3.attrition.v2 import file_sha256, optional_libs
print({'python': platform.python_version(), 'scikit-learn': sklearn.__version__, 'pandas': pd.__version__, 'numpy': np.__version__, 'seed': SEED})
print('optional libraries (used automatically when installed, NOT needed for any result here):', optional_libs())
print('config file:', (ROOT / 'config' / 'phase3_config.yaml').relative_to(ROOT))""")
    n.md("## B. Artifact inventory with checksums")
    n.code("""rows = []
for name, meta in (('resume_ranker_v2.joblib', 'resume_ranker_v2_metadata.json'), ('attrition_v2.joblib', 'attrition_v2_metadata.json'), ('attrition_v2_train_only.joblib', 'attrition_v2_metadata.json'), ('salary_band_v1.joblib', 'salary_band_v1_metadata.json')):
    m = json.load(open(P.METADATA / meta)); rec = m.get('sha256') or m['artifact_sha256'][name]; rows.append({'artifact': name, 'size_kb': round((P.MODELS / name).stat().st_size / 1024), 'metadata': meta, 'checksum_matches': file_sha256(P.MODELS / name) == rec})
pd.DataFrame(rows)""")
    n.md("## C. Evidence dashboard: the headline number of each task, read from the saved files")
    n.code("""q = pd.read_csv(P.task_dir(11) / 'ranking_quality_summary.csv').set_index('engine'); fa = json.load(open(P.task_dir(12) / 'fairness_results.json')); cv = pd.read_csv(P.task_dir(13) / 'cv_summary.csv').set_index('model')
v2m = json.load(open(P.METADATA / 'attrition_v2_metadata.json')); risk = pd.read_csv(P.PREDICTIONS / 'attrition_risk_2026.csv'); sm = json.load(open(P.METADATA / 'salary_band_v1_metadata.json')); ex = json.load(open(P.task_dir(14) / 'explanation_run.json'))
dash = pd.DataFrame([
 ('11 resume ranking', 'precision@5 / NDCG@5 / hit rate@5', '%.3f / %.3f / %.3f' % tuple(q.loc['final engine (v2, masked)', ['precision@5', 'ndcg@5', 'hit_rate@5']]), 'random: %.3f / %.3f / %.3f' % tuple(q.loc['random ordering', ['precision@5', 'ndcg@5', 'hit_rate@5']])),
 ('12 fairness audit', 'mean twin gap before -> after', '%.6f -> %.6f' % (fa['before']['mean_abs_gap'], fa['after']['mean_abs_gap']), 'max %.4f -> %.4f; NDCG@5 %.3f -> %.3f' % (fa['before']['max_abs_gap'], fa['after']['max_abs_gap'], fa['quality_before']['ndcg@5'], fa['quality_after']['ndcg@5'])),
 ('13 attrition v2', 'CV ROC-AUC baseline vs v2 blend', '%.3f vs %.3f' % (cv.loc['Logistic Regression (balanced) [baseline]', 'roc_auc_mean'], cv.loc['attrition-v2 candidate: LR + tuned HGB blend', 'roc_auc_mean']), v2m['verdict']['text'][:60] + '...'),
 ('14 risk report', 'rows / bands / format checks', '%d / %s / %d of %d passed' % (len(risk), risk.risk_band.value_counts().to_dict(), sum(c['passed'] for c in ex['checks']), len(ex['checks'])), 'additivity error %.1e' % ex['additivity_max_error']),
 ('15 pay bands', 'test MAE (Rs) / R2 / band coverage', '%.0f / %.3f / %.3f' % (sm['test_metrics'][sm['model']]['MAE_rupees'], sm['test_metrics'][sm['model']]['R2'], sm['band_coverage_test']), 'role-average baseline MAE %.0f' % sm['test_metrics']['baseline: average offered salary by role']['MAE_rupees'])],
 columns=['task', 'measure', 'result', 'reference']); dash""")
    n.md("## D. One walkthrough per task")
    n.code("""# Task 11: rank resumes for one job through the API contract
from cortexhr.phase3.resume.api import handle_rank_request
ranker = joblib.load(P.MODELS / 'resume_ranker_v2.joblib'); resp = handle_rank_request({'job_id': 'J05', 'top_k': 3}, ranker)
print('job', resp['job_id'], '| engine', resp['engine_version'], '| never used:', resp['never_used'])
for c in resp['candidates']: print(c['rank'], c['resume_id'], c['score'], c['score_breakdown'], '| missing required:', c['missing_required_skills'])""")
    n.code("""# Task 12: the twin test on one pair
from cortexhr.phase3.common.io import read_csv
pairs = read_csv('resume_bias_pairs'); labels = read_csv('resume_job_labels'); r1 = joblib.load(P.MODELS / 'resume_ranker_v2.joblib'); p = pairs.iloc[0]; job = labels[labels.resume_id == p.resume_id_original].job_id.iloc[0]
a, b = r1.score_pair(p.resume_id_original, job), r1.score_pair(p.resume_id_twin, job); print(p.pair_id, job, '| original score %.6f | twin score %.6f | gap %.6f' % (a['score'], b['score'], b['score'] - a['score']))""")
    n.code("""# Task 13 and 14: one employee, score + reasons
i = int(risk.risk_score_12m.idxmax()); r = risk.loc[i]; print(r.employee_id, r.risk_band, r.risk_score_12m); [print('  -', r[f'top_reason_{k}']) for k in (1, 2, 3)]; print(r.model_note[:140] + '...')""")
    n.code("""# Task 15: one band
from cortexhr.phase3.salary import model as S
art = joblib.load(P.MODELS / 'salary_band_v1.joblib'); d = S.load_salary().iloc[[10]]; print(S.recommend_band(art['model'], d, art['band_multipliers'], art['features']).round(0).to_dict('records')[0], '| actual offer', int(d.offered_annual_fixed.iloc[0]))""")
    n.md("## E. Workflow coverage: the 38 stages per task, counted from the notebooks")
    n.code("""import re
NB_ = {11: ['01', '02', '03', '04'], 12: ['05', '06'], 13: ['07', '08', '09'], 14: ['10', '11'], 15: ['12', '13', '14']}; rows = []
for t, pre in NB_.items():
    st, na = set(), set()
    for p_ in pre:
        for c in json.load(open(next((ROOT / 'notebooks').glob(p_ + '_*.ipynb')), encoding='utf-8'))['cells']:
            s = ''.join(c['source']); m = re.match(r'## Stage (\\d+):', s) if c['cell_type'] == 'markdown' else None
            if m: st.add(int(m.group(1))); na |= {int(m.group(1))} if 'Not applicable here, and why:' in s else set()
    rows.append({'task': t, 'notebooks': ', '.join(pre), 'stages covered': len(st), 'missing': sorted(set(range(1, 39)) - st) or 'none', 'marked not applicable (with reason)': sorted(na) or 'none'})
pd.DataFrame(rows)""")
    n.md("## F. Requirement traceability")
    n.code("""tp = ROOT / 'docs' / 'requirements_traceability.csv'
if tp.exists():
    t = pd.read_csv(tp); print(len(t), 'requirements tracked;', t.status.value_counts().to_dict()); print(t.groupby('task').status.value_counts().unstack(fill_value=0).to_string())
else: print('run: python scripts/make_docs.py  (writes docs/requirements_traceability.*)')""")
    n.md("## G. Known limitations (honest list)\n\n1. The main attrition-v2 production blend remains LR + HistGradientBoosting; a real LightGBM candidate is separately trained, tuned and evaluated on the same Task 3 folds when installed, and was not a proven improvement. 2. Task 14 uses grouped Monte Carlo Shapley for all 430 employees and a real SHAP PermutationExplainer cross-check on a reproducible 10-row sample when installed. 3. The restricted `unlock_CX-12` and `unlock_CX-15` files are used for measurement only and are not committed; they were supplied and validated in the final audit, Task 12 and Task 15 were rerun, and Task 13 group counts were cross-checked while its full retraining rerun exceeded the execution time limit. 4. The attrition v2 gain over the baseline is not proven. 5. Resume ranking evidence rests on 10 jobs and 555 human labels. 6. Pay bands learn from historical offers that may contain past bias.")
    return n


if __name__ == "__main__":
    from pathlib import Path
    print(nb15().save(Path(__file__).resolve().parents[1] / "notebooks" / "15_phase3_complete_workflow.ipynb"))
