from _common import *
import json
from cortexhr.phase3.common.paths import ROOT, task_dir, METADATA
sys.path.insert(0, str(ROOT / "scripts"))

EXPECTED = {11: ["ranking_quality_per_job.csv", "ranking_quality_summary.csv", "misranking_analysis.csv", "top5_per_job_all_resumes.csv", "weight_experiments.csv"],
            12: ["before_after_gap.csv", "ranking_quality_before_after.csv", "twin_pair_results.csv", "ablation_masking.csv", "fairness_results.json"],
            13: ["tuning_log.csv", "fold_metrics.csv", "cv_summary.csv", "paired_differences.csv", "calibration_comparison.csv", "test_metrics.csv"],
            14: ["attrition_risk_2026.csv", "shap_full430_group_values.csv", "explanation_run.json", "manual_review_10_rows.csv"],
            15: ["cv_metrics.csv", "test_metrics.csv", "tuning_log.csv", "band_recommendations_test.csv"]}
NOTEBOOKS = {11: ["01", "02", "03", "04"], 12: ["05", "06"], 13: ["07", "08", "09"], 14: ["10", "11"], 15: ["12", "13", "14"]}


def test_every_task_has_its_evidence_files():
    for t, files in EXPECTED.items():
        for f in files: assert (task_dir(t) / f).exists(), (t, f)


def _nb(prefix):
    return json.loads(next((ROOT / "notebooks").glob(f"{prefix}_*.ipynb")).read_text(encoding="utf-8"))


def test_each_task_covers_all_38_workflow_stages():
    import re
    for t, prefixes in NOTEBOOKS.items():
        stages = set()
        for p in prefixes:
            for c in _nb(p)["cells"]:
                if c["cell_type"] == "markdown":
                    m = re.match(r"## Stage (\d+):", "".join(c["source"])); stages |= {int(m.group(1))} if m else set()
        assert stages == set(range(1, 39)), (t, sorted(set(range(1, 39)) - stages))


def test_all_notebooks_were_executed_without_errors_and_have_outputs():
    nbs = sorted((ROOT / "notebooks").glob("*.ipynb")); assert len(nbs) >= 14
    for p in nbs:
        nb = json.loads(p.read_text(encoding="utf-8")); code = [c for c in nb["cells"] if c["cell_type"] == "code"]
        assert code and all(c["execution_count"] for c in code), p.name
        assert not any(o.get("output_type") == "error" for c in code for o in c["outputs"]), p.name
        assert sum(1 for c in code if c["outputs"]) >= len(code) * 0.7, p.name


def test_na_stages_explain_why():
    for p in sorted((ROOT / "notebooks").glob("*.ipynb")):
        for c in json.loads(p.read_text(encoding="utf-8"))["cells"]:
            s = "".join(c["source"])
            if c["cell_type"] == "markdown" and s.startswith("## Stage") and "Not applicable here, and why:" in s: assert len(s.split("Not applicable here, and why:")[1].strip()) > 40


def test_ranking_summary_file_agrees_with_a_fresh_computation():
    from cortexhr.phase3.resume.metrics import attach_labels, evaluate
    d = data(); v1, v2 = rankers(); per, agg = evaluate(attach_labels(v2.score_all(d["resume_job_labels"][["resume_id", "job_id"]]), d["resume_job_labels"]), 5)
    s = pd.read_csv(task_dir(11) / "ranking_quality_summary.csv").set_index("engine").loc["final engine (v2, masked)"]
    for k in ("precision@5", "recall@5", "hit_rate@5", "ndcg@5"): assert abs(s[k] - agg[k]) < 5e-4, k


def test_fairness_statement_matches_the_saved_results():
    res = json.loads((task_dir(12) / "fairness_results.json").read_text()); st = (ROOT / "docs" / "fairness" / "fairness_statement.md").read_text(encoding="utf-8")
    assert f"{res['before']['mean_abs_gap']:.6f}" in st and f"{res['after']['max_abs_gap']:.6f}" in st and "Before real use" in st


def test_five_misranks_or_the_real_number_are_reported():
    m = pd.read_csv(task_dir(11) / "misranking_analysis.csv"); assert 1 <= len(m) <= 5 and {"what_went_wrong", "root_cause", "possible_improvement"} <= set(m.columns)


def test_resume_ranker_artifact_and_metadata():
    from cortexhr.phase3.attrition.v2 import file_sha256
    meta = json.loads((METADATA / "resume_ranker_v2_metadata.json").read_text()); assert meta["engine_version"].startswith("resume-ranker-v2") and "candidate_name" in meta["never_used"]
    assert meta["sha256"] == file_sha256(ROOT / "artifacts" / "models" / "resume_ranker_v2.joblib")
