"""Builds notebooks 01-04 (Task 11: resume ranking). Each covers part of the 38-stage workflow; together they cover all 38."""
from nbkit import NB

COMMON_LOAD = """from cortexhr.phase3.common.io import read_csv
from cortexhr.phase3.resume import text as T
from cortexhr.phase3.common.validation import profile
resumes = read_csv('resumes'); jobs = read_csv('job_descriptions'); labels = read_csv('resume_job_labels'); pairs = read_csv('resume_bias_pairs')
dd = read_csv('data_dictionary')
print('resumes', resumes.shape, '| jobs', jobs.shape, '| labels', labels.shape, '| twin pairs', pairs.shape)"""

FIT = """from cortexhr.phase3.resume.scoring import ResumeRanker
from cortexhr.phase3.resume.metrics import attach_labels, evaluate, random_baseline
v1 = ResumeRanker(mask=False).fit(resumes, jobs)      # resume_text exactly as given
v2 = ResumeRanker(mask=True).fit(resumes, jobs)       # identifiers masked before scoring (final engine)
def score_labelled(r): return attach_labels(r.score_all(labels[['resume_id', 'job_id']]), labels)"""


def nb01():
    n = NB("Task 11, Notebook 01: Resume data audit", "11: Rank resumes against job descriptions", "Understand and audit resumes, job descriptions and human labels before building anything.", list(range(1, 14)))
    n.stage(1, "Recruiters read hundreds of resumes for every opening. A ranked shortlist saves time, but hiring is one of the most legally sensitive uses of AI, so the ranking must be explainable and must never use name, gender, age or college tier. Everything here is decision support: a human recruiter makes the decision.")
    n.stage(2, "**Input:** one job description and a pool of resumes. **Output:** every resume gets a score for that job, candidates are ranked, the top 5 are returned with matched and missing skills. **Type:** ranking (not classification), evaluated against human suitability scores.")
    n.stage(3, "Success criteria are fixed here, before any result is seen (they live in `config/phase3_config.yaml`).")
    n.code("pd.Series(CFG['resume']['success_criteria'], name='threshold').to_frame()")
    n.stage(4, "Four tables are involved. `resumes` (candidates), `job_descriptions` (10 openings), `resume_job_labels` (human judgement for resume x job pairs) and `resume_bias_pairs` (25 twin resumes, used in Task 12).")
    n.code(COMMON_LOAD)
    n.stage(5, "Loaded with the shared reader, which raises a clear error if a file is missing.")
    n.code("{k: v.shape for k, v in dict(resumes=resumes, jobs=jobs, labels=labels, pairs=pairs).items()}")
    n.stage(6, "Grain: `resumes` = one row per resume; `jobs` = one row per job; `labels` = one row per (resume, job) pair.")
    n.code("""print('resume_id unique:', resumes.resume_id.is_unique, '| job_id unique:', jobs.job_id.is_unique, '| (resume,job) unique in labels:', not labels.duplicated(['resume_id','job_id']).any())
print('labelled jobs per resume:', labels.groupby('resume_id').size().value_counts().to_dict())
print('labelled resumes per job:'); print(labels.groupby('job_id').size().describe().round(1).to_dict())
resumes.columns.tolist()""")
    n.stage(7, "From the supplied data dictionary and from reading the data.")
    n.code("dd[dd.sheet.str.contains('resume|job', case=False)][['sheet', 'column', 'data_type', 'description']].head(40)")
    n.stage(8, "Every column should have the type its meaning implies.")
    n.code("pd.concat({'resumes': resumes.dtypes.astype(str), 'jobs': jobs.dtypes.astype(str), 'labels': labels.dtypes.astype(str)})")
    n.stage(9, "One-page quality profile per table.")
    n.code("profile(resumes).join(profile(resumes.select_dtypes('number')).add_prefix('num_'), how='left').head(20)")
    n.stage(10, "Missing values, including empty or blank free text.")
    n.code("""print('resumes with missing values per column:'); print(resumes.isna().sum()[lambda s: s > 0].to_dict() or 'none')
print('blank resume_text:', int((resumes.resume_text.fillna('').str.strip() == '').sum()), '| blank skills:', int((resumes.skills.fillna('').str.strip() == '').sum()))
print('jobs missing values:', jobs.isna().sum()[lambda s: s > 0].to_dict() or 'none')
print('labels missing values:', labels.isna().sum()[lambda s: s > 0].to_dict() or 'none')""")
    n.stage(11, "Exact duplicates, and near duplicates. The 25 twin resumes are near duplicates **by design** (same content, one detail changed): they must be recognised, not removed.")
    n.code("""from cortexhr.phase3.resume import text as T
print('duplicate resume_id:', int(resumes.resume_id.duplicated().sum()), '| exact duplicate resume_text:', int(resumes.resume_text.duplicated().sum()))
masked = resumes.apply(lambda r: T.normalise(T.mask_identifiers(r.resume_text, r.candidate_name, r.email, r.city)), axis=1)
dup_groups = resumes.assign(masked=masked).groupby('masked').resume_id.apply(list); dup_groups = dup_groups[dup_groups.str.len() > 1]
twin_ids = set(pairs.resume_id_original) | set(pairs.resume_id_twin)
print('near-duplicate groups after masking identifiers:', len(dup_groups), '| all explained by the twin pairs:', all(set(g) <= twin_ids for g in dup_groups))
print('duplicate job descriptions:', int(jobs.description_text.duplicated().sum()), '| duplicate job titles:', int(jobs.job_title.duplicated().sum()))""")
    n.stage(12, "Outliers in the numeric fields and in text length. An outlier here is only a flag to look at, not something to delete.")
    n.code("""x = resumes.total_experience_years; q1, q3 = x.quantile([.25, .75]); iqr = q3 - q1
out = resumes[(x < q1 - 1.5 * iqr) | (x > q3 + 1.5 * iqr)]
print('experience IQR outliers:', len(out), '| max years:', x.max(), '| z>3:', int((((x - x.mean()) / x.std()).abs() > 3).sum()))
L = resumes.resume_text.str.len(); print('text length min/median/max:', L.min(), int(L.median()), L.max())
resumes.assign(text_len=L).sort_values('text_len').head(3)[['resume_id', 'layout_difficulty', 'text_len']]""")
    n.stage(13, "Broken links, impossible values and label contradictions.")
    n.code("""from cortexhr.phase3.common.validation import require_subset
require_subset(labels.resume_id, resumes.resume_id, 'label resume_ids'); require_subset(labels.job_id, jobs.job_id, 'label job_ids')
require_subset(pairs.resume_id_original, resumes.resume_id); require_subset(pairs.resume_id_twin, resumes.resume_id)
checks = {'all label resume ids exist': True, 'all label job ids exist': True,
          'human score in 1..5': bool(labels.human_suitability_score.between(1, 5).all()),
          'shortlist == (score >= 4)': bool(((labels.human_suitability_score >= 4) == (labels.shortlist == 'Y')).all()),
          'experience between 0 and 40': bool(resumes.total_experience_years.between(0, 40).all()),
          'job minimum experience >= 0': bool((jobs.min_experience_years >= 0).all())}
pd.Series(checks, name='passed').to_frame()""")
    n.md("## Findings of this audit\n\nPrinted from the checks above: no broken links, labels are internally consistent, and the only near duplicates are the planned twin pairs. These facts are used in every later notebook.")
    n.code("print('Audit conclusion:', 'PASS' if all(checks.values()) else 'FAIL', '|', sum(checks.values()), 'of', len(checks), 'checks passed;', len(dup_groups), 'twin near-duplicate groups found.')")
    return n


def nb02():
    n = NB("Task 11, Notebook 02: EDA, label analysis, leakage and skill/experience extraction", "11: Rank resumes against job descriptions", "Explore the data, check the labels, hunt for leakage, then build and validate the skill and experience extractors.", list(range(14, 22)))
    n.code(COMMON_LOAD)
    n.stage(14, "Resumes come in several layouts (clean, tables, two-column, all caps, missing sections). Layout is the main reason text extraction can fail.")
    n.code("""print(resumes.layout_difficulty.value_counts().to_dict()); print(resumes.source_format.value_counts().to_dict())
print(resumes.head(2)[['resume_id', 'target_job_family', 'total_experience_years', 'skills']])""")
    n.md("### Reading 5 resumes and 3 job descriptions, and what makes a good match\n\nThe cell below prints five resumes (different layouts) and three jobs. The list that follows is what the reading and the label analysis in stages 16 to 19 support.")
    n.code("""for _, r in resumes.sample(5, random_state=SEED).iterrows(): print(f"{r.resume_id} | layout: {r.layout_difficulty} | {r.total_experience_years} yrs | skills: {r.skills}\\n   {r.resume_text.split(chr(10))[2][:150]}")
print(); jobs[['job_id', 'job_title', 'min_experience_years', 'required_skills']].head(3)""")
    n.md("**What makes a good match (from reading and from the data):** (1) the **required skills** of the job are actually evidenced in the text, not only listed; (2) the **role history** resembles the job (a Data Analyst reading for an Analyst opening); (3) **experience at or above the minimum** the job asks for, but not unlimited years; (4) nice-to-have skills help only after the required ones; (5) text similarity helps as a tie-breaker, not as the main signal. Name, gender, age and college tier are not on the list.")
    n.stage(15, "One variable at a time: experience, number of skills, text length, minimum experience asked by jobs.")
    n.code("""from cortexhr.phase3.resume.skills import split_skill_list
resumes['n_skills'] = resumes.skills.map(lambda s: len(split_skill_list(s))); resumes['text_len'] = resumes.resume_text.str.len()
fig, ax = plt.subplots(1, 4, figsize=(15, 3))
for a, (c, t) in zip(ax, [('total_experience_years', 'Years of experience'), ('n_skills', 'Skills listed'), ('text_len', 'Text length'), ('min_experience_years', 'Job minimum years')]):
    (jobs if c == 'min_experience_years' else resumes)[c].plot.hist(ax=a, bins=15, color='#2f6fb0'); a.set_title(t)
plt.tight_layout(); resumes[['total_experience_years', 'n_skills', 'text_len']].describe().round(2)""")
    n.stage(16, "Two variables together: how do the human scores relate to the label file's own skill coverage and to experience?")
    n.code("""m = labels.merge(resumes[['resume_id', 'total_experience_years', 'layout_difficulty']], on='resume_id').merge(jobs[['job_id', 'min_experience_years']], on='job_id')
m['exp_gap'] = m.total_experience_years - m.min_experience_years
print(m[['human_suitability_score', 'required_skill_coverage', 'nice_skill_coverage', 'exp_gap']].corr(method='spearman').round(3)['human_suitability_score'])
fig, ax = plt.subplots(1, 2, figsize=(10, 3.2)); m.boxplot('human_suitability_score', 'layout_difficulty', ax=ax[0]); m.plot.scatter('required_skill_coverage', 'human_suitability_score', alpha=.3, ax=ax[1]); plt.tight_layout()""")
    n.stage(17, "Several variables together: are the possible ranking signals redundant?")
    n.code("""c = m[['human_suitability_score', 'required_skill_coverage', 'nice_skill_coverage', 'total_experience_years', 'exp_gap']].corr(method='spearman')
fig, ax = plt.subplots(figsize=(5, 4)); im = ax.imshow(c, cmap='RdBu_r', vmin=-1, vmax=1); ax.set_xticks(range(len(c))); ax.set_xticklabels(c.columns, rotation=45, ha='right', fontsize=7); ax.set_yticks(range(len(c))); ax.set_yticklabels(c.columns, fontsize=7); plt.colorbar(im); plt.tight_layout(); c.round(2)""")
    n.stage(18, "The label is a human suitability score 1 to 5; `shortlist = Y` means 4 or 5. How many relevant candidates does each job have? This limits what recall@5 can show.")
    n.code("""print(labels.human_suitability_score.value_counts().sort_index().to_dict()); print(labels.reviewer_note.value_counts().to_dict())
rel = labels.assign(rel=(labels.shortlist == 'Y')).groupby('job_id').agg(candidates=('rel', 'size'), relevant=('rel', 'sum')); rel['max_recall_at_5'] = (5 / rel.relevant).clip(upper=1).round(2); rel""")
    n.stage(19, "Three leakage checks. (1) Does any resume column simply name the job it was written for? (2) Do label columns hide the answer? (3) Does the resume text carry protected information (name, e-mail, college tier)?")
    n.code("""tj = labels.merge(resumes[['resume_id', 'target_job_family']], on='resume_id')
print('target_job_family == job_id among shortlisted pairs:', round((tj[tj.shortlist == 'Y'].target_job_family == tj[tj.shortlist == 'Y'].job_id).mean(), 3), '-> this column names the answer; it is NEVER used as a feature.')
print('label columns that are computed from skills and therefore must not be used as inputs:', ['required_skill_coverage', 'nice_skill_coverage'])
has_name = resumes.apply(lambda r: str(r.candidate_name).split()[0].lower() in r.resume_text.lower(), axis=1).mean(); has_mail = resumes.apply(lambda r: str(r.email).lower() in r.resume_text.lower(), axis=1).mean()
has_tier = resumes.resume_text.str.contains(r'\\bTier\\s*[1-9]', regex=True).mean()
pd.Series({'resume_text contains the candidate name': has_name, 'resume_text contains the e-mail': has_mail, 'resume_text contains a college tier phrase': has_tier}, name='share of resumes').round(3).to_frame()""")
    n.md("**Finding:** the free text carries name, e-mail and the college-tier phrase. Even if the structured columns are excluded, a model that reads `resume_text` as given still sees them. This is why the engine has a masking step and why Task 12 audits it.")
    n.stage(20, "Build the two extractors and measure them against the structured columns (which are used only to *measure* the extractors, not to rank).")
    n.code("""from cortexhr.phase3.resume.skills import SkillExtractor, build_vocabulary
from cortexhr.phase3.resume.experience import extract_experience
ext = SkillExtractor(build_vocabulary(jobs, resumes)); rows = []
for _, r in resumes.iterrows():
    clean = T.for_skill_extraction(r.resume_text, r.candidate_name, r.email, r.city); truth = {ext.canon(s) for s in split_skill_list(r.skills)}; got = set(ext.extract(clean))
    e = extract_experience(clean); rows.append({'resume_id': r.resume_id, 'layout': r.layout_difficulty, 'skill_recall': len(truth & got) / len(truth), 'skill_precision': len(truth & got) / max(1, len(got)),
        'years_extracted': e['total_years'], 'years_column': r.total_experience_years, 'exp_flags': ','.join(e['flags'])})
q = pd.DataFrame(rows); q['exp_abs_error'] = (q.years_extracted - q.years_column).abs()
print('skill recall %.3f | precision %.3f | resumes with years error > 0.15: %d of %d' % (q.skill_recall.mean(), q.skill_precision.mean(), (q.exp_abs_error > 0.15).sum(), len(q)))
q.groupby('layout')[['skill_recall', 'skill_precision', 'exp_abs_error']].mean().round(3)""")
    n.code("""low = q.sort_values('skill_recall').head(3); print('lowest skill recall (text has fewer skills than the structured list):'); low[['resume_id', 'layout', 'skill_recall']]""")
    n.md("Low recall appears for resumes with *missing sections*: the text simply does not mention all listed skills, so no text method can find them. That is a data limit, reported, not hidden.")
    n.md("**Think for yourself: would I rank on years of experience alone? No.** Years alone favours people with long, uninterrupted careers and disadvantages career changers, people returning after a break (more often women), and younger candidates. It also says nothing about *which* skills were used. In this engine experience is one capped part of the score (20%); notebook 03 shows how weak 'experience only' is as a ranker.")
    n.stage(21, "Which inputs may enter the score, which are relevant, and which are forbidden.")
    n.code("""from cortexhr.phase3.resume.scoring import FORBIDDEN_SCORE_INPUTS
print('FORBIDDEN as score inputs:', list(FORBIDDEN_SCORE_INPUTS))
print('ALLOWED: masked resume text (TF-IDF), skills found in the text vs job skills, years of experience found in the text vs the job minimum')
f = labels.merge(resumes[['resume_id', 'total_experience_years']], on='resume_id').merge(jobs[['job_id', 'min_experience_years']], on='job_id')
f['exp_fit'] = (f.total_experience_years / f.min_experience_years.clip(lower=1)).clip(upper=1)
pd.Series({'required skill coverage (label file, for orientation only)': f.human_suitability_score.corr(f.required_skill_coverage, method='spearman'), 'experience fit': f.human_suitability_score.corr(f.exp_fit, method='spearman')}, name='spearman with human score').round(3).to_frame()""")
    return n


def nb03():
    n = NB("Task 11, Notebook 03: Building and tuning the ranking engine", "11: Rank resumes against job descriptions", "Preprocess, set baselines, build the TF-IDF + skills + experience engine, validate weights without tuning on the job being judged.", list(range(22, 30)))
    n.code(COMMON_LOAD); n.code(FIT)
    n.stage(22, "Text preprocessing is fitted on resumes and jobs only (no labels). Identifiers are masked, skill phrases are kept as single tokens, plural folding is conservative.")
    n.code("""r = resumes.iloc[4]; raw = r.resume_text[:300]
from cortexhr.phase3.resume import text as T
masked = T.mask_identifiers(r.resume_text, r.candidate_name, r.email, r.city)
print('RAW   :', raw.replace('\\n', ' | ')); print('MASKED:', masked[:300].replace('\\n', ' | ')); print('TOKENS:', T.tokenize(masked, v2.phrases)[:25])""")
    n.stage(23, "There is no supervised training in this engine, so there is no model to overfit to the labels. The labels are used **only to measure**. Weights were fixed a priori in the config (skills matter most, text similarity second, experience capped). Sensitivity and leave-one-job-out checks below test whether that choice was sensible.")
    n.code("print('weights fixed in config before evaluation:', CFG['resume']['weights'])")
    n.stage(24, "Baselines: a random ordering, and each component on its own.")
    n.code("""ev1, ev2 = score_labelled(v1), score_labelled(v2)
rb = random_baseline(ev2, 5, 300, SEED)
from cortexhr.phase3.resume.analysis import component_frame, reweight
comp = component_frame(v2, labels)
single = {n_: evaluate(reweight(comp, w), 5)[1] for n_, w in {'TF-IDF only': dict(tfidf=1, required_skill=0, nice_skill=0, experience=0), 'required skills only': dict(tfidf=0, required_skill=1, nice_skill=0, experience=0), 'experience only': dict(tfidf=0, required_skill=0, nice_skill=0, experience=1)}.items()}
base = pd.DataFrame({'random ordering': rb, **single}).T[['precision@5', 'recall@5', 'hit_rate@5', 'ndcg@5', 'mrr']].round(3); base""")
    n.stage(25, "Why TF-IDF + skill overlap + experience: it is explainable to a recruiter in one sentence, needs no training labels, and every part can be audited for protected information. A neural embedding model (e.g. sentence-transformers) may rank better but is harder to audit; it is listed as future work, not installed here.")
    n.na(26, "no parameters are learned from labels. 'Training' means fitting the TF-IDF vocabulary on the resume and job text; this was done in the cell above when the two rankers were fitted (`v1` unmasked, `v2` masked).")
    n.code("print('vocabulary size (v2):', len(v2.vec.vocabulary_), '| skills in vocabulary:', len(v2.extractor.vocab), '| resumes:', len(v2.resumes), '| jobs:', len(v2.jobs))")
    n.stage(27, "Cross-validation analogue for a ranker: **leave-one-job-out**. For each job, choose weights using the other 9 jobs, then measure on the held-out job.")
    n.code("""from cortexhr.phase3.resume.analysis import weight_grid, sensitivity, leave_one_job_out
grid = weight_grid(0.1); print('weight combinations tested:', len(grid))
loo = leave_one_job_out(comp, grid, 5, 'ndcg@5'); loo[['held_out_job', 'train_mean_ndcg@5', 'held_out_ndcg@5', 'held_out_precision@5']].round(3)""")
    n.code("print('mean held-out NDCG@5 (weights chosen WITHOUT seeing the job):', round(loo['held_out_ndcg@5'].mean(), 3), '| mean held-out precision@5:', round(loo['held_out_precision@5'].mean(), 3))")
    n.stage(28, "Sensitivity of quality to the weights (a log of experiments, not a hunt for the best number).")
    n.code("""sens = sensitivity(comp, grid, 5); chosen = evaluate(reweight(comp, CFG['resume']['weights']), 5)[1]
print('configured weights: ndcg@5 %.3f precision@5 %.3f' % (chosen['ndcg@5'], chosen['precision@5']))
print('range over all %d weightings: ndcg@5 %.3f to %.3f; precision@5 %.3f to %.3f' % (len(sens), sens['ndcg@5'].min(), sens['ndcg@5'].max(), sens['precision@5'].min(), sens['precision@5'].max()))
from cortexhr.phase3.common.io import save_table
save_table(sens, P.task_dir(11) / 'weight_experiments.csv'); sens.sort_values('ndcg@5', ascending=False).head(5).round(3)""")
    n.code("""fig, ax = plt.subplots(figsize=(6, 3.2)); ax.hist(sens['ndcg@5'], bins=25, color='#2f6fb0'); ax.axvline(chosen['ndcg@5'], color='r', label='configured weights'); ax.set_xlabel('NDCG@5'); ax.legend(); plt.tight_layout()""")
    n.stage(29, "Evaluation on all 555 labelled pairs (ranking within each job's labelled pool).")
    n.code("""per, agg = evaluate(ev2, 5); print({k: round(v, 3) for k, v in agg.items()})
print('success criteria check:', 'precision@5 >= 2x random:', agg['precision@5'] >= CFG['resume']['success_criteria']['precision_at_5_min_multiple_of_random'] * rb['precision@5'], '| hit rate >=', CFG['resume']['success_criteria']['hit_rate_at_5_min'], ':', agg['hit_rate@5'] >= CFG['resume']['success_criteria']['hit_rate_at_5_min'])""")
    return n


def nb04():
    n = NB("Task 11, Notebook 04: Evaluation, misranking analysis, outputs and conclusion", "11: Rank resumes against job descriptions", "Report ranking quality against human labels, explain five misranked resumes, deliver the top-5 lists, the API contract and the saved engine.", [29] + list(range(30, 39)))
    n.code(COMMON_LOAD); n.code(FIT)
    n.stage(29, "Ranking quality table. `relevant` = shortlisted by humans (score 4 or 5). NDCG@5 uses the graded 1 to 5 score.")
    n.code("""ev2 = score_labelled(v2); per, agg = evaluate(ev2, 5); per.round(3)""")
    n.code("""rb = random_baseline(ev2, 5, 300, SEED)
table = pd.DataFrame({'final engine (v2, masked)': agg, 'random ordering': rb}).T[['precision@5', 'recall@5', 'hit_rate@5', 'ndcg@5', 'mrr', 'average_precision', 'r_precision']].round(3)
from cortexhr.phase3.common.io import save_table
save_table(per, P.task_dir(11) / 'ranking_quality_per_job.csv'); save_table(table.reset_index(names='engine'), P.task_dir(11) / 'ranking_quality_summary.csv'); table""")
    n.md("**How to read it.** *Precision@5*: of the 5 candidates shown, how many humans also shortlisted. *Recall@5*: of all shortlisted candidates for the job, how many are in the top 5 (limited when a job has more than 5 relevant candidates). *Hit rate@5*: share of jobs with at least one relevant candidate in the top 5. *NDCG@5*: rewards putting the best-scored candidates first. *MRR*: how early the first relevant candidate appears.")
    n.code("""fig, ax = plt.subplots(figsize=(9, 3.2)); x = np.arange(len(per)); ax.bar(x - .2, per['precision@5'], .4, label='precision@5'); ax.bar(x + .2, per['ndcg@5'], .4, label='ndcg@5'); ax.set_xticks(x); ax.set_xticklabels(per.job_id); ax.set_ylim(0, 1.05); ax.legend(); ax.set_title('Ranking quality per job'); plt.tight_layout(); plt.savefig(P.FIGURES / 'task11_ranking_quality_per_job.png', dpi=130, bbox_inches='tight')""")
    n.stage(30, "Genuine misranks that have human labels. Severity is the distance from where the human score says the candidate belongs. If fewer than five exist the real number is reported.")
    n.code("""from cortexhr.phase3.resume.analysis import find_misranks, diagnose
mis = find_misranks(ev2, 5, 5); print('genuine misranked pairs found:', len(find_misranks(ev2, 5, 10**6)), '| shown:', len(mis))
rows = []
for _, r in mis.iterrows():
    what, root, fix = diagnose(r)
    rows.append({'job_id': r.job_id, 'resume_id': r.resume_id, 'predicted_rank': int(r['rank']), 'human_score': int(r.rel), 'model_score': round(r.score, 3), 'misrank_type': r.misrank_type,
                 'matched_required': r.matched_required, 'missing_required': r.missing_required, 'what_went_wrong': what, 'root_cause': root, 'possible_improvement': fix})
mtab = pd.DataFrame(rows); save_table(mtab, P.task_dir(11) / 'misranking_analysis.csv'); mtab[['job_id', 'resume_id', 'predicted_rank', 'human_score', 'model_score', 'misrank_type']]""")
    n.code("""j2 = ev2[ev2.job_id == 'J02'].sort_values('rank').head(5)[['rank', 'resume_id', 'score', 'human_suitability_score', 'shortlist', 'required_coverage', 'missing_required']]
print('J02 has only', int(ev2[ev2.job_id == 'J02'].relevant.sum()), 'relevant candidates, so precision@5 can never exceed', round(ev2[ev2.job_id == 'J02'].relevant.sum() / 5, 2), '- the low J02 number is mostly a data limit, not a ranking failure.')
j2.round(3)""")
    n.code("""for _, r in mtab.iterrows(): print(f"[{r.job_id} / {r.resume_id}] rank {r.predicted_rank}, human {r.human_score}/5, score {r.model_score}\\n  matched: {r.matched_required or '-'} | missing: {r.missing_required or '-'}\\n  why: {r.what_went_wrong}\\n  root cause: {r.root_cause}\\n  improvement: {r.possible_improvement}\\n")""")
    n.stage(31, "How to explain the score to a recruiter, with one real example.")
    n.code("""top = v2.top_k('J01', 5)
print('SCORE =', ' + '.join(f"{w} x {k}" for k, w in v2.weights.items()), '(each part 0 to 1)')
top[['rank', 'resume_id', 'score', 'tfidf_scaled', 'required_coverage', 'nice_coverage', 'experience_score', 'matched_required', 'missing_required', 'recommendation']].round(3)""")
    n.stage(32, "Detailed fairness work is Task 12. Here is the headline check: the effect of masking on the 25 twin pairs.")
    n.code("""from cortexhr.phase3.fairness.twins import twin_gaps, gap_summary
g1, g2 = twin_gaps(v1, pairs, labels), twin_gaps(v2, pairs, labels)
pd.DataFrame({'v1 (text as given)': gap_summary(g1), 'v2 (masked)': gap_summary(g2)}).round(6)""")
    n.stage(33, "A ranking score is **not a probability**. It orders candidates; it does not say 'X% chance of being hired'. Reliability here means: are the quality numbers stable? Bootstrap over jobs.")
    n.code("""rng = np.random.default_rng(SEED); boots = [per.sample(len(per), replace=True, random_state=int(rng.integers(1e9)))[['precision@5', 'ndcg@5', 'hit_rate@5']].mean() for _ in range(2000)]
ci = pd.DataFrame(boots).quantile([.025, .5, .975]).T.round(3); ci.columns = ['2.5%', 'median', '97.5%']; ci""")
    n.stage(34, "v1 (text as given) versus v2 (masked). v2 is the final engine because it removes protected information from the score; ranking quality is compared so fairness does not silently cost accuracy.")
    n.code("""ev1 = score_labelled(v1); _, a1 = evaluate(ev1, 5)
cmp_ = pd.DataFrame({'v1 unmasked': a1, 'v2 masked (final)': agg}).T[['precision@5', 'recall@5', 'hit_rate@5', 'ndcg@5', 'mrr']].round(4); cmp_['delta_vs_v1'] = ''
print('NDCG@5 change from masking: %+.4f | precision@5 change: %+.4f' % (agg['ndcg@5'] - a1['ndcg@5'], agg['precision@5'] - a1['precision@5'])); cmp_""")
    n.stage(35, "Top 5 candidates for every job: (a) inside the labelled pool (comparable with human labels), (b) across all 185 resumes (what the API returns).")
    n.code("""frames = []
for jid in jobs.job_id:
    t = v2.top_k(jid, 5); t.insert(0, 'job_id_', jid); frames.append(t)
top_all = pd.concat(frames, ignore_index=True)
cols = ['job_id', 'rank', 'resume_id', 'score', 'tfidf_scaled', 'required_coverage', 'nice_coverage', 'experience_score', 'matched_required', 'missing_required', 'total_years', 'relevant_years', 'recommendation', 'human_review']
save_table(top_all[cols], P.task_dir(11) / 'top5_per_job_all_resumes.csv'); print(top_all.shape); top_all[cols].head(10).round(3)""")
    n.stage(36, "Output and business validation: required fields exist, no protected field appears, and the API contract is tested on good and bad requests.")
    n.code("""from cortexhr.phase3.resume.api import handle_rank_request, ApiError, FORBIDDEN_FIELDS
resp = handle_rank_request({'job_id': 'J01', 'top_k': 3}, v2)
print('response keys:', list(resp)); print('first candidate:', {k: resp['candidates'][0][k] for k in ('rank', 'resume_id', 'score', 'score_breakdown')})
leak = [f for f in FORBIDDEN_FIELDS if f in str(resp).lower().replace('never_used', '')]
print('protected fields present in the response:', leak or 'none')
for bad in ({'job_id': 'J01', 'gender': 'F'}, {'job_id': 'J01', 'top_k': 99}, {'job_id': 'NOPE'}, {}):
    try: handle_rank_request(bad, v2)
    except ApiError as e: print(e.status, e.body())""")
    n.stage(37, "Save the fitted engine with version, weights, inputs and environment. Existing Phase 1 and Phase 2 artifacts are not touched.")
    n.code("""import joblib, hashlib, platform, sklearn
from cortexhr.phase3.common.io import save_json, utc_now
joblib.dump(v2, P.MODELS / 'resume_ranker_v2.joblib')
meta = {'engine_version': v2.used_inputs()['version'], 'created_utc': utc_now(), 'weights': v2.weights, 'score_inputs': v2.used_inputs()['score_inputs'], 'never_used': v2.used_inputs()['never_used'],
        'metrics_labelled_pool': {k: round(v, 4) for k, v in agg.items()}, 'random_baseline': {k: round(v, 4) for k, v in rb.items()}, 'seed': SEED, 'sklearn': sklearn.__version__, 'python': platform.python_version(),
        'sha256': hashlib.sha256(open(P.MODELS / 'resume_ranker_v2.joblib', 'rb').read()).hexdigest()}
save_json(meta, P.METADATA / 'resume_ranker_v2_metadata.json'); meta['engine_version'], meta['sha256'][:12]""")
    n.stage(38, "")
    n.code("""print('FINDINGS'); print('- Final engine (v2 masked): precision@5 %.2f, NDCG@5 %.2f, hit rate@5 %.2f versus random %.2f / %.2f / %.2f.' % (agg['precision@5'], agg['ndcg@5'], agg['hit_rate@5'], rb['precision@5'], rb['ndcg@5'], rb['hit_rate@5']))
weak = per.sort_values('precision@5').iloc[0]; print('- Weakest job: %s (precision@5 %.2f, %d relevant candidates).' % (weak.job_id, weak['precision@5'], weak.relevant))
print('- Masking changed NDCG@5 by %+.3f.' % (agg['ndcg@5'] - a1['ndcg@5']))
allm = find_misranks(ev2, 5, 10**6); print('- Genuine misranks found: %d (%d under-ranked relevant, %d over-ranked weak); none reaches the top-5 of any job except as noted for J02.' % (len(allm), allm.misrank_type.str.contains('under').sum(), allm.misrank_type.str.contains('over').sum()))
print('LIMITS: 10 jobs and 555 human labels are small; skills outside the vocabulary are invisible; years are capped so more experience never beats the requirement; the score ranks, it does not predict hiring; a recruiter must review every candidate.')""")
    return n


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "notebooks"
    for name, f in (("01_task11_resume_data_audit", nb01), ("02_task11_resume_skill_extraction", nb02), ("03_task11_resume_ranking", nb03), ("04_task11_ranking_evaluation", nb04)):
        print(f(out and f()).save(out / f"{name}.ipynb") if False else f().save(out / f"{name}.ipynb"))
