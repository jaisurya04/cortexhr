"""Builds notebooks 05-06 (Task 12: resume screener fairness audit)."""
from nbkit import NB

LOAD = """from cortexhr.phase3.common.io import read_csv, save_table, save_json
from cortexhr.phase3.common.validation import profile
from cortexhr.phase3.resume import text as T
from cortexhr.phase3.resume.scoring import ResumeRanker
from cortexhr.phase3.resume.metrics import attach_labels, evaluate
from cortexhr.phase3.fairness.twins import validate_pairs, twin_gaps, gap_summary, attach_protected, vault_resume_attributes
resumes = read_csv('resumes'); jobs = read_csv('job_descriptions'); labels = read_csv('resume_job_labels'); pairs = read_csv('resume_bias_pairs'); dd = read_csv('data_dictionary')
attrs = vault_resume_attributes()   # restricted file: used ONLY to interpret results after scoring; None if the file is not present
print('twin pairs:', len(pairs), '| restricted attribute file available:', attrs is not None)"""

FIT = """v1 = ResumeRanker(mask=False).fit(resumes, jobs)    # the screener as first built: resume_text exactly as given
v2 = ResumeRanker(mask=True).fit(resumes, jobs)     # after the fix: identifiers masked
def labelled_eval(r): return evaluate(attach_labels(r.score_all(labels[['resume_id', 'job_id']]), labels), 5)"""


def nb05():
    n = NB("Task 12, Notebook 05: Twin-resume audit, baseline", "12: Audit your screener for fairness", "Validate the 25 twin pairs, measure the score gap of the screener as first built, and find out where any gap comes from.", list(range(1, 20)) + [24])
    n.stage(1, "A screener that scores a resume differently because of a name has failed before the first interview. Companies need a *documented* fairness test before an HR model goes live. This notebook is that test for the Task 11 screener.")
    n.stage(2, "**Counterfactual test.** Take 25 pairs of resumes that are identical except for one detail (a gender-coded name, sometimes also the college tier). Score both. If the only difference is the protected detail, any score difference is unfair treatment. **Output:** the gap per pair, the average gap, and where it comes from.")
    n.stage(3, "Pass mark, fixed in the config before measuring: the average and maximum twin gap after the fix must be at most the tolerance, and ranking quality must not fall by more than the allowed amount.")
    n.code(LOAD)
    n.code("pd.Series({'tolerance for a practical gap': CFG['fairness']['tolerance'], **{k: v for k, v in CFG['resume']['success_criteria'].items() if k in ('max_twin_gap_after_fix', 'max_ndcg_drop_after_fix')}}, name='value').to_frame()")
    n.stage(4, "Files: `resume_bias_pairs` (25 pairs), `resumes` (both twins are in it), `resume_job_labels` (the twins were also scored by humans on 3 jobs each), `job_descriptions`, and the restricted `unlock_CX-12` file with the hidden protected attributes. The restricted file is never an input to any score.")
    n.stage(5, "")
    n.code("{'pairs': pairs.shape, 'resumes': resumes.shape, 'labels': labels.shape, 'twin resumes labelled': int(labels.resume_id.isin(pairs.resume_id_twin).sum())}")
    n.stage(6, "One row per pair; each pair gives two resumes and, through the labels, up to 3 jobs, so 25 pairs give 75 comparisons.")
    n.code("pairs.head(3)")
    n.stage(7, "")
    n.code("dd[dd.sheet.str.contains('bias', case=False)][['sheet', 'column', 'data_type', 'description']]")
    n.stage(8, "")
    n.code("pairs.dtypes.astype(str).to_frame('dtype')")
    n.stage(9, "")
    n.code("profile(pairs)")
    n.stage(10, "")
    n.code("print('missing values in pairs:', pairs.isna().sum().sum(), '| twin ids found in resumes:', pairs.resume_id_twin.isin(resumes.resume_id).all(), pairs.resume_id_original.isin(resumes.resume_id).all())")
    n.stage(11, "A resume must belong to at most one pair, and a pair must not be its own twin.")
    n.code("ids = pd.concat([pairs.resume_id_original, pairs.resume_id_twin]); print('resume ids used in pairs:', len(ids), '| unique:', ids.nunique(), '| pair_id unique:', pairs.pair_id.is_unique, '| original == twin anywhere:', bool((pairs.resume_id_original == pairs.resume_id_twin).any()))")
    n.stage(12, "For this table an outlier means an unusual pair: one where more than the expected details differ. Measured as how many characters of the resume text differ.")
    n.code("""import difflib
r = resumes.set_index('resume_id'); size = []
for _, p in pairs.iterrows():
    a, b = r.loc[p.resume_id_original, 'resume_text'], r.loc[p.resume_id_twin, 'resume_text']
    size.append(sum(len(x) for x in [a, b]) - 2 * sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()))
pairs['changed_characters'] = size; pairs['changed_characters'].describe().round(1).to_frame().T""")
    n.stage(13, "Are the twins really identical apart from the expected details? This check compares every column.")
    n.code("""v = validate_pairs(pairs, resumes); print('pairs with an unexpected difference:', int((v.unexpected_differences != '').sum()), '| skills identical in all pairs:', bool(v.skills_actually_identical.all()), '| experience identical in all pairs:', bool(v.experience_identical.all()))
v.groupby(['attribute_changed', 'columns_that_differ']).size().to_frame('pairs')""")
    n.stage(14, "")
    n.code("pairs.attribute_changed.value_counts().to_frame('pairs')")
    n.stage(15, "")
    n.code("pairs.expected_score_difference.value_counts().to_frame('pairs'); ")
    n.stage(16, "Which details change in which pairs, using the restricted attribute file (interpretation only).")
    n.code("""if attrs is None: print('restricted file not present: skipped')
else:
    a = attrs; flip = pd.DataFrame({'gender_original': a.loc[pairs.resume_id_original, 'inferred_gender'].values, 'gender_twin': a.loc[pairs.resume_id_twin, 'inferred_gender'].values,
                                   'tier_original': a.loc[pairs.resume_id_original, 'college_tier'].values, 'tier_twin': a.loc[pairs.resume_id_twin, 'college_tier'].values})
    print('pairs where the gender-coded name flips:', int((flip.gender_original != flip.gender_twin).sum()), 'of', len(flip), '| pairs where college tier also changes:', int((flip.tier_original != flip.tier_twin).sum()))
    print(pd.crosstab(flip.gender_original, flip.gender_twin))""")
    n.stage(17, "")
    n.code("""if attrs is not None:
    print(pd.crosstab([flip.gender_original + '->' + flip.gender_twin], flip.tier_original + '->' + flip.tier_twin).to_string())
else: print('skipped (restricted file not present)')""")
    n.stage(18, "The 'label' for a fairness test is the *expected* gap, which is zero for every pair. A second reference: did the **human reviewers** score twins differently?")
    n.code("""hl = labels.set_index(['resume_id', 'job_id']).human_suitability_score; rows = []
for _, p in pairs.iterrows():
    for job in sorted(set(labels[labels.resume_id == p.resume_id_original].job_id) & set(labels[labels.resume_id == p.resume_id_twin].job_id)):
        rows.append({'pair_id': p.pair_id, 'job_id': job, 'human_diff_twin_minus_original': hl[(p.resume_id_twin, job)] - hl[(p.resume_id_original, job)]})
hd = pd.DataFrame(rows); print('expected gap in the pairs file:', pairs.expected_score_difference.unique().tolist(), '| human score differences between twins:', hd.human_diff_twin_minus_original.value_counts().to_dict())""")
    n.stage(19, "Where can a name or a college tier enter a text-based score? Compare the words of the two twins.")
    n.code("""o, t = pairs.iloc[0].resume_id_original, pairs.iloc[0].resume_id_twin
to, tt = set(T.tokenize(r.loc[o, 'resume_text'])), set(T.tokenize(r.loc[t, 'resume_text']))
print('pair', pairs.iloc[0].pair_id, '| words only in the original:', sorted(to - tt)); print('words only in the twin:', sorted(tt - to))
diffs = []
for _, p in pairs.iterrows():
    a_, b_ = set(T.tokenize(r.loc[p.resume_id_original, 'resume_text'])), set(T.tokenize(r.loc[p.resume_id_twin, 'resume_text'])); diffs.append(len(a_ ^ b_))
print('differing word types per pair: min %d, median %d, max %d' % (min(diffs), np.median(diffs), max(diffs)))""")
    n.md("**Leakage finding.** The twins differ in the name, the e-mail (built from the name) and, in some pairs, the college-tier phrase. A TF-IDF model reading `resume_text` as given sees those words. Excluding the *columns* `candidate_name` and `college_tier` is not enough, because the same information sits in the free text.")
    n.stage(24, "The baseline for fairness is **the screener as first built** (v1): the Task 11 engine with `resume_text` used as given and no masking. Its gap is the number to beat.")
    n.code(FIT)
    n.code("""g1 = twin_gaps(v1, pairs, labels); s1 = gap_summary(g1); pd.Series(s1, name='v1 (before the fix)').to_frame()""")
    n.code("""print('Interpretation: the gap is defined as twin score minus original score (score scale 0 to 1, so 0.001 = 0.1 score points). Mean absolute gap %.6f, maximum %.6f, non-zero in %d of %d comparisons.' % (s1['mean_abs_gap'], s1['max_abs_gap'], s1['comparisons_with_nonzero_gap'], s1['comparisons']))
print('Mean signed gap %.6f: positive would mean twins score higher on average; near zero with mixed signs means no systematic favouring.' % s1['mean_signed_gap(twin-original)'])""")
    return n


def nb06():
    n = NB("Task 12, Notebook 06: Reducing the gap, before and after, fairness statement", "12: Audit your screener for fairness", "Find which detail causes the gap, remove it, re-measure all 25 pairs, check ranking quality did not collapse, and write an honest fairness statement.", list(range(20, 24)) + list(range(25, 39)))
    n.code(LOAD); n.code(FIT)
    n.stage(20, "Features here are the *transformations* that remove protected information: mask the name, the e-mail, the city and the college-tier phrase before scoring. Each is a separate switch so that their effects can be measured one by one (ablation).")
    n.code("""o = pairs.iloc[0]; ro, rt = resumes.set_index('resume_id').loc[o.resume_id_original], resumes.set_index('resume_id').loc[o.resume_id_twin]
mo, mt = T.mask_identifiers(ro.resume_text, ro.candidate_name, ro.email, ro.city), T.mask_identifiers(rt.resume_text, rt.candidate_name, rt.email, rt.city)
print('texts equal after masking (tokens):', T.prepare_document(mo) == T.prepare_document(mt)); print(mo[:220].replace('\\n', ' | '))""")
    n.stage(21, "Proxy review: which text elements could carry gender, age or background, and what is decided about each?")
    n.code("""txt = resumes.resume_text.str.lower(); print('resumes mentioning a career gap / break / maternity / sabbatical:', int(txt.str.contains('career gap|career break|maternity|sabbatical|gap year').sum()), '| mentioning a graduation year or age:', int(txt.str.contains(r'born|age \\d|aged').sum()), '| mentioning pronouns (he / she):', int(txt.str.contains(r'\\b(he|she|his|her)\\b').sum()))""")
    n.code("""proxy = pd.DataFrame([
 ('candidate name', 'gender, religion, region', 'REMOVE: the twins differ here', 'masked'),
 ('e-mail address', 'built from the name', 'REMOVE', 'masked'),
 ('city', 'region, socio-economic background', 'REMOVE: not job-relevant for text scoring', 'masked'),
 ('"Tier N institution" phrase', 'college tier / background', 'REMOVE: never used', 'masked'),
 ('degree names (B.E., MCA ...)', 'weak background signal', 'KEEP: jobs state education preferences; reviewed, no twin difference', 'kept'),
 ('skills, role titles, years', 'job-relevant evidence', 'KEEP', 'kept'),
 ('graduation year / age', 'age', 'NOT PRESENT in the resumes (checked)', 'n/a')], columns=['element', 'could proxy for', 'decision', 'status'])
has_year = resumes.resume_text.str.contains(r'\\b(19|20)\\d{2}\\b', regex=True).mean(); print('share of resumes containing a 4-digit year:', round(has_year, 3)); proxy""")
    n.stage(22, "Fixed rule-based preprocessing; nothing is learned, so it cannot overfit to the twins.")
    n.code("print('masking switches:', v2.mask_parts, '| digits are also dropped by the tokenizer (documented in docs/methodology.md)')")
    n.stage(23, "Design: a paired (counterfactual) test. Each pair is its own control. There is no train/test split because nothing is trained on the pairs, and no tuning is done on them: the fix is a rule decided from the proxy review above.")
    n.na(25, "this is a test of an existing screener, not a new prediction model. The 'method' is counterfactual twin testing plus group-gap measurement with the supplied `group_gap()` helper.")
    n.na(26, "no model is trained in this task; the two screeners were fitted in the cell above (vocabulary only, no labels).")
    n.stage(27, "Stability of the gap: per job and per bootstrap draw of the 25 pairs.")
    n.code("""g1, g2 = twin_gaps(v1, pairs, labels), twin_gaps(v2, pairs, labels)
print(g1.groupby('job_id').abs_gap.mean().round(6).to_dict())
rng = np.random.default_rng(SEED); pp = g1.pair_id.unique(); boots = [g1[g1.pair_id.isin(rng.choice(pp, len(pp)))].abs_gap.mean() for _ in range(2000)]
print('v1 mean absolute gap, 95%% bootstrap interval over pairs: %.6f to %.6f' % (np.percentile(boots, 2.5), np.percentile(boots, 97.5)))
sb = [g1[g1.pair_id.isin(rng.choice(pp, len(pp)))].gap.mean() for _ in range(2000)]; slo, shi = np.percentile(sb, [2.5, 97.5])
direction = 'no systematic direction detected (the interval includes zero)' if slo <= 0 <= shi else 'a systematic direction (the interval excludes zero)'
print('mean SIGNED gap, 95%% bootstrap interval: %.6f to %.6f -> %s' % (slo, shi, direction))""")
    n.stage(28, "Not a hyper-parameter search. It is an **ablation**: which masked element removes the gap?")
    n.code("""rows = []
for label, parts in [('none (v1)', ()), ('name + e-mail only', ('name', 'email')), ('college-tier phrase only', ('tier',)), ('city only', ('city',)), ('everything (v2)', ('name', 'email', 'city', 'tier'))]:
    rk = ResumeRanker(mask=bool(parts), mask_parts=parts).fit(resumes, jobs); g = twin_gaps(rk, pairs, labels)
    rows.append({'masked': label, 'mean_abs_gap': g.abs_gap.mean(), 'max_abs_gap': g.abs_gap.max(), 'nonzero_comparisons': int((g.abs_gap > 1e-9).sum())})
abl = pd.DataFrame(rows); save_table(abl, P.task_dir(12) / 'ablation_masking.csv'); abl""")
    n.code("""print('Reading: removing name and e-mail removes the whole gap; masking only the tier phrase changes nothing here because the tokenizer drops digits (Tier 1 and Tier 3 become the same word). The tier phrase is still masked: another text model (for example an embedding model) might read the digit.')""")
    n.stage(29, "Before and after, all 25 pairs, plus ranking quality against the human labels.")
    n.code("""s1, s2 = gap_summary(g1), gap_summary(g2)
ba = pd.DataFrame({'before (v1)': s1, 'after (v2)': s2}); ba['change'] = ba['after (v2)'] - ba['before (v1)']; ba['improvement_%'] = np.where(ba['before (v1)'] != 0, (ba['before (v1)'] - ba['after (v2)']) / ba['before (v1)'] * 100, np.nan); save_table(ba.reset_index(names='measure'), P.task_dir(12) / 'before_after_gap.csv'); ba.round(6)""")
    n.code("""per1, q1 = labelled_eval(v1); per2, q2 = labelled_eval(v2)
qual = pd.DataFrame({'before (v1)': q1, 'after (v2)': q2}).T[['precision@5', 'recall@5', 'hit_rate@5', 'ndcg@5', 'mrr']]; qual.loc['change'] = qual.loc['after (v2)'] - qual.loc['before (v1)']; save_table(qual.reset_index(names='screener'), P.task_dir(12) / 'ranking_quality_before_after.csv'); qual.round(4)""")
    n.code("""fig, ax = plt.subplots(1, 3, figsize=(14, 3.4))
ax[0].hist(g1.gap, bins=25, color='#c0392b', alpha=.85); ax[0].axvline(0, color='k', lw=.8); ax[0].set_title('Twin score gap BEFORE (v1)'); ax[0].set_xlabel('twin score - original score')
ax[1].hist(g2.gap, bins=25, color='#2e8b57', alpha=.85); ax[1].set_xlim(ax[0].get_xlim()); ax[1].set_title('Twin score gap AFTER (v2)'); ax[1].set_xlabel('twin score - original score')
q = qual.loc[['before (v1)', 'after (v2)'], ['precision@5', 'ndcg@5', 'hit_rate@5']]; q.T.plot.bar(ax=ax[2], color=['#c0392b', '#2e8b57'], rot=0); ax[2].set_ylim(0, 1.05); ax[2].set_title('Ranking quality before / after')
plt.tight_layout(); plt.savefig(P.FIGURES / 'task12_twin_gap_before_after.png', dpi=130, bbox_inches='tight')""")
    n.stage(30, "The pairs with the largest gap before the fix, and whether any ranking decision changed.")
    n.code("""big = attach_protected(g1, pairs, attrs).sort_values('abs_gap', ascending=False).head(8)
cols = ['pair_id', 'job_id', 'score_original', 'score_twin', 'gap', 'rank_original', 'rank_twin'] + (['gender_original', 'gender_twin', 'tier_original', 'tier_twin'] if attrs is not None else []); big[cols].round(5)""")
    n.code("""pool = {j: v1.score_job(j, list(labels[labels.job_id == j].resume_id)) for j in labels.job_id.unique()}
changed = 0
for _, row in g1.iterrows():
    s = pool[row.job_id].set_index('resume_id'); orig = pairs.set_index('pair_id').loc[row.pair_id, 'resume_id_original']; tw = pairs.set_index('pair_id').loc[row.pair_id, 'resume_id_twin']
    lo, hi = sorted([row.score_original, row.score_twin]); between = ((s.score > lo) & (s.score < hi) & ~s.index.isin([orig, tw])).sum(); changed += int(between > 0)
print('comparisons where the gap is large enough to put another candidate between the twins:', changed, 'of', len(g1))""")
    n.stage(31, "Which words explain the gap? The twins differ only in name and e-mail words, so only those words can contribute.")
    n.code("""row = g1.sort_values('abs_gap', ascending=False).iloc[0]; p = pairs.set_index('pair_id').loc[row.pair_id]
o_, t_ = v1._prep[p.resume_id_original]['doc'], v1._prep[p.resume_id_twin]['doc']; so_, st_ = set(o_.split()), set(t_.split())
idf = dict(zip(v1.vec.get_feature_names_out(), v1.vec.idf_)); only = sorted(((idf.get(w, 0), w) for w in (so_ ^ st_)), reverse=True)[:8]
print('largest-gap pair', row.pair_id, row.job_id, '| gap %.5f' % row.gap); print('words that exist in only one twin, with their idf weight:', [(w, round(i, 2)) for i, w in only])""")
    n.stage(32, "Group view with the supplied `group_gap()` helper, on the **full pool of 185 resumes** per job, using the restricted attributes only to form groups. The gap is the difference between the best and worst group mean score.")
    n.code("""import importlib.util
from cortexhr.phase3.explanations.risk_report import load_helpers
H = load_helpers()
if attrs is None: print('restricted file not present: group view skipped')
else:
    out = []
    for name, rk in (('v1', v1), ('v2', v2)):
        for jid in jobs.job_id:
            s = rk.score_job(jid).set_index('resume_id').score; gg = attrs.loc[s.index]
            for grp in ('inferred_gender', 'college_tier'):
                means, gap = H.group_gap(s.values, gg[grp].values); out.append({'screener': name, 'job_id': jid, 'attribute': grp, 'group_gap': gap, **{f'mean_{k}': v for k, v in means.items()}})
    grp = pd.DataFrame(out); summary = grp.groupby(['screener', 'attribute']).group_gap.agg(['mean', 'max']).round(4); save_table(grp, P.task_dir(12) / 'group_gaps_full_pool.csv'); print(summary.to_string())""")
    n.code("""grp_txt = 'restricted attribute file not present, group view not run'
if attrs is not None:
    piv = grp.groupby(['screener', 'attribute']).group_gap.mean().unstack(0); print(piv.round(4).to_string())
    grp_txt = ('average group gap by college tier %.4f before and %.4f after masking; by gender %.4f before and %.4f after (best minus worst group mean score, 0 to 1 scale)'
               % (piv.loc['college_tier', 'v1'], piv.loc['college_tier', 'v2'], piv.loc['inferred_gender', 'v1'], piv.loc['inferred_gender', 'v2']))
    print('Reading:', grp_txt + '. These gaps mix real differences in skills and experience between people with differences caused by the screener, so they are a warning signal, not proof. Masking narrowed them a little; it does not remove them, and it should not be expected to.')
else: print(grp_txt)""")
    n.stage(33, "How reliable is the 'zero'? After masking, the two twins have identical scoring text, so a zero gap is guaranteed by construction. It shows the *leak is closed*, not that the screener is fair in every respect. Check the tie-break too: when scores are identical, which twin is ranked first?")
    n.code("""print('share of comparisons where the twin is ranked above the original: before %.2f | after %.2f (identical scores are ordered by a neutral hash, so about 0.5 is expected)' % (g1.twin_ranked_higher.mean(), g2.twin_ranked_higher.mean()))
print('after the fix, comparisons with exactly equal scores:', int((g2.abs_gap == 0).sum()), 'of', len(g2))""")
    n.stage(34, "Decision: use v2. It removes the measured gap, ranking quality barely changes, and it removes protected information from the score rather than hoping it does not matter.")
    n.code("""crit = CFG['resume']['success_criteria']; dn = q2['ndcg@5'] - q1['ndcg@5']
chk = pd.DataFrame([('mean twin gap after fix <= tolerance', s2['mean_abs_gap'], crit['max_twin_gap_after_fix'], s2['mean_abs_gap'] <= crit['max_twin_gap_after_fix']),
                    ('max twin gap after fix <= tolerance', s2['max_abs_gap'], crit['max_twin_gap_after_fix'], s2['max_abs_gap'] <= crit['max_twin_gap_after_fix']),
                    ('NDCG@5 did not drop by more than allowed', -dn, crit['max_ndcg_drop_after_fix'], -dn <= crit['max_ndcg_drop_after_fix'])], columns=['criterion', 'measured', 'limit', 'passed']); chk""")
    n.stage(35, "The per-pair results (75 rows) for both screeners, with hidden attributes attached for interpretation.")
    n.code("""res = pd.concat([attach_protected(g1, pairs, attrs).assign(screener='v1 before'), attach_protected(g2, pairs, attrs).assign(screener='v2 after')])
save_table(res, P.task_dir(12) / 'twin_pair_results.csv'); print(res.shape); res[res.screener == 'v1 before'].head(5).round(6)""")
    n.stage(36, "Validation: all 25 pairs and 75 comparisons present in both screeners; no hidden attribute was used as a score input.")
    n.code("""ok = {'25 pairs in both': g1.pair_id.nunique() == 25 and g2.pair_id.nunique() == 25, '75 comparisons in both': len(g1) == 75 and len(g2) == 75,
      'score inputs exclude protected fields': set(v2.used_inputs()['never_used']) >= {'candidate_name', 'gender', 'age', 'college_tier'}, 'all success criteria passed': bool(chk.passed.all())}
pd.Series(ok, name='ok').to_frame()""")
    n.stage(37, "Evidence saved as tables, plus the fairness statement written from the numbers computed above.")
    n.code("""hl = labels.set_index(['resume_id', 'job_id']).human_suitability_score
hdiff = [hl[(p.resume_id_twin, j)] - hl[(p.resume_id_original, j)] for _, p in pairs.iterrows() for j in sorted(set(labels[labels.resume_id == p.resume_id_original].job_id) & set(labels[labels.resume_id == p.resume_id_twin].job_id))]
human_same = all(d == 0 for d in hdiff); n_pairs = len(pairs); n_tier = int(pairs.attribute_changed.str.contains('tier').sum())
print('human reviewers scored all', len(hdiff), 'twin comparisons identically:', human_same)""")
    n.code("""stmt = f'''# Fairness statement: CortexHR resume screener (Task 12)

**What was tested.** 25 twin resume pairs (75 resume-job comparisons) that are identical in skills and experience and differ only in a gender-coded name ({n_pairs} pairs) and, in {n_tier} of them, also the college-tier phrase. Both twins were scored by the screener on every job they were labelled for.

**What was found (before the fix).** The first screener read the resume text as given. The score of the twin differed from the original in {s1['comparisons_with_nonzero_gap']} of {s1['comparisons']} comparisons (mean absolute gap {s1['mean_abs_gap']:.6f}, maximum {s1['max_abs_gap']:.6f} on a 0 to 1 scale). The mean signed gap was {s1['mean_signed_gap(twin-original)']:.6f}: {direction}. The score was reacting to the candidate's name and e-mail even if it did not favour one side. Human reviewers scored all twins identically: {human_same}.

**What was changed.** Name, e-mail, city and the college-tier phrase are removed before scoring. The ablation shows the whole gap came from name and e-mail words.

**After the fix.** Mean absolute gap {s2['mean_abs_gap']:.6f}, maximum {s2['max_abs_gap']:.6f}, non-zero in {s2['comparisons_with_nonzero_gap']} comparisons. Ranking quality moved from NDCG@5 {q1['ndcg@5']:.3f} to {q2['ndcg@5']:.3f} (change {dn:+.3f}) and precision@5 from {q1['precision@5']:.3f} to {q2['precision@5']:.3f}.

**Equal accuracy is not equal treatment:** two groups can have the same average accuracy while an individual still receives a different score or crosses a shortlist threshold because of a protected detail; accuracy compares aggregate outcomes, while treatment asks whether comparable individuals are handled consistently.

**What remains.** (1) A zero gap on 25 twin pairs is guaranteed by construction once the identical text is masked; it proves the leak is closed, not that the screener is fair in every respect. (2) Other proxies were not tested: writing style, hobbies, career gaps, language, employer names, degree names. (3) Group gaps in the full pool of 185 resumes still exist ({grp_txt}); they reflect real differences in skills and experience as well as any bias, so they need review with the recruiter, not a mechanical fix. (4) Ten jobs and 555 labels are a small basis. (5) A human must review every candidate.

**Before real use.** Test with more pairs and more attributes (age-coded years, career gaps, region), re-test after every change to the text processing, monitor shortlist rates by group, and keep a documented human override.
'''
(P.ROOT / 'docs' / 'fairness').mkdir(parents=True, exist_ok=True); (P.ROOT / 'docs' / 'fairness' / 'fairness_statement.md').write_text(stmt, encoding='utf-8')
save_json({'before': s1, 'after': s2, 'quality_before': q1, 'quality_after': q2, 'ablation': abl.to_dict('records')}, P.task_dir(12) / 'fairness_results.json'); print(stmt)""")
    n.stage(38, "")
    n.md("**Why equal accuracy is not equal treatment.** A screener can be equally accurate for two groups and still treat two *individuals* differently: if a name shifts a score by a little, some people fall below the line and others rise above it, even though average accuracy is the same. Accuracy is a property of a group of decisions; treatment is a property of one decision. The twin test measures treatment.")
    n.md("**Trade-off: who decides?** If a fix lowered ranking quality a little, that trade-off is not for the developer alone. My view: the decision belongs to the business owner of the hiring process together with HR, legal and, ideally, employee representatives, after being shown the numbers in plain words (here, masking changed NDCG@5 by a tiny amount, so no real trade-off arose). A developer should refuse to ship a screener that reacts to protected details, whatever the accuracy gain.")
    n.code("""print('FINDINGS'); print('- Before: non-zero twin gap in %d of %d comparisons (mean %.6f, max %.6f), caused by name and e-mail words in the free text.' % (s1['comparisons_with_nonzero_gap'], s1['comparisons'], s1['mean_abs_gap'], s1['max_abs_gap']))
print('- After masking: mean %.6f, max %.6f; NDCG@5 change %+.3f.' % (s2['mean_abs_gap'], s2['max_abs_gap'], dn)); print('- Human reviewers gave identical scores to all twins:', human_same, '| direction of the gap:', direction)
print('LIMITS: 25 pairs; zero gap after masking is by construction; other proxies untested; see docs/fairness/fairness_statement.md.')""")
    return n


if __name__ == "__main__":
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "notebooks"
    for name, f in (("05_task12_fairness_baseline", nb05), ("06_task12_fairness_improvement", nb06)):
        print(f().save(out / f"{name}.ipynb"))
