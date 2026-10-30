"""Chatbot (33-intent NLU) track, part 1: SETUP and stages 1-19."""

SETUP = r'''
import sys, json, time, hashlib, platform, re, warnings, pathlib
warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
ROOT = pathlib.Path.cwd().resolve()
while ROOT != ROOT.parent and not (ROOT/'src'/'cortexhr').exists(): ROOT = ROOT.parent
sys.path.insert(0, str(ROOT/'src'))
from cortexhr.phase2 import chatbot as CB, service as S
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedGroupKFold
from sklearn import metrics as M
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
pd.set_option('display.width', 200); pd.set_option('display.max_columns', 60); pd.set_option('display.max_colwidth', 80)

CSV = ROOT/'03_Data'/'csv'; REP = ROOT/'reports'/'phase2'; WF = REP/'workflow38'; WF.mkdir(parents=True, exist_ok=True)
MOCK = ROOT/'05_Starter'/'mock_services'/'mock_data'
SEED = 42
TRAIN_RAW = pd.read_csv(CSV/'chatbot_nlu_training.csv')       # untouched copy of the official training file
TEST_RAW = pd.read_csv(CSV/'chatbot_test_set.csv')            # official held-out test: NOT used for any modelling decision before stage 27
RESP = pd.read_csv(CSV/'chatbot_responses.csv').fillna(''); FAQ = pd.read_csv(CSV/'hr_faq.csv')
RESULTS = {}

def clean_text(s): return ' '.join(str(s if s is not None and s == s else '').split())
def word_vec(): return TfidfVectorizer(analyzer='word', ngram_range=(1, 2), min_df=1, strip_accents='unicode', sublinear_tf=True, lowercase=True, token_pattern=r'(?u)\b\w+\b')
def char_vec(ngmax=5): return TfidfVectorizer(analyzer='char_wb', ngram_range=(2, ngmax), min_df=1, sublinear_tf=True, lowercase=True)
def feats(kind='word+char', ngmax=5):
    return {'word': word_vec, 'char': lambda: char_vec(ngmax)}[kind]() if kind != 'word+char' else FeatureUnion([('word_tfidf', word_vec()), ('char_tfidf', char_vec(ngmax))])
def lr(C=2.0, cw='balanced'): return LogisticRegression(max_iter=3000, class_weight=cw, C=C, solver='lbfgs', random_state=SEED)
def make_model(clf=None, kind='word+char', ngmax=5): return Pipeline([('features', feats(kind, ngmax)), ('classifier', clf if clf is not None else lr())])
def near_groups(texts, intents, th=0.5):
    """Group templated paraphrases: same intent and word-TF-IDF cosine >= th are one family (connected components)."""
    X = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit_transform(texts); Sm = (X @ X.T).toarray(); it = np.asarray(intents).astype(str)
    return connected_components(csr_matrix((Sm >= th) & (it[:, None] == it[None, :])))[1]
def scores(y, p): return {'accuracy': round(M.accuracy_score(y, p), 4), 'macro_F1': round(M.f1_score(y, p, average='macro', zero_division=0), 4), 'weighted_F1': round(M.f1_score(y, p, average='weighted', zero_division=0), 4)}
print('train rows', len(TRAIN_RAW), '| intents', TRAIN_RAW.intent.nunique(), '| test rows (sealed until stage 27)', len(TEST_RAW), '| seed', SEED)
'''

NOTES = {}
CODE = {}

# ------------------------------------------------------------------------------------------------ 1
NOTES[1] = "writes the business case for the HR assistant as data: users, current process, what a wrong answer costs, privacy and legal constraints and whether ML is needed at all."
CODE[1] = r'''
ASSUMPTIONS = {'cost_wrong_answer': 5, 'cost_fallback': 1, 'unit': 'relative cost points (ASSUMPTION - confirm with HR operations)'}
business = pd.DataFrame([
 ['Problem', 'Employees ask HR the same questions all day (leave balance, payslip date, policies); HR staff answer them by hand.'],
 ['Objective / outcome', 'An assistant that understands the question (intent), answers policy questions from the FAQ, fetches the signed-in user\'s own data and politely refuses everything else.'],
 ['Stakeholders / end users', 'Employees (ask), HR staff (get fewer routine tickets, own the policy answers), HR/Admin roles (extra permissions), security/legal (privacy), IT (runs it).'],
 ['Current process', 'Email/phone/tickets to HR; policy handbook PDF.'],
 ['Limitations of current process', 'Slow, office-hours only, inconsistent answers, repeated work.'],
 ['Business questions', 'What does the user want (33 intents)? Is the model sure enough to answer? Is this user allowed to see it?'],
 ['False positive cost (wrong intent, confident)', f"Wrong answer to a policy question or, worse, the wrong private-data call. Assumed cost {ASSUMPTIONS['cost_wrong_answer']} points."],
 ['False negative cost (fallback when it could answer)', f"User must rephrase or ask HR. Assumed cost {ASSUMPTIONS['cost_fallback']} point."],
 ['Measurable success', 'High macro-F1 on messages the model has never seen, near-perfect refusal of other people\'s pay, fallback rather than guessing on unclear messages.'],
 ['Constraints', 'Salary and payslip data are confidential; an employee may only see their own; the statistical model must NEVER decide authorization (rules do); synthetic data only.'],
 ['Is ML necessary?', 'Partly: 33 intents with many phrasings need a classifier, but permissions, same-name handling, FAQ answers and the pay-privacy guard are deterministic rules.'],
], columns=['topic', 'answer'])
display(business); RESULTS['assumptions'] = ASSUMPTIONS
'''

# ------------------------------------------------------------------------------------------------ 2
NOTES[2] = "defines the task precisely: multiclass intent classification of one user message, with timing, available information, output and how the output is used."
CODE[2] = r'''
INTENTS = sorted(TRAIN_RAW.intent.astype(str).unique())
problem = pd.DataFrame([
 ['Target variable', 'intent (one label per message)'], ['Classes', f'{len(INTENTS)} intents, e.g. leave_balance, payslip_status, hr_policy_leave, salary_of_other, out_of_scope, greet ...'],
 ['Binary or multiclass', 'Multiclass (33 classes)'], ['Prediction unit', 'One user message (one turn)'], ['When predicted', 'Every time the user sends a message, before any data is fetched'],
 ['Horizon', 'None - classification of the current message (follow-up memory is handled by dialogue rules, not by the model)'],
 ['Information at prediction time', 'The message text (the role and identity come from the directory, never from the text)'], ['Expected output', 'intent + confidence (probability of the top class) + the next-best intents'],
 ['How it is used', 'confidence >= threshold -> answer/route; below -> safe fallback. Permission and privacy rules run AFTER and can override the model.']], columns=['item', 'definition'])
display(problem); assert len(INTENTS) == 33 and {'text', 'intent'} <= set(TRAIN_RAW.columns)
print('message is the only model input; no user role, id or salary value is available to the model.')
'''

# ------------------------------------------------------------------------------------------------ 3
NOTES[3] = "fixes the evaluation plan before modelling: primary metric, secondary metrics, minimum performance, baselines to beat, error costs, latency, explainability and fairness requirements."
CODE[3] = r'''
ACCEPT = {'primary_metric': 'macro-F1 on UNSEEN messages (grouped cross-validation, then the sealed test set once)', 'secondary_metrics': 'accuracy, weighted-F1, per-intent recall, top-3 accuracy, PR-AUC (macro), fallback coverage and error rate',
          'minimum_macro_F1': 0.70, 'minimum_accuracy': 0.75, 'applies_to': 'the SEALED test file (stage 27); grouped CV and validation are conservative development estimates used to rank models', 'min_recall_salary_of_other': 0.90, 'baselines_to_beat': 'majority class (1/33) and a hand-written keyword rule',
          'error_costs': RESULTS['assumptions'], 'what_matters': 'hard classification + a trustworthy confidence (for the fallback); calibrated probabilities are useful, ranking alone is not enough',
          'latency_ms_per_message_max': 200, 'explainability': 'top n-grams per intent can be shown', 'fairness_reliability': 'typos, short messages and Tanglish must not collapse accuracy; private-data refusal must not depend on the model',
          'note_on_90pct': 'a 90%+ goal was discussed in the project; the README states it is NOT reached (81.06% held-out). We report the measured number.'}
display(pd.Series(ACCEPT, name='frozen before modelling').astype(str).to_frame()); RESULTS['accept'] = ACCEPT
'''

# ------------------------------------------------------------------------------------------------ 4
NOTES[4] = "inventories the data files with fingerprints, records provenance and limitations, runs a privacy scan of the training text and proves the original data is untouched."
CODE[4] = r'''
def sha(p): return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()[:16]
files = ['chatbot_nlu_training.csv', 'chatbot_test_set.csv', 'chatbot_responses.csv', 'hr_faq.csv', 'chatbot_dialogues.csv', 'chatbot_nlu_training_expanded.csv']
inv = pd.DataFrame([{'file': f, 'rows': len(pd.read_csv(CSV/f)), 'sha256_16': sha(CSV/f), 'role': {'chatbot_nlu_training.csv': 'training (official)', 'chatbot_test_set.csv': 'held-out test (sealed)', 'chatbot_responses.csv': 'answers + allowed roles', 'hr_faq.csv': 'FAQ answers',
                    'chatbot_dialogues.csv': 'multi-turn scenarios', 'chatbot_nlu_training_expanded.csv': 'development-only augmentation (not used here)'}[f]} for f in files if (CSV/f).exists()]); display(inv); inv.to_csv(WF/'chatbot_source_manifest.csv', index=False)
prov = {'owner / method': 'CortexHR training dataset; messages written from templates (greeting prefixes, typos, fillers) - synthetic', 'time period': 'none (no timestamps)', 'relevance': 'Covers the 33 intents the bot must route, plus out_of_scope and chitchat',
        'labels': 'intent present for every message; test labels in expected_intent', 'privacy': 'No real employee messages; scanning for personal data below', 'limitations': 'Templated and tiny (26 per intent); real users will phrase things differently; test set has only 4 per intent'}
display(pd.Series(prov, name='provenance').to_frame())
pii = {'e-mail': r'[\w.+-]+@[\w-]+\.[\w.]+', 'phone': r'\b\d{10}\b', 'employee id': r'\bEMP-\d{3,6}\b'}; txt = TRAIN_RAW.text.astype(str)
print('personal-data scan of the training text:', {k: int(txt.str.contains(v, regex=True).sum()) for k, v in pii.items()})
assert TRAIN_RAW.equals(pd.read_csv(CSV/'chatbot_nlu_training.csv')); print('untouched original verified; fingerprints saved')
'''

# ------------------------------------------------------------------------------------------------ 5
NOTES[5] = "loads and inspects the training and test files: shape, samples, types, duplicates, label values, text length and the expected schema."
CODE[5] = r'''
print('train', TRAIN_RAW.shape, '| test', TEST_RAW.shape); display(TRAIN_RAW.head(6)); display(TEST_RAW.head(4))
print('dtypes:', TRAIN_RAW.dtypes.astype(str).to_dict()); print('duplicate column names:', int(TRAIN_RAW.columns.duplicated().sum()), '| exact duplicate rows (text+intent):', int(TRAIN_RAW.duplicated(['text', 'intent']).sum()), '| duplicate example_id:', int(TRAIN_RAW.example_id.duplicated().sum()))
print('target column: intent (train) / expected_intent (test)'); print('unique intents train/test:', TRAIN_RAW.intent.nunique(), TEST_RAW.expected_intent.nunique())
display(TRAIN_RAW.text.str.len().describe().round(1).to_frame('characters per message').T); display(TRAIN_RAW.entities.notna().sum() if 'entities' in TRAIN_RAW else 0)
print('schema check: required columns present ->', {'text', 'intent'} <= set(TRAIN_RAW.columns) and {'text', 'expected_intent'} <= set(TEST_RAW.columns))
'''

# ------------------------------------------------------------------------------------------------ 6
NOTES[6] = "builds the data dictionary and an intent catalogue (policy, personal data, HR-only, social, safety, not-connected) with allowed roles, and records assumptions and open questions."
CODE[6] = r'''
dictionary = pd.DataFrame([['example_id', 'identifier', 'unique id of a training message', 'yes'], ['text', 'text (model input)', 'the user message', 'yes'], ['intent', 'target (categorical, 33 values)', 'what the user wants', 'no (label)'],
    ['entities', 'text/json (optional)', 'slot values such as month or leave type (used by dialogue rules, not by the classifier)', 'partly'], ['test_id / expected_intent / expected_entities', 'identifier / target / text', 'held-out test file', 'evaluation only']],
    columns=['column', 'type', 'meaning', 'available at prediction time']); display(dictionary); dictionary.to_csv(WF/'chatbot_data_dictionary.csv', index=False)
roles = RESP.set_index('intent').allowed_roles.to_dict()
def family(i):
    if i in ('greet', 'thanks', 'goodbye', 'chitchat', 'help'): return 'social / help'
    if i == 'salary_of_other': return 'safety (must refuse)'
    if i in CB.FAQ_INTENTS: return 'policy FAQ'
    if i in CB.NOT_CONNECTED: return 'later-phase model (not connected)'
    if i == 'out_of_scope': return 'out of scope'
    return 'personal data (needs login)' if roles.get(i, 'ALL') != 'ALL' else 'other'
cat = pd.DataFrame({'intent': INTENTS}); cat['family'] = cat.intent.map(family); cat['allowed_roles'] = cat.intent.map(roles); INTENT_FAMILY = cat.set_index('intent').family.to_dict(); display(cat.groupby('family').intent.apply(lambda s: ', '.join(s)).to_frame('intents')); cat.to_csv(WF/'chatbot_intent_catalogue.csv', index=False)
print('features available at prediction time: only the message text. Created after the target event: none.')
print('business rules: private-data intents need login and the right role; salary_of_other is refused for everyone; below-threshold confidence -> safe fallback.')
print('open questions: (1) who confirms the intent list is complete? (2) should Tanglish (Tamil in English letters) be an officially supported input? (3) which languages must be covered?')
'''

# ------------------------------------------------------------------------------------------------ 7
NOTES[7] = "audits label and text quality: empty/duplicate texts, label conflicts, unexpected labels, class balance, length outliers and train/test coverage, then writes an issue log with resolutions."
CODE[7] = r'''
t = TRAIN_RAW.copy(); t['norm'] = t.text.map(lambda s: clean_text(s).casefold())
conf = t.groupby('norm').intent.nunique(); conflicts = int((conf > 1).sum()); dup_text = int(t.duplicated('norm').sum()); lens = t.text.str.len(); z = (lens - lens.mean()) / lens.std()
unexpected = sorted(set(TRAIN_RAW.intent.astype(str)) - set(RESP.intent)); unseen_in_train = sorted(set(TEST_RAW.expected_intent.astype(str)) - set(TRAIN_RAW.intent.astype(str)))
typo_hits = int(t.text.str.contains(r'urgenlty|laeve|plz|pls|thx|abt', case=False, regex=True).sum()); nonascii = int(t.text.map(lambda s: any(ord(c) > 127 for c in str(s))).sum())
issues = pd.DataFrame([['missing / empty text', int(t.text.isna().sum() + (t.text.astype(str).str.strip() == '').sum()), 'none found'], ['missing label', int(t.intent.isna().sum()), 'none found'], ['duplicate normalised text', dup_text, 'keep one per (text,intent) in cleaning; none conflict'],
    ['same text, different intent (label conflict)', conflicts, 'would be resolved with the data owner; none found'], ['intents without a response template', len(unexpected), 'every intent has a response row' if not unexpected else str(unexpected)],
    ['test intents unseen in training', len(unseen_in_train), 'none'], ['messages with deliberate typos/slang', typo_hits, 'KEEP: real users make typos; they are useful training signal'], ['non-ASCII messages', nonascii, 'keep; vectorizer strips accents'],
    ['length outliers (|z| > 3)', int((z.abs() > 3).sum()), 'keep; short messages (hi, bye) are legitimate'], ['class balance', 'min %d / max %d per intent' % (t.intent.value_counts().min(), t.intent.value_counts().max()), 'perfectly balanced - imbalance is not a data problem here (stage 18)'],
    ['label quality', 'templated', 'labels are consistent, but templated paraphrases are near-duplicates -> inflates ordinary cross-validation (stage 9)']], columns=['check', 'result', 'resolution']); display(issues); issues.to_csv(WF/'chatbot_issue_log.csv', index=False)
print('test file class balance:', TEST_RAW.expected_intent.value_counts().describe()[['min', 'max']].to_dict()); print('Nothing was deleted or relabelled; fixes (whitespace, exact duplicates) are applied to a copy in stage 10.')
RESULTS['issues'] = issues.set_index('check').result.astype(str).to_dict()
'''

# ------------------------------------------------------------------------------------------------ 8
NOTES[8] = "explores the target distribution, message lengths, vocabulary per intent, which intents are lexically close (confusion hypotheses), templated patterns and subgroup differences, and summarises implications."
CODE[8] = r'''
vc = TRAIN_RAW.intent.value_counts(); print('target distribution: every intent has', int(vc.min()), 'examples (', round(100 / 33, 2), '% each)')
lens = TRAIN_RAW.text.str.len(); fam = TRAIN_RAW.intent.map(INTENT_FAMILY)
fig, ax = plt.subplots(1, 3, figsize=(17, 4)); ax[0].hist(lens, bins=30); ax[0].set(title='message length (characters)'); pd.DataFrame({'len': lens, 'family': fam}).boxplot(by='family', column='len', ax=ax[1], rot=60); ax[1].set_title('length by intent family'); plt.suptitle('')
words = TRAIN_RAW.text.str.lower().str.findall(r'\w+').explode().value_counts().head(15); words[::-1].plot.barh(ax=ax[2], title='most frequent words'); plt.tight_layout(); plt.show()
v = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words='english').fit(TRAIN_RAW.text); X = v.transform(TRAIN_RAW.text); cent = np.vstack([np.asarray(X[(TRAIN_RAW.intent == i).to_numpy()].mean(axis=0)).ravel() for i in INTENTS]); cent /= np.linalg.norm(cent, axis=1, keepdims=True); cs = cent @ cent.T; np.fill_diagonal(cs, 0)
pairs = pd.DataFrame([(INTENTS[i], INTENTS[j], round(cs[i, j], 3)) for i in range(33) for j in range(i + 1, 33)], columns=['intent A', 'intent B', 'centroid cosine']).sort_values('centroid cosine', ascending=False); display(pairs.head(8))
fillers = {'polite prefix (please/can you)': TRAIN_RAW.text.str.contains(r'\bplease\b|\bcan you\b', case=False).mean(), 'trailing filler (urgently/pls)': TRAIN_RAW.text.str.contains(r'urgen|\bpls\b|\bplz\b', case=False).mean(), 'question mark': TRAIN_RAW.text.str.contains(r'\?').mean()}; display(pd.Series(fillers, name='share of messages').round(3).to_frame())
display(pd.DataFrame({'mean chars': lens.groupby(fam).mean().round(1), 'intents': TRAIN_RAW.groupby(fam).intent.nunique()}))
print('Hypotheses to test later (not causes): H1 policy intents with similar wording are the main confusion (check in stage 24); H2 very short messages are harder; H3 the templated wording makes ordinary CV far too optimistic (tested in stage 9).')
print('Findings: balanced classes; short messages (median ~36 chars); heavy template reuse (polite prefixes, typos); the closest intent pairs above are the likely confusions.')
RESULTS['close_pairs'] = pairs.head(5).values.tolist()
'''

# ------------------------------------------------------------------------------------------------ 9
NOTES[9] = "investigates leakage specific to templated text: train/test overlap, near-duplicate families inside training (which inflate ordinary cross-validation), and confirms vectorizers are fitted inside the pipeline."
CODE[9] = r'''
a = TRAIN_RAW.text.map(lambda s: clean_text(s).casefold()); b = TEST_RAW.text.map(lambda s: clean_text(s).casefold()); print('exact train/test text overlap:', len(set(a) & set(b)), 'messages')
vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(TRAIN_RAW.text); Xa, Xb = vec.transform(TRAIN_RAW.text), vec.transform(TEST_RAW.text); sim_tt = np.asarray((Xb @ Xa.T).max(axis=1).todense()).ravel(); Saa = (Xa @ Xa.T).toarray(); np.fill_diagonal(Saa, 0)
print(f'near-duplicates (cosine >= 0.8) of a TRAIN message: {(Saa.max(1) >= 0.8).mean() * 100:.0f}% of training messages vs {(sim_tt >= 0.8).mean() * 100:.0f}% of TEST messages -> the test set is much less templated than the training set')
GROUPS_ALL = near_groups(TRAIN_RAW.text, TRAIN_RAW.intent, 0.5); print('paraphrase families in training (same intent, cosine >= 0.5):', GROUPS_ALL.max() + 1, 'families for', len(TRAIN_RAW), 'messages (average', round(len(TRAIN_RAW) / (GROUPS_ALL.max() + 1), 1), 'per family)')
from sklearn.model_selection import cross_val_score, StratifiedKFold
plain = cross_val_score(make_model(), TRAIN_RAW.text, TRAIN_RAW.intent, cv=StratifiedKFold(5, shuffle=True, random_state=SEED), scoring='f1_macro')
grouped = cross_val_score(make_model(), TRAIN_RAW.text, TRAIN_RAW.intent, cv=StratifiedGroupKFold(5, shuffle=True, random_state=SEED), groups=GROUPS_ALL, scoring='f1_macro')
leak = pd.DataFrame({'ordinary stratified CV (paraphrases split across folds)': plain, 'grouped CV (families kept together)': grouped}).round(3); display(leak.T.assign(mean=leak.mean().round(3)))
print('LEAKAGE FOUND: ordinary CV macro-F1 ~%.2f vs grouped CV ~%.2f. A paraphrase of the same template sits in both train and validation folds, so ordinary CV mostly measures memorisation.' % (plain.mean(), grouped.mean()))
print('Decision: ALL validation in this workflow is grouped by paraphrase family (StratifiedGroupKFold); the official test file (little overlap) is the reality check.')
chk = pd.DataFrame([['target encoded in a feature', 'none: the only input is the message text', True], ['feature unavailable at prediction time', 'none', True], ['train/test exact overlap', f'{len(set(a) & set(b))} messages', len(set(a) & set(b)) == 0],
    ['vectorizer fitted on validation/test text', 'TF-IDF lives inside the Pipeline; fitted per training fold only', True], ['target-derived aggregates', 'none', True], ['near-duplicate families across folds', 'handled by grouped CV (above)', True], ['temporal leakage', 'not applicable (no timestamps)', True]], columns=['check', 'finding / decision', 'ok']); display(chk); assert chk.ok.all()
RESULTS['leakage'] = {'plain_cv_macro_F1': round(float(plain.mean()), 3), 'grouped_cv_macro_F1': round(float(grouped.mean()), 3)}
'''

# ------------------------------------------------------------------------------------------------ 10
NOTES[10] = "cleans a COPY of the training data (whitespace, exact duplicates), keeps typos and short messages on purpose, resolves label conflicts (none), validates against the response table and records every decision."
CODE[10] = r'''
TR = TRAIN_RAW[['example_id', 'text', 'intent']].copy(); TR['text'] = TR.text.map(clean_text); TR['intent'] = TR.intent.astype(str).str.strip()
before = len(TR); TR = TR.drop_duplicates(['text', 'intent']).reset_index(drop=True); removed = before - len(TR)
conf = TR.assign(n=TR.text.str.casefold()).groupby('n').intent.nunique(); assert (conf == 1).all()
decisions = pd.DataFrame([['duplicates', f'{removed} exact (text, intent) duplicates removed from the working copy'], ['data types', 'text -> str, intent -> str'], ['categorical values', 'intent labels trimmed; 33 distinct values unchanged'], ['units / formats', 'whitespace collapsed; case is NOT changed here (the vectorizer lower-cases)'],
    ['impossible values', 'none (no empty texts)'], ['outliers', 'very short messages are legitimate (hi, bye) and kept'], ['rare observations', 'typos/slang kept on purpose (real users type them)'], ['missing values', 'none'], ['conflicting records', 'none (a text never has two intents)'], ['original data', 'unchanged on disk and in TRAIN_RAW']], columns=['topic', 'decision']); display(decisions); decisions.to_csv(WF/'chatbot_cleaning_decisions.csv', index=False)
assert TR.text.str.len().min() > 0 and set(TR.intent) <= set(RESP.intent) and TRAIN_RAW.equals(pd.read_csv(CSV/'chatbot_nlu_training.csv')); print('cleaned copy validated:', len(TR), 'rows,', TR.intent.nunique(), 'intents, all intents have a response template; original untouched')
GROUPS = near_groups(TR.text, TR.intent, 0.5); print('paraphrase families recomputed on the cleaned copy:', GROUPS.max() + 1)
'''

# ------------------------------------------------------------------------------------------------ 11
NOTES[11] = "documents the text features (word and character n-grams), explains why date/aggregation/interaction features do not apply, and compares word-only, character-only and combined features with grouped cross-validation."
CODE[11] = r'''
feat_doc = pd.DataFrame([['word TF-IDF, 1-2 grams', 'captures intent phrases such as "casual leave", "pay slip"'], ['character TF-IDF, 2-5 grams inside word boundaries', 'robust to typos, short messages and word forms (laeve, leaves)'], ['sublinear tf + lower-casing + accent stripping', 'damps repeated words, ignores case/accents'],
    ['date/time features', 'not applicable (no timestamps)'], ['frequency / recency / aggregation', 'not applicable (one message at a time; dialogue memory is handled by rules)'], ['rare-category grouping', 'not applicable (text, not categories)'], ['interaction features', 'not used: n-grams already capture word combinations'],
    ['business concepts as features', 'month, leave type and person name are extracted by the dialogue rules (slots), not fed to the classifier'], ['future information', 'none: the model sees only the current message']], columns=['feature / idea', 'purpose or reason it is not used']); display(feat_doc)
print('availability at inference: the same Pipeline vectorizes any incoming string, including empty/unseen words ->', make_model().fit(TR.text, TR.intent).predict(['zzzz', ''])[:2])
rows = []; cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
for kind in ['word', 'char', 'word+char']:
    f1, ac, dim = [], [], []
    for tr_i, va_i in cv.split(TR.text, TR.intent, GROUPS):
        m = make_model(kind=kind).fit(TR.text.iloc[tr_i], TR.intent.iloc[tr_i]); p = m.predict(TR.text.iloc[va_i]); f1.append(M.f1_score(TR.intent.iloc[va_i], p, average='macro')); ac.append(M.accuracy_score(TR.intent.iloc[va_i], p)); dim.append(len(m.named_steps['features'].get_feature_names_out()) if hasattr(m.named_steps['features'], 'get_feature_names_out') else 0)
    rows.append({'feature set': kind, 'n_features (last fold)': dim[-1], 'macro_F1 mean': round(np.mean(f1), 3), 'macro_F1 std': round(np.std(f1), 3), 'accuracy mean': round(np.mean(ac), 3)})
fs = pd.DataFrame(rows); display(fs); RESULTS['feature_sets'] = fs.set_index('feature set')['macro_F1 mean'].to_dict(); print('Feature sets compared on grouped validation folds only (never the test set).')
'''

# ------------------------------------------------------------------------------------------------ 12
NOTES[12] = "creates a stratified, paraphrase-family-grouped 80/20 split of the OFFICIAL TRAINING file for development, keeps the official test file sealed, and records proportions, seed and leakage checks."
CODE[12] = r'''
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED); tr_i, va_i = next(sgkf.split(TR.text, TR.intent, GROUPS))
Xtr, ytr, Xva, yva = TR.text.iloc[tr_i].reset_index(drop=True), TR.intent.iloc[tr_i].reset_index(drop=True), TR.text.iloc[va_i].reset_index(drop=True), TR.intent.iloc[va_i].reset_index(drop=True)
split = pd.DataFrame({'split': ['train (development)', 'validation (development)', 'official test (sealed)'], 'rows': [len(Xtr), len(Xva), len(TEST_RAW)], 'intents': [ytr.nunique(), yva.nunique(), TEST_RAW.expected_intent.nunique()]}); split['share_%'] = (split.rows / split.rows.sum() * 100).round(1); display(split)
print('independent unit: a paraphrase FAMILY (not a single message). Method: stratified by intent AND grouped by family, because a random split would leak paraphrases.')
print('family overlap train/validation:', len(set(GROUPS[tr_i]) & set(GROUPS[va_i])), '| exact text overlap:', len(set(Xtr.str.casefold()) & set(Xva.str.casefold())), '| intents missing from validation:', len(set(ytr) - set(yva)))
vt = pd.DataFrame({'train per intent': ytr.value_counts(), 'validation per intent': yva.value_counts()}).fillna(0).astype(int); display(vt.describe().loc[['min', 'mean', 'max']].round(1))
TEST_FINGERPRINT = hashlib.sha256(''.join(TEST_RAW.text.astype(str)).encode()).hexdigest()[:16]; print('seed', SEED, '| test file fingerprint (checked again in stage 27):', TEST_FINGERPRINT, '| the test file has NOT been used for any decision so far')
print('Reflects the real use case: production sees phrasings it has never seen, which is exactly what a family-grouped split simulates.'); assert not (set(GROUPS[tr_i]) & set(GROUPS[va_i]))
'''

# ------------------------------------------------------------------------------------------------ 13
NOTES[13] = "assembles the reusable preprocessing + model pipeline (cleaning function, word + character TF-IDF, classifier), fits it on the training split only and proves it copes with empty, unseen, very long and non-English input."
CODE[13] = r'''
print('separate features from target: X = message text (str), y = intent. Numerical/categorical imputation and scaling: not applicable to text.')
pipe = make_model().fit(Xtr.map(clean_text), ytr); print('steps:', [n for n, _ in pipe.steps], '| sub-vectorizers:', [n for n, _ in pipe.named_steps['features'].transformer_list])
fe = pipe.named_steps['features']; w = dict(fe.transformer_list)['word_tfidf']; print('vocabulary learned from TRAIN only:', len(w.vocabulary_), 'word n-grams;', 'a word that only appears in validation is simply ignored:', sorted(set(' '.join(Xva).lower().split()) - set(w.vocabulary_))[:6])
weird = ['', '   ', None, 'asdkjhasdkjhasd', 'leave 😀😀', 'x ' * 2500, '12345', 'எனக்கு leave balance evlo irukku', 'LEAVE BALANCE??!!', 'salary of EMP-0003']
out = pd.DataFrame({'input': [str(x)[:30] for x in weird], 'predicted': pipe.predict([clean_text(x) for x in weird]), 'confidence': pipe.predict_proba([clean_text(x) for x in weird]).max(axis=1).round(3)}); display(out)
print('All odd inputs return a prediction with a (low) confidence instead of an exception; the confidence threshold (stage 21) turns the low ones into a safe fallback.')
print('Feature selection: none applied. The same fitted pipeline is used for validation, test and production inference (stage 29 wraps it).')
'''

# ------------------------------------------------------------------------------------------------ 14
NOTES[14] = "builds four baselines (majority class, stratified random, hand-written keyword rules, simple interpretable word-only Naive Bayes / Logistic Regression) and evaluates them on the validation split with confusion analysis."
CODE[14] = r'''
from sklearn.dummy import DummyClassifier
KEY = {'greet': ['hello', 'hi', 'hey', 'good morning'], 'thanks': ['thank'], 'goodbye': ['bye'], 'leave_balance': ['leave balance', 'leaves left', 'how many leaves', 'leaves do i have'], 'payslip_status': ['payslip', 'pay slip'], 'payslip_date': ['salary date', 'payday', 'when is salary'], 'hr_policy_leave': ['leave policy'],
       'wfh_policy': ['work from home', 'wfh'], 'pf_info': ['pf', 'provident'], 'attendance_summary': ['attendance'], 'my_manager': ['manager'], 'holiday_list': ['holiday'], 'salary_of_other': ['salary of', 'salary for'], 'help': ['help'], 'reimbursement_policy': ['reimburse', 'expense']}
def keyword_rule(t):
    t = t.lower(); hits = {i: sum(k in t for k in ks) for i, ks in KEY.items()}; best = max(hits, key=hits.get); return best if hits[best] > 0 else 'out_of_scope'
base = {'majority class (dummy)': DummyClassifier(strategy='most_frequent').fit(Xtr, ytr).predict(Xva), 'stratified random (dummy)': DummyClassifier(strategy='stratified', random_state=SEED).fit(Xtr, ytr).predict(Xva), 'keyword rules (hand-written, 15 intents)': np.array([keyword_rule(t) for t in Xva]),
        'Multinomial NB, word 1-2 grams (simple ML)': make_model(MultinomialNB(alpha=0.3), 'word').fit(Xtr, ytr).predict(Xva), 'Logistic Regression, word only (interpretable ML)': make_model(lr(), 'word').fit(Xtr, ytr).predict(Xva)}
tab = pd.DataFrame({k: scores(yva, p) for k, p in base.items()}).T; display(tab); RESULTS['baselines'] = tab.to_dict('index')
print('Chance level = 1/33 =', round(1 / 33, 3), '| the minimum acceptable accuracy from stage 3 =', ACCEPT['minimum_accuracy'])
p = base['Logistic Regression, word only (interpretable ML)']; cmx = pd.crosstab(pd.Series(yva, name='actual'), pd.Series(p, name='predicted')); off = [(a, b, int(cmx.loc[a, b])) for a in cmx.index for b in cmx.columns if a != b and cmx.loc[a, b] > 0]; print('confusion matrix recorded (33x33); most frequent confusions of the word-only LR:', sorted(off, key=lambda x: -x[2])[:5])
print('Weaknesses: keyword rules only know 15 intents and fail on new wording; word-only models break on typos; every baseline is the reference for stages 15-20.')
'''

# ------------------------------------------------------------------------------------------------ 15
NOTES[15] = "lists the candidate algorithms with reasons for use or exclusion (gradient boosting and neural nets are explained, not blindly run), considers complexity, speed and explainability and picks five candidates."
CODE[15] = r'''
cands = pd.DataFrame([
 ['Logistic Regression (word+char TF-IDF)', 'interpretable baseline, gives probabilities', 'INCLUDED (current production recipe)'], ['Decision Tree', 'rule-like splits on n-grams', 'INCLUDED as a transparent contrast (expected to overfit sparse text)'], ['Random Forest', 'non-linear patterns', 'INCLUDED (200 trees)'],
 ['Gradient boosting (XGBoost/LightGBM)', 'strong on dense tabular data', 'NOT used: thousands of sparse n-gram columns and 26 examples per class favour linear models; would be slow and overfit'],
 ['Linear SVM (calibrated)', 'strong linear text classifier', 'INCLUDED, wrapped with calibration so it can give probabilities for the fallback'], ['Multinomial / Complement Naive Bayes', 'fast, works with TF-IDF counts', 'INCLUDED (Complement NB suits small text data)'],
 ['Neural network / transformer', 'handles meaning and paraphrase', 'NOT justified with 858 templated rows and a 200 ms / no-GPU target; revisit with real user data'],
 ['k-Nearest Neighbours', 'memorises templates', 'NOT used: would look excellent on templated data and fail on new wording']], columns=['candidate', 'why consider it', 'decision']); display(cands)
CANDS = {'Logistic Regression': lambda: make_model(lr()), 'Linear SVM (calibrated)': lambda: make_model(CalibratedClassifierCV(LinearSVC(C=1.0, class_weight='balanced'), cv=3)), 'Complement NB': lambda: make_model(ComplementNB(alpha=0.3)),
         'Random Forest': lambda: make_model(RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1, class_weight='balanced')), 'Decision Tree': lambda: make_model(DecisionTreeClassifier(random_state=SEED, class_weight='balanced'))}
cost = pd.DataFrame([['Logistic Regression', 'low', 'fast', 'high (coefficients)', 'yes'], ['Linear SVM (calibrated)', 'low', 'fast', 'medium', 'yes (via calibration)'], ['Complement NB', 'lowest', 'fastest', 'medium (word weights)', 'yes (poorly calibrated)'], ['Random Forest', 'medium', 'slower', 'low', 'yes'], ['Decision Tree', 'low', 'fast', 'high (but unstable)', 'yes']],
                    columns=['model', 'complexity', 'training/inference cost', 'explainability', 'predict_proba']); display(cost); print('Selected manageable candidate set:', list(CANDS)); print('predict_proba is REQUIRED: the confidence-based safe fallback depends on it.')
'''

# ------------------------------------------------------------------------------------------------ 16
NOTES[16] = "fits every candidate on the training split inside its pipeline, times it, captures warnings, compares training with validation accuracy and saves a reproducible configuration."
CODE[16] = r'''
import warnings as _w
rows, fitted = [], {}
for name, mk in CANDS.items():
    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter('always'); t0 = time.time(); m = mk().fit(Xtr, ytr); fit_s = time.time() - t0
    t0 = time.time(); pv = m.predict(Xva); pred_ms = (time.time() - t0) / len(Xva) * 1000; fitted[name] = m
    rows.append({'model': name, 'fit_s': round(fit_s, 2), 'predict_ms_per_message': round(pred_ms, 2), 'train_accuracy': round(M.accuracy_score(ytr, m.predict(Xtr)), 3), 'validation_accuracy': round(M.accuracy_score(yva, pv), 3), 'validation_macro_F1': round(M.f1_score(yva, pv, average='macro'), 3), 'warnings': len(caught)})
tr_tab = pd.DataFrame(rows); tr_tab['train-validation gap'] = (tr_tab.train_accuracy - tr_tab.validation_accuracy).round(3); display(tr_tab)
print('class weights: used (balanced) where the model supports them - classes are balanced anyway, so the effect is tested in stage 18. Resampling: not used (balanced classes).')
print('convergence/training warnings:', int(tr_tab.warnings.sum()), '| a large train-validation gap means the model memorises templates - the tree models show it most clearly.')
cfg = {'vectorizers': 'word TF-IDF (1-2) + char_wb TF-IDF (2-5), sublinear, lowercase, strip accents', 'seed': SEED, 'candidates': {k: str(v().named_steps['classifier'])[:110] for k, v in CANDS.items()}, 'split': 'family-grouped stratified 80/20 on the official training file', 'sklearn': __import__('sklearn').__version__}
(WF/'chatbot_training_config.json').write_text(json.dumps(cfg, indent=2)); print('reproducible configuration saved ->', (WF/'chatbot_training_config.json').relative_to(ROOT))
'''

# ------------------------------------------------------------------------------------------------ 17
NOTES[17] = "compares all candidates with paraphrase-family-grouped, stratified 5-fold cross-validation on identical folds (vectorizers refitted inside every fold), reporting mean and variation and checking over-fitting; the sealed test file is never used."
CODE[17] = r'''
cv = StratifiedGroupKFold(5, shuffle=True, random_state=SEED); folds = list(cv.split(TR.text, TR.intent, GROUPS)); rows = []; t0 = time.time()
for name, mk in CANDS.items():
    f1, ac, trn = [], [], []
    for a_, b_ in folds:
        m = mk().fit(TR.text.iloc[a_], TR.intent.iloc[a_]); p = m.predict(TR.text.iloc[b_]); f1.append(M.f1_score(TR.intent.iloc[b_], p, average='macro')); ac.append(M.accuracy_score(TR.intent.iloc[b_], p)); trn.append(M.accuracy_score(TR.intent.iloc[a_], m.predict(TR.text.iloc[a_])))
    rows.append({'model': name, 'macro_F1 mean': round(np.mean(f1), 3), 'macro_F1 std': round(np.std(f1), 3), 'accuracy mean': round(np.mean(ac), 3), 'worst fold macro_F1': round(min(f1), 3), 'train accuracy': round(np.mean(trn), 3), 'overfit gap': round(np.mean(trn) - np.mean(ac), 3)})
cvt = pd.DataFrame(rows).sort_values('macro_F1 mean', ascending=False).reset_index(drop=True); display(cvt); cvt.to_csv(WF/'chatbot_grouped_cv.csv', index=False); RESULTS['cv'] = cvt.set_index('model')[['macro_F1 mean', 'macro_F1 std', 'accuracy mean']].to_dict('index')
print(f'5 grouped folds x {len(CANDS)} models in {time.time() - t0:.0f}s | identical folds for every model | TF-IDF refitted inside each fold | official test file never touched')
print('Stability: the spread (std, worst fold) is wide because folds hold out whole paraphrase families; differences under ~0.03 are noise. Overfitting: train accuracy near 1.0 for every model - the gap column is the honest signal.')
'''

# ------------------------------------------------------------------------------------------------ 18
NOTES[18] = "checks imbalance honestly: the data are perfectly balanced, so it tests whether class weights matter, reports per-intent recall instead of accuracy alone, PR-AUC and calibration effects, and explains why oversampling/SMOTE are not applied."
CODE[18] = r'''
vc = TR.intent.value_counts(); print('class counts: min', int(vc.min()), 'max', int(vc.max()), '| imbalance ratio', round(vc.max() / vc.min(), 2), '| the TEST file is balanced as well (4 per intent)')
print('Does imbalance affect the business goal? Not in the data - but in production some intents (greet, leave_balance) will be asked far more often than others (attrition_risk_query). That is a prior-shift issue handled by monitoring (stage 34), not by resampling.')
rows = []
for label, cw in [('class_weight=None', None), ('class_weight=balanced', 'balanced')]:
    f1, mn, ece_l, pr = [], [], [], []
    for a_, b_ in folds:
        m = make_model(lr(cw=cw)).fit(TR.text.iloc[a_], TR.intent.iloc[a_]); P_ = m.predict_proba(TR.text.iloc[b_]); p = m.classes_[P_.argmax(1)]; y_ = TR.intent.iloc[b_].to_numpy(); f1.append(M.f1_score(y_, p, average='macro'))
        mn.append(pd.Series(y_ == p).groupby(y_).mean().min()); conf = P_.max(1); ok = (p == y_); ece_l.append(sum(abs(ok[(conf >= lo) & (conf < hi)].mean() - conf[(conf >= lo) & (conf < hi)].mean()) * ((conf >= lo) & (conf < hi)).mean() for lo, hi in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0.1, 1.01, 10)) if ((conf >= lo) & (conf < hi)).any()))
        Y = pd.get_dummies(pd.Categorical(y_, categories=m.classes_)).to_numpy(); pr.append(M.average_precision_score(Y, P_, average='macro'))
    rows.append({'setting': label, 'macro_F1': round(np.mean(f1), 3), 'worst intent recall': round(np.mean(mn), 3), 'PR-AUC (macro)': round(np.mean(pr), 3), 'ECE (calibration error)': round(np.mean(ece_l), 3)})
imb = pd.DataFrame(rows); display(imb); print('Accuracy alone would hide a collapsed intent; "worst intent recall" and macro metrics expose it.')
print('Random over/under-sampling and SMOTE: NOT applied - classes are already equal; SMOTE on sparse text n-grams produces meaningless blends of words. If an imbalanced real log is added later, oversample inside training folds only.')
print('Decision from validation results (not assumptions):', 'keep class_weight=balanced' if imb.iloc[1]['macro_F1'] >= imb.iloc[0]['macro_F1'] - 0.005 else 'class_weight=None is better', '(differences are small because the data are balanced).')
RESULTS['imbalance'] = imb.set_index('setting').to_dict('index')
'''

# ------------------------------------------------------------------------------------------------ 19
NOTES[19] = "tunes the two most important hyper-parameters of the chosen model (regularisation C and the maximum character n-gram) with grouped cross-validation, optimising the declared primary metric, comparing with the untuned production recipe and checking consistency across folds."
CODE[19] = r'''
Cs, ngs = [0.5, 1, 2, 5, 10, 20], [4, 5]; t0 = time.time(); res = {(C, ng): [] for C in Cs for ng in ngs}
for a_, b_ in folds:
    for ng in ngs:
        fe = feats('word+char', ng).fit(TR.text.iloc[a_]); Xa, Xb = fe.transform(TR.text.iloc[a_]), fe.transform(TR.text.iloc[b_])
        for C in Cs: res[(C, ng)].append(M.f1_score(TR.intent.iloc[b_], lr(C=C).fit(Xa, TR.intent.iloc[a_]).predict(Xb), average='macro'))
tune = pd.DataFrame([{'C': C, 'max char n-gram': ng, 'macro_F1 mean': round(np.mean(v), 4), 'std': round(np.std(v), 4), 'by fold': [round(x, 3) for x in v]} for (C, ng), v in res.items()]).sort_values('macro_F1 mean', ascending=False).reset_index(drop=True); display(tune.head(8))
prod = tune[(tune.C == 2) & (tune['max char n-gram'] == 5)].iloc[0]; best = tune.iloc[0]; wins = sum(b > p for b, p in zip(best['by fold'], prod['by fold']))
print(f'search: {len(Cs)} C values x {len(ngs)} n-gram settings, {len(folds)} grouped folds, {time.time() - t0:.0f}s (vectorizer fitted once per fold and setting, so nothing leaks)')
print(f"best macro-F1 {best['macro_F1 mean']} (C={best.C}, char n-gram max={best['max char n-gram']}) vs untuned production recipe {prod['macro_F1 mean']} (C=2, 5) -> gain {best['macro_F1 mean'] - prod['macro_F1 mean']:+.4f}; better in {wins} of {len(folds)} folds")
CONSISTENT = bool(best['macro_F1 mean'] - prod['macro_F1 mean'] >= 0.01 and wins >= 4)
SELECTED = {'C': float(best.C), 'ngmax': int(best['max char n-gram'])} if CONSISTENT else {'C': 2.0, 'ngmax': 5}
print('Guard against over-tuning: only 12 settings, one declared metric (macro-F1), and the gain must be >= 0.01 AND consistent in >= 4 of 5 folds to be adopted.')
print('SELECTED configuration for stages 20-27:', SELECTED, '(' + ('tuned setting adopted' if CONSISTENT else 'gain not consistent -> the production recipe C=2, 5-grams stays') + ')')
RESULTS['tuning'] = {'best': {'C': float(best.C), 'ngmax': int(best['max char n-gram']), 'macro_F1': float(best['macro_F1 mean'])}, 'production_macro_F1': float(prod['macro_F1 mean']), 'selected': SELECTED, 'adopted_tuned': CONSISTENT}
'''
