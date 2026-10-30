from .common import NB, SETUP, checks_cell

FRESH = {
 'apply_leave_howto': ["how do I put in for a leave", "steps to raise a leave application"],
 'attendance_summary': ["give me my attendance numbers for last month", "how many days did I come to office in September"],
 'attrition_risk_query': ["which employees might resign in Finance", "show people with high attrition risk in Support"],
 'chitchat': ["who are you", "what's your name"],
 'goodbye': ["talk to you later", "that's all, goodbye"],
 'greet': ["good morning", "hello there"],
 'headcount_forecast': ["what will our headcount look like next quarter", "predict number of employees for Operations"],
 'help': ["what kind of questions can you answer", "how can you assist me"],
 'holiday_list': ["which dates are public holidays", "when is the next company holiday"],
 'hr_policy_leave': ["how many paid leaves do we get in a year", "what is the leave rule for carry forward"],
 'insurance_benefits': ["does the health insurance cover my parents", "what is the medical cover amount"],
 'leave_balance': ["how many leaves are left for me", "check my remaining casual leave"],
 'leave_request_status': ["was my leave approved", "status of the leave I applied"],
 'my_department': ["which department do I belong to", "tell me my dept"],
 'my_manager': ["who is my reporting manager", "name of my boss"],
 'notice_period_policy': ["how long is the notice period if I resign", "what is the resignation notice duration"],
 'out_of_scope': ["recommend a good movie", "what is the capital of France"],
 'payroll_flags': ["show suspicious payslips this month", "any payroll errors flagged"],
 'payslip_date': ["on which date is salary credited", "when will I receive my pay this month"],
 'payslip_status': ["download my payslip for August", "is my September payslip available"],
 'pf_info': ["how much PF is deducted from my salary", "what is the employer contribution to provident fund"],
 'probation_policy': ["how long is the probation period", "when does probation end for new joiners"],
 'reimbursement_policy': ["how do I claim travel expenses", "what is the process to get reimbursed"],
 'resume_score_request': ["rank this candidate resume for Data Engineer", "evaluate this CV against the job description"],
 'salary_band_query': ["what is the salary range for a Data Scientist", "pay band for senior accountant"],
 'salary_of_other': ["how much does Anita Verma earn", "show me Rahul Gupta's payslip"],
 'tax_declaration': ["where do I submit my investment proofs", "how is income tax calculated on my pay"],
 'team_members': ["who all are in my team", "list the people reporting to me"],
 'thanks': ["thank you so much", "appreciate the help"],
 'unusual_attendance': ["were there any odd entries in my attendance", "did the system flag anything in my attendance"],
 'wfh_policy': ["how many days can I work from home", "is remote work allowed"],
 'who_to_contact': ["who should I email for HR questions", "contact details of the payroll team"],
 'working_hours': ["what time does the office open", "what are the standard working hours"],
}


def build():
    n = NB()
    n.md("""# Task 9 (Day 9): Teach the machine to understand HR questions
**Objective.** Train an intent classifier on the chatbot messages and answer FAQ questions with a safe fallback.
**Why it matters.** Employees ask the same HR questions every day. A good assistant saves hours and gives consistent answers.
**You will learn.** Turning text into numbers (TF-IDF), intent classification, confusion matrices, confidence thresholds and fallbacks, cleaning messy text.
**Rule.** The official 132-message test set is used to *report*, never to choose settings. Settings are chosen by 5-fold cross-validation on the 858 training messages. A wrong confident answer is worse than "I am not sure".""")
    n.code(SETUP)
    n.code("""from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report""")

    n.stage(1, "Data Collection", "Training messages, the held-out test set and the FAQ answers.")
    n.code("""train = H.load('chatbot_nlu_training'); test = H.load('chatbot_test_set'); faq = H.load('hr_faq'); resp = H.load('chatbot_responses')
print('training:', train.shape, '| intents:', train.intent.nunique(), '| test:', test.shape, '| FAQ rows:', len(faq), '| reply templates:', len(resp))
assert len(train) == 858 and train.intent.nunique() == 33 and len(test) == 132""")

    n.stage(2, "Data Understanding", "Read 20 messages, including the messy ones, and list the 33 intents.")
    n.code("""train.sample(20, random_state=C.SEED)[['text', 'intent']].reset_index(drop=True)""")
    n.code("""train.groupby('intent').size().rename('messages').to_frame().T""")

    n.stage(3, "Data Quality Audit", "Duplicates, empties, class balance, and the most important check: no test message may appear in training.")
    n.code("""audit = {'empty/blank messages': int(train.text.fillna('').str.strip().eq('').sum()), 'duplicate training texts': int(train.text.duplicated().sum()),
         'test texts also present in training (leakage)': int(test.text.isin(train.text).sum()), 'messages per intent (min / max)': f'{train.intent.value_counts().min()} / {train.intent.value_counts().max()}',
         'test messages per intent (min / max)': f'{test.expected_intent.value_counts().min()} / {test.expected_intent.value_counts().max()}',
         'training intents missing from test': sorted(set(train.intent) - set(test.expected_intent))}
pd.Series(audit, name='result').to_frame()""")

    n.stage(4, "Exploratory Data Analysis (EDA)", "How long are messages, and where is the language messy?")
    n.code("""train['n_words'] = train.text.str.split().str.len()
fig, ax = plt.subplots(1, 2, figsize=(14, 3.8))
ax[0].hist(train.n_words, bins=range(1, 17)); ax[0].set_title('words per training message')
train.groupby('intent').n_words.mean().sort_values().plot.barh(ax=ax[1], title='mean words per intent'); fig.tight_layout();""")
    n.md("Short social messages (`greet`, `thanks`, `goodbye`, `help`, `chitchat`) have only one to three words, so they carry little signal and are the most likely to be confused with each other. Many messages also contain typos (`levae`, `paylsip`, `ruels`) and filler (`pls`, `urgently`, `quick question:`).")

    n.stage(5, "Data Cleaning", "Light cleaning only: lowercase, collapse spaces, strip accents. Typos are NOT fixed by a dictionary; character n-grams absorb them.")
    n.code("""def clean(s): return re.sub(r'\\s+', ' ', str(s)).strip()
demo = pd.DataFrame({'raw': ['  how   to apply casual  levae ', 'is my June paylsip ready pls', 'pf withdrawal ruels'], })
demo['cleaned'] = demo.raw.map(clean); demo""")
    n.code("""# Why no dictionary: 'levae' and 'leave' share most character n-grams, so the character TF-IDF sees them as near-identical
vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5)).fit(train.text.map(clean))
a, b, c = vec.transform(['apply casual levae', 'apply casual leave', 'what is the weather'])
print('cosine(levae vs leave):', round(float((a @ b.T).toarray()[0, 0]), 3), '| cosine(levae vs unrelated):', round(float((a @ c.T).toarray()[0, 0]), 3))""")

    n.stage(6, "Feature Engineering", "TF-IDF: words and character pieces turned into weighted numbers.")
    n.md("**TF-IDF in two lines.** *Term frequency* counts how often a word appears in a message; *inverse document frequency* down-weights words that appear in almost every message (like \"my\" or \"please\"), so rare, telling words (\"payslip\", \"probation\") get the biggest numbers. Our features combine word 1-2-grams with character 2-5-grams, which helps with short messages and typos.")
    n.code("""prod = CB.train_intent_model(CSV / 'chatbot_nlu_training.csv')
feats = prod.named_steps['features']; lr = prod.named_steps['classifier']
names = np.array([f'{k.split("_")[0]}:{t}' for k, v in feats.transformer_list for t in v.get_feature_names_out()])
print('feature count:', len(names), '(word + character TF-IDF)')
pd.DataFrame({i: names[np.argsort(-lr.coef_[k])[:6]] for k, i in enumerate(lr.classes_) if i in ['payslip_status', 'probation_policy', 'leave_balance', 'thanks']})""")

    n.stage(7, "Train-Test Split", "The official test set stays sealed. Every choice below uses stratified 5-fold cross-validation on the training messages only.")
    n.code("""skf = StratifiedKFold(5, shuffle=True, random_state=C.SEED); Xtr, ytr = train.text.map(clean).tolist(), train.intent.astype(str).tolist()
Xte, yte = test.text.map(clean).tolist(), test.expected_intent.astype(str).tolist()
print('CV folds:', skf.get_n_splits(), '| train messages:', len(Xtr), '| sealed test messages:', len(Xte))""")

    n.stage(8, "Model Building", "TF-IDF + logistic regression, compared with simpler variants by cross-validation (training data only).")
    n.code("""def pipe(feature, clf): return Pipeline([('features', feature), ('classifier', clf)])
cands = {'word TF-IDF + LR': pipe(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), LogisticRegression(max_iter=3000, class_weight='balanced', C=2.0)),
         'char TF-IDF + LR': pipe(TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), sublinear_tf=True), LogisticRegression(max_iter=3000, class_weight='balanced', C=2.0)),
         'word TF-IDF + ComplementNB': pipe(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True), ComplementNB(alpha=0.3)),
         'word+char TF-IDF + LR (production)': clone(prod)}
rows, oof = [], {}
for name, m in cands.items():
    pred = cross_val_predict(m, Xtr, ytr, cv=skf)
    rows.append({'model': name, 'cv accuracy': round(accuracy_score(ytr, pred), 4), 'cv macro-F1': round(f1_score(ytr, pred, average='macro'), 4)})
cv = pd.DataFrame(rows).sort_values('cv accuracy', ascending=False).reset_index(drop=True); cv""")
    n.md("The production model is chosen from this **cross-validation table**, not from the test set. It also exposes `predict_proba`, which the safe fallback needs (a plain linear SVM does not).\n\n**Warning: cross-validation overstates accuracy here.** CV says about 99% but the sealed test set says 81%. The 858 training messages are noisy variants of a small number of templates (26 per intent, with typos and filler added), so every fold contains near-copies of the other folds. CV is still fine for *ranking* candidates, but the held-out and fresh-message numbers are the honest estimate of new wording.")

    n.stage(9, "Model Evaluation", "The sealed test set is scored ONCE with the production model: accuracy, macro-F1, per-intent results and the confusion matrix.")
    n.code("""pred_te = prod.predict(Xte); acc = accuracy_score(yte, pred_te)
print(f'HELD-OUT accuracy {acc:.4f} ({sum(np.array(pred_te) == np.array(yte))}/{len(yte)})  | macro-F1 {f1_score(yte, pred_te, average="macro"):.4f}')
rep = pd.DataFrame(classification_report(yte, pred_te, output_dict=True, zero_division=0)).T.round(3); rep.iloc[:-3].sort_values('f1-score').head(8)""")
    n.code("""labels = sorted(set(yte) | set(pred_te)); cm = confusion_matrix(yte, pred_te, labels=labels)
fig, ax = plt.subplots(figsize=(13, 11)); ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, xticks_rotation=90, colorbar=False); ax.set_title('Intent confusion matrix (held-out test)'); fig.tight_layout();""")
    n.code("""summary = pd.read_csv(OUT / 'chatbot_metric_summary.csv').set_index('metric').value
print('matches the recorded report value:', abs(acc - float(summary['heldout_accuracy'])) < 1e-9, '| honest status: below 90%, so do not claim 90%')""")

    n.stage(10, "Hyperparameter Tuning", "The confidence threshold for the safe fallback, chosen on out-of-fold training predictions (never on the test set).")
    n.code("""proba = cross_val_predict(clone(prod), Xtr, ytr, cv=skf, method='predict_proba'); classes = np.array(sorted(set(ytr)))
top, pred_cv = proba.max(axis=1), classes[proba.argmax(axis=1)]; rows = []
for t in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60]:
    ans = top >= t; rows.append({'threshold': t, 'answered %': round(100 * ans.mean(), 1), 'accuracy when answered %': round(100 * (pred_cv[ans] == np.array(ytr)[ans]).mean(), 1), 'sent to fallback %': round(100 * (~ans).mean(), 1)})
sweep = pd.DataFrame(rows); sweep""")
    n.md(f"The bot uses **0.30** (`HRChatbot(confidence_threshold=0.30)`): it answers most messages and is almost always right when it answers; the rest get the polite fallback with a way to reach HR. A higher threshold is safer but sends more genuine questions to fallback. Abstentions are reported separately and never counted as correct. Because these CV numbers are optimistic (see the warning in Stage 8), expect a larger share of real messages to fall back: on the 66 fresh messages in Stage 11 about a quarter fall below 0.30.")

    n.stage(11, "Model Interpretation", "Error analysis on the held-out predictions (analysis only, no tuning), then a small independent check.")
    n.code("""errs = pd.DataFrame({'text': Xte, 'expected': yte, 'predicted': pred_te}); errs = errs[errs.expected != errs.predicted]
pairs = errs.groupby(['expected', 'predicted']).size().rename('count').reset_index().sort_values('count', ascending=False)
print(len(errs), 'errors'); pairs.head(8)""")
    n.think("which two intents get confused most? What would you change in the training data?",
            """errs.head(12)""",
            "The most repeated confusion is `help` predicted as `chitchat`, and short social messages (`greet`, `thanks`) landing in neighbouring intents: they are one or two words, so the model has almost nothing to go on. The fix is data, not a bigger model: add more *short, slangy* greetings, thanks and help requests (\"yo\", \"cheers\", \"what can I ask you\") and a few more `who_to_contact` / `payslip_date` phrasings. New examples must be written from policy knowledge or real chat logs, never copied from the test set, and then re-checked on a NEW test set. The bot also has a closed list of social phrases as an auditable safety net.")
    n.md("**Independent mini-check.** 66 fresh messages (2 per intent) hand-written for this notebook. They are not from any official file, they are small, and they only estimate how the model behaves on new wording; they do not replace a proper independently labelled test set.")
    n.code("""fresh = {FRESH}
fx = [(m, k) for k, v in fresh.items() for m in v]; Xf, yf = [m for m, _ in fx], [k for _, k in fx]
pf = prod.predict(Xf); pp = prod.predict_proba(Xf).max(axis=1)
print(f'fresh-message accuracy {accuracy_score(yf, pf):.3f} ({int((np.array(pf) == np.array(yf)).sum())}/{len(yf)}) | answered above 0.30 confidence: {100 * (pp >= 0.30).mean():.0f}%')
pd.DataFrame({'text': Xf, 'expected': yf, 'predicted': pf, 'confidence': pp.round(2)}).query('expected != predicted').reset_index(drop=True)""".replace("{FRESH}", repr(FRESH)))

    n.stage(12, "Final Conclusions and Recommendations")
    n.code("""bot = CB.HRChatbot(prod, CSV / 'chatbot_responses.csv', CSV / 'hr_faq.csv', users_csv=MOCK / 'demo_users.csv')
demo_msgs = ['what is the notice period', 'how is LOP calculated', 'asdkjh qwerty zxcv', 'tell me a joke']
convs = [(m, bot.reply(m, 'nb9', 'EMPLOYEE', 'EMP-0002')) for m in demo_msgs]
for m, r in convs: print(f"YOU: {m}\\nBOT: {r['reply']}\\n     [intent={r['intent']} confidence={r['confidence']:.2f} service_called={r['service_called']}]\\n")
print(f'''MODEL: word+char TF-IDF + logistic regression. Held-out accuracy {acc:.1%} on 132 messages (33 intents); fresh-message check {accuracy_score(yf, pf):.1%} on 66.
FALLBACK: threshold 0.30 chosen by cross-validation; low-confidence messages get a polite reply and never a data call.
HONEST STATUS: below 90%. Next step: more short/slangy training examples and an independently labelled test set.''')""")
    n.checklist(checks_cell("""checks = {
    "'Test accuracy reported and errors inspected'": len(errs) == len(yte) - sum(np.array(pred_te) == np.array(yte)) and len(errs) > 0,
    "'Low-confidence messages get a polite fallback'": any(r['reply'] == CB.SAFE_FALLBACK for _, r in convs),
    "'The /v1/chat endpoint works in your API (HTTP needs FastAPI: run pytest -q tests_phase2/test_api.py)'": 'MANUAL',
    "'You can explain TF-IDF in two lines (Stage 6)'": True,
}"""))
    n.footer("Accuracy and confusion matrix (Stage 9, `reports/phase2/chatbot_confusion_matrix.png`) and 3 example conversations (`reports/phase2/chat_transcripts.md`).",
             "[scikit-learn](https://github.com/scikit-learn/scikit-learn) | [spaCy](https://github.com/explosion/spaCy) | [Rasa](https://github.com/RasaHQ/rasa)",
             "a wrong confident answer is worse than 'I am not sure'. Teams tune the fallback with care.",
             "the bot knows the topic but not the details. Task 10 adds names, dates, memory and permissions.",
             "I built a language model service that knows when it is unsure.")
    return n.save("P2_T09_hr_intent_chatbot.ipynb")
