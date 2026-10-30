"""Intent classification and policy-aware demo HR chatbot."""
from __future__ import annotations
import re
import os
try:  # httpx is only needed to call the mock HR service; classification and tests work without it
    import httpx
except ImportError:  # pragma: no cover
    httpx = None
import datetime as _dt
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SOCIAL_PHRASES = {
    'greet': {'hi','hello','hey','hiya','yo','namaste','good morning','good afternoon','good evening'},
    'thanks': {'thanks','thank you','thx','thanks a lot','many thanks','cheers','thank you so much'},
    'goodbye': {'bye','goodbye','ok bye','see you','see you later','bye bye'},
}

MAX_SESSIONS = 1000                 # oldest conversations are dropped so memory cannot grow without limit
FAQ_MIN_SIMILARITY = 0.20           # below this the FAQ answer is not used (a wrong confident answer is worse than "not sure")
FAQ_MIN_MARGIN = 0.03               # best FAQ must beat the runner-up by this much
FAQ_RESCUE_INTENTS = {'out_of_scope', 'chitchat', 'help', 'who_to_contact', 'payslip_date', 'greet', 'thanks', 'goodbye'}
FAQ_INTENTS = {'hr_policy_leave','working_hours','payslip_date','apply_leave_howto','reimbursement_policy','who_to_contact','wfh_policy','notice_period_policy','insurance_benefits','pf_info','probation_policy','tax_declaration'}
# Intents whose back-end models belong to later phases. The bot says so instead of inventing numbers.
NOT_CONNECTED = {'salary_band_query': 'salary band guidance', 'attrition_risk_query': 'attrition risk', 'headcount_forecast': 'headcount forecasting', 'resume_score_request': 'resume scoring'}
# Small, auditable synonym table so "wfh", "lop", "hike" and similar words find the right FAQ.
FAQ_ALIASES = [(r'\bloss of pay\b|\blop\b','lop'),(r'\bunpaid leave\b|\bleave without pay\b','unpaidleave'),(r'\bwork(ing)? from home\b|\bwfh\b|\bremote(ly)?\b|\bhybrid\b','wfh'),
 (r'\bprovident fund\b|\bpf\b','pf'),(r'\bprofessional tax\b|\bpt\b','proftax'),(r'\bincome tax\b|\btds\b','tds'),(r'\bform ?16\b','form16'),
 (r'\bincrements?\b|\bhikes?\b|\braise\b|\brevis(ed|ion|ions)\b|\bappraisal\b','revision'),(r'\bdress ?code\b|\bwear\b|\battire\b|\bclothes\b','dresscode'),
 (r'\bbuy ?out\b','buyout'),(r'\bbonus\b|\bincentive\b','bonus'),(r'\bsaturdays?\b','saturday'),(r'\bcredited\b|\bpay ?day\b|\bget paid\b','paid'),
 (r'\bresign(ing|ation)?\b','notice'),(r'\bsick days?\b','sick leave'),(r'\bcasual days?\b','casual leave'),(r'\b(medical|health) (insurance|cover)\b','insurance'),
 (r'\bchild(ren)?\b|\bspouse\b|\bparents?\b','dependent'),(r'\bexpenses?\b|\breimburse\w*\b|\bclaims?\b','expense'),(r'\bbank (account|details)\b','bankdetails')]
FAQ_GENERIC = {'policy', 'period', 'pay', 'work', 'works', 'much', 'many', 'long', 'get', 'have', 'does'}
FAQ_STOP = list('a an the do does is are i we you my me to of for in on it what how when who which can will calculated calculate decided computed'.split())
NAME_STOP = {'january','february','march','april','may','june','july','august','september','october','november','december','jan','feb','mar','apr','jun','jul','aug','sep','sept','oct','nov','dec',
             'last','this','next','previous','current','month','year','the','my','me','hr','payroll','salary','pay','payslip','details'}
LEAVE_WORDS = {'casual': ('cl_balance','casual leave (CL)'), 'sick': ('sl_balance','sick leave (SL)'), 'earned': ('el_balance','earned leave (EL)')}
SAFE_FALLBACK = "I’m not confident I understood. Please rephrase, or ask about leave, attendance, payslips or HR policies."

# ---- Task 10 privacy guard: anyone else's pay is refused, however it is phrased ---------------------------------
PAY_WORDS = r"\b(salary|salaries|pay|paid|payslips?|income|compensation|ctc|earns?|earning|earnings|wages?|package|take[- ]home|stipend|remuneration)\b"
BULK_WORDS = r"\b(everyone|everybody|all (the )?(employees|staff|people|colleagues|workers)|every employee|each employee|other (employees|people|colleagues)|employees['’]|whole (team|company)|entire (team|company))\b"
INJECTION_WORDS = r"ignore (all |the |any |your )?(previous |prior |above |earlier )?(instructions|rules|policy|policies)|\b(pretend|act as|you are now|developer mode|jailbreak|override)\b"
RELATION_WORDS = r"\b(manager|boss|supervisor|team ?lead|lead|colleague|co-?worker|teammate|team ?mate|friend|neighbou?r|him|his|her|hers|their|someone|somebody|anyone|another)\b"
OWN_CONTEXT = r"\b(approve|approval|policy|process|apply|appraisal|revision|hike|increment|calculated|structure|slab|raise|who|see|view|access|visible|know)\b"
FULL_MONTHS = ['january','february','august','september','october','november','december']
REFUSE_OTHER_PAY = 'I can’t disclose another employee’s salary or payslip. You can ask about your own payroll or general salary-band policy.'

def train_intent_model(training_csv):
    df = pd.read_csv(training_csv)
    if not {'text','intent'}.issubset(df.columns): raise ValueError('Training CSV must contain text and intent columns')
    # Word n-grams capture intent phrases; character n-grams add robustness to
    # spelling variation, short messages, and morphological variants. Both
    # vectorizers are fit inside the pipeline, preventing vocabulary leakage.
    model = Pipeline([
        ('features', FeatureUnion([
            ('word_tfidf', TfidfVectorizer(
                analyzer='word', ngram_range=(1, 2), min_df=1,
                strip_accents='unicode', sublinear_tf=True,
                lowercase=True, token_pattern=r'(?u)\b\w+\b',
            )),
            ('char_tfidf', TfidfVectorizer(
                analyzer='char_wb', ngram_range=(2, 5), min_df=1,
                sublinear_tf=True, lowercase=True,
            )),
        ])),
        ('classifier', LogisticRegression(
            max_iter=3000, class_weight='balanced', C=2.0,
            solver='lbfgs', random_state=42,
        )),
    ])
    clean_text = df.text.fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    model.fit(clean_text, df.intent.astype(str))
    return model

def evaluate_intent_model(model, test_csv):
    df = pd.read_csv(test_csv)
    clean_text = df.text.fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    pred = model.predict(clean_text)
    labels = sorted(set(df.expected_intent.astype(str)) | set(pred.astype(str)))
    return {'accuracy': float(accuracy_score(df.expected_intent, pred)), 'report': classification_report(df.expected_intent, pred, output_dict=True, zero_division=0), 'confusion_matrix': confusion_matrix(df.expected_intent, pred, labels=labels), 'labels': labels, 'predictions': pred}

class _SimpleResponse:
    def __init__(self, status_code, body): self.status_code, self._body = status_code, body
    def json(self):
        import json as _json
        return _json.loads(self._body)


def _http_request(method, url, params=None, headers=None, timeout=2.0):
    """HTTP call via httpx when installed, otherwise the standard library, so the bot also works without httpx."""
    if httpx is not None:
        return httpx.request(method, url, params=params, headers=headers, timeout=timeout)
    import urllib.error, urllib.parse, urllib.request
    full = url + ('?' + urllib.parse.urlencode(params) if params else '')
    req = urllib.request.Request(full, method=method, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return _SimpleResponse(resp.status, resp.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        return _SimpleResponse(exc.code, exc.read().decode('utf-8', 'replace'))


def _allowed(role, allowed_roles):
    allowed = [x.strip().upper() for x in str(allowed_roles or 'ALL').split(',')]
    role = str(role or 'EMPLOYEE').upper()
    return 'ALL' in allowed or role in allowed

def _month_from_text(text, previous=None):
    t = text.lower()
    month_map = {'january':'January','february':'February','march':'March','april':'April','may':'May','june':'June','july':'July','august':'August','september':'September','october':'October','november':'November','december':'December','jan':'January','feb':'February','mar':'March','apr':'April','jun':'June','jul':'July','aug':'August','sep':'September','sept':'September','oct':'October','nov':'November','dec':'December'}
    for key, val in month_map.items():
        if re.search(r'\b'+key+r'\b', t): return val
    if 'last month' in t: return 'last_month'
    if 'this month' in t: return 'this_month'
    import difflib                                   # typo tolerance: "septmber", "decmber" (long month names only)
    for token in re.findall(r'[a-z]{6,}', t):
        close = difflib.get_close_matches(token, FULL_MONTHS, n=1, cutoff=0.85)
        if close: return close[0].capitalize()
    return previous

def _name_from_text(text):
    """A capitalised person name next to salary/pay words.  Months, 'Last', 'My' and similar are not names."""
    pay = r'(?i:salary|pay|payslip|income|compensation|ctc|package|earnings?|wages?)'
    m = re.search(r'\b(?i:salary|pay|payslip|details|ctc|package)\s+(?i:of|for)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})', text)
    if not m: m = re.search(r'\b([A-Z][a-z]+\s+[A-Z][a-z]+)(?:[\'’]s)?\s+' + pay + r'\b', text)
    if not m: return None
    words = m.group(1).split()
    if any(w.lower() in NAME_STOP for w in words): return None
    return m.group(1)


def _stem_tokens(text):
    return {w[:-1] if len(w) > 3 and w.endswith('s') and not w.endswith('ss') else w for w in re.findall(r'[a-z0-9]+', text)}


def _faq_norm(text):
    t = str(text).lower()
    for pattern, repl in FAQ_ALIASES: t = re.sub(pattern, repl, t)
    return t


DEMO_TODAY = _dt.date(2026, 10, 27)          # mock HR data ends September 2026, so "last month" means 2026-09


class HRChatbot:
    """Policy-aware demo HR assistant: intent model + auditable rules + mock HR service + permissions.

    Order of decisions in ``reply``: identity and role (from the directory, never from the client) -> refuse other
    people's pay -> same-name handling -> intent model and small rules -> follow-ups and slots -> FAQ lookup ->
    confidence fallback -> role check -> data call -> plain-language answer.
    """

    def __init__(self, model, responses_csv, faq_csv, confidence_threshold=0.30, service_url=None, users_csv=None,
                 employees_csv=None, payroll_flags_provider=None, attendance_flags_provider=None, today=None):
        self.model = model
        self.responses = pd.read_csv(responses_csv).fillna('')
        self.response_map = self.responses.set_index('intent').to_dict('index')
        self.faq = pd.read_csv(faq_csv).fillna('')
        self.threshold = confidence_threshold
        self.service_url = (service_url or os.getenv('CORTEXHR_MOCK_SERVICE_URL', 'http://127.0.0.1:9000')).rstrip('/')
        self.users = pd.read_csv(users_csv).fillna('') if users_csv and Path(users_csv).exists() else None
        self.sessions = OrderedDict()
        if employees_csv is None and users_csv is not None:
            employees_csv = Path(users_csv).with_name('employees.csv')
        self.employees = pd.read_csv(employees_csv).fillna('') if employees_csv and Path(employees_csv).exists() else None
        self.payroll_flags_provider = payroll_flags_provider          # callable(month 'YYYY-MM') -> dict, or None
        self.attendance_flags_provider = attendance_flags_provider    # callable(attendance summary dict) -> dict, or None
        self.today = today or DEMO_TODAY
        from sklearn.pipeline import FeatureUnion as _FU
        self._faq_vec = _FU([('w', TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=FAQ_STOP)),
                             ('c', TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True))])
        self._faq_matrix = self._faq_vec.fit_transform([_faq_norm(q) for q in self.faq.question])
        # Words that tell FAQs apart (appear in fewer than 3 questions). A match must share one, so the generic
        # question "what is the leave policy" is not answered with the maternity-leave FAQ.
        tokens = [_stem_tokens(_faq_norm(q)) - set(FAQ_STOP) for q in self.faq.question]
        counts = {}
        for t in tokens:
            for w in t: counts[w] = counts.get(w, 0) + 1
        self._faq_distinct = [{w for w in t if counts[w] < 3 and w not in FAQ_GENERIC} for t in tokens]

    # ------------------------------------------------------------------ helpers

    def asks_for_other_pay(self, text, demo_user):
        """True when the message wants someone else's (or everyone's) pay, or tries to talk the bot out of its rules."""
        low = text.lower()
        if not re.search(PAY_WORDS, low): return False
        if re.search(INJECTION_WORDS, low) or re.search(BULK_WORDS, low): return True
        own = str(demo_user).upper()
        if {x.upper() for x in re.findall(r'\bEMP-\d{3,6}\b', text, re.I)} - {own}: return True
        if _name_from_text(text): return True
        people = self.find_people(text)
        if len(people) and not (people.employee_id.astype(str) == str(demo_user)).all(): return True
        return bool(re.search(RELATION_WORDS, low) and not re.search(OWN_CONTEXT, low))

    def reset(self, session_id): self.sessions.pop(session_id, None)

    def _session(self, session_id):
        if session_id in self.sessions: self.sessions.move_to_end(session_id)
        else:
            self.sessions[session_id] = {'slots': {}, 'last_intent': None}
            while len(self.sessions) > MAX_SESSIONS: self.sessions.popitem(last=False)
        return self.sessions[session_id]

    def find_people(self, text):
        """People from the employee directory whose full name appears in the text (exact, case-insensitive)."""
        if self.employees is None: return pd.DataFrame()
        low = text.lower()
        names = [n for n in self.employees.full_name.unique() if n.lower() in low]
        if not names: return pd.DataFrame()
        return self.employees[self.employees.full_name == max(names, key=len)]

    def faq_match(self, text):
        """Best FAQ row for a message: (row_index, similarity 0..1, margin over the runner-up)."""
        sims = (self._faq_matrix @ self._faq_vec.transform([_faq_norm(text)]).T).toarray().ravel() / 2.0
        order = np.argsort(-sims)
        return int(order[0]), float(sims[order[0]]), float(sims[order[0]] - sims[order[1]])

    def _shares_distinct_word(self, text, faq_index):
        import difflib
        words = _stem_tokens(_faq_norm(text)) - set(FAQ_STOP)
        return any(difflib.get_close_matches(d, words, n=1, cutoff=0.8) for d in self._faq_distinct[faq_index])

    def _month_param(self, month):
        if month == 'last_month':
            first = self.today.replace(day=1) - _dt.timedelta(days=1); return f'{first.year}-{first.month:02d}'
        if month == 'this_month': return f'{self.today.year}-{self.today.month:02d}'
        if re.match(r'^\d{4}-\d{2}$', str(month)): return str(month)
        names = ['January','February','March','April','May','June','July','August','September','October','November','December']
        return f'{self.today.year}-{names.index(month) + 1:02d}' if month in names else None

    def _out(self, intent, reply, state, confidence, called=False):
        return {'intent': intent, 'reply': reply, 'slots': state['slots'].copy(), 'confidence': float(confidence), 'service_called': called}

    def _directory_name(self, employee_id):
        if self.employees is None or not employee_id: return None
        row = self.employees[self.employees.employee_id == employee_id]
        return str(row.iloc[0].full_name) if len(row) else None

    # ------------------------------------------------------------------ plain-language answers from service data
    def _describe(self, intent, payload, month_param, state, text):
        if intent == 'leave_balance':
            wanted = [k for k in LEAVE_WORDS if k in state['slots'].get('leave_type', [])] or list(LEAVE_WORDS)
            parts = [f"{payload.get(LEAVE_WORDS[k][0], 'n/a')} {LEAVE_WORDS[k][1]}" for k in wanted]
            return 'You have ' + ', '.join(parts[:-1]) + (' and ' if len(parts) > 1 else '') + parts[-1] + ' day(s) left.'
        if intent == 'payslip_status':
            return (f"Your payslip for {month_param} is available. Gross pay Rs {float(payload.get('gross_pay', 0)):,.0f}, deductions "
                    f"Rs {float(payload.get('total_deductions', 0)):,.0f}, net pay Rs {float(payload.get('net_pay', 0)):,.0f}.")
        if intent == 'attendance_summary':
            return (f"In {month_param} you were present {payload.get('present_days')} of {payload.get('scheduled_days')} scheduled days "
                    f"({float(payload.get('attendance_rate', 0)):.0%}), with {payload.get('late_arrivals', 0)} late arrival(s).")
        if intent == 'unusual_attendance':
            provider = self.attendance_flags_provider
            if provider is None:
                from .service import attendance_flag_from_summary as provider
            info = provider(payload)
            if info.get('flagged'):
                return f"Your attendance for {month_param} was flagged for review: {info['reason_text']} Flags are leads to check, not conclusions."
            return f"Nothing unusual was flagged in your attendance for {month_param}. This is a screening check, not a guarantee."
        if intent == 'my_manager':
            mid = payload.get('manager_id')
            if not mid or str(mid) == 'nan': return 'No manager is recorded for you.'
            name = self._directory_name(str(mid))
            return f"Your manager is {name} ({mid})." if name else f"Your manager's employee id is {mid}."
        if intent == 'my_department': return f"Your department id is {payload.get('department_id', 'not available')}."
        if intent == 'team_members':
            return ('Your direct reports: ' + ', '.join(str(x.get('full_name', x.get('employee_id'))) for x in payload) + '.') if payload else 'You have no direct reports on record.'
        if intent == 'leave_request_status':
            if not payload: return 'You have no leave requests on record.'
            r = payload[0]
            return (f"Your latest leave request ({r.get('leave_type_code')}, {r.get('days')} day(s), {r.get('start_date')} to {r.get('end_date')}) "
                    f"is {str(r.get('status', 'unknown')).lower()}.")
        if intent == 'holiday_list':
            upcoming = sorted([h for h in payload if str(h.get('holiday_date')) >= self.today.isoformat()], key=lambda h: h['holiday_date'])
            if upcoming:
                h = upcoming[0]; return f"The next holiday is {h['holiday_name']} on {h['holiday_date']} ({h.get('weekday', '')})."
            last = sorted(payload, key=lambda h: h['holiday_date'])[-1] if payload else None
            return (f"No upcoming holiday is published after {self.today.isoformat()}. The last published one was {last['holiday_name']} on {last['holiday_date']}."
                    if last else 'No holiday calendar is available.')
        return SAFE_FALLBACK

    # ------------------------------------------------------------------ main entry point
    def reply(self, message, session_id='demo-session', user_role='EMPLOYEE', demo_user='EMP-0001'):
        text = str(message or '').strip()
        # Trust the role from the identity directory, never from a client-supplied role field.
        if self.users is not None:
            identity = self.users[self.users.employee_id.astype(str) == str(demo_user)]
            if identity.empty:
                return {'intent':'out_of_scope','reply':'Invalid demo identity. Choose a valid X-Demo-User from mock_data/demo_users.csv.','slots':{},'confidence':0.0,'service_called':False}
            user_role = str(identity.iloc[0].role).upper()
        user_role = str(user_role).upper()
        if not text: return {'intent':'out_of_scope','reply':'Please enter a message.','slots':{},'confidence':0.0,'service_called':False}
        state = self._session(session_id)
        lower = text.lower()
        # Strong policy override: never disclose another person's salary, regardless of model confidence or role.
        if self.asks_for_other_pay(text, demo_user):
            return self._out('salary_of_other', REFUSE_OTHER_PAY, state, 1.0)
        # Same-name design: a person is never guessed (see README). Employees get one neutral answer for any other
        # person; HR/ADMIN get a clarifying question listing only directory facts and the bot waits for an id.
        pending = state.get('pending_people')
        if pending:
            chosen = next((e for e in pending if e.lower() in lower), None)
            if chosen:
                state['slots']['employee_id'] = chosen; state.pop('pending_people', None)
                person = self.employees[self.employees.employee_id == chosen].iloc[0]
                return self._out(state.get('pending_intent', 'out_of_scope'), f"Thanks, I will use {chosen} ({person.full_name}, department {person.department_id}). The demo HR service only returns the signed-in user's own data, so I cannot fetch their records here.", state, 1.0)
            return self._out(state.get('pending_intent', 'out_of_scope'), 'I still need to know which person you mean. Please reply with one of these employee ids: ' + ', '.join(pending) + '.', state, 1.0)
        people = self.find_people(text)
        if len(people) and user_role not in {'HR', 'ADMIN'}:
            people = people[people.employee_id != str(demo_user)]          # asking about yourself is fine
        if len(people) and user_role not in {'HR', 'ADMIN'}:
            return self._out('out_of_scope', 'I can only help with your own information. For anyone else, please contact HR.', state, 1.0)
        if len(people) > 1:
            ids = people.employee_id.tolist()
            state['pending_people'] = ids; state['pending_intent'] = 'attendance_summary' if 'attendance' in lower else 'out_of_scope'
            options = '; '.join(f"{r.employee_id} ({r.department_id}, {r.work_location})" for r in people.itertuples())
            return self._out(state['pending_intent'], f"I found {len(people)} employees named {people.iloc[0].full_name}: {options}. Which one do you mean? Please reply with the employee id.", state, 1.0)

        probabilities = self.model.predict_proba([text])[0]
        classes = self.model.classes_
        idx = int(probabilities.argmax()); intent = str(classes[idx]); confidence = float(probabilities[idx])
        # Small, auditable phrase overrides for high-risk / well-defined task intents.
        if re.match(r'^(salary|pay|payslip)\s+(for|of)\s+(' + '|'.join(['last month','this month','january','february','march','april','may','june','july','august','september','october','november','december']) + r')\b', lower) and 'band' not in lower: intent, confidence = 'payslip_status', max(confidence, 0.9)
        elif 'salary band' in lower: intent, confidence = 'salary_band_query', max(confidence, 0.9)
        elif 'forecast headcount' in lower or 'headcount forecast' in lower: intent, confidence = 'headcount_forecast', max(confidence, 0.9)
        elif 'unusual payroll' in lower or 'payroll anomaly' in lower: intent, confidence = 'payroll_flags', max(confidence, 0.9)
        elif re.match(r"^(what is|what's|show|tell me|check|see|get|give me)\b.*\bmy (salary|pay|payslip|ctc)\b", lower): intent, confidence = 'payslip_status', max(confidence, 0.9)
        elif 'payslip' in lower and any(k in lower for k in ['my', 'ready', 'show', 'march', 'last month', 'download']): intent, confidence = 'payslip_status', max(confidence, 0.9)
        elif re.search(r'\bunusual\b.*\b(days?|attendance)\b|\b(days?|attendance)\b.*\bunusual\b', lower) and 'payroll' not in lower: intent, confidence = 'unusual_attendance', max(confidence, 0.9)
        elif 'attendance' in lower and any(k in lower for k in ['my', 'show', 'last month', 'summary']): intent, confidence = 'attendance_summary', max(confidence, 0.9)
        elif re.search(r'\b(casual|sick|earned)\b', lower) and re.search(r'\bleaves?\b', lower) and any(k in lower for k in ['how many', 'left', 'balance', 'remaining', 'have']) and 'policy' not in lower and 'do we get' not in lower and 'do i get' not in lower:
            intent, confidence = 'leave_balance', max(confidence, 0.9)
        elif 'who is my manager' in lower: intent, confidence = 'my_manager', max(confidence, 0.9)
        # Closed-class social phrases: a short, auditable whole-message lookup.
        social = re.sub(r'[^a-z ]', '', lower).strip()
        for social_intent, phrases in SOCIAL_PHRASES.items():
            if social in phrases: intent, confidence = social_intent, 0.95
        # Short continuations inherit the previous data intent ("and last month?", "and sick leave?").
        last_intent = state.get('last_intent')
        continuation = lower.startswith(('and ', 'what about', 'also ', 'for ')) or lower in {'show details', 'why is it high'} or 'last month' in lower
        followup_intents = {'leave_balance','attendance_summary','unusual_attendance','payslip_status','leave_request_status','attrition_risk_query','payroll_flags','salary_band_query','headcount_forecast'}
        if last_intent in followup_intents and continuation:
            intent = last_intent; confidence = max(confidence, 0.8)
        # A bare month ("September") answers the month question the bot just asked.
        awaiting = state.get('awaiting_month')
        if awaiting and _month_from_text(text) and (confidence < 0.8 or intent != awaiting):
            intent, confidence = awaiting, max(confidence, 0.85)
        month = _month_from_text(text, state['slots'].get('month'))
        if month: state['slots']['month'] = month
        name = _name_from_text(text)
        if name: state['slots']['employee_name'] = name
        types = [k for k in LEAVE_WORDS if re.search(r'\b' + k + r'\b', lower)]
        if types: state['slots']['leave_type'] = types
        elif not continuation: state['slots'].pop('leave_type', None)

        # FAQ lookup from hr_faq.  Policy intents use the specific FAQ only when it clearly matches; a stray
        # out-of-scope or low-confidence message is rescued only by a clear match, otherwise the fallback applies.
        faq_i, faq_sim, faq_margin = self.faq_match(text)
        clear = faq_sim >= FAQ_MIN_SIMILARITY and faq_margin >= FAQ_MIN_MARGIN and self._shares_distinct_word(text, faq_i)
        rescue = clear and (confidence < self.threshold or intent in FAQ_RESCUE_INTENTS) and intent not in {'greet', 'thanks', 'goodbye'} and not (confidence >= 0.8 and intent not in FAQ_INTENTS)
        exact_faq = clear and faq_sim >= 0.8 and intent not in {'greet', 'thanks', 'goodbye', 'salary_of_other'} and not re.search(r'\b(left|remaining|balance|available)\b', lower)      # near-verbatim FAQ wording wins; personal-balance wording never does
        if exact_faq or rescue or (intent in FAQ_INTENTS and confidence >= self.threshold and clear and (faq_sim >= 0.45 or confidence < 0.6)):
            row = self.faq.iloc[faq_i]
            state['last_intent'] = None; state.pop('awaiting_month', None)
            return self._out(intent if intent in FAQ_INTENTS else 'faq_lookup', str(row.answer), state, max(confidence, faq_sim))

        if confidence < self.threshold:
            return self._out(intent, SAFE_FALLBACK, state, confidence)
        row = self.response_map.get(intent, {})
        if not row: return self._out(intent, SAFE_FALLBACK, state, confidence)
        if not _allowed(user_role, row.get('allowed_roles', 'ALL')):
            state['last_intent'] = intent
            return self._out(intent, 'You do not have permission to access that information. Please contact HR if you believe this is an error.', state, confidence)

        reply = str(row.get('response_template') or '').strip()
        called = False
        if intent in NOT_CONNECTED:
            reply = f"I understood this as a request for {NOT_CONNECTED[intent]}, but that feature is not connected in this version yet. I won’t guess an answer."
        elif intent == 'salary_of_other':
            reply = 'I can’t disclose another employee’s salary or payslip.'
        elif intent == 'payroll_flags':
            if not state['slots'].get('month'):
                reply = 'Which month should I check? Please say a month, such as “September”, or “last month”.'
            else:
                month_param = self._month_param(state['slots']['month'])
                provider = self.payroll_flags_provider
                if provider is None:
                    from .service import payroll_flag_summary as provider
                info = provider(month_param)
                called = True
                reply = (f"No scored payroll run is available for {month_param}." if info.get('count') is None else
                         f"{info['count']} payslips were flagged for {month_param}. Top reason: {info['top_reason']}. Estimated money at risk: Rs {info['money_at_risk']:,.0f}. These are review candidates, not confirmed errors.")
        elif intent in {'leave_balance','leave_request_status','my_manager','my_department','team_members','payslip_status','attendance_summary','unusual_attendance','holiday_list'}:
            needs_month = intent in {'payslip_status', 'attendance_summary', 'unusual_attendance'}
            if needs_month and not state['slots'].get('month'):
                reply = 'Which month should I check? You can say “last month” or name a month.'
            else:
                month_param = self._month_param(state['slots'].get('month')) if needs_month else None
                if needs_month and month_param is None:
                    reply = 'I did not recognise that month. Please name a month, such as “September”, or say “last month”.'
                else:
                    paths = {'leave_balance': ('/api/v1/leave/balance', {}), 'leave_request_status': ('/api/v1/leave/requests', {'limit': '1'}),
                             'my_manager': ('/api/v1/employees/me', {}), 'my_department': ('/api/v1/employees/me', {}),
                             'team_members': ('/api/v1/employees/me/team', {}), 'holiday_list': ('/api/v1/holidays', {}),
                             'payslip_status': (f'/api/v1/payslips/me/{month_param}', {}),
                             'attendance_summary': ('/api/v1/attendance/summary', {'month': month_param}),
                             'unusual_attendance': ('/api/v1/attendance/summary', {'month': month_param})}
                    path, params = paths[intent]
                    try:
                        response = _http_request('GET', self.service_url + path, params=params, headers={'X-Demo-User': str(demo_user)}, timeout=2.0)
                        called = True
                        if response.status_code == 200:
                            reply = self._describe(intent, response.json(), month_param, state, text)
                        elif response.status_code == 404:
                            reply = f"I could not find that record{(' for ' + month_param) if month_param else ''}. The demo data covers the last six months."
                        elif response.status_code == 401:
                            reply = 'I could not verify your identity with the HR service, so I did not look anything up.'
                        else:
                            reply = f'The HR service could not complete this request (HTTP {response.status_code}). Please try again or contact HR.'
                    except (RuntimeError, OSError) + ((httpx.HTTPError,) if httpx is not None else ()):
                        reply = 'I could not reach the local mock HR service. Start it with `uvicorn mock_server:app --app-dir 05_Starter/mock_services --port 9000`, then try again.'
        elif '{' in reply and '}' in reply:
            reply = 'I understood your request, but I cannot fetch those details in this version. Please check the HR portal or contact HR.'
        if reply.startswith('Which month') or reply.startswith('I did not recognise that month'): state['awaiting_month'] = intent
        else: state.pop('awaiting_month', None)
        state['last_intent'] = intent
        return self._out(intent, reply, state, confidence, called)
