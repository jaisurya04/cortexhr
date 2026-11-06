"""Resume text processing: section parsing, identifier masking, normalisation, tokenising.

Design notes
- Protected / irrelevant details (name, e-mail, city, college tier) are REMOVED before any scoring.
- NLTK / spaCy are not required: a small stop-word list and conservative suffix rules are used (documented in docs/methodology.md).
- Skill phrases are protected: "machine learning" becomes the single token "machine_learning" so it is not split or stemmed.
"""
from __future__ import annotations
import re
import unicodedata

STOPWORDS = frozenset("""a about above after again all also am an and any are as at be because been before being below between both but by
can could did do does doing down during each few for from further had has have having he her here hers him his how i if in into is it its
just me more most my no nor not now of off on once only or other our out over own same she should so some such than that the their them then
there these they this those through to too under until up very was we were what when where which while who whom why will with would you your
experienced motivated strengths include includes including years year yrs yr experience""".split())

_SECTION_HEADINGS = ("SUMMARY", "EXPERIENCE", "SKILLS", "EDUCATION", "CERTIFICATIONS", "EXPERIENCE NOTES", "PROJECTS", "OBJECTIVE")
_TIER_RE = re.compile(r"\btier\s*[1-9]\s*(?:institution|college|university)?\b", re.I)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def normalise_unicode(text: str) -> str:
    t = unicodedata.normalize("NFKC", str(text))
    return t.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")


def is_blank(text) -> bool:
    return text is None or (isinstance(text, float) and text != text) or not str(text).strip()


def parse_sections(text: str) -> dict:
    """Split a resume into sections. Layout-robust: if headings are not found the whole text is returned as 'body'."""
    t = normalise_unicode(text)
    pat = re.compile(r"(?m)^\s*(%s)\s*$" % "|".join(re.escape(h) for h in _SECTION_HEADINGS), re.I)
    marks = [(m.start(), m.end(), m.group(1).upper()) for m in pat.finditer(t)]
    if not marks:
        return {"body": t}
    out = {"header": t[:marks[0][0]].strip()}
    for i, (s, e, name) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(t)
        out[name.lower().replace(" ", "_")] = (out.get(name.lower().replace(" ", "_"), "") + " " + t[e:end]).strip()
    return out


def mask_identifiers(text: str, name: str | None = None, email: str | None = None, city: str | None = None, tier: bool = True, strip_emails: bool = True) -> str:
    """Remove name, e-mail, city and college-tier phrases. Inputs are used only to know WHAT to remove."""
    t = normalise_unicode(text)
    if not is_blank(email):
        t = re.sub(re.escape(str(email)), " ", t, flags=re.I)
    if strip_emails:
        t = _EMAIL_RE.sub(" ", t)                  # any other e-mail address
    for val in (name, city):
        if not is_blank(val):
            t = re.sub(r"\b%s\b" % re.escape(str(val).strip()), " ", t, flags=re.I)
            for part in str(val).split():          # first name / surname alone
                if len(part) > 2:
                    t = re.sub(r"\b%s\b" % re.escape(part), " ", t, flags=re.I)
    if tier:
        t = _TIER_RE.sub(" ", t)                    # "Tier 2 institution"
    return re.sub(r"[ \t]+", " ", t)


def normalise(text: str) -> str:
    t = normalise_unicode(text).lower()
    t = re.sub(r"[|•·●▪]", " ", t)
    t = re.sub(r"[^a-z0-9+#.\-/_\s]", " ", t)        # keep c++, c#, node.js, scikit-learn, ci/cd
    return re.sub(r"\s+", " ", t).strip()


def protect_phrases(text: str, phrases) -> str:
    """Join multi-word skills into one token (machine learning -> machine_learning). Longest phrase first."""
    t = text
    for p in sorted({p.lower() for p in phrases if " " in p or "-" in p}, key=len, reverse=True):
        t = re.sub(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(p), p.replace(" ", "_"), t)
    return t


def lemma(tok: str) -> str:
    """Conservative plural folding; protected skill tokens (with _ - . + #) are never touched."""
    if len(tok) <= 3 or any(c in tok for c in "_-.+#/") or tok.isdigit():
        return tok
    if tok.endswith("ies") and len(tok) > 4: return tok[:-3] + "y"
    if tok.endswith(("sses", "ches", "shes", "xes")): return tok[:-2]
    if tok.endswith("s") and not tok.endswith(("ss", "us", "is")): return tok[:-1]
    return tok


def tokenize(text: str, phrases=(), remove_stopwords=True) -> list[str]:
    t = protect_phrases(normalise(text), phrases)
    toks = re.findall(r"[a-z0-9_+#./-]+", t)
    toks = [x.strip(".-/") for x in toks]
    toks = [lemma(x) for x in toks if x and not x.isdigit() and len(x) > 1]
    return [x for x in toks if not (remove_stopwords and x in STOPWORDS)]


def prepare_document(text: str, phrases=()) -> str:
    return " ".join(tokenize(text, phrases))


_ROLE_COMPANY_RE = re.compile(r"(?m)^([^\n]*?[,|]\s*)[A-Za-z][A-Za-z0-9&.\- ]{2,60}?(\s*\(\s*\d+(?:\.\d+)?\s*yrs?\s*\))", re.I)


def for_skill_extraction(text: str, name=None, email=None, city=None, tier: bool = True, strip_emails: bool = True) -> str:
    """Masked text with employer names removed from role lines ("Harbor Analytics" must not count as an analytics skill)."""
    t = mask_identifiers(text, name, email, city, tier, strip_emails)
    return _ROLE_COMPANY_RE.sub(lambda m: m.group(1).rstrip(",| ") + " " + m.group(2), t)
