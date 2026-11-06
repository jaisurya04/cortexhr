"""Years-of-experience extraction from resume text. Duration-based (the resumes give '(2.4 yrs)' per role, not dates)."""
from __future__ import annotations
import re
from .text import normalise_unicode, tokenize

_ROLE_RE = re.compile(r"^[\s|*\-]*([A-Za-z][A-Za-z0-9/&.\- ]{2,60}?)\s*[,|]\s*([A-Za-z][A-Za-z0-9&.\- ]{2,60}?)\s*\(\s*(\d+(?:\.\d+)?)\s*yrs?\s*\)\s*(?:summary|skills|experience|education|certifications)?\s*$", re.I)
_STATED_RE = re.compile(r"with\s+(\d+(?:\.\d+)?)\s+years?\s+of\s+experience", re.I)
_TITLE_NOISE = {"senior", "junior", "lead", "associate", "assistant", "sr", "jr", "principal", "head", "chief", "intern", "trainee"}


def _lines(text: str) -> list[str]:
    """Layout-robust line list: two-column pages are split at runs of 3+ spaces, glued sentences at '.X'."""
    out = []
    for ln in normalise_unicode(text).splitlines():
        for seg in re.split(r"\s{3,}", ln):
            out.extend(re.split(r"(?<=[a-z])\.(?=[A-Z])", seg))
    return [x for x in out if x.strip()]


def extract_experience(text) -> dict:
    """Return roles, summed years (de-duplicated), the years stated in the summary and quality flags."""
    flags = []
    if text is None or (isinstance(text, float) and text != text) or not str(text).strip():
        return {"roles": [], "total_years": None, "stated_years": None, "flags": ["empty_text"]}
    t = normalise_unicode(text)
    roles, seen = [], set()
    for ln in _lines(t):
        m = _ROLE_RE.match(ln)
        if not m: continue
        title = m.group(1).strip(" -|,")
        key = (title.lower(), m.group(2).strip().lower(), float(m.group(3)))
        if key in seen: flags.append("duplicate_role_ignored"); continue
        seen.add(key); roles.append({"title": title, "company": m.group(2).strip(), "years": float(m.group(3))})
    stated = _STATED_RE.search(t); stated_years = float(stated.group(1)) if stated else None
    total = round(sum(r["years"] for r in roles), 2) if roles else None
    if total is None and stated_years is not None: flags.append("used_stated_years"); total = stated_years
    if total is None: flags.append("no_experience_found")
    if total is not None and stated_years is not None and abs(total - stated_years) > 0.15: flags.append("stated_vs_roles_mismatch")
    if total is not None and (total < 0 or total > 50): flags.append("implausible_years"); total = None
    return {"roles": roles, "total_years": total, "stated_years": stated_years, "flags": sorted(set(flags))}


def _title_tokens(title: str) -> set[str]:
    return {t for t in tokenize(title, remove_stopwords=True) if t not in _TITLE_NOISE}


def title_similarity(role_title: str, job_title: str) -> float:
    a, b = _title_tokens(role_title), _title_tokens(job_title)
    return len(a & b) / len(b) if a and b else 0.0


def relevant_years(roles: list[dict], job_title: str, threshold: float = 0.5) -> float:
    """Years in roles whose title shares at least `threshold` of the job title's core words."""
    return round(sum(r["years"] for r in roles if title_similarity(r["title"], job_title) >= threshold), 2)


def experience_score(total_years, rel_years, min_years: float) -> float:
    """0..1. Half from total years, half from relevant years, each capped at the job minimum, so more years never beats the requirement."""
    need = max(float(min_years), 1.0)
    tot = min(1.0, (total_years or 0.0) / need); rel = min(1.0, (rel_years or 0.0) / need)
    return round(0.5 * tot + 0.5 * rel, 6)
