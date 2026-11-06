"""Skill extraction from free text (resume and job description) with exact and synonym matching."""
from __future__ import annotations
import re
from dataclasses import dataclass
import pandas as pd

# Small, explicit synonym map (canonical lower-case name -> alternative spellings). Documented in docs/feature_definitions.md.
DEFAULT_SYNONYMS = {
    "excel": ["ms excel", "microsoft excel"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "postgresql": ["postgres"],
    "rest apis": ["rest api", "restful api", "restful apis"],
    "machine learning": ["ml"],
    "google analytics": ["ga4"],
    "social media marketing": ["smm"],
    "model deployment": ["deploying models"],
    "data visualization": ["data visualisation"],
    "problem solving": ["problem-solving"],
}


def split_skill_list(value) -> list[str]:
    if value is None or (isinstance(value, float) and value != value): return []
    return [s.strip() for s in re.split(r"[;|,]", str(value)) if s.strip()]


def build_vocabulary(jobs: pd.DataFrame, resumes: pd.DataFrame | None = None) -> list[str]:
    """Skill ontology = every skill named in job descriptions plus the structured skill lists. Built WITHOUT labels."""
    vocab = set()
    for c in ("required_skills", "nice_to_have_skills"):
        if c in jobs:
            for v in jobs[c]: vocab |= {s.lower() for s in split_skill_list(v)}
    if resumes is not None and "skills" in resumes:
        for v in resumes["skills"]: vocab |= {s.lower() for s in split_skill_list(v)}
    return sorted(vocab)


@dataclass(frozen=True)
class Match:
    skill: str
    kind: str          # "exact" or "synonym"


class SkillExtractor:
    """Dictionary matcher. Longest phrase first; a matched span is consumed so "customer communication"
    does not also count as "communication". Synonyms are folded onto one canonical name."""

    def __init__(self, vocabulary, synonyms: dict | None = None):
        syn = {k.lower(): [a.lower() for a in v] for k, v in (synonyms or DEFAULT_SYNONYMS).items()}
        self.synonyms = syn
        self._alt_to_canon = {a: c for c, alts in syn.items() for a in alts}
        # a vocabulary item that is itself a synonym (e.g. "ms excel") is folded onto its canonical skill
        self.vocab = sorted({self.canon(v) for v in vocabulary}, key=lambda s: (-len(s), s))
        forms = {c: c for c in self.vocab}
        self._forms = [(c, c, "exact") for c in self.vocab] + [(a, c, "synonym") for a, c in self._alt_to_canon.items()]
        self._forms.sort(key=lambda t: (-len(t[0]), t[0]))
        self._pat = {f: self._compile(f) for f, _, _ in self._forms}

    def canon(self, skill: str) -> str:
        s = str(skill).strip().lower()
        return self._alt_to_canon.get(s, s)

    @staticmethod
    def _compile(s: str):
        return re.compile(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(s).replace(r"\ ", r"[\s_-]+"), re.I)

    def extract(self, text) -> dict[str, str]:
        """Return {canonical_skill: 'exact'|'synonym'}; 'exact' wins if both spellings occur."""
        if text is None or (isinstance(text, float) and text != text): return {}
        t = str(text).lower(); found: dict[str, str] = {}
        for form, canon, kind in self._forms:
            pat = self._pat[form]
            if pat.search(t):
                if found.get(canon) != "exact": found[canon] = kind
                t = pat.sub(" ", t)                      # consume the span
        return found


def job_skill_sets(job_row, extractor: "SkillExtractor | None" = None) -> tuple[list[str], list[str]]:
    c = extractor.canon if extractor else (lambda x: x.lower())
    req = [c(s) for s in split_skill_list(job_row["required_skills"])]
    nice = [c(s) for s in split_skill_list(job_row.get("nice_to_have_skills"))]
    return req, nice


def coverage(found: dict, required: list[str], nice: list[str]) -> dict:
    """Matched / missing skills and coverage fractions (exact and synonym matches are both counted and labelled)."""
    m_req = [s for s in required if s in found]; m_nice = [s for s in nice if s in found]
    return {
        "matched_required": m_req, "missing_required": [s for s in required if s not in found],
        "matched_nice": m_nice, "missing_nice": [s for s in nice if s not in found],
        "required_coverage": len(m_req) / len(required) if required else 0.0,
        "nice_coverage": len(m_nice) / len(nice) if nice else 0.0,
        "synonym_matches": [s for s in m_req + m_nice if found.get(s) == "synonym"],
    }
