"""Deterministic PII redaction. Runs before any text reaches an LLM, a trace or disk.

Regex + heuristics on purpose: an LLM-based redactor would itself need to see the PII.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
PHONE = re.compile(r"(?<![\w-])(?:\+?\d{1,3}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}(?![\w-])")
# Any link: scheme/www links, plus bare lowercase domains (github.com/x, jane.dev).
# Lowercase-only TLDs keep tech names like "ASP.NET" and "Node.js" intact.
URL = re.compile(
    r"(?:https?://|www\.)\S+"
    r"|\b[a-z0-9-]+(?:\.[a-z0-9-]+)*\.(?:com|io|dev|me|net|org|ai|app|co|xyz|site|page)\b(?:/\S*)?"
)
STREET_ADDRESS = re.compile(
    r"\b\d{1,6}\s+(?:[A-Z][\w.]*\s+){1,4}"
    r"(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Way|Place|Pl)\b\.?"
    r"(?:,?\s*(?:Apt|Suite|Unit|#)\s*\w+)?"
)
STATE_ZIP = re.compile(r"\b[A-Z]{2}\s+\d{5}(?:-\d{4})?\b")

# A resume header line usually starts with the candidate's name: "Jane Q. Doe | jane@x.com | ...".
HEADER_SEPARATORS = re.compile(r"\s*[|•·,–—]\s*|\s{3,}")
NAME_LINE = re.compile(r"[A-Z][a-zA-Z'’-]+(?:\s+[A-Z][a-zA-Z.'’-]*){1,3}")
NOT_A_NAME = {"resume", "curriculum", "vitae", "cv", "summary", "profile", "experience", "contact"}


@dataclass
class Redacted:
    text: str
    counts: Counter = field(default_factory=Counter)  # placeholder -> number of replacements


def _sub(pattern: re.Pattern, placeholder: str, result: Redacted) -> None:
    result.text, n = pattern.subn(placeholder, result.text)
    if n:
        result.counts[placeholder] += n


def _candidate_name(text: str) -> str | None:
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    first_segment = HEADER_SEPARATORS.split(first_line)[0]
    if not NAME_LINE.fullmatch(first_segment):
        return None
    if {word.lower().strip(".") for word in first_segment.split()} & NOT_A_NAME:
        return None
    return first_segment


def _redact_contact_info(result: Redacted, *, urls: bool) -> None:
    # Emails before URLs so "jane@site.dev" isn't half-eaten by the domain pattern.
    _sub(EMAIL, "[EMAIL]", result)
    if urls:
        _sub(URL, "[URL]", result)
    _sub(PHONE, "[PHONE]", result)


def redact_resume(text: str) -> Redacted:
    """Remove name, email, phone, links and street address from resume text."""
    result = Redacted(text)
    if name := _candidate_name(text):
        # Full name first, then each part on its own ("Jane" in "Jane led the team").
        parts = [name] + [p for p in name.split() if len(p.strip(".")) > 2]
        for part in parts:
            _sub(re.compile(rf"\b{re.escape(part)}\b"), "[NAME]", result)
    _redact_contact_info(result, urls=True)
    _sub(STREET_ADDRESS, "[ADDRESS]", result)
    _sub(STATE_ZIP, "[ADDRESS]", result)
    return result


def redact_jd(text: str) -> Redacted:
    """Remove contact details (e.g. a recruiter's email/phone) but keep company links for research."""
    result = Redacted(text)
    _redact_contact_info(result, urls=False)
    return result
