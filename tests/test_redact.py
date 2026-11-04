import pytest

from interview_prep.ingest.redact import redact_jd, redact_resume

from .conftest import PERSONA_RESUME

PII = ["Priya", "Raman", "priya.raman@example.com", "555-0134", "linkedin.com/in/priyaraman",
       "priyaraman.dev", "github.com/priyar", "123 Market Street", "94105"]


@pytest.fixture
def redacted():
    return redact_resume(PERSONA_RESUME)


@pytest.mark.parametrize("pii", PII)
def test_resume_pii_removed(redacted, pii):
    assert pii not in redacted.text


@pytest.mark.parametrize(
    "kept",
    ["Python 3.12", "ASP.NET", "Node.js", "10000 requests", "40%", "2016 - 2020", "Ledgerly", "San Francisco"],
)
def test_resume_content_kept(redacted, kept):
    assert kept in redacted.text


def test_counts_report_what_was_replaced(redacted):
    assert redacted.counts["[EMAIL]"] == 1
    assert redacted.counts["[PHONE]"] == 1
    assert redacted.counts["[URL]"] == 3
    assert redacted.counts["[NAME]"] == 2  # header + "Priya enjoys..."
    assert redacted.counts["[ADDRESS]"] == 2  # street + state/ZIP


@pytest.mark.parametrize("phone", ["+1 415 555 0134", "415.555.0134", "4155550134", "(415)555-0134"])
def test_phone_formats(phone):
    assert redact_resume(f"Contact: {phone}").text == "Contact: [PHONE]"


def test_header_words_are_not_treated_as_a_name():
    assert redact_resume("Curriculum Vitae\nJane did things").counts["[NAME]"] == 0


def test_jd_keeps_company_links_but_drops_recruiter_contact():
    result = redact_jd("Apply at https://acme.com/careers or email jo@acme.com, 212-555-0100.")
    assert result.text == "Apply at https://acme.com/careers or email [EMAIL], [PHONE]."
