import pytest

from interview_prep.ingest import parse_jd, parse_resume
from interview_prep.ingest.pdf import extract_text
from interview_prep.schemas import ParsedJD, ResumeProfile, RoleTrack, Seniority

from .conftest import FIXTURES, FakeStructuredLLM, make_pdf

PROFILE = ResumeProfile(
    summary="Backend engineer.", skills=["Python"], total_years_experience=4, seniority=Seniority.MID
)
JD = ParsedJD(
    job_title="Backend Software Engineer", company_name="DataSync Inc.",
    role_track=RoleTrack.BACKEND_SWE, seniority=Seniority.MID, required_skills=["Python"],
)


def test_extract_text_reads_pdf(persona_pdf):
    assert "Ledgerly" in extract_text(persona_pdf)


def test_extract_text_rejects_pdf_without_text():
    with pytest.raises(ValueError, match="No text"):
        extract_text(make_pdf(""))


def test_parse_resume_never_sends_pii_to_llm(persona_pdf):
    llm = FakeStructuredLLM(PROFILE)
    parsed = parse_resume(persona_pdf, llm=llm)

    assert parsed.profile == PROFILE
    (prompt,) = llm.prompts
    assert "Ledgerly" in prompt
    for pii in ["Priya", "priya.raman@example.com", "555-0134", "priyaraman"]:
        assert pii not in prompt
    assert parsed.redacted.text in prompt  # the UI shows the user exactly what the LLM saw


def test_parse_jd_redacts_contact_details():
    jd_text = (FIXTURES / "jds" / "backend_mid_datasync.txt").read_text()
    llm = FakeStructuredLLM(JD)

    assert parse_jd(jd_text + "\nQuestions? recruiter@datasync.example.com", llm=llm) == JD
    assert "DataSync Inc." in llm.prompts[0]
    assert "recruiter@" not in llm.prompts[0]


def test_parse_jd_rejects_empty_text():
    with pytest.raises(ValueError, match="empty"):
        parse_jd("   ", llm=FakeStructuredLLM(JD))
