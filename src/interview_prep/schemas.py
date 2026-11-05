"""Pydantic models shared across the app. Field descriptions double as LLM extraction instructions."""

from enum import StrEnum

from pydantic import BaseModel, Field


class RoleTrack(StrEnum):
    BACKEND_SWE = "backend_swe"
    AI_LLM_ENGINEER = "ai_llm_engineer"
    DATA_ENGINEER = "data_engineer"
    FULLSTACK_SWE = "fullstack_swe"
    FRONTEND_SWE = "frontend_swe"
    GENERALIST_SWE = "generalist_swe"
    DATA_ANALYST = "data_analyst"
    OTHER = "other"


# Tracks with a v1 question bank and rubrics; the others are recognised but not yet interviewable.
SUPPORTED_TRACKS = (RoleTrack.BACKEND_SWE, RoleTrack.AI_LLM_ENGINEER, RoleTrack.DATA_ENGINEER)


class Seniority(StrEnum):
    NEW_GRAD = "new_grad"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"


class FeedbackMode(StrEnum):
    COACHING = "coaching"  # brief score + tip after each answer, plus the end report
    MOCK = "mock"  # end report only


class SessionLength(StrEnum):
    QUICK = "quick"  # ~5 main questions
    FULL = "full"  # ~12 main questions across rounds


class Experience(BaseModel):
    title: str = Field(description="Job title held.")
    company: str = Field(description="Employer name.")
    start: str | None = Field(default=None, description="Start date as written, e.g. 'Jan 2021'.")
    end: str | None = Field(default=None, description="End date as written, or 'Present'.")
    highlights: list[str] = Field(
        default_factory=list,
        description="Concrete achievements and responsibilities, keeping numbers and technologies.",
    )


class Project(BaseModel):
    name: str
    description: str
    technologies: list[str] = Field(default_factory=list)


class ResumeProfile(BaseModel):
    summary: str = Field(description="Two-sentence professional summary of the candidate.")
    skills: list[str] = Field(description="Technical skills, tools, languages and frameworks mentioned anywhere.")
    experience: list[Experience] = Field(default_factory=list, description="Work experience, most recent first.")
    projects: list[Project] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list, description="Degrees and institutions, one per entry.")
    total_years_experience: float = Field(
        description="Total years of professional (non-internship) experience, estimated from dates."
    )
    seniority: Seniority = Field(
        description="new_grad: <1 year, junior: 1-2, mid: 3-5, senior: 6+ years or clear senior titles."
    )


class ParsedJD(BaseModel):
    job_title: str = Field(description="The official title of the role.")
    company_name: str | None = Field(
        default=None,
        description="The hiring company. Null if not stated or only described vaguely (e.g. 'a leading fintech').",
    )
    company_website: str | None = Field(default=None, description="Company website, only if stated in the JD.")
    role_track: RoleTrack = Field(
        description="Closest track. backend_swe: APIs/services/databases. ai_llm_engineer: LLM apps, RAG, agents. "
        "data_engineer: pipelines/warehousing/SQL. fullstack_swe, frontend_swe, generalist_swe, data_analyst "
        "as named. other: none of these."
    )
    seniority: Seniority = Field(description="Level implied by title and years. Staff and above map to senior.")
    min_years_experience: int | None = Field(default=None, description="Minimum years required, if stated.")
    required_skills: list[str] = Field(description="Essential skills, tools or technologies.")
    preferred_skills: list[str] = Field(default_factory=list, description="Nice-to-have skills.")
    responsibilities: list[str] = Field(default_factory=list, description="Primary responsibilities of the role.")


class SessionSetup(BaseModel):
    """What the user confirmed on the Setup page; the input to research and planning."""

    company_name: str
    job_title: str
    role_track: RoleTrack
    seniority: Seniority
    feedback_mode: FeedbackMode
    length: SessionLength


# --- Question bank (Phase 2) ---------------------------------------------------------------------------------


class Round(StrEnum):
    """Interview rounds; each has its own rubric (SPEC §8)."""

    SCREEN = "screen"  # motivation and fit
    BEHAVIORAL = "behavioral"
    TECHNICAL = "technical"  # coding concepts, domain depth, SQL, ...
    SYSTEM_DESIGN = "system_design"


SHARED_TRACK = "shared"  # bank questions (screen + behavioral) asked on every track


class BankQuestion(BaseModel):
    """One curated seed question. The interviewer adapts it to the resume and asks its own follow-ups."""

    id: str = Field(pattern=r"^[a-z]+-\d{3}$", description="Stable ID such as 'be-007'; never reused.")
    track: str = Field(description=f"A supported RoleTrack value, or '{SHARED_TRACK}'.")
    round: Round
    difficulty: int = Field(ge=1, le=3, description="1: new grad/junior, 2: mid, 3: senior.")
    topics: list[str] = Field(min_length=1, description="Lowercase tags used for retrieval and weak-area memory.")
    question: str = Field(min_length=10)
    what_good_looks_like: str = Field(
        min_length=20, description="Key points a strong answer covers; guides grading and follow-ups."
    )


class QuestionHit(BankQuestion):
    score: float = Field(description="Relevance to the query, 0-1 (cosine similarity).")


class ResumeChunk(BaseModel):
    text: str
    chunk_index: int
    score: float = Field(description="Relevance to the query, 0-1 (cosine similarity).")
