import streamlit as st

from interview_prep.ingest import parse_jd, parse_resume
from interview_prep.schemas import SUPPORTED_TRACKS, FeedbackMode, Seniority, SessionLength, SessionSetup

st.set_page_config(page_title="Setup · AI Interview Prep", page_icon="🎯")
st.title("Setup")

with st.form("inputs"):
    resume_file = st.file_uploader("Resume (PDF)", type="pdf")
    jd_text = st.text_area("Job description", height=250, placeholder="Paste the full job description…")
    left, right = st.columns(2)
    mode = left.radio(
        "Feedback",
        list(FeedbackMode),
        format_func=lambda m: {"coaching": "Coaching: tip after each answer", "mock": "Mock: report at the end"}[m],
    )
    length = right.radio(
        "Length",
        list(SessionLength),
        format_func=lambda n: {"quick": "Quick (~5 questions)", "full": "Full (~12 questions)"}[n],
    )
    analyze = st.form_submit_button("Analyze", type="primary")

if analyze:
    if not resume_file or not jd_text.strip():
        st.error("Add both a resume PDF and a job description.")
    else:
        try:
            with st.status("Analyzing…", expanded=True) as status:
                st.write("Extracting resume text and redacting contact info…")
                resume = parse_resume(resume_file.getvalue())
                st.write("Parsing the job description…")
                jd = parse_jd(jd_text)
                status.update(label="Analysis complete", state="complete", expanded=False)
        except Exception as exc:  # surface parse and model errors to the user instead of a stack trace
            st.error(f"Couldn't analyze: {exc}")
        else:
            st.session_state.update(resume=resume, jd=jd, mode=mode, length=length)
            st.session_state.pop("setup", None)

if "resume" not in st.session_state:
    st.stop()

resume, jd = st.session_state.resume, st.session_state.jd
profile = resume.profile

st.subheader("Your profile")
redactions = ", ".join(f"{n} × {label}" for label, n in resume.redacted.counts.items()) or "nothing found"
st.caption(f"Redacted before sending to the AI: {redactions}")
with st.expander("What the AI sees (redacted resume)"):
    st.text(resume.redacted.text)
st.write(profile.summary)
st.write(f"**Seniority:** {profile.seniority} · **Experience:** {profile.total_years_experience:g} years")
st.write("**Skills:** " + ", ".join(profile.skills))
for job in profile.experience:
    st.markdown(f"- **{job.title}**, {job.company} ({job.start or '?'} – {job.end or '?'})")

st.subheader("The role")
st.write(f"**{jd.job_title}** · detected track `{jd.role_track}` · level `{jd.seniority}`")
st.write("**Required:** " + ", ".join(jd.required_skills))
if jd.preferred_skills:
    st.write("**Preferred:** " + ", ".join(jd.preferred_skills))

st.subheader("Confirm before research")
if jd.role_track not in SUPPORTED_TRACKS:
    st.warning(
        f"`{jd.role_track}` interviews aren't supported yet. Pick the closest supported track below."
    )
with st.form("confirm"):
    company = st.text_input(
        "Company to research", value=jd.company_name or "", help="Research is spent on this name, so check it."
    )
    tracks = list(SUPPORTED_TRACKS)
    track = st.selectbox(
        "Interview track", tracks, index=tracks.index(jd.role_track) if jd.role_track in tracks else 0
    )
    levels = list(Seniority)
    seniority = st.selectbox("Level", levels, index=levels.index(jd.seniority))
    confirmed = st.form_submit_button("Confirm", type="primary")

if confirmed:
    if not company.strip():
        st.error("Enter the company name; the JD didn't state one clearly.")
    else:
        st.session_state.setup = SessionSetup(
            company_name=company.strip(),
            job_title=jd.job_title,
            role_track=track,
            seniority=seniority,
            feedback_mode=st.session_state.mode,
            length=st.session_state.length,
        )

if setup := st.session_state.get("setup"):
    st.success(f"Ready to research **{setup.company_name}** (company research arrives in Phase 3).")
    st.json(setup.model_dump(mode="json"), expanded=False)
