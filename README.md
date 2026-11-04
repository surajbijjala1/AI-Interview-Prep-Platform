# AI Interview Prep Platform

Upload your resume and a job description. The agent researches the company's interview process on the public web,
then runs an adaptive, graded mock interview one question at a time.

A learning project for **RAG, LangChain/LangGraph, LangSmith and MCP**, built in phases on free tiers only.
The full design is in [docs/SPEC.md](docs/SPEC.md). Per-phase walkthroughs are in [docs/learning/](docs/learning/).

| Phase | Status |
|---|---|
| 1. Foundation: config, model switch, PII redaction, resume/JD parsing, Setup page | ✅ |
| 2. Custom MCP server + RAG question bank | ⏳ |
| 3. Research agent (Tavily / Fetch / DuckDuckGo MCP) | ⏳ |
| 4. LangGraph interview loop | ⏳ |
| 5. Grading, report, memory | ⏳ |
| 6. LangSmith evals + polish | ⏳ |

## Getting started

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

```bash
cp .env.example .env
```

Fill in `GOOGLE_API_KEY` (from a free-tier [AI Studio](https://aistudio.google.com/app/apikey) project) and, for
tracing, `LANGSMITH_API_KEY`. Then:

```bash
uv run streamlit run app/Home.py
```

The Home page shows which services are configured.

### Running on a local model

Set `LLM_PROVIDER=ollama` in `.env` (default model `llama3.2`) and start Ollama:

```bash
ollama serve
```

Local 3B models are noticeably weaker. For example, they often miss the company name, which is why the Setup page
asks you to confirm it.

### Tests

No network or API quota needed:

```bash
uv run pytest
```

## Privacy

Your name, email, phone, links and street address are removed from the resume **before** any text reaches a model
or a LangSmith trace. The Setup page shows exactly what the AI sees. The uploaded PDF is never written to disk.
Tests use fictional personas only.

## Project layout

```
app/                    Streamlit UI (Home + pages/)
src/interview_prep/     config, llm (provider switch), schemas, ingest/ (pdf, redact, resume, jd)
tests/                  pytest with fake LLMs; fixtures/ holds synthetic JDs
docs/                   SPEC.md, learning/phase-N.md
playground/             your local scratch space (gitignored)
```
