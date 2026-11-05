# AI Interview Prep Platform

[![Tests](https://github.com/surajbijjala1/AI-Interview-Prep-Platform/actions/workflows/tests.yml/badge.svg)](https://github.com/surajbijjala1/AI-Interview-Prep-Platform/actions/workflows/tests.yml)

Upload your resume and a job description. The agent researches the company's interview process on the public web,
then runs an adaptive, graded mock interview one question at a time.

A learning project for **RAG, LangChain/LangGraph, LangSmith and MCP**, built in phases on free tiers only.
The full design is in [docs/SPEC.md](docs/SPEC.md). Per-phase walkthroughs are in [docs/learning/](docs/learning/).

| Phase | Status |
|---|---|
| 1. Foundation: config, model switch, PII redaction, resume/JD parsing, Setup page | ✅ |
| 2. Custom MCP server + RAG question bank | ✅ |
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

### Ollama (always needed)

Embeddings always run locally on [Ollama](https://ollama.com), so start it and pull the embedding model once:

```bash
ollama serve
```

```bash
ollama pull nomic-embed-text
```

### Running on a local model

To use Ollama for the chat model too, set `LLM_PROVIDER=ollama` in `.env` (default model `llama3.2`).

Local 3B models are noticeably weaker. For example, they often miss the company name, which is why the Setup page
asks you to confirm it.

### Tests

No network or API quota needed. GitHub Actions runs the same suite on every pull request and on pushes to `main`
([.github/workflows/tests.yml](.github/workflows/tests.yml)):

```bash
uv run pytest
```

## MCP server

`interview-prep` is our own MCP server. It searches the question bank, extracts and redacts resumes, indexes and
searches them, and deletes data. The app's **Question bank** page calls it over MCP, and so can Claude Desktop and
Claude Code. Setup and example prompts are in [docs/mcp-clients.md](docs/mcp-clients.md). To run it by hand:

```bash
uv run interview-prep-mcp
```

The question bank lives in [data/question_bank/](data/question_bank/). After editing it, check retrieval quality
(needs Ollama):

```bash
uv run python evals/bank_retrieval.py
```

## Privacy

Your name, email, phone, links and street address are removed from the resume **before** any text reaches a model
or a LangSmith trace. The Setup page shows exactly what the AI sees. The uploaded PDF is never written to disk.
Indexed resume text (already redacted) and the SQLite database stay on your machine in `.data/` (gitignored).
Tests use fictional personas only.

## Project layout

```
app/                    Streamlit UI (Home + pages/)
src/interview_prep/     config, llm (provider switch), embeddings, schemas, ingest/ (pdf, redact, resume, jd),
                        question_bank, storage/ (SQLite + Chroma), mcp_server/, mcp_client
data/question_bank/     curated questions: shared (screen + behavioral) and one file per track
evals/                  retrieval checks (LangSmith evals arrive in Phase 6)
tests/                  pytest with fake LLMs and fake embeddings; fixtures/ holds synthetic JDs
docs/                   SPEC.md, learning/phase-N.md
.data/                  local SQLite DB and Chroma vectors (gitignored)
playground/             your local scratch space (gitignored)
```
