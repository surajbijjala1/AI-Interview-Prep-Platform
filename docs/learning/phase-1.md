# Phase 1: Foundation

**What you can do now:** upload a resume PDF and paste a JD. The app redacts your contact info, extracts a
structured profile and role, and asks you to confirm the company, track and level.

## Concepts

### 1. Structured output: turning text into typed objects
`ingest/resume.py` and `ingest/jd.py` build a chain: `prompt | llm.with_structured_output(Schema)`.

- `with_structured_output` hands the model a JSON schema generated from a Pydantic class
  (`schemas.py`). The model must reply in that shape, and LangChain validates the reply into a Python object.
- The **`Field(description=...)` text is part of the prompt.** The model reads it. Look at `ParsedJD.role_track`:
  the description defines each track. Changing it changes how jobs get classified.
- `|` is LCEL (LangChain Expression Language). Each piece is a *Runnable*, and piping them makes a new Runnable
  with `.invoke`, `.batch` and `.stream` built in.

### 2. One factory, many providers
`llm.py:get_chat_model` uses `init_chat_model(model, model_provider=...)`, so the rest of the code never imports a
provider class. `LLM_PROVIDER=ollama` swaps Gemini for local `llama3.2` with no other changes.

- **Rate limiter:** `InMemoryRateLimiter` is a token bucket. At 10 requests/min, a token is added every 6 seconds,
  and each model call waits for a token. One limiter is shared by all Gemini instances (`lru_cache`), so the cap is
  app-wide. The bucket starts empty, so the first call waits up to 6 s.
- **Retries:** the Google SDK retries 429s and 5xx errors with exponential backoff (`max_retries=6`). Errors that
  can't be retried, like 402 "credits depleted", fail immediately.

### 3. Redact before the model, not after
`ingest/redact.py` is plain regex plus a heuristic, on purpose: an LLM-based redactor would itself have to see the
PII, and so would the trace of that call. Key ideas:

- **Order matters:** emails are removed before URLs, or `jane@site.dev` would be half-matched as a domain.
- **Name detection:** the first segment of the first line, if it looks like 2–4 capitalized words. Then each name
  part is removed everywhere ("Priya led the migration").
- **Precision vs. recall:** lowercase-only TLDs keep `ASP.NET` and `Node.js`. State + ZIP (`CA 94105`) is matched
  instead of any 5-digit number, so "10000 requests" survives.
- The invariant has **one owner**: `parse_resume` takes PDF bytes, not text, so there is no code path that sends
  unredacted text to the model.

### 4. Testing LLM code without an LLM
`tests/conftest.py:FakeStructuredLLM` returns a canned object and **records every prompt it receives**. That lets
`test_parse_resume_never_sends_pii_to_llm` assert on the exact text the model would have seen. That's a privacy
guarantee checked on every test run, at zero quota.

## Code map

| File | Role |
|---|---|
| `src/interview_prep/config.py` | Settings from env/.env (pydantic-settings); loads `.env` so LangSmith sees its vars |
| `src/interview_prep/llm.py` | Provider switch + shared rate limiter |
| `src/interview_prep/schemas.py` | All Pydantic models; field descriptions are extraction instructions |
| `src/interview_prep/ingest/` | `pdf.py` → `redact.py` → `resume.py` / `jd.py` |
| `app/Home.py`, `app/pages/1_Setup.py` | Streamlit UI; multipage via the `pages/` folder |

## Things worth noticing

- Streamlit **reruns the whole script** on every interaction. Results survive in `st.session_state`, and
  `st.form` batches inputs so typing doesn't trigger reruns. In Phase 4, interview state moves to the
  LangGraph checkpointer instead.
- Run the app with `LLM_PROVIDER=ollama` and compare against Gemini: the 3B model misses the explicit
  "DataSync Inc." company name and "3+ years". You'll measure this properly in Phase 6 evals.

## Exercises (optional)

1. **Close a redaction gap.** Many resumes start with a title line: `RESUME\nJane Doe | jane@x.com`. Write a
   failing test in `tests/test_redact.py` showing that "Jane" survives today, then fix `_candidate_name` so it
   skips header-word lines and checks the next non-empty line. Keep the existing tests green.
2. **Change extraction behavior through the schema only.** Make `ParsedJD` also extract `work_mode`
   (`remote | hybrid | onsite | null`) by adding a field with a good description. Don't touch the prompt. Then run
   the app and check that the result appears in the JD parse.
