# Phase 2: Custom MCP server + RAG

**What you can do now:** search a curated bank of 105 interview questions by meaning, from the app's Question bank
page or from Claude Desktop / Claude Code. You can also index a redacted resume and search it. Everything runs
locally, at no cost.

## Concepts

### 1. MCP: one server, many clients
An MCP server exposes **tools** (functions with JSON-schema inputs), **resources** (readable data addressed by URI)
and **prompts**. Clients discover them at runtime (`list_tools`) and call them by name. Our server
(`mcp_server/server.py`) is written once and used by three clients: the Streamlit page, Claude Desktop and Claude
Code. None of them import our Python code. They only speak the protocol.

- **FastMCP turns type hints into the protocol.** `Annotated[int, Field(ge=1, le=20)]` becomes
  `{"type": "integer", "minimum": 1, "maximum": 20}` in the tool's input schema, which is what the client and the
  LLM see. The docstring becomes the tool description. Return types (`list[QuestionHit]`) become `structuredContent`.
- **stdio transport:** the client starts the server as a subprocess and talks JSON-RPC over stdin/stdout. So
  **nothing in the server may `print()`**: a stray print corrupts the protocol stream. Logs go to stderr.
- **Annotations** (`readOnlyHint`, `destructiveHint`) tell clients how careful to be. `delete_all_data` also requires
  `confirm=true`, so a model can't wipe data with a single ambiguous call.
- **The server never calls an LLM.** It owns data and retrieval, and the client's model does the reasoning. That's why
  it needs no API keys, and why Claude Desktop can use it with Claude as the brain.

### 2. RAG building blocks
`storage/vectors.py` is the retrieval half of RAG. Generation arrives in Phase 4, when the interviewer turns hits into
questions.

- **Embeddings** map text to vectors whose closeness tracks meaning. We use `nomic-embed-text` on Ollama, which is
  local, free and private.
- **Task prefixes matter.** nomic was trained with `search_document:` and `search_query:` prefixes
  (`embeddings.py`). In the resume check, removing them took correct top hits from 2 of 5 queries to 1 of 5
  ("mentoring" started missing).
- **Metadata filters + similarity:** each question is stored with `track`, `round` and `difficulty`, so a search is
  "nearest vectors *among* backend or shared, technical, difficulty 2". Chroma applies the `where` filter before
  ranking. `exclude_ids` is how Phase 4 avoids repeating questions.
- **Cosine distance:** collections use `{"hnsw": {"space": "cosine"}}`, so relevance = 1 − distance, in 0..1.
- **What gets embedded is a design choice.** `embedding_text()` is the question plus its topic tags. Adding
  the "what good looks like" text didn't improve the retrieval check, so the simpler version stays.

### 3. Keeping an index honest
Vectors from different models live in different spaces, so mixing them gives silent garbage. Each Chroma collection
stores `embedding_model` and, for the bank, a `fingerprint` (a hash of every question plus
`EMBEDDING_TEXT_VERSION`). On open, a mismatch means drop and rebuild (`VectorStore._store`). Edit a question and
the index fixes itself on the next search. Switch `EMBEDDING_MODEL` and it re-embeds.

### 4. Chunking is where resume RAG wins or loses
The first version split the persona resume into 2 chunks of 600 characters. The first chunk (contact header +
summary + job) won *every* query. The fixes, in `chunk_resume()`:

1. PDF extraction turns blank lines into `"\n \n"`, which hid section breaks from the splitter.
2. Contact-header lines (`[NAME] | [EMAIL] | …`) carry no meaning, so they're dropped.
3. Split at section headings first. A plain `RecursiveCharacterTextSplitter` packs small neighbors together
   (SKILLS + EDUCATION), and that bigger chunk then wins unrelated queries.

Even then, a one-line EDUCATION chunk doesn't match the query "education" well: pure vector search struggles with very
short, keyword-like text. That's the textbook case for **hybrid search** (exercise 2).

### 5. Measure retrieval instead of eyeballing it
`evals/bank_retrieval.py` runs 12 queries with the IDs a human would accept and prints hit@1 and hit@3. It caught a
real content gap: "a time I disagreed with my manager" missed until `gen-016` was tagged `disagreement`. Score:
11/12 hit@1, 12/12 hit@3. Re-run it whenever you edit the bank or `embedding_text()`.

### 6. Schema migrations
SQLite stores its schema version in `PRAGMA user_version`. `MIGRATIONS` in `storage/db.py` is an append-only list,
and on open, every migration past the stored version runs in its own transaction. Phase 2 has one table, `sessions`.
Later phases append migrations for research cache, turns and reports, and nobody deletes their `.data/` folder.

## Code map

| File | Role |
|---|---|
| `data/question_bank/*.json` | The bank: 30 shared (screen + behavioral) + 25 per track. Reviewed in this PR |
| `src/interview_prep/question_bank.py` | Load + validate, `embedding_text()`, `fingerprint()` |
| `src/interview_prep/embeddings.py` | Ollama embeddings with nomic task prefixes |
| `src/interview_prep/storage/db.py` | SQLite + migrations |
| `src/interview_prep/storage/vectors.py` | Chroma collections, bank sync/search, resume chunk/index/search, deletion |
| `src/interview_prep/mcp_server/server.py` | The MCP server: tools, resource, annotations, error messages |
| `src/interview_prep/mcp_client.py` | How the app connects to MCP servers (Phase 3 adds web research servers) |
| `app/pages/6_Question_Bank.py` | Test page: every action is a real MCP call; expand "MCP call" to see the raw exchange |
| `.mcp.json`, `docs/mcp-clients.md` | Claude Code / Claude Desktop setup |
| `tests/test_mcp_server.py` | Protocol-level tests with an in-memory client; `KeywordEmbeddings` stands in for Ollama |

## Things worth noticing

- On the test page, each call takes a few seconds (up to ~7 s cold), mostly starting the server subprocess, because
  the page opens a fresh MCP session per click. Phase 3 keeps sessions open for the whole run. That's the Streamlit + async spike in the
  spec's risk table.
- `index_resume` re-scrubs contact info even though callers should send redacted text (defense in depth), but it
  deliberately doesn't re-run *name* detection. Read `scrub_contact_info` to see why.
- The tests never touch Ollama. `KeywordEmbeddings` (bag of words) is a bad embedding but a deterministic one,
  enough to test filters, ranking plumbing and the MCP contract in CI.

## Exercises (optional)

1. **Add an MCP prompt.** FastMCP supports `@mcp.prompt()`. Add a prompt `practice_question(track, topic)` that
   returns instructions like "Use search_question_bank to pick one {track} question about {topic}, ask it, then grade
   my answer against what_good_looks_like." Then try it from Claude Desktop. Test
   it with `client.list_prompts()` in `tests/test_mcp_server.py`.
2. **Hybrid resume search.** Add BM25 (for example `rank_bm25`, or LangChain's `BM25Retriever`) over the session's
   chunks, and merge with the vector results using reciprocal rank fusion: `score = Σ 1/(60 + rank)`. Write a test
   where the query "education" must return the EDUCATION chunk for the persona. Does it also fix "cloud and container
   tools" → SKILLS?
