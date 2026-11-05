# Using the `interview-prep` MCP server from Claude

Our MCP server runs locally over stdio. Any MCP client can start it. It needs Ollama for embeddings, but no API keys:

```bash
ollama serve
```

The first search embeds the question bank (~105 questions, a few seconds). Later starts reuse the index in `.data/`.

## Tools

| Tool | What it does | Changes data? |
|---|---|---|
| `search_question_bank(query, track?, round?, difficulty?, exclude_ids?, k?)` | Semantic search over the curated bank | No |
| `get_question(question_id)` | One question plus what a good answer covers | No |
| `extract_resume_text(pdf_path)` | Resume PDF → text with name, email, phone, links and address redacted | No |
| `index_resume(session_id, text)` | Chunk and embed a resume for a session | Yes |
| `search_resume(session_id, query, k?)` | Most relevant resume excerpts for a topic | No |
| `delete_session(session_id)` | Delete one session and its resume chunks | **Deletes** |
| `delete_all_data(confirm)` | Delete every session and resume chunk (the bank stays) | **Deletes** |

Resource: `question-bank://{track}` returns every question for `shared`, `backend_swe`, `ai_llm_engineer` or
`data_engineer` as JSON.

Each tool carries MCP annotations (`readOnlyHint`, `destructiveHint`), so clients can treat deletes with extra care.

## Claude Code

The repo ships a project-scoped config, [`.mcp.json`](../.mcp.json). Start `claude` in the repo folder, approve
the `interview-prep` server when asked, then check it:

```bash
claude mcp get interview-prep
```

Inside Claude Code, `/mcp` shows the server and its tools.

## Claude Desktop

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (Settings → Developer → Edit Config) and add
an `mcpServers` entry next to the keys already in the file. Claude Desktop doesn't inherit your shell's `PATH`, so use
the absolute path that `which uv` prints, and the absolute path of your checkout:

```json
{
  "mcpServers": {
    "interview-prep": {
      "command": "/absolute/path/to/uv",
      "args": [
        "--directory", "/absolute/path/to/AI-Interview-Prep-Platform",
        "run", "--quiet", "interview-prep-mcp"
      ]
    }
  }
}
```

Quit and reopen Claude Desktop. The server appears under the tools (sliders) icon in a new chat. Server logs are in
`~/Library/Logs/Claude/mcp-server-interview-prep.log`.

## Things to try

- "Find me three senior-level Kafka or messaging questions from my interview bank."
- "Give me a mid-level system design question for a data engineer, and tell me what a good answer covers."
- "Extract the text of `~/Documents/resume.pdf`, index it as session `practice-1`, then show the parts most relevant
  to distributed systems." (Only the redacted text is stored, in this repo's `.data/` folder.)
- "Delete session `practice-1`."

Claude itself is the LLM here: our server only returns data, and Claude decides how to use it. The app does the same
through `langchain-mcp-adapters` (`src/interview_prep/mcp_client.py`).

## Troubleshooting

| Symptom | Fix |
|---|---|
| "Ollama isn't running" | Run `ollama serve`, and `ollama pull nomic-embed-text` once |
| Server fails to start in Claude Desktop | Check the absolute `uv` path and the log file above |
| Tools missing in Claude Code | Approve the project server; `claude mcp reset-project-choices` re-asks |
| Search results feel stale after editing the bank | Nothing to do: the index rebuilds when the bank or embedding model changes |
