"""Test page for our MCP server: every action here is a real MCP tool call over stdio."""

import asyncio
import time
import uuid

import streamlit as st

from interview_prep.mcp_client import get_mcp_client
from interview_prep.question_bank import BANK_TRACKS
from interview_prep.schemas import Round

st.set_page_config(page_title="Question bank · AI Interview Prep", page_icon="🗂️")
st.title("Question bank")
st.caption(
    "A test page for the `interview-prep` MCP server. Each button starts the server as a subprocess and calls one "
    "of its tools over stdio, through langchain-mcp-adapters: the same path the interview agent will use."
)


class ToolError(Exception):
    pass


async def _call(name: str, args: dict):
    async with get_mcp_client().session("interview-prep") as session:
        return await session.call_tool(name, args)


def call_tool(name: str, **args):
    """Call a tool and return its structured result. Shows the raw call under the result."""
    args = {k: v for k, v in args.items() if v not in (None, [], "")}
    started = time.perf_counter()
    result = asyncio.run(_call(name, args))
    elapsed = time.perf_counter() - started
    with st.expander(f"MCP call: `{name}` ({elapsed:.1f}s)"):
        st.json({"tool": name, "arguments": args, "isError": result.isError}, expanded=True)
        st.caption("Response (structuredContent):")
        st.json(result.structuredContent or [c.model_dump() for c in result.content], expanded=False)
    if result.isError:
        raise ToolError(result.content[0].text)
    data = result.structuredContent
    return data["result"] if set(data) == {"result"} else data  # lists come wrapped as {"result": [...]}


@st.cache_data(ttl=600, show_spinner="Asking the server for its tools…")
def list_tools() -> list[dict]:
    async def run():
        async with get_mcp_client().session("interview-prep") as session:
            return (await session.list_tools()).tools

    return [{"name": t.name, "description": t.description} for t in asyncio.run(run())]


with st.expander("Tools this server exposes"):
    for tool in list_tools():
        st.markdown(f"**`{tool['name']}`**: {tool['description']}")

bank_tab, resume_tab = st.tabs(["Search the bank", "Search your resume"])

with bank_tab:
    with st.form("bank"):
        query = st.text_input("Query", placeholder="e.g. kafka delivery guarantees")
        c1, c2, c3 = st.columns(3)
        track = c1.selectbox("Track", [None, *BANK_TRACKS[1:]], format_func=lambda t: t or "Any")
        round_ = c2.selectbox("Round", [None, *Round], format_func=lambda r: r or "Any")
        difficulty = c3.selectbox("Difficulty", [None, 1, 2, 3], format_func=lambda d: str(d) if d else "Any")
        k = st.slider("Results", 1, 10, 5)
        search = st.form_submit_button("Search", type="primary")

    if search and query.strip():
        try:
            hits = call_tool(
                "search_question_bank", query=query, track=track, round=round_, difficulty=difficulty, k=k
            )
        except ToolError as exc:
            st.error(str(exc))
        else:
            if not hits:
                st.info("No questions match those filters.")
            for hit in hits:
                st.markdown(
                    f"**{hit['question']}**  \n`{hit['id']}` · {hit['round']} · difficulty {hit['difficulty']} · "
                    f"relevance {hit['score']:.2f} · _{', '.join(hit['topics'])}_"
                )
                st.caption(f"What good looks like: {hit['what_good_looks_like']}")

with resume_tab:
    if "resume" not in st.session_state:
        st.info("Analyze a resume on the Setup page first; this tab indexes its redacted text.")
        st.page_link("pages/1_Setup.py", label="Go to Setup", icon="➡️")
    else:
        # A throwaway session per browser tab, so this never mixes with real interview sessions.
        session_id = st.session_state.setdefault("scratch_session_id", f"scratch-{uuid.uuid4().hex[:8]}")
        st.write(f"Scratch session `{session_id}`")
        left, right = st.columns(2)
        if left.button("Index my redacted resume"):
            try:
                result = call_tool("index_resume", session_id=session_id, text=st.session_state.resume.redacted.text)
                st.session_state.resume_indexed = True
                st.success(f"Stored {result['chunks']} chunks.")
            except ToolError as exc:
                st.error(str(exc))
        if right.button("Delete this session's data"):
            result = call_tool("delete_session", session_id=session_id)
            st.session_state.resume_indexed = False
            st.success(f"Deleted {result['resume_chunks']} chunks.")

        if st.session_state.get("resume_indexed"):
            with st.form("resume"):
                resume_query = st.text_input("Find resume excerpts about…", placeholder="e.g. message queues")
                find = st.form_submit_button("Search", type="primary")
            if find and resume_query.strip():
                try:
                    for chunk in call_tool("search_resume", session_id=session_id, query=resume_query, k=3):
                        st.markdown(f"**Chunk {chunk['chunk_index']}** · relevance {chunk['score']:.2f}")
                        st.text(chunk["text"])
                except ToolError as exc:
                    st.error(str(exc))
