"""Exercise the server over the real MCP protocol, using an in-memory client instead of a subprocess."""

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from interview_prep.mcp_server import build_server
from tests.conftest import PERSONA_RESUME

pytestmark = pytest.mark.anyio


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def client(storage):
    async with create_connected_server_and_client_session(build_server(storage)) as session:
        yield session


async def call(client, tool: str, **args):
    result = await client.call_tool(tool, args)
    assert not result.isError, result.content[0].text
    return result.structuredContent


async def call_error(client, tool: str, **args) -> str:
    result = await client.call_tool(tool, args)
    assert result.isError
    return result.content[0].text


async def test_lists_tools_with_safety_annotations(client):
    tools = {t.name: t for t in (await client.list_tools()).tools}
    assert set(tools) == {
        "search_question_bank", "get_question", "extract_resume_text", "index_resume", "search_resume",
        "delete_session", "delete_all_data",
    }
    assert tools["search_question_bank"].annotations.readOnlyHint is True
    assert tools["delete_all_data"].annotations.destructiveHint is True
    # The JSON schema the client (and the LLM) sees includes our field descriptions and limits.
    k = tools["search_question_bank"].inputSchema["properties"]["k"]
    assert k["maximum"] == 20


async def test_search_question_bank(client):
    result = await call(client, "search_question_bank", query="kafka exactly once delivery", track="backend_swe", k=3)
    hits = result["result"]
    assert len(hits) == 3
    assert hits[0]["id"] == "be-007"
    assert {"question", "what_good_looks_like", "score"} <= set(hits[0])


async def test_search_rejects_unknown_track(client):
    assert "Unknown track 'frontend_swe'" in await call_error(
        client, "search_question_bank", query="react", track="frontend_swe"
    )


async def test_get_question(client):
    assert (await call(client, "get_question", question_id="de-006"))["topics"] == ["sql", "gaps and islands"]
    assert "No question" in await call_error(client, "get_question", question_id="zz-999")


async def test_question_bank_resource(client):
    result = await client.read_resource("question-bank://ai_llm_engineer")
    questions = json.loads(result.contents[0].text)
    assert len(questions) >= 25 and all(q["track"] == "ai_llm_engineer" for q in questions)


async def test_extract_resume_text_redacts(client, tmp_path, persona_pdf):
    pdf = tmp_path / "persona.pdf"
    pdf.write_bytes(persona_pdf)
    result = await call(client, "extract_resume_text", pdf_path=str(pdf))
    for pii in ("Priya", "Raman", "priya.raman@example.com", "555-0134", "123 Market Street"):
        assert pii not in result["text"]
    assert result["redactions"]["[EMAIL]"] == 1
    assert "Not a PDF" in await call_error(client, "extract_resume_text", pdf_path=str(tmp_path / "missing.pdf"))


async def test_index_resume_scrubs_contact_info(client, embeddings):
    unredacted = PERSONA_RESUME  # a careless caller skipped extract_resume_text
    result = await call(client, "index_resume", session_id="s1", text=unredacted)
    assert result["chunks"] >= 2 and result["redactions"]["[EMAIL]"] == 1
    assert not any("priya.raman@example.com" in text for text in embeddings.calls)

    hits = (await call(client, "search_resume", session_id="s1", query="kafka rabbitmq"))["result"]
    assert "Kafka" in hits[0]["text"]
    assert "Call index_resume first" in await call_error(client, "search_resume", session_id="nope", query="x")


async def test_delete_tools(client):
    await call(client, "index_resume", session_id="s1", text="Python developer.")
    await call(client, "index_resume", session_id="s2", text="Go developer.")
    assert await call(client, "delete_session", session_id="s1") == {"sessions": 1, "resume_chunks": 1}
    assert "confirm=true" in await call_error(client, "delete_all_data", confirm=False)
    assert await call(client, "delete_all_data", confirm=True) == {"sessions": 1, "resume_chunks": 1}


async def test_explains_when_ollama_is_down(storage):
    class OllamaDown(type(storage.vectors.embeddings)):
        def _embed(self, text):
            raise ConnectionError("Failed to connect to Ollama.")

    storage.vectors.embeddings = OllamaDown()
    async with create_connected_server_and_client_session(build_server(storage)) as client:
        assert "ollama serve" in await call_error(client, "search_question_bank", query="caching")
