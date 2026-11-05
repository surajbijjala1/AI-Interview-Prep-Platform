"""Connections to MCP servers for langchain-mcp-adapters. Phase 3 adds Tavily, Fetch and DuckDuckGo here."""

import os
import sys

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.sessions import StdioConnection

from interview_prep.config import ROOT

# The stdio client passes only a minimal environment (PATH, HOME, ...) to servers, so forward our settings.
# The server also reads .env itself.
_FORWARDED_ENV_PREFIXES = ("OLLAMA_", "EMBEDDING_", "DATA_DIR")


def interview_prep_connection() -> StdioConnection:
    """Our server, run as a subprocess with this same Python environment."""
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-m", "interview_prep.mcp_server"],
        "cwd": str(ROOT),
        "env": {k: v for k, v in os.environ.items() if k.startswith(_FORWARDED_ENV_PREFIXES)},
    }


def get_mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient({"interview-prep": interview_prep_connection()})
