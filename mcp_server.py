"""MCP server for using YouTube research from any MCP-compatible client.

Run with: python mcp_server.py
The server uses stdio, which is the transport expected by most local MCP clients.
"""

import sys
from typing import Any

from mcp.server.fastmcp import FastMCP

from research_pipeline import YouTubeResearcher

mcp = FastMCP("youtube-research-assistant")


def _researcher() -> YouTubeResearcher:
    """Create the shared research service only when a tool is invoked."""
    return YouTubeResearcher()


@mcp.tool()
def search_youtube(topic: str, limit: int = 5) -> list[dict[str, str]]:
    """Find YouTube videos relevant to a research topic (maximum 10)."""
    return _researcher().search_videos(topic, max(1, min(limit, 10)))


@mcp.tool()
def get_transcript(video_id: str) -> dict[str, str | None]:
    """Fetch an English YouTube transcript by video ID, when captions are available."""
    transcript, error = _researcher().fetch_transcript(video_id)
    return {"transcript": transcript, "error": error}


@mcp.tool()
def create_research_brief(topic: str, limit: int = 5) -> dict[str, Any]:
    """Create a cited, transcript-grounded YouTube research brief using LangGraph."""
    result = _researcher().research(topic, max(1, min(limit, 10)))
    # Large transcript indexes are useful in the Streamlit app but unnecessarily
    # inflate MCP responses; the source summaries and brief remain available here.
    return {key: value for key, value in result.items() if key != "index"}


if __name__ == "__main__":
    if sys.stdin.isatty():
        print(
            "This is an MCP stdio server, not an interactive command. "
            "Add it to an MCP client's configuration; the client will launch it.",
            file=sys.stderr,
        )
        raise SystemExit(0)
    mcp.run(transport="stdio")
