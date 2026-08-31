# YouTube Research Assistant

A standalone Streamlit project that turns a topic into a transcript-grounded YouTube research brief.

The research workflow is implemented with LangGraph, uses LangChain's Groq chat
model integration, and is also available to other AI clients as MCP tools.

## What it does

1. Searches YouTube for the top five results.
2. Fetches available English transcripts.
3. Summarizes each usable source with Groq.
4. Synthesizes a comprehensive brief with `[Video N]` citations.
5. Answers follow-up questions using retrieved chunks from the combined transcript index.

Videos with captions that are disabled, restricted, or unavailable are shown in the source list but are excluded from the evidence base.

## Setup and run

```bash
cd youtube_research_assistant
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set `GROQ_API_KEY` in `.env`, then run:

```bash
streamlit run app.py
```

The default model is `openai/gpt-oss-20b`. If your `.env` was created from an
older version of this project, replace
`GROQ_MODEL=llama-3.1-8b-instant` with `GROQ_MODEL=openai/gpt-oss-20b` (or
another model enabled for your Groq account).

The project does not download video media. It uses `yt-dlp` for search and `youtube-transcript-api` for captions.

## MCP server

The repository exposes three MCP tools: `search_youtube`, `get_transcript`, and
`create_research_brief`. They are backed by the same LangGraph workflow as the
Streamlit app.

This server uses the MCP **stdio** transport. Do not run `python mcp_server.py`
in a terminal or type into it: the server accepts only JSON-RPC messages from an
MCP client. Instead, add this configuration to your MCP client; it will launch
the server as a subprocess and connect to it automatically:

```json
{
  "mcpServers": {
    "youtube-research": {
      "command": "python",
      "args": ["/absolute/path/to/youtube_research_assistant/mcp_server.py"]
    }
  }
}
```
