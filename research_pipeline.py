"""Transcript-grounded YouTube research workflow."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Sequence, TypedDict

import yt_dlp
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, START, StateGraph

logger = logging.getLogger(__name__)


class ResearchState(TypedDict, total=False):
    """Data that moves through the LangGraph research workflow."""

    topic: str
    limit: int
    videos: List[Dict[str, Any]]
    index: List[Dict[str, Any]]
    brief: str


class YouTubeResearcher:
    """Search, summarize, synthesize, and answer questions from video transcripts.

    The multi-step research operation is orchestrated by LangGraph. Individual
    methods remain public so they can also be safely exposed as MCP tools.
    """

    def __init__(self) -> None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set. Add it to .env or your environment.")
        # Groq retired llama-3.1-8b-instant for free/developer accounts in 2026.
        # Keep this configurable for accounts with different model permissions.
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self.llm = ChatGroq(api_key=api_key, model=self.model, temperature=0.2)
        self.workflow = self._build_workflow()

    def research(self, topic: str, limit: int = 5) -> Dict[str, Any]:
        """Run the LangGraph workflow and return a transcript-grounded brief."""
        result = self.workflow.invoke({"topic": topic, "limit": limit})
        return {"topic": result["topic"], "videos": result["videos"],
                "index": result["index"], "brief": result["brief"]}

    def _build_workflow(self):
        workflow = StateGraph(ResearchState)
        workflow.add_node("collect_sources", self._collect_sources)
        workflow.add_node("build_evidence_index", self._build_evidence_index)
        workflow.add_node("write_brief", self._write_brief)
        workflow.add_edge(START, "collect_sources")
        workflow.add_edge("collect_sources", "build_evidence_index")
        workflow.add_edge("build_evidence_index", "write_brief")
        workflow.add_edge("write_brief", END)
        return workflow.compile()

    def _collect_sources(self, state: ResearchState) -> ResearchState:
        videos: List[Dict[str, Any]] = []
        for position, video in enumerate(self.search_videos(state["topic"], state["limit"]), start=1):
            transcript, error = self.fetch_transcript(video["id"])
            item = {**video, "position": position, "transcript": transcript,
                    "transcript_error": error, "summary": None}
            if transcript:
                item["summary"] = self.summarize(state["topic"], item, transcript)
            videos.append(item)
        return {"videos": videos}

    def _build_evidence_index(self, state: ResearchState) -> ResearchState:
        return {"index": self.build_index(state["videos"])}

    def _write_brief(self, state: ResearchState) -> ResearchState:
        return {"brief": self.synthesize(state["topic"], state["videos"])}

    def search_videos(self, topic: str, limit: int) -> List[Dict[str, str]]:
        options = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
        with yt_dlp.YoutubeDL(options) as downloader:
            results = downloader.extract_info(f"ytsearch{limit}:{topic}", download=False)
        return [{"id": entry["id"], "title": entry.get("title") or "Untitled video",
                 "channel": entry.get("channel") or entry.get("uploader") or "Unknown channel",
                 "url": entry.get("webpage_url") or f"https://www.youtube.com/watch?v={entry['id']}",
                 "duration": entry.get("duration_string") or "Unknown duration"}
                for entry in results.get("entries", []) if entry.get("id")]

    def fetch_transcript(self, video_id: str) -> tuple[str | None, str | None]:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi
            result = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
            snippets = getattr(result, "snippets", result)
            text = " ".join(item.text if hasattr(item, "text") else item.get("text", "") for item in snippets)
            text = re.sub(r"\s+", " ", text).strip()
            return (text, None) if text else (None, "The transcript was empty.")
        except Exception as exc:
            logger.info("Transcript unavailable for %s: %s", video_id, exc)
            return None, f"Transcript unavailable: {exc}"

    def summarize(self, topic: str, video: Dict[str, Any], transcript: str) -> str:
        prompt = f"""Summarize this YouTube transcript for research on {topic}.

Video: {video['title']} — {video['channel']}
Transcript:
{transcript[:18000]}

Give central claims, supporting details, caveats, and useful takeaways. Use only the transcript."""
        return self._complete(prompt, 900)

    def synthesize(self, topic: str, videos: Sequence[Dict[str, Any]]) -> str:
        sources = [f"[Video {v['position']}: {v['title']} | {v['channel']}]\n{v['summary']}"
                   for v in videos if v.get("summary")]
        if not sources:
            return "No usable transcripts were available, so no transcript-grounded brief could be created."
        prompt = f"""Create a comprehensive research brief on {topic} using only the summaries below.
Include an executive summary, key findings, agreement/disagreement, caveats, and open questions.
Cite substantive claims with [Video N].

SOURCE SUMMARIES
{chr(10).join(sources)}"""
        return self._complete(prompt, 1800)

    def answer_question(self, question: str, index: Sequence[Dict[str, Any]]) -> str:
        ranked = sorted(index, key=lambda item: self._relevance(question, item["text"]), reverse=True)[:8]
        if not ranked:
            return "There are no indexed transcripts available for this research session."
        evidence = "\n\n".join(f"[Video {c['position']}: {c['title']}]\n{c['text']}" for c in ranked)
        prompt = f"""Answer using only the transcript excerpts. If they do not support an answer, say so.
Cite claims with [Video N].

Question: {question}

TRANSCRIPT EXCERPTS
{evidence}"""
        return self._complete(prompt, 850)

    @staticmethod
    def build_index(videos: Sequence[Dict[str, Any]], chunk_size: int = 1400) -> List[Dict[str, Any]]:
        return [{"position": video["position"], "title": video["title"],
                 "text": video["transcript"][start:start + chunk_size]}
                for video in videos if video.get("transcript")
                for start in range(0, len(video["transcript"]), chunk_size)]

    @staticmethod
    def _relevance(question: str, text: str) -> int:
        terms = set(re.findall(r"[a-zA-Z0-9]{3,}", question.lower()))
        return sum(text.lower().count(term) for term in terms)

    def _complete(self, prompt: str, max_tokens: int) -> str:
        response = self.llm.invoke([HumanMessage(content=prompt)], max_tokens=max_tokens)
        content = response.content
        return content.strip() if isinstance(content, str) else str(content).strip()
