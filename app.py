"""Standalone Streamlit app for YouTube transcript research."""

import os

import streamlit as st
from dotenv import load_dotenv

from research_pipeline import YouTubeResearcher

load_dotenv()
st.set_page_config(page_title="YouTube Research Assistant", page_icon="🎬", layout="wide")


def main() -> None:
    st.session_state.setdefault("research", None)
    st.session_state.setdefault("messages", [])

    st.title("🎬 YouTube Research Assistant")
    st.write("Search the top five YouTube results, synthesize their transcripts, then ask follow-up questions.")
    with st.sidebar:
        st.header("Workflow")
        st.markdown("1. Search YouTube\n2. Fetch English transcripts\n3. Summarize sources\n4. Write a cited brief\n5. Ask the transcript index")
        st.caption("Videos without usable captions remain listed but are excluded from the evidence base.")
        if st.button("Clear research session", use_container_width=True):
            st.session_state.research = None
            st.session_state.messages = []
            st.rerun()

    topic = st.text_input("Research topic", placeholder="e.g., the science of intermittent fasting")
    if st.button("Build research brief", type="primary", disabled=not topic.strip()):
        if not os.getenv("GROQ_API_KEY"):
            st.error("Set GROQ_API_KEY in .env before starting research.")
        else:
            try:
                with st.spinner("Searching, fetching transcripts, summarizing, and synthesizing..."):
                    st.session_state.research = YouTubeResearcher().research(topic.strip())
                    st.session_state.messages = []
            except Exception as exc:
                st.error(f"The research workflow could not be completed: {exc}")

    research = st.session_state.research
    if not research:
        st.info("Enter a topic to begin a research session.")
        return

    st.header(f"Research brief: {research['topic']}")
    st.markdown(research["brief"])
    st.subheader("Source videos")
    for video in research["videos"]:
        with st.expander(f"{video['position']}. {video['title']} — {video['channel']}"):
            st.markdown(f"[{video['url']}]({video['url']}) · {video['duration']}")
            if video.get("summary"):
                st.markdown(video["summary"])
            else:
                st.warning(video["transcript_error"])

    st.divider()
    st.subheader("Ask the transcript index")
    st.caption("Answers are grounded in retrieved transcript excerpts and cite their source videos.")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if question := st.chat_input("Ask a follow-up question"):
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant transcript excerpts..."):
                try:
                    answer = YouTubeResearcher().answer_question(question, research["index"])
                except Exception as exc:
                    answer = f"I could not answer from the transcript index: {exc}"
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
