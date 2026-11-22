"""Streamlit chat UI for workspace_agent.

    streamlit run my-adk-agent/adk_service/streamlit_app.py
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

# Local development: pick up workspace_agent/.env. Real environment variables win (Cloud Run).
# Tests set WORKSPACE_AGENT_LOAD_DOTENV=0 so real secrets never enter the test process.
if os.environ.get("WORKSPACE_AGENT_LOAD_DOTENV", "1") == "1":
    load_dotenv(PROJECT_ROOT / "workspace_agent" / ".env", override=False)

from adk_service.service import AgentReply, WorkspaceService  # noqa: E402

EXAMPLES = [
    "I want a quiz on curveball related rates problems that test my concepts",
    "Summarize my Q3 report into a slidedeck",
    'Move salmon.pdf to drive folder "Atlantic Salmon Research Project"',
]


@st.cache_resource
def get_service() -> WorkspaceService:
    """One Runner per server process, shared by every browser session."""
    return WorkspaceService()


def init_state(service: WorkspaceService) -> None:
    if "user_id" not in st.session_state:
        st.session_state.user_id = f"user-{uuid.uuid4().hex[:12]}"
        st.session_state.session_id = service.new_session(st.session_state.user_id)
        st.session_state.history = []   # list of (role, text)
        st.session_state.pending = None


def record(reply: AgentReply) -> None:
    for text in reply.messages:
        st.session_state.history.append(("assistant", text))
    for err in reply.errors:
        st.session_state.history.append(("assistant", f"⚠️ Something went wrong: {err}"))
    st.session_state.pending = reply.pending
    if not reply.messages and not reply.errors and not reply.pending:
        st.session_state.history.append(("assistant", "Done."))


def send(service: WorkspaceService, text: str) -> None:
    st.session_state.history.append(("user", text))
    with st.spinner("Working on it…"):
        record(service.send(st.session_state.user_id, st.session_state.session_id, text))


def answer(service: WorkspaceService, approved: bool, feedback: str) -> None:
    pending = st.session_state.pending
    st.session_state.history.append(("user", ("Approved" if approved else "Cancelled") + (f": {feedback}" if feedback else "")))
    with st.spinner("Working on it…"):
        record(service.respond(st.session_state.user_id, st.session_state.session_id, pending, approved, feedback))


def main() -> None:
    st.set_page_config(page_title="Workspace Agent", page_icon="📁")
    st.title("Workspace Agent")
    st.caption("Create Docs, Slides, PDFs and infographics, and organize your Google Drive folders.")

    service = get_service()
    init_state(service)

    with st.sidebar:
        st.subheader("Try")
        for example in EXAMPLES:
            if st.button(example, width="stretch", disabled=st.session_state.pending is not None):
                send(service, example)
        if st.button("New conversation", type="secondary"):
            for key in ("user_id", "session_id", "history", "pending"):
                st.session_state.pop(key, None)
            st.rerun()

    for role, text in st.session_state.history:
        with st.chat_message(role):
            st.markdown(text)

    pending = st.session_state.pending
    if pending is not None:
        with st.chat_message("assistant"):
            st.markdown(pending.message)
            feedback = st.text_input("Feedback (optional)", key="feedback")
            approve_col, cancel_col = st.columns(2)
            if approve_col.button("Approve", type="primary", width="stretch"):
                answer(service, True, feedback)
                st.rerun()
            if cancel_col.button("Cancel", width="stretch"):
                answer(service, False, feedback)
                st.rerun()

    prompt = st.chat_input("Ask the Workspace Agent…", disabled=pending is not None)
    if prompt:
        send(service, prompt)
        st.rerun()


main()
