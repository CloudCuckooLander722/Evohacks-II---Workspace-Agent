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

import mimetypes  # noqa: E402

from adk_service.service import AgentReply, Attachment, WorkspaceService  # noqa: E402
from tools.workspace_tools import (  # noqa: E402
    complete_google_sign_in,
    exchange_google_sign_in,
    google_sign_in_status,
    google_sign_in_url,
    per_user_sign_in,
)

UPLOAD_TYPES = ["pdf", "xlsx", "docx", "csv", "txt", "md", "json", "png", "jpg", "jpeg"]
# Attachments travel inline in the Gemini request, which is capped at about 20 MB in total.
# (A comment, not a docstring: Streamlit "magic" renders bare string literals on the page.)
MAX_UPLOAD_BYTES = 15 * 1024 * 1024

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
        st.session_state.uploader_key = 0


def record(reply: AgentReply) -> None:
    for text in reply.messages:
        st.session_state.history.append(("assistant", text))
    for err in reply.errors:
        st.session_state.history.append(("assistant", f"⚠️ Something went wrong: {err}"))
    st.session_state.pending = reply.pending
    if not reply.messages and not reply.errors and not reply.pending:
        st.session_state.history.append(("assistant", "Done."))


def to_attachments(files) -> list[Attachment]:
    return [Attachment(name=f.name, data=f.getvalue(),
                       mime_type=f.type or mimetypes.guess_type(f.name)[0] or "application/octet-stream")
            for f in files or []]


def send(service: WorkspaceService, text: str, attachments: list[Attachment] = ()) -> None:
    shown = text + "".join(f"\n\n📎 {a.name}" for a in attachments)
    st.session_state.history.append(("user", shown))
    if sum(len(a.data) for a in attachments) > MAX_UPLOAD_BYTES:
        st.session_state.history.append(("assistant", "⚠️ Attachments are too large; keep them under 15 MB in total."))
        return
    with st.spinner("Working on it…"):
        record(service.send(st.session_state.user_id, st.session_state.session_id, text, attachments,
                            credentials=st.session_state.get("google_credentials")))


def answer(service: WorkspaceService, approved: bool, feedback: str) -> None:
    pending = st.session_state.pending
    st.session_state.history.append(("user", ("Approved" if approved else "Cancelled") + (f": {feedback}" if feedback else "")))
    with st.spinner("Working on it…"):
        record(service.respond(st.session_state.user_id, st.session_state.session_id, pending, approved, feedback,
                               credentials=st.session_state.get("google_credentials")))


def handle_google_redirect() -> None:
    """Google's consent screen redirects back here with ?code=&state= (or ?error=)."""
    params = st.query_params
    if "code" in params and "state" in params:
        try:
            if per_user_sign_in():
                # Public mode: this browser session's own account, kept in memory only.
                st.session_state.google_credentials = exchange_google_sign_in(params["code"], params["state"])
            else:
                complete_google_sign_in(params["code"], params["state"])
            st.success("Signed in to Google. You can use the agent now.")
        except Exception as e:
            st.error(f"Google sign-in failed: {e}")
        st.query_params.clear()
        st.session_state.pop("sign_in_url", None)   # its state is used up; make a fresh one if needed
    elif "error" in params:
        st.error(f"Google sign-in was cancelled or failed: {params['error']}")
        st.query_params.clear()


def google_account_panel() -> None:
    try:
        if per_user_sign_in():
            status = "signed_in" if st.session_state.get("google_credentials") else "signed_out"
        else:
            status = google_sign_in_status()
    except Exception as e:
        st.error(f"Google credentials problem: {e}")
        return
    if status == "signed_out":
        st.warning("Sign in so the agent can use your Google Drive.")
        if "sign_in_url" not in st.session_state:
            st.session_state.sign_in_url = google_sign_in_url()
        st.link_button("Sign in with Google", st.session_state.sign_in_url, type="primary", width="stretch")
    elif status == "signed_in":
        st.caption("✅ Signed in to Google")
        if per_user_sign_in() and st.button("Sign out", width="stretch"):
            for key in ("google_credentials", "sign_in_url"):
                st.session_state.pop(key, None)
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Workspace Agent", page_icon="📁")
    st.title("Workspace Agent")
    st.caption("Create Docs, Slides, PDFs and infographics, and organize your Google Drive folders.")

    handle_google_redirect()
    service = get_service()
    init_state(service)

    with st.sidebar:
        google_account_panel()
        st.subheader("Try")
        for example in EXAMPLES:
            if st.button(example, width="stretch", disabled=st.session_state.pending is not None
                         or (per_user_sign_in() and not st.session_state.get("google_credentials"))):
                send(service, example)
        st.subheader("Attach")
        files = st.file_uploader("Files to send with your next message (e.g. a report to summarize)",
                                 type=UPLOAD_TYPES, accept_multiple_files=True,
                                 key=f"uploader-{st.session_state.uploader_key}")
        if st.button("New conversation", type="secondary"):
            for key in ("user_id", "session_id", "history", "pending", "uploader_key"):
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

    needs_sign_in = per_user_sign_in() and not st.session_state.get("google_credentials")
    prompt = st.chat_input("Sign in with Google (sidebar) to start" if needs_sign_in else "Ask the Workspace Agent…",
                           disabled=pending is not None or needs_sign_in)
    if prompt:
        send(service, prompt, to_attachments(files))
        st.session_state.uploader_key += 1   # clear the picker; the files went with this message
        st.rerun()


main()
