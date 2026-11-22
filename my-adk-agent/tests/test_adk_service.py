"""Tests for adk_service: the Runner-backed WorkspaceService and the Streamlit UI.

Uses the fake Gemini and fake Drive fixtures from conftest.py, so no network or credentials.
"""

import sys
import threading
from pathlib import Path

import pytest
from google.adk.events import Event
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from adk_service.service import (  # noqa: E402
    APP_NAME, AgentReply, Attachment, PendingConfirmation, WorkspaceService, collect_reply,
)
from conftest import set_router  # noqa: E402

APP_FILE = Path(__file__).resolve().parent.parent / "adk_service" / "streamlit_app.py"


# ===========================================================================
# collect_reply
# ===========================================================================

def text_event(author, text, **kw):
    return Event(author=author, content=types.Content(role="model", parts=[types.Part(text=text)]), **kw)


def test_collect_reply_keeps_only_workflow_text():
    events = [
        text_event("router_agent", '{"category": "UNKNOWN"}', output={"category": "UNKNOWN"}),
        text_event(APP_NAME, "Created folder 'AP Chem'."),
    ]
    reply = collect_reply(events)
    assert reply.messages == ["Created folder 'AP Chem'."]
    assert reply.errors == [] and reply.pending is None


def test_collect_reply_detects_confirmation_request():
    call = types.FunctionCall(id="call-1", name="adk_request_input",
                              args={"interruptId": "confirm_document", "message": "Create this document?"})
    event = Event(author=APP_NAME, content=types.Content(role="model", parts=[types.Part(function_call=call)]))
    pending = collect_reply([event]).pending
    assert pending == PendingConfirmation("call-1", "adk_request_input", "confirm_document", "Create this document?")


def test_collect_reply_ignores_agent_tool_calls_and_partials():
    tool_call = types.FunctionCall(id="c", name="search_drive_files", args={"query": "x"})
    events = [
        Event(author="content_agent", content=types.Content(role="model", parts=[types.Part(function_call=tool_call)])),
        text_event(APP_NAME, "half a sente", partial=True),
    ]
    assert collect_reply(events) == AgentReply()


def test_collect_reply_reports_tool_errors():
    event = Event(author=APP_NAME, output={"status": "error", "tool": "move_document", "error": "ConnectionError: offline"})
    assert collect_reply([event]).errors == ["move_document: ConnectionError: offline"]


# ===========================================================================
# WorkspaceService (the Runner)
# ===========================================================================

@pytest.fixture
def service():
    svc = WorkspaceService()
    yield svc
    svc.close()


def test_service_uses_runner_with_workspace_agent(service):
    from google.adk.runners import Runner
    assert isinstance(service.runner, Runner)
    assert service.runner.agent.name == "workspace_agent"
    assert service.runner.app_name == APP_NAME


def test_service_document_creation_confirm_then_create(service, fake_gemini, fake_drive):
    set_router(fake_gemini, "DOCUMENT_CREATION")
    sid = service.new_session("u1")

    reply = service.send("u1", sid, "Make me a related rates quiz")
    assert reply.pending is not None
    assert reply.pending.interrupt_id == "confirm_document"
    assert "1. 4 ft/s" in reply.pending.message          # answer key shown for review
    assert fake_drive == []                              # nothing created before approval

    reply = service.respond("u1", sid, reply.pending, approved=True)
    assert reply.pending is None
    assert ("create", "DOC", "Related Rates Quiz", "f-calc") in fake_drive
    assert any("Created 'Related Rates Quiz'" in m for m in reply.messages)


def test_service_document_creation_cancel(service, fake_gemini, fake_drive):
    set_router(fake_gemini, "DOCUMENT_CREATION")
    sid = service.new_session("u1")
    pending = service.send("u1", sid, "Make me a quiz").pending
    reply = service.respond("u1", sid, pending, approved=False, feedback="make it harder")
    assert fake_drive == []
    assert "Cancelled" in reply.messages[-1] and "make it harder" in reply.messages[-1]


def test_service_folder_deletion_needs_confirmation(service, fake_gemini, fake_drive):
    set_router(fake_gemini, "FOLDER_DELETION")
    sid = service.new_session("u1")
    reply = service.send("u1", sid, "Delete my Old Stuff folder")
    assert reply.pending is not None and "trash" in reply.pending.message
    assert fake_drive == []
    service.respond("u1", sid, reply.pending, approved=True)
    assert fake_drive == [("trash", "f-old")]


@pytest.mark.parametrize("category, expected_log, expected_text", [
    ("FOLDER_CREATION", [("mkdir", "AP Chem")], "Created folder 'AP Chem'"),
    ("DOCUMENT_MOVEMENT", [("mkdir", "Atlantic Salmon Research Project"),
                           ("mv", "file1", "f-Atlantic Salmon Research Project")], "Moved 'salmon.pdf'"),
    ("UNKNOWN", [], "I can create a Doc"),
])
def test_service_single_step_requests(service, fake_gemini, fake_drive, category, expected_log, expected_text):
    set_router(fake_gemini, category)
    reply = service.send("u1", service.new_session("u1"), "request")
    assert reply.pending is None
    assert fake_drive == expected_log
    assert expected_text in reply.messages[-1]


def test_service_surfaces_tool_errors(service, fake_gemini, fake_drive, monkeypatch):
    import workspace_agent.workspace_agent as wa
    monkeypatch.setattr(wa, "search_drive_files", lambda q: {"status": "error", "tool": "search_drive_files",
                                                             "error": "ConnectionError: offline"})
    set_router(fake_gemini, "DOCUMENT_MOVEMENT")
    reply = service.send("u1", service.new_session("u1"), "move it")
    assert reply.errors == ["search_drive_files: ConnectionError: offline"]


def test_service_sends_attachments(service, fake_gemini, fake_drive):
    set_router(fake_gemini, "DOCUMENT_MOVEMENT")
    reply = service.send("u1", service.new_session("u1"), "Move this PDF to Atlantic Salmon Research Project",
                         [Attachment(name="salmon.pdf", data=b"%PDF", mime_type="application/pdf")])
    assert ("upload", "salmon.pdf", "application/pdf", "f-Atlantic Salmon Research Project") in fake_drive
    assert reply.messages[-1].startswith("Uploaded 'salmon.pdf'")


def test_service_sessions_are_independent(service, fake_gemini, fake_drive):
    set_router(fake_gemini, "DOCUMENT_CREATION")
    a, b = service.new_session("ua"), service.new_session("ub")
    pending_a = service.send("ua", a, "quiz").pending
    pending_b = service.send("ub", b, "quiz").pending
    service.respond("ub", b, pending_b, approved=False)
    assert fake_drive == []
    service.respond("ua", a, pending_a, approved=True)
    assert ("create", "DOC", "Related Rates Quiz", "f-calc") in fake_drive


def test_service_handles_concurrent_callers(service, fake_gemini, fake_drive):
    """Streamlit serves each browser tab on its own thread; the service must handle that."""
    set_router(fake_gemini, "UNKNOWN")
    results, errors = [], []

    def worker(i):
        try:
            results.append(service.send(f"u{i}", service.new_session(f"u{i}"), "hi"))
        except Exception as e:  # pragma: no cover - reported via assertion
            errors.append(e)
    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=60)
    assert errors == [] and len(results) == 5
    assert all("I can create a Doc" in r.messages[-1] for r in results)


# ===========================================================================
# Streamlit UI
# ===========================================================================

@pytest.fixture
def app(monkeypatch, fake_gemini, fake_drive):
    import streamlit as st
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("WORKSPACE_AGENT_LOAD_DOTENV", "0")
    st.cache_resource.clear()
    at = AppTest.from_file(str(APP_FILE), default_timeout=60)
    at.run()
    yield at
    st.cache_resource.clear()


def button(at, label):
    return next(b for b in at.button if b.label == label)


def chat_texts(at):
    return [(m.name, " ".join(md.value for md in m.markdown)) for m in at.chat_message]


def test_ui_renders(app):
    assert not app.exception
    assert app.title[0].value == "Workspace Agent"
    assert len(app.chat_input) == 1
    assert app.session_state.session_id


def test_ui_document_creation_approve(app, fake_gemini, fake_drive):
    set_router(fake_gemini, "DOCUMENT_CREATION")
    app.chat_input[0].set_value("Make me a related rates quiz").run()
    assert not app.exception
    assert app.session_state.pending is not None
    assert app.chat_input[0].disabled                 # can't send while a confirmation is open
    assert any("Answer key" in text for _, text in chat_texts(app))
    assert fake_drive == []

    button(app, "Approve").click().run()
    assert not app.exception
    assert app.session_state.pending is None
    assert ("create", "DOC", "Related Rates Quiz", "f-calc") in fake_drive
    texts = chat_texts(app)
    assert texts[0] == ("user", "Make me a related rates quiz")
    assert any("Created 'Related Rates Quiz'" in text for role, text in texts if role == "assistant")


def test_ui_document_creation_cancel_with_feedback(app, fake_gemini, fake_drive):
    set_router(fake_gemini, "DOCUMENT_CREATION")
    app.chat_input[0].set_value("quiz").run()
    app.text_input[0].set_value("too easy").run()
    button(app, "Cancel").click().run()
    assert fake_drive == []
    texts = chat_texts(app)
    assert ("user", "Cancelled: too easy") in texts
    assert "Cancelled; nothing was created" in texts[-1][1]


def test_ui_example_button_and_errors(app, fake_gemini, fake_drive, monkeypatch):
    import workspace_agent.workspace_agent as wa
    monkeypatch.setattr(wa, "search_drive_files", lambda q: {"status": "error", "tool": "search_drive_files",
                                                             "error": "offline"})
    set_router(fake_gemini, "DOCUMENT_MOVEMENT")
    button(app, 'Move salmon.pdf to drive folder "Atlantic Salmon Research Project"').click().run()
    assert not app.exception
    assert any("Something went wrong: search_drive_files: offline" in text for _, text in chat_texts(app))


def test_ui_new_conversation_resets(app, fake_gemini, fake_drive):
    set_router(fake_gemini, "UNKNOWN")
    app.chat_input[0].set_value("hello").run()
    old_session = app.session_state.session_id
    assert chat_texts(app)
    button(app, "New conversation").click().run()
    assert not app.exception
    assert chat_texts(app) == []
    assert app.session_state.session_id != old_session
