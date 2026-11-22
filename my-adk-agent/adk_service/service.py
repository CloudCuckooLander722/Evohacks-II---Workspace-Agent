"""Runs workspace_agent through an ADK Runner and turns its events into simple replies.

WorkspaceService owns one Runner and a background event loop, so synchronous callers (like the
Streamlit UI, which reruns its script on every interaction) can use it with plain function calls:

    service = WorkspaceService()
    session_id = service.new_session("user-1")
    reply = service.send("user-1", session_id, "Create a quiz on related rates")
    if reply.pending:                       # the workflow is waiting for confirmation
        reply = service.respond("user-1", session_id, reply.pending, approved=True)
"""

from __future__ import annotations

import asyncio
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from google.adk.runners import Runner  # noqa: E402
from google.adk.sessions import BaseSessionService, InMemorySessionService  # noqa: E402
from google.genai import types  # noqa: E402

from tools.workspace_tools import use_credentials  # noqa: E402
from workspace_agent import root_agent  # noqa: E402

APP_NAME = "workspace_agent"
REQUEST_INPUT_CALL = "adk_request_input"


@dataclass
class Attachment:
    """A file the user attached to a message, sent to the agent as inline bytes."""
    name: str
    data: bytes
    mime_type: str


@dataclass
class PendingConfirmation:
    """The workflow paused and asked the user something (e.g. 'Create this document?')."""
    call_id: str
    name: str
    interrupt_id: str
    message: str


@dataclass
class AgentReply:
    messages: list[str] = field(default_factory=list)
    """User-facing text from the workflow, in order."""
    errors: list[str] = field(default_factory=list)
    """Tool failures reported by workflow nodes."""
    pending: Optional[PendingConfirmation] = None
    """Set when the workflow is waiting for the user's confirmation."""


def collect_reply(events: list[Any], root_name: str = APP_NAME) -> AgentReply:
    """Turn a run's events into an AgentReply.

    - Text from the root workflow (its function nodes) is user-facing; text from sub-agents is
      their structured JSON output, which is skipped.
    - An adk_request_input function call means the workflow paused for confirmation.
    - An output dict with status 'error' is a tool failure.
    """
    reply = AgentReply()
    for event in events:
        if getattr(event, "partial", False):
            continue
        parts = event.content.parts if event.content and event.content.parts else []
        for part in parts:
            call = part.function_call
            if call and call.name == REQUEST_INPUT_CALL:
                args = call.args or {}
                reply.pending = PendingConfirmation(
                    call_id=call.id,
                    name=call.name,
                    interrupt_id=args.get("interruptId") or args.get("interrupt_id") or "",
                    message=args.get("message") or "",
                )
            elif part.text and event.author == root_name and not part.thought:
                reply.messages.append(part.text)
        output = getattr(event, "output", None)
        if isinstance(output, dict) and output.get("status") == "error":
            reply.errors.append(f"{output.get('tool', 'tool')}: {output.get('error', 'unknown error')}")
    return reply


class WorkspaceService:
    """Thread-safe synchronous facade over an ADK Runner for workspace_agent."""

    def __init__(self, agent=root_agent, app_name: str = APP_NAME,
                 session_service: Optional[BaseSessionService] = None):
        self.app_name = app_name
        self.root_name = agent.name
        self.runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=session_service or InMemorySessionService(),
        )
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name="adk-runner", daemon=True)
        self._thread.start()

    # -- async API ----------------------------------------------------------

    async def anew_session(self, user_id: str) -> str:
        session = await self.runner.session_service.create_session(app_name=self.app_name, user_id=user_id)
        return session.id

    async def _arun(self, user_id: str, session_id: str, message: types.Content,
                    credentials: Any = None) -> AgentReply:
        if credentials is None:
            events = [e async for e in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=message)]
        else:
            # The context var is set inside this task, so every task the run starts inherits it
            # and concurrent users' runs each see only their own credentials.
            with use_credentials(credentials):
                events = [e async for e in self.runner.run_async(user_id=user_id, session_id=session_id, new_message=message)]
        return collect_reply(events, self.root_name)

    async def asend(self, user_id: str, session_id: str, text: str,
                    attachments: Sequence[Attachment] = (), credentials: Any = None) -> AgentReply:
        parts = [types.Part(text=text)] + [
            types.Part(inline_data=types.Blob(data=a.data, mime_type=a.mime_type, display_name=a.name))
            for a in attachments
        ]
        return await self._arun(user_id, session_id, types.Content(role="user", parts=parts), credentials)

    async def arespond(self, user_id: str, session_id: str, pending: PendingConfirmation,
                       approved: bool, feedback: Optional[str] = None, credentials: Any = None) -> AgentReply:
        response = types.FunctionResponse(
            id=pending.call_id, name=pending.name, response={"approved": approved, "feedback": feedback or None}
        )
        return await self._arun(user_id, session_id, types.Content(role="user", parts=[types.Part(function_response=response)]),
                                credentials)

    # -- sync API (runs on the service's own event loop) ----------------------

    def _call(self, coro, timeout: Optional[float] = 600):
        return asyncio.run_coroutine_threadsafe(coro, self._loop).result(timeout)

    def new_session(self, user_id: str) -> str:
        return self._call(self.anew_session(user_id))

    def send(self, user_id: str, session_id: str, text: str,
             attachments: Sequence[Attachment] = (), credentials: Any = None) -> AgentReply:
        return self._call(self.asend(user_id, session_id, text, attachments, credentials))

    def respond(self, user_id: str, session_id: str, pending: PendingConfirmation,
                approved: bool, feedback: Optional[str] = None, credentials: Any = None) -> AgentReply:
        return self._call(self.arespond(user_id, session_id, pending, approved, feedback, credentials))

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
