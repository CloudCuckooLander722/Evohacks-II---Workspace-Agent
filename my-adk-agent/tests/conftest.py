"""Shared test fixtures: an in-memory fake Drive and a fake Gemini with canned per-agent output."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import workspace_agent.workspace_agent as wa  # noqa: E402

DOC = {"title": "Related Rates Quiz", "file_type": "DOC", "folder_topic": "Calculus I",
       "answer_key": "1. 4 ft/s", "content": "1. A ladder...", "works_cited": ["Stewart"]}

CANNED = {
    "content_agent": DOC,
    "folder_creation_agent": {"folder_name": "AP Chem"},
    "folder_deletion_agent": {"folder_name": "Old Stuff"},
    "move_agent": {"file_query": "salmon.pdf", "folder_name": "Atlantic Salmon Research Project"},
}

# A distinctive phrase from each agent's instruction, used to tell which agent is calling the model.
INSTRUCTION_KEYS = {
    "You sort a Google": "router_agent",
    "You handle DOCUMENT_CREATION": "content_agent",
    "wants to create": "folder_creation_agent",
    "wants to delete": "folder_deletion_agent",
    "wants to move": "move_agent",
}


def set_router(fake_gemini: dict, category: str, confidence: float = 0.95, missing_info=None) -> None:
    """Choose what the fake router_agent answers."""
    fake_gemini["router_agent"] = {"category": category, "confidence": confidence,
                                   "request": "request", "missing_info": missing_info}


@pytest.fixture
def fake_drive(monkeypatch):
    """Replace the Drive functions workspace_agent uses with an in-memory Drive; returns the action log."""
    log = []
    folders = {"Old Stuff": "f-old", "Calculus I": "f-calc"}
    monkeypatch.setattr(wa, "find_drive_folder",
                        lambda name: {"id": folders.get(name), "name": name, "score": 1.0})

    def create(name):
        folders[name] = f"f-{name}"
        log.append(("mkdir", name))
        return {"id": folders[name], "name": name}
    monkeypatch.setattr(wa, "create_drive_folder", create)
    monkeypatch.setattr(wa, "delete_drive_folder", lambda fid: log.append(("trash", fid)))
    monkeypatch.setattr(wa, "search_drive_files",
                        lambda q: {"files": [{"id": "file1", "name": q}]})
    monkeypatch.setattr(wa, "move_drive_file", lambda fid, folder: log.append(("mv", fid, folder)))
    for kind in list(wa.CREATORS):
        monkeypatch.setitem(wa.CREATORS, kind,
                            lambda title, content, fid, kind=kind: log.append(("create", kind, title, fid)) or {"id": "doc"})
    return log


@pytest.fixture
def fake_gemini(monkeypatch):
    """Patch Gemini so each agent returns its canned JSON; set the router's answer with set_router()."""
    from google.adk.models.google_llm import Gemini
    from google.adk.models.llm_response import LlmResponse
    from google.genai import types

    canned = dict(CANNED)

    async def generate(self, req, stream=False):
        instruction = str(req.config.system_instruction)
        agent = next(name for key, name in INSTRUCTION_KEYS.items() if key in instruction)
        yield LlmResponse(content=types.Content(role="model", parts=[types.Part(text=json.dumps(canned[agent]))]))

    monkeypatch.setenv("GOOGLE_API_KEY", "fake")
    monkeypatch.setattr(Gemini, "generate_content_async", generate)
    return canned
