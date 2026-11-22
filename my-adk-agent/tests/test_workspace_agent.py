"""Offline tests for workspace_agent, tools, and agents.

No network or credentials needed: Google API clients are replaced with mocks, and Gemini is
replaced with a fake model that returns canned structured output per agent.

Run from the repo root:
    
"""

import asyncio
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.tool_errors import report_tool_error  # noqa: E402
import tools.workspace_tools as t  # noqa: E402
import workspace_agent.workspace_agent as wa  # noqa: E402
from conftest import DOC, set_router as router  # noqa: E402


# ===========================================================================
# agents/tool_errors.py
# ===========================================================================

def test_report_tool_error_returns_error_payload():
    result = report_tool_error("my_tool", ValueError("boom"))
    assert result == {"status": "error", "tool": "my_tool", "error": "ValueError: boom"}


# ===========================================================================
# tools/workspace_tools.py — pure helpers
# ===========================================================================

def test_query_escaping():
    assert t._q("Bob's \\ file") == "Bob\\'s \\\\ file"


def test_sections_split_on_headings_and_rules():
    md = "# Intro\nSome text\n- **a**\n\n## Results\n1. x\n---\nloose"
    sections = t._sections(md)
    assert [h for h, _ in sections] == ["Intro", "Results", ""]
    assert sections[0][1][:2] == ["Some text", "- **a**"]
    assert sections[2][1] == ["loose"]


def test_sections_without_headings_is_single_section():
    assert t._sections("just text\nmore") == [("", ["just text", "more"])]
    assert t._sections("") == []


def test_clean_body_strips_markdown():
    assert t._clean_body(["- **bold** item", "2. `code`", "", "  plain  "]) == "bold item\ncode\nplain"


@pytest.mark.parametrize("a, b, minimum", [
    ("Atlantic Salmon Research Project", "atlantic salmon research project", 1.0),
    ("Calculus 2", "Calculus II", 0.75),
    ("AP Chem", "AP Chemistry", 0.75),          # abbreviation
    ("Calculus II Quizzes", "Calculus II", 0.75),  # containment
])
def test_similarity_matches(a, b, minimum):
    assert t._similarity(a, b) >= minimum


@pytest.mark.parametrize("a, b", [("Biology", "AP Chemistry"), ("History", "Calculus II"), ("", "x")])
def test_similarity_non_matches(a, b):
    assert t._similarity(a, b) < t.FOLDER_MATCH_THRESHOLD


FOLDERS = [
    {"id": "1", "name": "Atlantic Salmon Research Project"},
    {"id": "2", "name": "Calculus II"},
    {"id": "3", "name": "AP Chemistry"},
]


def test_rank_folders_orders_best_first():
    ranked = t.rank_folders("calculus 2", FOLDERS)
    assert ranked[0]["id"] == "2"
    assert ranked == sorted(ranked, key=lambda f: f["score"], reverse=True)
    assert all(0 <= f["score"] <= 1 for f in ranked)


def test_find_drive_folder_match_and_no_match(monkeypatch):
    monkeypatch.setattr(t, "_list_folders", lambda: FOLDERS)
    assert t.find_drive_folder("Atlantic Salmon Research Project")["id"] == "1"
    miss = t.find_drive_folder("Medieval History")
    assert miss["id"] is None
    assert len(miss["candidates"]) == 3


# ===========================================================================
# tools/workspace_tools.py — Google API calls (mocked)
# ===========================================================================

@pytest.fixture
def drive(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(t, "_drive", lambda: mock)
    return mock


@pytest.fixture
def slides(monkeypatch):
    mock = MagicMock()
    monkeypatch.setattr(t, "_slides", lambda: mock)
    return mock


@pytest.fixture
def fresh_credentials():
    t._credentials.cache_clear()
    yield
    t._credentials.cache_clear()


def write_json(tmp_path, name, data):
    path = tmp_path / name
    path.write_text(json.dumps(data))
    return path


@pytest.mark.parametrize("data, kind", [
    ({"type": "service_account", "client_email": "x"}, "service_account"),
    ({"type": "authorized_user", "refresh_token": "r"}, "authorized_user"),
    ({"installed": {"client_id": "c"}}, "oauth_client"),
    ({"web": {"client_id": "c"}}, "oauth_client"),
])
def test_credentials_kind(tmp_path, data, kind):
    assert t.credentials_kind(write_json(tmp_path, "c.json", data)) == kind


def test_credentials_kind_rejects_other_json(tmp_path):
    with pytest.raises(ValueError):
        t.credentials_kind(write_json(tmp_path, "c.json", {"api_key": "x"}))


def test_credentials_without_config_or_adc_raise(monkeypatch, fresh_credentials):
    import google.auth
    from google.auth.exceptions import DefaultCredentialsError

    def no_adc(scopes=None):
        raise DefaultCredentialsError("none")
    monkeypatch.delenv("GOOGLE_CREDENTIALS_PATH", raising=False)
    monkeypatch.setattr(google.auth, "default", no_adc)
    with pytest.raises(RuntimeError, match="No Google credentials"):
        t._credentials()


def test_credentials_fall_back_to_adc(monkeypatch, fresh_credentials):
    import google.auth
    seen = {}
    sentinel = object()

    def adc(scopes=None):
        seen["scopes"] = scopes
        return sentinel, "project"
    monkeypatch.delenv("GOOGLE_CREDENTIALS_PATH", raising=False)
    monkeypatch.setattr(google.auth, "default", adc)
    assert t._credentials() is sentinel
    assert seen["scopes"] == t.SCOPES


def test_credentials_authorized_user_file(tmp_path, monkeypatch, fresh_credentials):
    path = write_json(tmp_path, "token.json", {
        "type": "authorized_user", "client_id": "c", "client_secret": "s", "refresh_token": "r"})
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", str(path))
    creds = t._credentials()
    assert creds.refresh_token == "r"
    assert set(creds.scopes) == set(t.SCOPES)


def test_credentials_oauth_client_runs_consent_flow(tmp_path, monkeypatch, fresh_credentials):
    path = write_json(tmp_path, "client.json", {"installed": {"client_id": "c"}})
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", str(path))
    called = []
    monkeypatch.setattr(t, "_oauth_user_credentials", lambda p: called.append(p) or "user-creds")
    assert t._credentials() == "user-creds"
    assert called == [str(path)]


WEB_CLIENT = {"web": {"client_id": "c", "client_secret": "s", "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                      "token_uri": "https://oauth2.googleapis.com/token", "redirect_uris": ["https://app.example/"]}}


def test_credentials_stale_path_says_so(monkeypatch, fresh_credentials):
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", "/nowhere/credentials.json")
    with pytest.raises(FileNotFoundError, match="GOOGLE_CREDENTIALS_PATH is /nowhere"):
        t._credentials()


def test_web_client_needs_sign_in_then_uses_token(tmp_path, monkeypatch, fresh_credentials):
    path = write_json(tmp_path, "credentials.json", WEB_CLIENT)
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", str(path))
    monkeypatch.delenv("GOOGLE_OAUTH_TOKEN_FILE", raising=False)
    assert t.google_sign_in_status() == "signed_out"
    with pytest.raises(t.GoogleSignInRequired):
        t._credentials()

    # The consent URL uses the client's registered redirect and remembers the PKCE verifier.
    from urllib.parse import parse_qs, urlparse
    query = parse_qs(urlparse(t.google_sign_in_url()).query)
    assert query["redirect_uri"] == ["https://app.example/"]
    state = query["state"][0]
    assert state in t._PENDING_SIGN_INS

    # Finishing the sign-in stores the token next to the client file.
    from google_auth_oauthlib.flow import Flow
    def fake_fetch(self, **kw):
        assert kw == {"code": "abc"} and self.code_verifier
        self.oauth2session.token = {"access_token": "a", "refresh_token": "r", "token_type": "Bearer",
                                    "expires_in": 3600, "expires_at": 4102444800, "scope": t.SCOPES}
    monkeypatch.setattr(Flow, "fetch_token", fake_fetch)
    t.complete_google_sign_in("abc", state)
    assert (tmp_path / "token.json").exists() and state not in t._PENDING_SIGN_INS
    assert t.google_sign_in_status() == "signed_in"
    assert t._credentials().refresh_token == "r"

    with pytest.raises(ValueError, match="expired"):
        t.complete_google_sign_in("abc", "unknown-state")


def test_sign_in_not_needed_without_web_client(tmp_path, monkeypatch):
    monkeypatch.delenv("GOOGLE_CREDENTIALS_PATH", raising=False)
    assert t.google_sign_in_status() == "not_needed"
    monkeypatch.setenv("GOOGLE_CREDENTIALS_PATH", str(write_json(tmp_path, "c.json", {"installed": {"client_id": "c"}})))
    assert t.google_sign_in_status() == "not_needed"


def test_request_shapes_match_google_discovery_docs():
    """Build (without sending) every request the module makes; bad method/param names raise TypeError."""
    from google.auth.credentials import AnonymousCredentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload

    d = build("drive", "v3", credentials=AnonymousCredentials(), cache_discovery=False)
    s = build("slides", "v1", credentials=AnonymousCredentials(), cache_discovery=False)
    media = MediaIoBaseUpload(io.BytesIO(b"x"), mimetype="text/markdown")
    d.files().list(q="x", fields="f", pageSize=10, pageToken=None, supportsAllDrives=True,
                   includeItemsFromAllDrives=True)
    d.files().create(body={}, media_body=media, fields="id", supportsAllDrives=True)
    d.files().get(fileId="a", fields="id", supportsAllDrives=True)
    d.files().update(fileId="a", body={"trashed": True}, fields="id", supportsAllDrives=True)
    d.files().update(fileId="a", addParents="b", removeParents="c", fields="id", supportsAllDrives=True)
    d.files().delete(fileId="a", supportsAllDrives=True)
    d.files().export_media(fileId="a", mimeType="application/pdf")
    d.files().get_media(fileId="a", supportsAllDrives=True)
    s.presentations().create(body={})
    s.presentations().batchUpdate(presentationId="p", body={})
    s.presentations().pages().getThumbnail(presentationId="p", pageObjectId="g",
                                           thumbnailProperties_mimeType="PNG",
                                           thumbnailProperties_thumbnailSize="LARGE")


def test_list_files_paginates(drive):
    drive.files().list().execute.side_effect = [
        {"files": [{"id": "a"}], "nextPageToken": "tok"},
        {"files": [{"id": "b"}]},
    ]
    assert [f["id"] for f in t._list_files("q")] == ["a", "b"]


def test_list_drive_folders_success_and_error(monkeypatch):
    seen = {}

    def list_files(q, fields=None):
        seen["fields"] = fields
        return [{"id": "1", "name": "A", "capabilities": {"canAddChildren": True}},
                {"id": "2", "name": "Shared read-only", "capabilities": {"canAddChildren": False}},
                {"id": "3", "name": "No capabilities returned"}]
    monkeypatch.setattr(t, "_list_files", list_files)
    # Only folders the user can add files to (read-only shared folders cause 403 on upload).
    assert t.list_drive_folders() == {"folders": [{"id": "1", "name": "A"}]}
    assert "capabilities/canAddChildren" in seen["fields"]

    def fail(q, fields=None):
        raise ConnectionError("offline")
    monkeypatch.setattr(t, "_list_files", fail)
    assert t.list_drive_folders()["status"] == "error"


def test_create_drive_folder(drive):
    drive.files().create().execute.return_value = {"id": "f1", "name": "AP Chem"}
    assert t.create_drive_folder("AP Chem", parent_id="root") == {"id": "f1", "name": "AP Chem"}
    body = drive.files().create.call_args.kwargs["body"]
    assert body == {"name": "AP Chem", "mimeType": t.FOLDER_MIME, "parents": ["root"]}


def test_delete_drive_folder_trashes_instead_of_deleting(drive):
    drive.files().update().execute.return_value = {"id": "f1", "name": "Old", "trashed": True}
    assert t.delete_drive_folder("f1")["trashed"] is True
    assert drive.files().update.call_args.kwargs["body"] == {"trashed": True}
    drive.files().delete.assert_not_called()


def test_move_drive_file_replaces_parents(drive):
    drive.files().get().execute.return_value = {"parents": ["old", "dest"]}
    t.move_drive_file("file1", "dest")
    kwargs = drive.files().update.call_args.kwargs
    assert kwargs["addParents"] == "dest"
    assert kwargs["removeParents"] == "old"


def test_search_drive_files_falls_back_in_order(monkeypatch):
    queries = []

    def fake_list(q, limit=None):
        queries.append(q)
        return [{"id": "x", "name": "salmon report.pdf"}] if "fullText" in q else []
    monkeypatch.setattr(t, "_list_files", fake_list)

    result = t.search_drive_files("salmon report.pdf")
    assert result["files"][0]["id"] == "x"
    assert "name='salmon report.pdf'" in queries[0]
    assert "name contains 'salmon report.pdf'" in queries[1]
    assert "name contains 'salmon' and name contains 'report'" in queries[2]
    assert "fullText contains" in queries[3]
    assert all(f"mimeType!='{t.FOLDER_MIME}'" in q for q in queries)


def test_search_drive_files_returns_error_dict(monkeypatch):
    def fail(q, limit=None):
        raise ConnectionError("offline")
    monkeypatch.setattr(t, "_list_files", fail)
    assert t.search_drive_files("x")["status"] == "error"


@pytest.mark.parametrize("mime, expected", [
    (t.DOC_MIME, "doc text"),
    (t.SLIDES_MIME, "doc text"),
    (t.SHEET_MIME, "doc text"),
    ("text/plain", "doc text"),
    ("application/pdf", "pdf text"),
])
def test_read_drive_file_by_type(drive, monkeypatch, mime, expected):
    drive.files().get().execute.return_value = {"id": "1", "name": "f", "mimeType": mime}
    monkeypatch.setattr(t, "_download", lambda req: b"doc text")
    monkeypatch.setattr(t, "_extract_pdf_text", lambda data: "pdf text")
    result = t.read_drive_file("1")
    assert result["content"] == expected
    assert result["truncated"] is False


def _xlsx_bytes() -> bytes:
    from openpyxl import Workbook
    wb = Workbook()
    wb.active.title = "Q3"
    wb.active.append(["region", "revenue"])
    wb.active.append(["north", 120])
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _docx_bytes() -> bytes:
    import zipfile
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("word/document.xml",
                   '<w:document><w:body><w:p><w:r><w:t>Salmon &amp; trout</w:t></w:r></w:p>'
                   '<w:p><w:r><w:t xml:space="preserve">Year two </w:t></w:r><w:r><w:t>results</w:t></w:r></w:p>'
                   '</w:body></w:document>')
    return buf.getvalue()


def test_extract_file_text():
    assert t.extract_file_text(_xlsx_bytes(), t.XLSX_MIME) == "## Q3\nregion,revenue\nnorth,120"
    assert t.extract_file_text(_docx_bytes(), t.DOCX_MIME) == "Salmon & trout\nYear two results"
    assert t.extract_file_text(b"a,b", "text/csv") == "a,b"
    assert t.extract_file_text(b"x", "application/zip") is None


def test_read_drive_file_reads_uploaded_excel(drive, monkeypatch):
    drive.files().get().execute.return_value = {"id": "1", "name": "q3.xlsx", "mimeType": t.XLSX_MIME}
    monkeypatch.setattr(t, "_download", lambda req: _xlsx_bytes())
    assert "north,120" in t.read_drive_file("1")["content"]


def test_read_drive_file_truncates_and_rejects_unknown_types(drive, monkeypatch):
    drive.files().get().execute.return_value = {"id": "1", "name": "f", "mimeType": "text/plain"}
    monkeypatch.setattr(t, "_download", lambda req: b"a" * (t.MAX_READ_CHARS + 5))
    result = t.read_drive_file("1")
    assert len(result["content"]) == t.MAX_READ_CHARS and result["truncated"] is True

    drive.files().get().execute.return_value = {"id": "1", "name": "f", "mimeType": "image/png"}
    assert t.read_drive_file("1")["status"] == "error"


def test_create_document_imports_markdown_as_google_doc(monkeypatch):
    calls = []
    monkeypatch.setattr(t, "_upload", lambda *a, **k: calls.append((a, k)) or {"id": "d"})
    t._create_document("Quiz", "# Q1", "folder")
    args, kwargs = calls[0]
    assert args == ("Quiz", b"# Q1", "text/markdown", "folder")
    assert kwargs["convert_to"] == t.DOC_MIME


def test_create_pdf_exports_uploads_and_removes_temp_doc(drive, monkeypatch):
    uploads = []

    def fake_upload(name, data, mime, folder_id, convert_to=None):
        uploads.append((name, data, mime, folder_id, convert_to))
        return {"id": f"id{len(uploads)}"}
    monkeypatch.setattr(t, "_upload", fake_upload)
    monkeypatch.setattr(t, "_download", lambda req: b"%PDF-1.7")

    t._create_pdf("Report", "# Hi", "folder")
    assert uploads[0][4] == t.DOC_MIME and uploads[0][3] is None   # temp doc, outside any folder
    assert uploads[1] == ("Report.pdf", b"%PDF-1.7", "application/pdf", "folder", None)
    assert drive.files().delete.call_args.kwargs["fileId"] == "id1"


def test_create_pdf_removes_temp_doc_even_on_failure(drive, monkeypatch):
    monkeypatch.setattr(t, "_upload", lambda *a, **k: {"id": "temp"})

    def fail(req):
        raise ConnectionError("export failed")
    monkeypatch.setattr(t, "_download", fail)
    with pytest.raises(ConnectionError):
        t._create_pdf("Report", "# Hi", "folder")
    assert drive.files().delete.call_args.kwargs["fileId"] == "temp"


def test_create_slidedeck_builds_one_slide_per_section(slides, monkeypatch):
    slides.presentations().create().execute.return_value = {
        "presentationId": "p1",
        "slides": [{"pageElements": [{"objectId": "title0", "shape": {"placeholder": {"type": "CENTERED_TITLE"}}}]}],
    }
    moved = []
    monkeypatch.setattr(t, "_move_into", lambda fid, folder: moved.append((fid, folder)) or {"id": fid})

    t._create_slidedeck("Deck", "# One\n- a\n# Two\n- b\n- c", "folder")
    requests = slides.presentations().batchUpdate.call_args.kwargs["body"]["requests"]
    kinds = [next(iter(r)) for r in requests]
    assert kinds.count("createSlide") == 2
    assert {"insertText": {"objectId": "title0", "text": "Deck"}} in requests
    texts = [r["insertText"]["text"] for r in requests if "insertText" in r]
    assert texts == ["Deck", "One", "a", "Two", "b\nc"]
    assert moved == [("p1", "folder")]


def test_create_infographic_uploads_png_and_removes_temp_deck(drive, slides, monkeypatch):
    slides.presentations().create().execute.return_value = {
        "presentationId": "p1", "slides": [{"objectId": "page1", "pageElements": [{"objectId": "old"}]}],
    }
    slides.presentations().pages().getThumbnail().execute.return_value = {"contentUrl": "https://img"}
    monkeypatch.setattr(t, "_credentials", lambda: object())
    import google.auth.transport.requests as gatr
    session = MagicMock()
    session.get.return_value = SimpleNamespace(content=b"\x89PNG", raise_for_status=lambda: None)
    monkeypatch.setattr(gatr, "AuthorizedSession", lambda creds: session)
    uploads = []
    monkeypatch.setattr(t, "_upload", lambda *a, **k: uploads.append(a) or {"id": "png"})

    t._create_infographic("Facts", "# A\nx\n# B\ny", "folder")
    requests = slides.presentations().batchUpdate.call_args.kwargs["body"]["requests"]
    assert {"deleteObject": {"objectId": "old"}} in requests
    assert sum("createShape" in r for r in requests) == 3          # title + 2 section cards
    assert uploads == [("Facts.png", b"\x89PNG", "image/png", "folder")]
    assert drive.files().delete.call_args.kwargs["fileId"] == "p1"


# ===========================================================================
# workspace_agent/workspace_agent.py — schemas and helpers
# ===========================================================================

def test_workspace_category_schema():
    ok = wa.WorkspaceCategory(category="FOLDER_DELETION", confidence=0.9, request="delete X")
    assert ok.missing_info is None
    for bad in [{"category": "NOPE", "confidence": 0.9, "request": "x"},
                {"category": "UNKNOWN", "confidence": 1.5, "request": "x"}]:
        with pytest.raises(Exception):
            wa.WorkspaceCategory(**bad)
    # Unknown keys are ignored (extra='forbid' would put additionalProperties in the Gemini schema).
    assert not hasattr(wa.WorkspaceCategory(category="UNKNOWN", confidence=0.5, request="x", extra=1), "extra")


@pytest.mark.parametrize("value, expected", [
    (None, None), ("", None), ("None", None), ("N/A", None), (" nothing is missing. ", None),
    ("which report to summarize", "which report to summarize"),
])
def test_missing_info_placeholders_become_none(value, expected):
    decision = wa.WorkspaceCategory(category="UNKNOWN", confidence=0.5, request="x", missing_info=value)
    assert decision.missing_info == expected


def test_gemini_schemas_have_no_additional_properties():
    from google.genai import _transformers
    for model in (wa.WorkspaceCategory, wa.DocumentCommand, wa.FolderCommand, wa.MoveCommand, wa.Confirmation):
        assert "additional_properties" not in json.dumps(_transformers.t_schema(None, model).model_dump(exclude_none=True, mode="json"))


def test_document_command_schema():
    doc = wa.DocumentCommand(title="T", file_type="SLIDES", folder_topic="F", content="c")
    assert doc.answer_key is None and doc.works_cited is None
    with pytest.raises(Exception):
        wa.DocumentCommand(title="T", file_type="VIDEO", folder_topic="F", content="c")


@pytest.mark.parametrize("response, expected", [
    ({"approved": True}, (True, None)),
    ({"approved": False, "feedback": "no"}, (False, "no")),
    (wa.Confirmation(approved=True, feedback="ok"), (True, "ok")),
    ("Yes", (True, None)),
    ("nope", (False, None)),
    (None, (False, None)),
])
def test_is_approved(response, expected):
    assert wa._is_approved(response) == expected


def test_get_or_create_folder(monkeypatch):
    monkeypatch.setattr(wa, "find_drive_folder", lambda name: {"id": "1", "name": name})
    assert wa._get_or_create_folder("A") == ({"id": "1", "name": "A"}, False)
    monkeypatch.setattr(wa, "find_drive_folder", lambda name: {"id": None})
    monkeypatch.setattr(wa, "create_drive_folder", lambda name: {"id": "new", "name": name})
    assert wa._get_or_create_folder("A") == ({"id": "new", "name": "A"}, True)


def test_creators_cover_every_file_type():
    file_types = wa.DocumentCommand.model_fields["file_type"].annotation.__args__
    assert set(wa.CREATORS) == set(file_types)


# ===========================================================================
# workspace_agent/workspace_agent.py — plain function nodes
# ===========================================================================

@pytest.mark.parametrize("decision, route", [
    ({"category": "FOLDER_CREATION", "confidence": 0.9, "request": "r"}, "FOLDER_CREATION"),
    ({"category": "FOLDER_CREATION", "confidence": 0.5, "request": "r"}, "UNKNOWN"),
    ({"category": "DOCUMENT_CREATION", "confidence": 0.9, "request": "r", "missing_info": "no report"}, "UNKNOWN"),
    ({"category": "DOCUMENT_CREATION", "confidence": 0.9, "request": "r", "missing_info": "None"}, "DOCUMENT_CREATION"),
])
def test_route_request(decision, route):
    event = wa.route_request(decision)
    assert event.actions.route == route
    assert event.output == "r"


def test_handle_unknown_mentions_missing_info():
    event = wa.handle_unknown(SimpleNamespace(state={"router_decision": {"missing_info": "which report"}}))
    assert "which report" in event.content.parts[0].text
    event = wa.handle_unknown(SimpleNamespace(state={"router_decision": {"missing_info": "N/A"}}))
    assert "N/A" not in event.content.parts[0].text


def _request(*parts):
    from google.adk.models import LlmRequest
    from google.genai import types
    return LlmRequest(contents=[types.Content(role="user", parts=list(parts))])


def test_prepare_attachments_converts_and_labels_files():
    from google.genai import types
    xlsx = types.Part(inline_data=types.Blob(data=_xlsx_bytes(), mime_type=t.XLSX_MIME, display_name="q3.xlsx"))
    pdf = types.Part(inline_data=types.Blob(data=b"%PDF", mime_type="application/pdf", display_name="r.pdf"))
    req = _request(types.Part(text="Summarize this report"), xlsx, pdf)
    original = req.contents[0]
    wa.prepare_attachments(None, req)

    parts = req.contents[0].parts
    assert parts[0].text == "Summarize this report"
    assert parts[1].text == f"[Attached file: q3.xlsx ({t.XLSX_MIME})]" and "north,120" in parts[2].text
    assert parts[3].text.startswith("[Attached file: r.pdf")
    assert parts[4].inline_data.data == b"%PDF" and parts[4].inline_data.display_name is None
    # The session's own content keeps the original bytes and name.
    assert original.parts[1].inline_data.display_name == "q3.xlsx"


def test_prepare_attachments_unreadable_type_becomes_a_note():
    from google.genai import types
    req = _request(types.Part(inline_data=types.Blob(data=b"PK", mime_type="application/zip", display_name="a.zip")))
    wa.prepare_attachments(None, req)
    assert "can't be read" in req.contents[0].parts[0].text


def test_create_document_uses_existing_folder_and_adds_answer_key(fake_drive):
    event = wa.create_document(DOC)
    assert fake_drive == [("create", "DOC", "Related Rates Quiz", "f-calc"),
                          ("create", "DOC", "Related Rates Quiz - Answer Key", "f-calc")]
    assert "existing folder" in event.content.parts[0].text


def test_create_document_creates_missing_folder(fake_drive):
    wa.create_document({**DOC, "file_type": "SLIDES", "folder_topic": "New Topic", "answer_key": None})
    assert fake_drive == [("mkdir", "New Topic"), ("create", "SLIDES", "Related Rates Quiz", "f-New Topic")]


def test_create_document_reports_errors(fake_drive, monkeypatch):
    def fail(*a):
        raise ConnectionError("offline")
    monkeypatch.setitem(wa.CREATORS, "DOC", fail)
    assert wa.create_document(DOC).output["status"] == "error"


def test_create_folder_skips_existing(fake_drive):
    assert "already exists" in wa.create_folder({"folder_name": "Old Stuff"}).content.parts[0].text
    wa.create_folder({"folder_name": "AP Chem"})
    assert fake_drive == [("mkdir", "AP Chem")]


# ===========================================================================
# workspace_agent/workspace_agent.py — full workflow with a fake Gemini
# ===========================================================================

def run_workflow(reply=None, attachments=()):
    """Run one request through root_agent; answer the confirmation prompt with `reply` if one appears.

    `attachments` are (name, bytes, mime) files sent with the request.
    """
    from google.adk.runners import InMemoryRunner
    from google.genai import types

    async def main():
        runner = InMemoryRunner(agent=wa.root_agent, app_name="test")
        session = await runner.session_service.create_session(app_name="test", user_id="u")
        texts, prompt = [], None

        async def send(message):
            nonlocal prompt
            async for event in runner.run_async(user_id="u", session_id=session.id, new_message=message):
                for part in (event.content.parts if event.content else []):
                    if part.function_call:
                        prompt = part.function_call
                    if part.text and event.author != "user" and not part.text.lstrip().startswith("{"):
                        texts.append(part.text)

        files = [types.Part(inline_data=types.Blob(data=d, mime_type=m, display_name=n)) for n, d, m in attachments]
        await send(types.Content(role="user", parts=[types.Part(text="request"), *files]))
        asked = prompt
        if reply is not None and prompt:
            await send(types.Content(role="user", parts=[types.Part(function_response=types.FunctionResponse(
                id=prompt.id, name=prompt.name, response=reply))]))
        return texts, asked

    return asyncio.run(main())



def test_workflow_document_creation_approved(fake_gemini, fake_drive):
    router(fake_gemini, "DOCUMENT_CREATION")
    texts, prompt = run_workflow(reply={"approved": True})
    assert prompt is not None
    assert "1. 4 ft/s" in json.dumps(prompt.args)   # answer key shown for review
    assert ("create", "DOC", "Related Rates Quiz", "f-calc") in fake_drive
    assert "Created 'Related Rates Quiz'" in texts[-1]


def test_workflow_document_creation_rejected_creates_nothing(fake_gemini, fake_drive):
    router(fake_gemini, "DOCUMENT_CREATION")
    texts, _ = run_workflow(reply={"approved": False, "feedback": "harder"})
    assert fake_drive == []
    assert "Cancelled" in texts[-1] and "harder" in texts[-1]


def test_workflow_document_creation_waits_for_confirmation(fake_gemini, fake_drive):
    router(fake_gemini, "DOCUMENT_CREATION")
    _, prompt = run_workflow(reply=None)
    assert prompt is not None
    assert fake_drive == []


def test_workflow_folder_creation(fake_gemini, fake_drive):
    router(fake_gemini, "FOLDER_CREATION")
    texts, prompt = run_workflow()
    assert prompt is None
    assert fake_drive == [("mkdir", "AP Chem")]


def test_workflow_folder_deletion_requires_confirmation(fake_gemini, fake_drive):
    router(fake_gemini, "FOLDER_DELETION")
    _, prompt = run_workflow(reply={"approved": False})
    assert prompt is not None and fake_drive == []

    _, _ = run_workflow(reply={"approved": True})
    assert fake_drive == [("trash", "f-old")]


def test_workflow_document_movement(fake_gemini, fake_drive):
    router(fake_gemini, "DOCUMENT_MOVEMENT")
    texts, _ = run_workflow()
    assert ("mv", "file1", "f-Atlantic Salmon Research Project") in fake_drive
    assert "Moved 'salmon.pdf'" in texts[-1]


@pytest.mark.parametrize("category, confidence", [("UNKNOWN", 0.95), ("FOLDER_CREATION", 0.3)])
def test_workflow_unknown_or_low_confidence(fake_gemini, fake_drive, category, confidence):
    router(fake_gemini, category, confidence)
    texts, _ = run_workflow()
    assert fake_drive == []
    assert "I can create a Doc" in texts[-1]


def test_workflow_move_uploads_attached_file(fake_gemini, fake_drive):
    router(fake_gemini, "DOCUMENT_MOVEMENT")
    texts, prompt = run_workflow(attachments=[("salmon.pdf", b"%PDF", "application/pdf")])
    assert prompt is None
    assert fake_drive == [("mkdir", "Atlantic Salmon Research Project"),
                          ("upload", "salmon.pdf", "application/pdf", "f-Atlantic Salmon Research Project")]
    assert "Uploaded 'salmon.pdf'" in texts[-1]


def test_workflow_document_creation_with_attached_excel(fake_gemini, fake_drive):
    router(fake_gemini, "DOCUMENT_CREATION")
    texts, _ = run_workflow(reply={"approved": True}, attachments=[("q3.xlsx", _xlsx_bytes(), t.XLSX_MIME)])
    assert ("create", "DOC", "Related Rates Quiz", "f-calc") in fake_drive


@pytest.mark.parametrize("reply, moved", [({"approved": True}, True), ({"approved": False}, False)])
def test_workflow_move_confirms_best_of_several_matches(fake_gemini, fake_drive, monkeypatch, reply, moved):
    router(fake_gemini, "DOCUMENT_MOVEMENT")
    monkeypatch.setattr(wa, "search_drive_files", lambda q: {"files": [
        {"id": "old", "name": "salmon notes old.docx"}, {"id": "best", "name": "salmon.pdf.bak"}]})
    texts, prompt = run_workflow(reply=reply)
    assert prompt is not None and "salmon.pdf.bak" in json.dumps(prompt.args)
    assert (("mv", "best", "f-Atlantic Salmon Research Project") in fake_drive) is moved
    assert ("Moved 'salmon.pdf.bak'" if moved else "Cancelled") in texts[-1]


@pytest.mark.parametrize("found, text", [({"files": []}, "No Drive file"), ({"status": "error", "error": "offline"}, None)])
def test_workflow_move_no_match_or_error(fake_gemini, fake_drive, monkeypatch, found, text):
    router(fake_gemini, "DOCUMENT_MOVEMENT")
    monkeypatch.setattr(wa, "search_drive_files", lambda q: found)
    texts, prompt = run_workflow()
    assert prompt is None and fake_drive == []
    if text:
        assert text in texts[-1]
