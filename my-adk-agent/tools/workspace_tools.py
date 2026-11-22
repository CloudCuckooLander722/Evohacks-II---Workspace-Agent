"""Google Workspace tools used by workspace_agent.

Everything goes through the official Google API client (Drive v3, Slides v1).

Authentication: see _credentials() (GOOGLE_CREDENTIALS_PATH, else Application Default Credentials).

Functions without a leading underscore that the LLM calls directly (list_drive_folders,
search_drive_files, read_drive_file) never raise: they return an error dict from
report_tool_error so the agent can recover. Everything else raises on failure and the
workflow nodes catch and report it.
"""

from __future__ import annotations

import io
import json
import os
import re
import uuid
import zipfile
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Optional

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload, MediaIoBaseUpload

from agents.tool_errors import report_tool_error

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/presentations",
]

FOLDER_MIME = "application/vnd.google-apps.folder"
DOC_MIME = "application/vnd.google-apps.document"
SLIDES_MIME = "application/vnd.google-apps.presentation"
SHEET_MIME = "application/vnd.google-apps.spreadsheet"
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

FOLDER_MATCH_THRESHOLD = 0.75
"""Minimum rank_folders score for find_drive_folder to treat a folder as a match."""

MAX_READ_CHARS = 100_000
"""read_drive_file truncates content beyond this, to keep the LLM context bounded."""

FILE_FIELDS = "id, name, mimeType, parents, webViewLink"


# ---------------------------------------------------------------------------
# Auth / clients
# ---------------------------------------------------------------------------

def credentials_kind(path: str | os.PathLike) -> str:
    """Classify a Google credentials JSON: 'service_account', 'authorized_user', or 'oauth_client'."""
    info = json.loads(Path(path).read_text())
    if info.get("type") in ("service_account", "authorized_user"):
        return info["type"]
    if "installed" in info or "web" in info:
        return "oauth_client"
    raise ValueError(f"{path} is not a service-account key, OAuth client, or authorized-user token.")


class GoogleSignInRequired(RuntimeError):
    """The OAuth client is a web client and nobody has signed in through the app yet."""


def _client_type(client_file: str | os.PathLike) -> str:
    """'web' or 'installed' for an OAuth client JSON."""
    return "web" if "web" in json.loads(Path(client_file).read_text()) else "installed"


def _token_path(client_file: str | os.PathLike) -> Path:
    """Where the signed-in user's token is cached: GOOGLE_OAUTH_TOKEN_FILE, else next to the client JSON."""
    override = os.environ.get("GOOGLE_OAUTH_TOKEN_FILE")
    return Path(override) if override else Path(client_file).resolve().parent / "token.json"


def _oauth_user_credentials(client_file: str):
    """The signed-in user's credentials for an OAuth client JSON.

    Uses (and refreshes) the cached token. Without one, an "installed" (Desktop) client runs the
    local consent flow; a "web" client raises GoogleSignInRequired, because its sign-in has to go
    through the app's registered redirect URI (see google_sign_in_url).
    """
    from google.auth.exceptions import RefreshError
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials

    token_path = _token_path(client_file)
    creds = None
    if token_path.exists():
        creds = Credentials.from_authorized_user_file(str(token_path), SCOPES)
    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            token_path.write_text(creds.to_json())
            return creds
        except RefreshError:
            token_path.unlink(missing_ok=True)   # revoked or expired for good: sign in again

    if _client_type(client_file) == "web":
        raise GoogleSignInRequired("Not signed in to Google. Use 'Sign in with Google' in the app's sidebar first.")
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(client_file, SCOPES)
    creds = flow.run_local_server(port=int(os.environ.get("GOOGLE_OAUTH_PORT", "8080")), open_browser=False)
    token_path.write_text(creds.to_json())
    return creds


def _credentials_path() -> Optional[str]:
    """GOOGLE_CREDENTIALS_PATH, checked to exist so a stale path fails with a clear message."""
    path = os.environ.get("GOOGLE_CREDENTIALS_PATH")
    if path and not Path(path).is_file():
        raise FileNotFoundError(
            f"GOOGLE_CREDENTIALS_PATH is {path}, but that file doesn't exist. "
            "Fix it in workspace_agent/.env and restart the app."
        )
    return path


# ---------------------------------------------------------------------------
# Web sign-in (OAuth "web" client; the app's URL is the registered redirect URI)
# ---------------------------------------------------------------------------

_PENDING_SIGN_INS: dict[str, Optional[str]] = {}
"""OAuth state -> PKCE code verifier for sign-ins started but not finished. Process-wide, because
Google's redirect opens a new browser session."""


def _web_client_path() -> Optional[str]:
    path = _credentials_path()
    if path and credentials_kind(path) == "oauth_client" and _client_type(path) == "web":
        return path
    return None


def _web_flow(client_file: str, **kwargs):
    from google_auth_oauthlib.flow import Flow
    redirect = os.environ.get("GOOGLE_OAUTH_REDIRECT_URI") or \
        json.loads(Path(client_file).read_text())["web"]["redirect_uris"][0]
    return Flow.from_client_secrets_file(client_file, SCOPES, redirect_uri=redirect, **kwargs)


def google_sign_in_status() -> str:
    """'not_needed' (no web OAuth client configured), 'signed_in', or 'signed_out'."""
    path = _web_client_path()
    if not path:
        return "not_needed"
    return "signed_in" if _token_path(path).exists() else "signed_out"


def google_sign_in_url() -> str:
    """Start a sign-in: the Google consent URL, which redirects back to the app with ?code=&state=."""
    path = _web_client_path()
    if not path:
        raise RuntimeError("GOOGLE_CREDENTIALS_PATH is not a web OAuth client.")
    flow = _web_flow(path)
    url, state = flow.authorization_url(access_type="offline", prompt="consent")
    while len(_PENDING_SIGN_INS) >= 20:
        _PENDING_SIGN_INS.pop(next(iter(_PENDING_SIGN_INS)))
    _PENDING_SIGN_INS[state] = flow.code_verifier
    return url


def complete_google_sign_in(code: str, state: str) -> None:
    """Finish a sign-in from the redirect's query parameters and cache the user's token."""
    path = _web_client_path()
    if not path:
        raise RuntimeError("GOOGLE_CREDENTIALS_PATH is not a web OAuth client.")
    if state not in _PENDING_SIGN_INS:
        raise ValueError("This sign-in link has expired or didn't come from this app. Please sign in again.")
    flow = _web_flow(path, state=state, code_verifier=_PENDING_SIGN_INS.pop(state))
    # Google may report the granted scopes in a different form; don't treat that as an error.
    os.environ.setdefault("OAUTHLIB_RELAX_TOKEN_SCOPE", "1")
    flow.fetch_token(code=code)
    _token_path(path).write_text(flow.credentials.to_json())
    _credentials.cache_clear()


@lru_cache(maxsize=1)
def _credentials():
    """Load Google credentials.

    GOOGLE_CREDENTIALS_PATH points at one of:
    - a service-account key        -> used directly (sees only files shared with that account)
    - an authorized-user token     -> a user's refreshable login, e.g. token.json from a previous
                                      OAuth run; the way to reach a user's Drive on Cloud Run
    - an OAuth client              -> the signed-in user's cached token; a Desktop client runs the
                                      consent flow once, a web client needs the app's sign-in
    Without it, Application Default Credentials are used (on Cloud Run: the service's account).
    """
    path = _credentials_path()
    if path:
        kind = credentials_kind(path)
        if kind == "service_account":
            from google.oauth2 import service_account
            return service_account.Credentials.from_service_account_file(path, scopes=SCOPES)
        if kind == "authorized_user":
            from google.oauth2.credentials import Credentials
            return Credentials.from_authorized_user_file(path, SCOPES)
        return _oauth_user_credentials(path)

    import google.auth
    from google.auth.exceptions import DefaultCredentialsError
    try:
        creds, _ = google.auth.default(scopes=SCOPES)
        return creds
    except DefaultCredentialsError as e:
        raise RuntimeError(
            "No Google credentials configured. Set GOOGLE_CREDENTIALS_PATH to a service-account key, "
            "an OAuth client JSON, or an authorized-user token JSON."
        ) from e


@lru_cache(maxsize=1)
def _drive():
    return build("drive", "v3", credentials=_credentials(), cache_discovery=False)


@lru_cache(maxsize=1)
def _slides():
    return build("slides", "v1", credentials=_credentials(), cache_discovery=False)


def _q(value: str) -> str:
    """Escape a value for use inside a single-quoted Drive query string."""
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _list_files(q: str, limit: Optional[int] = None, fields: str = FILE_FIELDS) -> list[dict]:
    files, page_token = [], None
    while True:
        resp = _drive().files().list(
            q=q,
            fields=f"nextPageToken, files({fields})",
            pageSize=min(limit or 1000, 1000),
            pageToken=page_token,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token or (limit and len(files) >= limit):
            return files[:limit] if limit else files


def _upload(name: str, data: bytes, mime: str, folder_id: Optional[str], convert_to: Optional[str] = None) -> dict:
    """Upload bytes to Drive; with convert_to, Drive imports them as a native Google file."""
    metadata = {"name": name, "mimeType": convert_to or mime}
    if folder_id:
        metadata["parents"] = [folder_id]
    media = MediaIoBaseUpload(io.BytesIO(data), mimetype=mime, resumable=False)
    return _drive().files().create(
        body=metadata, media_body=media, fields=FILE_FIELDS, supportsAllDrives=True
    ).execute()


def _download(request) -> bytes:
    buf = io.BytesIO()
    downloader = MediaIoBaseDownload(buf, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return buf.getvalue()


def _move_into(file_id: str, folder_id: Optional[str]) -> dict:
    if not folder_id:
        return _drive().files().get(fileId=file_id, fields=FILE_FIELDS, supportsAllDrives=True).execute()
    return move_drive_file(file_id, folder_id)


# ---------------------------------------------------------------------------
# Markdown -> slide sections
# ---------------------------------------------------------------------------

def _sections(content: str) -> list[tuple[str, list[str]]]:
    """Split Markdown into (heading, body lines) sections on '#' headings or '---' rules."""
    sections: list[tuple[str, list[str]]] = []
    title, body = "", []
    for raw in content.splitlines():
        line = raw.rstrip()
        heading = re.match(r"^#{1,6}\s+(.*)", line)
        if heading or re.fullmatch(r"-{3,}|\*{3,}", line.strip()):
            if title or any(b.strip() for b in body):
                sections.append((title, body))
            title, body = (heading.group(1).strip() if heading else ""), []
        else:
            body.append(line)
    if title or any(b.strip() for b in body):
        sections.append((title, body))
    return sections


def _clean_body(lines: list[str]) -> str:
    """Strip Markdown list markers and emphasis; Slides renders plain text."""
    out = []
    for line in lines:
        if not line.strip():
            continue
        line = re.sub(r"^\s*([-*+]|\d+[.)])\s+", "", line)
        line = re.sub(r"(\*\*|__|`)", "", line)
        out.append(line.strip())
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Document creation (called by workflow code, not exposed to the LLM)
# ---------------------------------------------------------------------------

def _create_document(title: str, content: str, folder_id: Optional[str] = None) -> dict:
    """Create a Google Doc from Markdown content (Drive converts headings, lists, bold, etc.)."""
    return _upload(title, content.encode("utf-8"), "text/markdown", folder_id, convert_to=DOC_MIME)


def _create_slidedeck(title: str, content: str, folder_id: Optional[str] = None) -> dict:
    """Create a Google Slides deck: a title slide, then one slide per Markdown section."""
    slides = _slides()
    pres = slides.presentations().create(body={"title": title}).execute()
    pres_id = pres["presentationId"]

    # The new deck starts with one TITLE slide; fill it, then add a TITLE_AND_BODY slide per section.
    first = pres["slides"][0]
    requests = []
    for el in first.get("pageElements", []):
        ph = el.get("shape", {}).get("placeholder", {}).get("type")
        if ph in ("CENTERED_TITLE", "TITLE"):
            requests.append({"insertText": {"objectId": el["objectId"], "text": title}})

    sections = _sections(content) or [("", content.splitlines())]
    for i, (heading, body) in enumerate(sections):
        sid, tid, bid = f"s{i}_{uuid.uuid4().hex[:8]}", f"t{i}_{uuid.uuid4().hex[:8]}", f"b{i}_{uuid.uuid4().hex[:8]}"
        requests.append({
            "createSlide": {
                "objectId": sid,
                "slideLayoutReference": {"predefinedLayout": "TITLE_AND_BODY"},
                "placeholderIdMappings": [
                    {"layoutPlaceholder": {"type": "TITLE", "index": 0}, "objectId": tid},
                    {"layoutPlaceholder": {"type": "BODY", "index": 0}, "objectId": bid},
                ],
            }
        })
        if heading:
            requests.append({"insertText": {"objectId": tid, "text": heading}})
        text = _clean_body(body)
        if text:
            requests.append({"insertText": {"objectId": bid, "text": text}})
            requests.append({
                "createParagraphBullets": {
                    "objectId": bid,
                    "textRange": {"type": "ALL"},
                    "bulletPreset": "BULLET_DISC_CIRCLE_SQUARE",
                }
            })

    slides.presentations().batchUpdate(presentationId=pres_id, body={"requests": requests}).execute()
    return _move_into(pres_id, folder_id)


def _create_pdf(title: str, content: str, folder_id: Optional[str] = None) -> dict:
    """Create a PDF: import the Markdown as a temporary Google Doc, export it to PDF, upload the PDF."""
    temp = _upload(f"{title} (temp)", content.encode("utf-8"), "text/markdown", None, convert_to=DOC_MIME)
    try:
        pdf = _download(_drive().files().export_media(fileId=temp["id"], mimeType="application/pdf"))
    finally:
        _drive().files().delete(fileId=temp["id"], supportsAllDrives=True).execute()
    name = title if title.lower().endswith(".pdf") else f"{title}.pdf"
    return _upload(name, pdf, "application/pdf", folder_id)


# Infographic layout: a portrait page (points), with a title banner and stacked, colored section cards.
_INFO_W, _INFO_H, _MARGIN = 540, 960, 30
_PALETTE = [
    {"red": 0.16, "green": 0.38, "blue": 0.67},
    {"red": 0.13, "green": 0.55, "blue": 0.45},
    {"red": 0.85, "green": 0.47, "blue": 0.14},
    {"red": 0.55, "green": 0.27, "blue": 0.62},
    {"red": 0.78, "green": 0.22, "blue": 0.27},
]


def _box(page_id: str, x: float, y: float, w: float, h: float, text: str, size: int, fill, bold: bool) -> list[dict]:
    oid = f"box_{uuid.uuid4().hex[:10]}"
    pt = lambda v: {"magnitude": v, "unit": "PT"}  # noqa: E731
    reqs = [
        {"createShape": {
            "objectId": oid,
            "shapeType": "ROUND_RECTANGLE" if fill else "TEXT_BOX",
            "elementProperties": {
                "pageObjectId": page_id,
                "size": {"width": pt(w), "height": pt(h)},
                "transform": {"scaleX": 1, "scaleY": 1, "translateX": x, "translateY": y, "unit": "PT"},
            },
        }},
        {"insertText": {"objectId": oid, "text": text}},
        {"updateTextStyle": {
            "objectId": oid,
            "style": {
                "fontSize": pt(size),
                "bold": bold,
                "foregroundColor": {"opaqueColor": {"rgbColor": (
                    {"red": 1, "green": 1, "blue": 1} if fill else {"red": 0.1, "green": 0.1, "blue": 0.15}
                )}},
            },
            "fields": "fontSize,bold,foregroundColor",
        }},
    ]
    if fill:
        reqs.append({"updateShapeProperties": {
            "objectId": oid,
            "shapeProperties": {
                "shapeBackgroundFill": {"solidFill": {"color": {"rgbColor": fill}}},
                "outline": {"propertyState": "NOT_RENDERED"},
                "contentAlignment": "MIDDLE",
            },
            "fields": "shapeBackgroundFill.solidFill.color,outline.propertyState,contentAlignment",
        }})
    return reqs


def _create_infographic(title: str, content: str, folder_id: Optional[str] = None) -> dict:
    """Create a PNG infographic: lay out a portrait slide with colored section cards, render it, upload the PNG.

    The intermediate Google Slides file is deleted afterwards; only the PNG stays in Drive.
    """
    slides = _slides()
    pres = slides.presentations().create(body={
        "title": f"{title} (temp)",
        "pageSize": {"width": {"magnitude": _INFO_W, "unit": "PT"}, "height": {"magnitude": _INFO_H, "unit": "PT"}},
    }).execute()
    pres_id = pres["presentationId"]
    try:
        page_id = pres["slides"][0]["objectId"]
        requests = [{"deleteObject": {"objectId": el["objectId"]}} for el in pres["slides"][0].get("pageElements", [])]

        inner_w = _INFO_W - 2 * _MARGIN
        requests += _box(page_id, _MARGIN, _MARGIN, inner_w, 90, title, 26, None, True)

        sections = [(h, _clean_body(b)) for h, b in _sections(content)] or [("", content)]
        sections = sections[:8]
        top, gap = _MARGIN + 110, 14
        card_h = (_INFO_H - top - _MARGIN - gap * (len(sections) - 1)) / len(sections)
        for i, (heading, body) in enumerate(sections):
            text = f"{heading}\n{body}".strip() if heading else body
            size = 15 if len(text) < 180 else 12 if len(text) < 350 else 10
            requests += _box(page_id, _MARGIN, top + i * (card_h + gap), inner_w, card_h,
                             text, size, _PALETTE[i % len(_PALETTE)], False)

        slides.presentations().batchUpdate(presentationId=pres_id, body={"requests": requests}).execute()

        thumb = slides.presentations().pages().getThumbnail(
            presentationId=pres_id, pageObjectId=page_id,
            thumbnailProperties_mimeType="PNG", thumbnailProperties_thumbnailSize="LARGE",
        ).execute()
        from google.auth.transport.requests import AuthorizedSession
        png = AuthorizedSession(_credentials()).get(thumb["contentUrl"], timeout=60)
        png.raise_for_status()
    finally:
        _drive().files().delete(fileId=pres_id, supportsAllDrives=True).execute()

    name = title if title.lower().endswith(".png") else f"{title}.png"
    return _upload(name, png.content, "image/png", folder_id)


# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------

def _list_folders() -> list[dict]:
    """Folders the user can add files to.

    Folders shared with the user as viewer or commenter are left out: uploading or moving a file
    into one fails with 403 insufficientParentPermissions. A subject name that only matches such
    a folder then gets a new folder of the user's own.
    """
    folders = _list_files(f"mimeType='{FOLDER_MIME}' and trashed=false",
                          fields="id, name, capabilities/canAddChildren")
    return [
        {"id": f["id"], "name": f["name"]}
        for f in folders
        if f.get("capabilities", {}).get("canAddChildren")
    ]


def list_drive_folders() -> dict:
    """List the Google Drive folders the user can add files to.

    Returns:
        {"folders": [{"id": str, "name": str}, ...]}
    """
    try:
        return {"folders": _list_folders()}
    except Exception as e:
        return report_tool_error("list_drive_folders", e)


def create_drive_folder(name: str, parent_id: Optional[str] = None) -> dict:
    """Create a Drive folder with the given name. Returns {"id", "name"}."""
    body = {"name": name, "mimeType": FOLDER_MIME}
    if parent_id:
        body["parents"] = [parent_id]
    folder = _drive().files().create(body=body, fields="id, name", supportsAllDrives=True).execute()
    return {"id": folder["id"], "name": folder["name"]}


def _normalize(name: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", name.lower())


def _similarity(a: str, b: str) -> float:
    """Name similarity in [0, 1]: best of character similarity, word overlap, and containment."""
    ta, tb = _normalize(a), _normalize(b)
    if not ta or not tb:
        return 0.0
    ja, jb = " ".join(ta), " ".join(tb)
    if ja == jb:
        return 1.0
    char = SequenceMatcher(None, ja, jb).ratio()
    sa, sb = set(ta), set(tb)
    overlap = len(sa & sb) / len(sa | sb)

    def covered(small: set[str], big: set[str]) -> bool:
        # Every word in `small` appears in `big`, allowing abbreviations ("chem" ~ "chemistry").
        return all(any(w == v or (len(w) >= 3 and v.startswith(w)) for v in big) for w in small)

    contained = 0.85 if (covered(sa, sb) or covered(sb, sa)) else 0.0
    return max(char, overlap, contained)


def rank_folders(topic: str, folders: list[dict]) -> list[dict]:
    """Rank folders by similarity of their names to a topic, best first. Each gets a "score" in [0, 1]."""
    scored = [{**f, "score": round(_similarity(topic, f["name"]), 3)} for f in folders]
    return sorted(scored, key=lambda f: f["score"], reverse=True)


def find_drive_folder(topic: str) -> dict:
    """Find the existing Drive folder (one the user can add files to) that best matches a topic via similarity search.

    Returns {"id", "name", "score"} for the best match, or {"id": None, "candidates": [...]}
    when no folder scores at least FOLDER_MATCH_THRESHOLD.
    """
    ranked = rank_folders(topic, _list_folders())
    if ranked and ranked[0]["score"] >= FOLDER_MATCH_THRESHOLD:
        return ranked[0]
    return {"id": None, "candidates": ranked[:3]}


def delete_drive_folder(folder_id: str) -> dict:
    """Move the Drive folder (and its contents) to the trash. It can be restored from Drive's Trash for 30 days."""
    folder = _drive().files().update(
        fileId=folder_id, body={"trashed": True}, fields="id, name, trashed", supportsAllDrives=True
    ).execute()
    return {"id": folder["id"], "name": folder["name"], "trashed": folder["trashed"]}


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------

def search_drive_files(query: str) -> dict:
    """Search Google Drive for files (not folders) whose name matches the query; falls back to full-text search.

    Args:
        query: A file name or search terms, e.g. "salmon report" or "Q3 report.pdf".

    Returns:
        {"files": [{"id", "name", "mimeType", "parents", "webViewLink"}, ...]}, at most 20.
    """
    try:
        base = f"mimeType!='{FOLDER_MIME}' and trashed=false"
        terms = query.strip()
        files = _list_files(f"{base} and name='{_q(terms)}'", limit=20)
        if not files:
            files = _list_files(f"{base} and name contains '{_q(terms)}'", limit=20)
        if not files:
            stem = re.sub(r"\.[A-Za-z0-9]{2,4}$", "", terms)
            words = " and ".join(f"name contains '{_q(w)}'" for w in stem.split() if len(w) > 1)
            if words:
                files = _list_files(f"{base} and {words}", limit=20)
        if not files:
            files = _list_files(f"{base} and fullText contains '{_q(terms)}'", limit=20)
        return {"files": files}
    except Exception as e:
        return report_tool_error("search_drive_files", e)


def _extract_pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_xlsx_text(data: bytes) -> str:
    """Every sheet as CSV-like rows, each under a '## <sheet name>' heading."""
    from openpyxl import load_workbook
    wb = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    out = []
    for sheet in wb.worksheets:
        rows = [",".join("" if v is None else str(v) for v in row)
                for row in sheet.iter_rows(values_only=True) if any(v is not None for v in row)]
        out.append(f"## {sheet.title}\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(out)


def _extract_docx_text(data: bytes) -> str:
    """Paragraph text from a .docx (stdlib only: the body is word/document.xml)."""
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)
    texts = ("".join(re.findall(r"<w:t(?: [^>]*)?>([^<]*)</w:t>", p)) for p in paragraphs)
    from html import unescape
    return "\n".join(unescape(t) for t in texts if t.strip())


def is_text_mime(mime: str) -> bool:
    return mime.startswith("text/") or mime in ("application/json", "application/xml")


def extract_file_text(data: bytes, mime: str) -> Optional[str]:
    """Text content of a PDF, Excel, Word or plain-text file's bytes; None for unsupported types."""
    if mime == "application/pdf":
        return _extract_pdf_text(data)
    if mime == XLSX_MIME:
        return _extract_xlsx_text(data)
    if mime == DOCX_MIME:
        return _extract_docx_text(data)
    if is_text_mime(mime):
        return data.decode("utf-8", "replace")
    return None


def upload_drive_file(name: str, data: bytes, mime: str, folder_id: str) -> dict:
    """Upload a file (e.g. one the user attached) into a Drive folder, keeping its format."""
    return _upload(name, data, mime, folder_id)


def read_drive_file(file_id: str) -> dict:
    """Read the text content of a Google Drive file (Docs, Slides, Sheets, PDFs, Excel, Word, and plain-text files).

    Args:
        file_id: The Drive file id, e.g. from search_drive_files.

    Returns:
        {"id", "name", "mimeType", "content", "truncated"}
    """
    try:
        drive = _drive()
        meta = drive.files().get(fileId=file_id, fields="id, name, mimeType", supportsAllDrives=True).execute()
        mime = meta["mimeType"]
        if mime in (DOC_MIME, SLIDES_MIME):
            text = _download(drive.files().export_media(fileId=file_id, mimeType="text/plain")).decode("utf-8", "replace")
        elif mime == SHEET_MIME:
            text = _download(drive.files().export_media(fileId=file_id, mimeType="text/csv")).decode("utf-8", "replace")
        elif mime in ("application/pdf", XLSX_MIME, DOCX_MIME) or is_text_mime(mime):
            text = extract_file_text(_download(drive.files().get_media(fileId=file_id, supportsAllDrives=True)), mime)
        else:
            return report_tool_error("read_drive_file", ValueError(f"Can't read files of type {mime}."))
        return {
            "id": meta["id"],
            "name": meta["name"],
            "mimeType": mime,
            "content": text[:MAX_READ_CHARS],
            "truncated": len(text) > MAX_READ_CHARS,
        }
    except Exception as e:
        return report_tool_error("read_drive_file", e)


def move_drive_file(file_id: str, folder_id: str) -> dict:
    """Move a Drive file into the given folder, removing it from its previous folder(s)."""
    drive = _drive()
    current = drive.files().get(fileId=file_id, fields="parents", supportsAllDrives=True).execute()
    previous = ",".join(p for p in current.get("parents", []) if p != folder_id)
    return drive.files().update(
        fileId=file_id,
        addParents=folder_id,
        removeParents=previous or None,
        fields=FILE_FIELDS,
        supportsAllDrives=True,
    ).execute()
