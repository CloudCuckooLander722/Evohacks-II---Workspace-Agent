import sys
from pathlib import Path
from typing import Literal, Optional, Union

from google.adk import Context, Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmRequest
from google.adk.events import RequestInput
from google.adk.workflow import START, node
from google.genai import types
from pydantic import BaseModel, Field, field_validator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.tool_errors import report_tool_error
from tools.workspace_tools import (
    _create_document,
    _create_slidedeck,
    _create_pdf,
    _create_infographic,
    extract_file_text,
    upload_drive_file,
    list_drive_folders,
    create_drive_folder,
    find_drive_folder,
    delete_drive_folder,
    search_drive_files,
    read_drive_file,
    move_drive_file,
    rank_folders,
)

# Values a model sometimes writes into missing_info when it means "nothing is missing".
_NOTHING_MISSING = {"", "none", "null", "n/a", "na", "nothing", "no", "nothing is missing", "nothing missing"}


def _clean_missing_info(value: Optional[str]) -> Optional[str]:
    """None when missing_info is empty or a placeholder like 'None' or 'N/A'."""
    if value is None or value.strip().strip(".").lower() in _NOTHING_MISSING:
        return None
    return value.strip()


# PYDANTIC SCHEMAS

CategoryName = Literal[
    "DOCUMENT_CREATION",
    "FOLDER_CREATION",
    "FOLDER_DELETION",
    "DOCUMENT_MOVEMENT",
    "UNKNOWN"] 


class WorkspaceCategory(BaseModel):

    category: CategoryName

    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Probability from 0.0 to 1.0 indicating confidence in this category.",
    )

    request: str = Field(
        description=(
            "The full request restated as one self-contained instruction, with every detail the user "
            "gave or the conversation already settled: the file type (Doc, Slides, PDF, infographic), "
            "topic and contents, any source document to build it from, the target folder name if one "
            "was given, which existing file or folder it refers to, and whether an answer key is wanted."
        )
    )

    missing_info: Optional[str] = Field(
        default=None,
        description="What essential detail is missing (e.g. 'no report shared'), or null if nothing is missing.",
    )

    @field_validator("missing_info")
    @classmethod
    def _null_placeholders(cls, value: Optional[str]) -> Optional[str]:
        return _clean_missing_info(value)

class DocumentCommand(BaseModel):

    title: str = Field(description="A short, descriptive title for the document.")

    file_type: Literal["DOC", "SLIDES", "PDF", "INFOGRAPHIC"] = Field(
        description="""The type of document to create, as specified by the user.
                    - "DOC": Google Docs documents
                    - "SLIDES": Google Slides presentations
                    - "PDF": PDF documents
                    - "INFOGRAPHIC": Infographic or visual display of information.""")

    folder_topic: str = Field(
        description=(
            "A short Drive folder name for the subject this document belongs to, e.g. 'Calculus II', "
            "'AP Chemistry', 'Atlantic Salmon Research Project'. Use the folder the user named, if any."
        )
    )

    answer_key: Optional[str] = Field(
        default=None,
        description=(
            "For quizzes, problem sets, and worksheets: worked answers to every question, in Markdown. "
            "Null for documents that have no questions."
        )
    )

    works_cited: Optional[list[str]] = Field(
        default=None,
        description="Optional list of sources cited by the agent."
    )

    content: str = Field(
        description=("""
            Content (e.g. problems, data, summaries ) derived from the users' request. 
        """)
    )



class FolderCommand(BaseModel):

    folder_name: str = Field(description="The name of the Drive folder to create or delete, as the user gave it.")


class MoveCommand(BaseModel):

    file_query: str = Field(
        description="The name (or the best search terms) of the existing Drive file to move, e.g. 'salmon report.pdf'."
    )
    folder_name: str = Field(
        description="The name of the destination Drive folder, e.g. 'Atlantic Salmon Research Project'."
    )


class Confirmation(BaseModel):
    approved: bool = Field(description="True to go ahead, false to cancel.")
    feedback: Optional[str] = Field(default=None, description="Optional corrections or reasons for cancelling.")


# ATTACHMENTS

# Inline types Gemini reads natively. Everything else (Excel, Word, CSV, ...) is converted to text.
_NATIVE_INLINE_PREFIXES = ("application/pdf", "image/", "audio/", "video/")


def _attached_files(content: Optional[types.Content]) -> list[types.Blob]:
    """Files the user attached to a message (inline bytes)."""
    if not content or not content.parts:
        return []
    return [p.inline_data for p in content.parts if p.inline_data and p.inline_data.data]


def _attachment_parts(blob: types.Blob) -> list[types.Part]:
    """A labelled, Gemini-readable version of one attached file."""
    name = blob.display_name or "attachment"
    mime = blob.mime_type or "application/octet-stream"
    label = types.Part(text=f"[Attached file: {name} ({mime})]")
    if mime.startswith(_NATIVE_INLINE_PREFIXES):
        # The Gemini API rejects display_name on inline data, so send the bytes without it.
        return [label, types.Part(inline_data=types.Blob(data=blob.data, mime_type=mime))]
    try:
        text = extract_file_text(blob.data, mime)
    except Exception as e:
        text = None
        report_tool_error("read_attachment", e)
    if text is None:
        return [types.Part(text=f"[Attached file: {name} ({mime}) - this file type can't be read.]")]
    return [label, types.Part(text=text)]


def prepare_attachments(callback_context: CallbackContext, llm_request: LlmRequest):
    """before_model_callback: make attached files readable by Gemini.

    Builds new Content objects rather than editing parts in place, because the request's contents
    can share objects with the session's events, and the original bytes must stay there (e.g. so
    move_document can upload an attached file).
    """
    contents = []
    for content in llm_request.contents:
        if not _attached_files(content):
            contents.append(content)
            continue
        parts = []
        for part in content.parts:
            parts.extend(_attachment_parts(part.inline_data) if part.inline_data and part.inline_data.data else [part])
        contents.append(types.Content(role=content.role, parts=parts))
    llm_request.contents = contents
    return None


# AGENTS

ROUTER_CONFIDENCE_THRESHOLD = 0.7

router_agent = LlmAgent(
    name='router_agent',
    model='gemini-2.5-flash',
    instruction="""
        You sort a Google Workspace request into exactly one category:
        - DOCUMENT_CREATION: create a new Google Doc, Google Slides deck, PDF, or infographic,
          e.g. "I want a quiz on curveball related rates problems", "Summarize this report into a slidedeck",
          "Create a PDF for me or an infographic".
        - FOLDER_CREATION: create a new Drive folder.
        - FOLDER_DELETION: delete an existing Drive folder.
        - DOCUMENT_MOVEMENT: move an existing Drive file into a folder,
          e.g. 'Move this PDF to drive folder "Atlantic Salmon Research Project"'.
        - UNKNOWN: anything that fits none of the above.

        Files: the user may attach a file to their message (it appears as "[Attached file: <name> ...]")
        or refer to a file already in their Google Drive.
        - "this report", "this PDF", etc. with an attached file means the attached file. Name it in `request`
          (e.g. "Summarize the attached file 'Q3 report.xlsx' into a slidedeck").
        - A file described by name or topic (e.g. "my Q3 report", "salmon.pdf") will be searched for in Drive.
          That is not missing information.

        Set `confidence` to the probability (0.0 to 1.0) that your category is correct.
        Restate the request in `request` as one self-contained instruction.
        Set `missing_info` only when the request cannot be carried out without something the user has not
        given, e.g. "Summarize this report" with no file attached and no name or topic to search for.
        Otherwise leave it null.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0
    ),
    output_schema=WorkspaceCategory,
    output_key="router_decision",
    before_model_callback=prepare_attachments,
)

content_agent = LlmAgent(
    name='content_agent',
    model='gemini-3.5-flash',
    instruction="""
        You handle DOCUMENT_CREATION requests. You write the content of the document the user asked for
        and choose the Drive folder it belongs in.

        File types:
        - DOC: Google Docs documents (e.g. quizzes, problem sets, worksheets, notes).
        - SLIDES: Google Slides presentations (e.g. "Summarize this report into a slidedeck").
        - PDF: PDF documents.
        - INFOGRAPHIC: flyers, infographics, or other visual displays of information.

        Writing the content:
        - Quizzes / problem sets (e.g. "a quiz on curveball related rates problems that test my concepts"):
          generate problems that genuinely test conceptual understanding, not just rote computation,
          and organize them clearly. Put worked answers to every question in `answer_key`.
        - Summaries of a source document (e.g. a report into a slidedeck): if the user attached it, it is
          in the conversation already ("[Attached file: ...]"); use that. Otherwise find it with
          `search_drive_files` and read it with `read_drive_file` (Docs, Slides, Sheets, PDFs, Excel, Word,
          text). Include all of its important information, organized into slides or sections.
        - PDFs and infographics follow the same process as Google Docs.
        - List any sources you relied on in `works_cited`.

        Choosing the folder:
        - Set `folder_topic` to the folder the user named, or else a short subject name
          (e.g. 'Calculus II', 'Atlantic Salmon Research Project'). You may call `list_drive_folders`
          to reuse the exact name of an existing folder on the same subject.
        - Do not create anything yourself. After the user confirms your draft, the workflow matches
          `folder_topic` to an existing Drive folder via similarity search, creates the folder if none
          matches, and places the document inside it.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0
    ),
    tools=[search_drive_files, read_drive_file, list_drive_folders],
    output_schema=DocumentCommand,
    output_key="document_info",
    before_model_callback=prepare_attachments,
)

folder_creation_agent = LlmAgent(
    name='folder_creation_agent',
    model='gemini-2.5-flash',
    instruction="Extract the name of the Drive folder the user wants to create.",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    output_schema=FolderCommand,
    before_model_callback=prepare_attachments,
)

folder_deletion_agent = LlmAgent(
    name='folder_deletion_agent',
    model='gemini-2.5-flash',
    instruction="Extract the name of the Drive folder the user wants to delete.",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    output_schema=FolderCommand,
    before_model_callback=prepare_attachments,
)

move_agent = LlmAgent(
    name='move_agent',
    model='gemini-2.5-flash',
    instruction=(
        "Extract which Drive file the user wants to move and the name of the destination folder. "
        "If the user attached the file, use the attached file's name as `file_query`."
    ),
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    output_schema=MoveCommand,
    before_model_callback=prepare_attachments,
)


# HELPERS

CREATORS = {
    "DOC": _create_document,
    "SLIDES": _create_slidedeck,
    "PDF": _create_pdf,
    "INFOGRAPHIC": _create_infographic,
}


def _is_approved(response) -> tuple[bool, Optional[str]]:
    """Interpret a confirmation reply, structured or free text."""
    if isinstance(response, Confirmation):
        return response.approved, response.feedback
    if isinstance(response, dict):
        return bool(response.get("approved")), response.get("feedback")
    text = str(response or "").strip().lower()
    return text in {"y", "yes", "approve", "approved", "ok", "confirm", "go ahead"}, None


def _get_or_create_folder(name: str) -> tuple[dict, bool]:
    """Match a folder by similarity search; create it if no existing folder matches."""
    match = find_drive_folder(name)
    if match and match.get("id"):
        return match, False
    return create_drive_folder(name), True


# NODES

def route_request(node_input: dict):
    """Branch on the router's category; low-confidence or incomplete requests go to UNKNOWN."""
    decision = WorkspaceCategory.model_validate(node_input)
    category = decision.category
    if decision.confidence < ROUTER_CONFIDENCE_THRESHOLD or decision.missing_info:
        category = "UNKNOWN"
    return Event(output=decision.request, route=category)


def handle_unknown(ctx: Context):
    decision = ctx.state.get("router_decision") or {}
    missing = _clean_missing_info(decision.get("missing_info"))
    reason = f"I still need: {missing}." if missing else "I couldn't tell what you want me to do."
    return Event(message=(
        f"{reason} I can create a Doc, Slides deck, PDF or infographic, create or delete a Drive "
        "folder, or move a file into a folder. Please rephrase or add the missing detail."
    ))


@node(rerun_on_resume=True)
def confirm_document(ctx: Context, node_input: dict):
    """Show the draft (and answer key) and wait for the user's approval before creating anything."""
    doc = DocumentCommand.model_validate(node_input)
    response = ctx.resume_inputs.get("confirm_document")
    if response is None:
        preview = f"**{doc.title}** ({doc.file_type}) → folder '{doc.folder_topic}'\n\n{doc.content}"
        if doc.answer_key:
            preview += f"\n\n---\n**Answer key**\n\n{doc.answer_key}"
        yield RequestInput(
            interrupt_id="confirm_document",
            message=preview + "\n\nCreate this document? Approve, or cancel with feedback.",
            response_schema=Confirmation,
        )
        return

    approved, feedback = _is_approved(response)
    if approved:
        yield Event(output=doc.model_dump(), route="APPROVED")
    else:
        note = f" Feedback: {feedback}" if feedback else ""
        yield Event(message=f"Cancelled; nothing was created.{note}", route="REJECTED")


def create_document(node_input: dict):
    doc = DocumentCommand.model_validate(node_input)
    try:
        folder, created = _get_or_create_folder(doc.folder_topic)
        content = doc.content
        if doc.works_cited:
            content += "\n\nWorks Cited\n" + "\n".join(f"- {src}" for src in doc.works_cited)
        result = CREATORS[doc.file_type](doc.title, content, folder["id"])
        if doc.answer_key:
            CREATORS["DOC"](f"{doc.title} - Answer Key", doc.answer_key, folder["id"])
    except Exception as e:
        return Event(output=report_tool_error("create_document", e))
    where = f"new folder '{folder['name']}'" if created else f"existing folder '{folder['name']}'"
    return Event(message=f"Created '{doc.title}' in {where}.", output=result)


def create_folder(node_input: dict):
    cmd = FolderCommand.model_validate(node_input)
    try:
        folder, created = _get_or_create_folder(cmd.folder_name)
    except Exception as e:
        return Event(output=report_tool_error("create_folder", e))
    if not created:
        return Event(message=f"A matching folder already exists: '{folder['name']}'. Nothing was created.")
    return Event(message=f"Created folder '{folder['name']}'.", output=folder)


@node(rerun_on_resume=True)
def delete_folder(ctx: Context, node_input: dict):
    """Find the folder and require explicit confirmation before deleting it."""
    cmd = FolderCommand.model_validate(node_input)
    try:
        match = find_drive_folder(cmd.folder_name)
    except Exception as e:
        yield Event(output=report_tool_error("find_drive_folder", e))
        return
    if not match or not match.get("id"):
        yield Event(message=f"No Drive folder matches '{cmd.folder_name}'. Nothing was deleted.")
        return

    response = ctx.resume_inputs.get("confirm_delete")
    if response is None:
        yield RequestInput(
            interrupt_id="confirm_delete",
            message=f"Move the Drive folder '{match['name']}' and everything in it to the trash?",
            response_schema=Confirmation,
        )
        return

    approved, _ = _is_approved(response)
    if not approved:
        yield Event(message="Cancelled; nothing was deleted.")
        return
    try:
        delete_drive_folder(match["id"])
    except Exception as e:
        yield Event(output=report_tool_error("delete_drive_folder", e))
        return
    yield Event(message=f"Moved folder '{match['name']}' to the trash (restorable from Drive's Trash for 30 days).")


@node(rerun_on_resume=True)
def move_document(ctx: Context, node_input: dict):
    """Move a Drive file into a folder, or upload the file the user attached into it.

    When several Drive files match, the best match by name is offered for confirmation.
    """
    cmd = MoveCommand.model_validate(node_input)
    try:
        attached = _attached_files(ctx.user_content)
        if attached:
            folder, created = _get_or_create_folder(cmd.folder_name)
            for blob in attached:
                upload_drive_file(blob.display_name or cmd.file_query, blob.data, blob.mime_type, folder["id"])
            names = ", ".join(f"'{b.display_name or cmd.file_query}'" for b in attached)
            where = f"new folder '{folder['name']}'" if created else f"'{folder['name']}'"
            yield Event(message=f"Uploaded {names} to {where}.")
            return

        found = search_drive_files(cmd.file_query)
        if found.get("status") == "error":
            yield Event(output=found)
            return
        files = rank_folders(cmd.file_query, found.get("files", []))
        if not files:
            yield Event(message=f"No Drive file matches '{cmd.file_query}'. Nothing was moved.")
            return
        file = files[0]

        if len(files) > 1:
            response = ctx.resume_inputs.get("confirm_move")
            if response is None:
                others = ", ".join(f"'{f['name']}'" for f in files[1:5])
                yield RequestInput(
                    interrupt_id="confirm_move",
                    message=(f"Several files match '{cmd.file_query}'. Move the closest match, '{file['name']}', "
                             f"to '{cmd.folder_name}'?\n\nOther matches: {others}"),
                    response_schema=Confirmation,
                )
                return
            approved, _ = _is_approved(response)
            if not approved:
                yield Event(message="Cancelled; nothing was moved. Tell me the file's exact name and I'll move that one.")
                return

        folder, created = _get_or_create_folder(cmd.folder_name)
        move_drive_file(file["id"], folder["id"])
    except Exception as e:
        yield Event(output=report_tool_error("move_document", e))
        return
    where = f"new folder '{folder['name']}'" if created else f"'{folder['name']}'"
    yield Event(message=f"Moved '{file['name']}' to {where}.")


# WORKFLOW

root_agent = Workflow(
    name='workspace_agent',
    edges=[
        (START, router_agent, route_request, {
            "DOCUMENT_CREATION": content_agent,
            "FOLDER_CREATION": folder_creation_agent,
            "FOLDER_DELETION": folder_deletion_agent,
            "DOCUMENT_MOVEMENT": move_agent,
            "UNKNOWN": handle_unknown,
        }),
        (content_agent, confirm_document, {"APPROVED": create_document}),
        (folder_creation_agent, create_folder),
        (folder_deletion_agent, delete_folder),
        (move_agent, move_document),
    ],
)
