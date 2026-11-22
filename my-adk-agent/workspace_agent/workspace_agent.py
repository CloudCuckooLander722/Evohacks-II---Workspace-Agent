import sys
from pathlib import Path
from typing import Literal, Optional, Union

from google.adk import Context, Event, Workflow
from google.adk.agents import LlmAgent
from google.adk.events import RequestInput
from google.adk.workflow import START, node
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.tool_errors import report_tool_error
from tools.workspace_tools import (
    _create_document,
    _create_slidedeck,
    _create_pdf,
    _create_infographic,
    list_drive_folders,
    create_drive_folder,
    find_drive_folder,
    delete_drive_folder,
    search_drive_files,
    read_drive_file,
    move_drive_file,
    rank_folders,
)


# PYDANTIC SCHEMAS

CategoryName = Literal[
    "DOCUMENT_CREATION",
    "FOLDER_CREATION",
    "FOLDER_DELETION",
    "DOCUMENT_MOVEMENT",
    "UNKNOWN"] 


class WorkspaceCategory(BaseModel):
    model_config = ConfigDict(extra='forbid')

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

class DocumentCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')

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
    model_config = ConfigDict(extra='forbid')

    folder_name: str = Field(description="The name of the Drive folder to create or delete, as the user gave it.")


class MoveCommand(BaseModel):
    model_config = ConfigDict(extra='forbid')

    file_query: str = Field(
        description="The name (or the best search terms) of the existing Drive file to move, e.g. 'salmon report.pdf'."
    )
    folder_name: str = Field(
        description="The name of the destination Drive folder, e.g. 'Atlantic Salmon Research Project'."
    )


class Confirmation(BaseModel):
    approved: bool = Field(description="True to go ahead, false to cancel.")
    feedback: Optional[str] = Field(default=None, description="Optional corrections or reasons for cancelling.")


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

        Set `confidence` to the probability (0.0 to 1.0) that your category is correct.
        Restate the request in `request` as one self-contained instruction.
        If an essential detail is missing (e.g. which report to summarize, which file to move),
        describe it in `missing_info`; otherwise leave it null.
    """,
    generate_content_config=types.GenerateContentConfig(
        temperature=0.0
    ),
    output_schema=WorkspaceCategory,
    output_key="router_decision"
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
        - Summaries of a source document (e.g. a report into a slidedeck): find it with `search_drive_files`,
          read it with `read_drive_file`, and include all of its important information, organized into
          slides or sections.
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
    output_key="document_info"
)

folder_creation_agent = LlmAgent(
    name='folder_creation_agent',
    model='gemini-2.5-flash',
    instruction="Extract the name of the Drive folder the user wants to create.",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    output_schema=FolderCommand,
)

folder_deletion_agent = LlmAgent(
    name='folder_deletion_agent',
    model='gemini-2.5-flash',
    instruction="Extract the name of the Drive folder the user wants to delete.",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    output_schema=FolderCommand,
)

move_agent = LlmAgent(
    name='move_agent',
    model='gemini-2.5-flash',
    instruction="Extract which existing Drive file the user wants to move and the name of the destination folder.",
    generate_content_config=types.GenerateContentConfig(temperature=0.0),
    output_schema=MoveCommand,
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
    missing = decision.get("missing_info")
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


def move_document(node_input: dict):
    cmd = MoveCommand.model_validate(node_input)
    try:
        found = search_drive_files(cmd.file_query)
        if found.get("status") == "error":
            return Event(output=found)
        files = found.get("files", [])
        if not files:
            return Event(message=f"No Drive file matches '{cmd.file_query}'. Nothing was moved.")
        if len(files) > 1:
            names = ", ".join(f"'{f['name']}'" for f in files[:5])
            return Event(message=f"Several files match '{cmd.file_query}': {names}. Which one should I move?")
        file = files[0]
        folder, created = _get_or_create_folder(cmd.folder_name)
        move_drive_file(file["id"], folder["id"])
    except Exception as e:
        return Event(output=report_tool_error("move_document", e))
    where = f"new folder '{folder['name']}'" if created else f"'{folder['name']}'"
    return Event(message=f"Moved '{file['name']}' to {where}.")


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
