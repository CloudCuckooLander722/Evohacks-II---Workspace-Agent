# Evohacks-II---Workspace-Agent
A demonstration of an agent that can edit and modify files in Google Drive, whether it be files or PDFs.

## Setup

```bash
python -m venv .venv
.venv/bin/pip install -r my-adk-agent/requirements.txt
.venv/bin/pip install -r my-adk-agent/requirements-ui.txt && .venv/bin/pip install "watchdog==7.0.0"
```

(streamlit declares `watchdog<7` and google-adk needs `watchdog>=7`; pip warns, and that is expected —
see `my-adk-agent/requirements-ui.txt`.)

Put configuration in `my-adk-agent/workspace_agent/.env` (git-ignored):

- `GEMINI_API_KEY` — for the Gemini models.
- `GOOGLE_CREDENTIALS_PATH` — a Google credentials JSON with the Drive and Slides APIs enabled in its
  project. Any of: an OAuth client ("Desktop app"; the first Drive call prints a consent URL and caches
  the login in `token.json`), an authorized-user token (such as that `token.json`), or a service-account
  key (sees only files shared with it). If unset, Application Default Credentials are used.

Never commit credential files or tokens.

## Run

```bash
cd my-adk-agent
../.venv/bin/streamlit run adk_service/streamlit_app.py
```

`adk_service/service.py` runs `workspace_agent` through an ADK `Runner`; `adk_service/streamlit_app.py`
is the chat UI on top of it, with Approve/Cancel buttons for the agent's confirmation prompts.

## Tests

Offline tests (Google APIs and Gemini are mocked; no credentials needed):

```bash
.venv/bin/pip install -r my-adk-agent/requirements-dev.txt
.venv/bin/pytest my-adk-agent/tests
```

## Deploy to Cloud Run

```bash
gcloud auth login
cd my-adk-agent
PROJECT_ID=<project> DRIVE_CREDENTIALS_FILE=token.json ./deploy.sh
```

The script stores `GEMINI_API_KEY` and the Drive credentials in Secret Manager, builds the
`Dockerfile` with Cloud Build, and deploys one instance with session affinity (conversations are
kept in memory). The service requires IAM authentication unless `ALLOW_PUBLIC=1`: anyone who can
open it can change the linked Drive. On Cloud Run, use an authorized-user token or a service-account
key; the interactive OAuth consent flow cannot run there.
