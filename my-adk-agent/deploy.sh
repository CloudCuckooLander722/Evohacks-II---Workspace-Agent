#!/usr/bin/env bash
# Deploy the Streamlit UI + workspace_agent to Google Cloud Run.
#
# Usage (from my-adk-agent/):
#   PROJECT_ID=my-project ./deploy.sh
#
# Optional:
#   REGION=us-central1              Cloud Run region
#   SERVICE=workspace-agent         Cloud Run service name
#   DRIVE_CREDENTIALS_FILE=token.json
#       A Google credentials JSON for Drive (authorized-user token or service-account key).
#       Stored in Secret Manager and mounted at /secrets/google/credentials.json.
#       If omitted, the app uses the Cloud Run service account, which only sees Drive
#       folders that are shared with it.
#   ALLOW_PUBLIC=1                  Make the URL public. Default: callers need IAM
#                                   (roles/run.invoker), because anyone who can open the
#                                   app can create, move and trash files in the linked Drive.
#
# Requires: gcloud authenticated (gcloud auth login), and GEMINI_API_KEY in the environment
# (or in workspace_agent/.env).
set -euo pipefail

cd "$(dirname "$0")"

if [[ -f workspace_agent/.env ]]; then
  # Only pull the two values this script needs; nothing else from .env is exported.
  : "${GEMINI_API_KEY:=$(grep -E '^GEMINI_API_KEY=' workspace_agent/.env | head -1 | cut -d= -f2- | tr -d '"' || true)}"
  : "${PROJECT_ID:=$(grep -E '^PROJECT_ID=' workspace_agent/.env | head -1 | cut -d= -f2- | tr -d '"' || true)}"
fi

: "${PROJECT_ID:?Set PROJECT_ID}"
: "${GEMINI_API_KEY:?Set GEMINI_API_KEY}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-workspace-agent}"

gcloud config set project "$PROJECT_ID" >/dev/null
gcloud services enable run.googleapis.com cloudbuild.googleapis.com artifactregistry.googleapis.com \
  secretmanager.googleapis.com drive.googleapis.com slides.googleapis.com

PROJECT_NUMBER="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
RUNTIME_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

upsert_secret() {  # name, file
  if gcloud secrets describe "$1" >/dev/null 2>&1; then
    gcloud secrets versions add "$1" --data-file="$2" >/dev/null
  else
    gcloud secrets create "$1" --replication-policy=automatic --data-file="$2" >/dev/null
  fi
  gcloud secrets add-iam-policy-binding "$1" --member="serviceAccount:${RUNTIME_SA}" \
    --role=roles/secretmanager.secretAccessor >/dev/null
}

printf '%s' "$GEMINI_API_KEY" | upsert_secret "${SERVICE}-gemini-api-key" -
SECRETS="GEMINI_API_KEY=${SERVICE}-gemini-api-key:latest"
ENV_VARS="WORKSPACE_AGENT_LOAD_DOTENV=0"

if [[ -n "${DRIVE_CREDENTIALS_FILE:-}" ]]; then
  upsert_secret "${SERVICE}-drive-credentials" "$DRIVE_CREDENTIALS_FILE"
  SECRETS="${SECRETS},/secrets/google/credentials.json=${SERVICE}-drive-credentials:latest"
  ENV_VARS="${ENV_VARS},GOOGLE_CREDENTIALS_PATH=/secrets/google/credentials.json"
fi

AUTH_FLAG="--no-allow-unauthenticated"
[[ "${ALLOW_PUBLIC:-0}" == "1" ]] && AUTH_FLAG="--allow-unauthenticated"

# Sessions live in memory (InMemorySessionService), so keep one instance with sticky sessions;
# otherwise a confirmation could land on an instance that never saw the request.
gcloud run deploy "$SERVICE" \
  --source . \
  --region "$REGION" \
  --set-secrets "$SECRETS" \
  --set-env-vars "$ENV_VARS" \
  --session-affinity \
  --min-instances 0 \
  --max-instances 1 \
  --timeout 900 \
  --memory 1Gi \
  "$AUTH_FLAG"

gcloud run services describe "$SERVICE" --region "$REGION" --format='value(status.url)'
