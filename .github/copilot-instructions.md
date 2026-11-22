# Repository instructions

## Project context

- This repository is currently in an initial, documentation-only state.
- The project goal is a demonstration agent that can edit and modify files in Google Drive, including regular files and PDFs.
- Treat the README as the source of truth for the project purpose.
- Do not assume that a framework, package manager, entry point, test runner, or Google Drive SDK has been selected unless repository files establish it.

## Working in this repository

- Make small, complete changes and preserve existing behavior.
- Keep the README and directly related documentation accurate when adding setup or runtime behavior.
- Follow existing conventions once implementation files exist. If there is no convention, choose the simplest conventional structure for the detected language and framework.
- Do not create speculative directories, dependencies, or abstractions before they are needed.
- When introducing a build or test tool, update the repository documentation with the exact setup and command.

## Implementation priorities

- Implement Google Drive operations through the official and supported Google APIs or client libraries.
- Keep file/PDF handling and Google Drive integration boundaries clear.
- Design operations around identifiable Drive files and explicit requested edits; do not make broad or destructive changes by default.
- Add safeguards for overwrite, deletion, and permission-sensitive operations.
- Treat local files, Drive metadata, document content, and API responses as untrusted input.
- Preserve original documents when editing or exporting, and avoid destructive in-place transformations unless explicitly requested.

## Security and credentials

- Never commit credentials, OAuth secrets, access tokens, refresh tokens, service-account keys, or `.env` values.
- Use environment variables or the repository's established secret-management pattern.
- Keep credential setup examples clearly marked as placeholders.
- Never log tokens, authorization headers, private document contents, or personal data.
- Respect least-privilege access and request only the Google scopes required for the implementation.

## Quality and verification

- Add focused tests for behavior that is introduced or corrected, especially Drive file selection, edit operations, PDF handling, authentication failures, and permission errors.
- Use mocks or fixtures for external Google APIs; do not require live credentials for unit tests.
- Report the exact verification commands run and their results.
- If no build, lint, or test command exists, inspect the repository rather than inventing or claiming one.
- Do not present unexecuted steps or unverified behavior as complete.
