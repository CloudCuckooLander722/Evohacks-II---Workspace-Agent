"""Shared error reporting for workspace tools."""

import logging

logger = logging.getLogger(__name__)


def report_tool_error(tool_name: str, error: Exception) -> dict:
    """Log a tool failure and return an error payload the agent can read."""
    logger.exception("Tool %s failed: %s", tool_name, error)
    return {"status": "error", "tool": tool_name, "error": f"{type(error).__name__}: {error}"}
