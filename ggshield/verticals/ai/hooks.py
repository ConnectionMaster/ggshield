import hashlib
import json
import re
from typing import Any, Dict, List, Sequence, Set

from ggshield.verticals.ai.agents import Claude, Copilot, Cursor

from .models import Agent, EventType, HookPayload, Tool


HOOK_NAME_TO_EVENT_TYPE = {
    "userpromptsubmit": EventType.USER_PROMPT,
    "beforesubmitprompt": EventType.USER_PROMPT,
    "pretooluse": EventType.PRE_TOOL_USE,
    "posttooluse": EventType.POST_TOOL_USE,
}

TOOL_NAME_TO_TOOL = {
    "shell": Tool.BASH,  # Cursor
    "bash": Tool.BASH,  # Claude Code
    "run_in_terminal": Tool.BASH,  # Copilot
    "read": Tool.READ,  # Claude/Cursor
    "read_file": Tool.READ,  # Copilot
}


def lookup(data: Dict[str, Any], keys: Sequence[str], default: Any = None) -> Any:
    """Returns the value of the first key found in a dictionary."""
    for key in keys:
        if key in data:
            return data[key]
    return default


# Regex (and method) to look for any @file_path in the prompt.
# A list of test cases can be found in test_hooks.py.
_FILE_PATH_REGEX = re.compile(
    r'@"((?:[^"\\]|\\.)*)"'  # quoted: @"..."
    r"|"
    r"(?:\W|^)@([\w/\\.-]+)",  # unquoted: @path
    re.MULTILINE,
)


def find_filepaths(prompt: str) -> Set[str]:
    """Find all file paths in the prompt."""
    paths = set()
    for m in _FILE_PATH_REGEX.finditer(prompt):
        path = m.group(1) or m.group(2) or ""
        path = path.strip()
        # Don't include trailing dots in the path
        if path.endswith("."):
            path = path[:-1]
        if path:
            paths.add(path)
    return paths


def parse_hook_input(raw_content: str) -> list[HookPayload]:
    """Parse the input content. Raises a ValueError if the input is not valid.

    Returns:
        A list of payloads. Most of the time the list will contain only one payload,
        but in some cases files mentioned in the prompt will be read but the
        PreToolUse event will not be called. So we need to handle this case ourselves.
    """
    # Parse the content as JSON
    if not raw_content.strip():
        raise ValueError("Error: No input received on stdin")
    try:
        data = json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise ValueError(f"Error: Failed to parse JSON from stdin: {e}") from e

    payloads = []

    # Try to guess which AI coding assistant is calling us
    agent = _detect_agent(data)

    # Infer the event type
    event_name = lookup(data, ["hook_event_name", "hookEventName"], None)
    if event_name is None:
        raise ValueError("Error: couldn't find event type")
    event_type = HOOK_NAME_TO_EVENT_TYPE.get(event_name.lower(), EventType.OTHER)

    identifier = ""
    content = ""
    tool = None

    # Extract the identifier and content based on the event type
    if event_type == EventType.USER_PROMPT:
        content = data.get("prompt", "")
        # Look for files mentioned in the prompt that could be read
        # without triggering a PRE_TOOL_USE event.
        payloads.extend(_parse_user_prompt(content, event_type, agent))

    elif event_type == EventType.PRE_TOOL_USE:
        tool_name = data.get("tool_name", "").lower()
        tool = TOOL_NAME_TO_TOOL.get(tool_name, Tool.OTHER)
        tool_input = data.get("tool_input", {})
        # Select the content based on the tool
        if tool == Tool.BASH:
            content = tool_input.get("command", "")
            identifier = content
        elif tool == Tool.READ:
            # We only need to deal with the identifier, the content will be read by the Scannable
            identifier = lookup(tool_input, ["file_path", "filePath"], "")

    elif event_type == EventType.POST_TOOL_USE:
        tool_name = data.get("tool_name", "").lower()
        tool = TOOL_NAME_TO_TOOL.get(tool_name, Tool.OTHER)
        content = data.get("tool_output", "") or data.get("tool_response", {})
        # Claude Code returns a dict for the tool output
        if isinstance(content, (dict, list)):
            content = json.dumps(content)

    # If identifier was not set, hash the content
    if not identifier:
        identifier = hashlib.sha256((content or "").encode()).hexdigest()

    payloads.append(
        HookPayload(
            event_type=event_type,
            tool=tool,
            content=content,
            identifier=identifier,
            agent=agent,
        )
    )
    return payloads


def _detect_agent(data: Dict[str, Any]) -> Agent:
    """Detect the AI code assistant."""
    if "cursor_version" in data:
        return Cursor()
    elif "github.copilot-chat" in data.get("transcript_path", "").lower():
        return Copilot()
    # no .lower() here to reduce the risk of false positives (this is also why this check is last)
    elif "session_id" in data and "claude" in data.get("transcript_path", ""):
        return Claude()
    # No other agent is supported yet
    raise ValueError("Unsupported agent")


def _parse_user_prompt(
    content: str, event_type: EventType, agent: Agent
) -> List[HookPayload]:
    """Parse the user prompt for additional payloads that we may miss."""
    payloads = []
    # Scenario 1 (the only one we know about so far):
    # Code assistants don't always trigger a PRE_TOOL_USE event when
    # a file is mentioned in the prompt, especially with an "@" prefix.
    matches = find_filepaths(content)
    for match in matches:
        payloads.append(
            HookPayload(
                event_type=event_type,
                tool=Tool.READ,
                content="",
                identifier=match,
                agent=agent,
            )
        )
    return payloads
