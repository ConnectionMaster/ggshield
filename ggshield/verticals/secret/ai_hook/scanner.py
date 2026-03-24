import hashlib
import json
import re
from typing import Any, Dict, List, Sequence, Set

from notifypy import Notify

from ggshield.core.filter import censor_match
from ggshield.core.scanner_ui import create_message_only_scanner_ui
from ggshield.core.text_utils import pluralize, translate_validity
from ggshield.verticals.secret import SecretScanner
from ggshield.verticals.secret.ai_hook.copilot import Copilot
from ggshield.verticals.secret.secret_scan_collection import Secret

from .claude_code import Claude
from .cursor import Cursor
from .models import EventType, Flavor, Payload, Result, Tool


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


class AIHookScanner:
    """AI hook scanner.

    It is called with the payload of a hook event.
    Note that instead of having a base class with common method and a subclass per supported AI tool,
    we instead have a single class which detects which protocol to use (called "flavor").
    This is because some tools sloppily support hooks from others. For instance,
    Cursor will call hooks defined in the Claude Code format, but send payload in its own format.
    So we can't assume which tool will call us based on the command line/hook configuration only.

    Raises:
        ValueError: If the input is not valid.
    """

    def __init__(self, scanner: SecretScanner):
        self.scanner = scanner

    def scan(self, content: str) -> int:
        """Scan the content, print the result and return the exit code."""

        payloads = self._parse_input(content)
        result = self._scan_payloads(payloads)
        payload = result.payload

        # Special case: in post-tool use, the action is already done: at least notify the user
        if result.block and payload.event_type == EventType.POST_TOOL_USE:
            self._send_secret_notification(
                result.nbr_secrets, payload.tool or Tool.OTHER, payload.flavor.name
            )

        return payload.flavor.output_result(result)

    def _parse_input(self, raw_content: str) -> list[Payload]:
        """Parse the input content. Raises a ValueError if the input is not valid.

        Returns:
            A list of payloads. Most of the time the list will contain only one payload,
            but in some cases ("@" mention of files in Claude Code being the only known case so far)
            files mentioned in the prompt will be read but the PreToolUse event will not be called.
            So we need to handle this case ourselves.
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
        flavor = self._detect_flavor(data)

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
            payloads.extend(self._parse_user_prompt(content, event_type, flavor))

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
            Payload(
                event_type=event_type,
                tool=tool,
                content=content,
                identifier=identifier,
                flavor=flavor,
            )
        )
        return payloads

    def _scan_payloads(self, payloads: List[Payload]) -> Result:
        """Scan payloads for secrets using the SecretScanner.

        Returns:
            The result of the first blocking payload, or a non-blocking result.
            Raises a ValueError if the list is empty (we must have at least one to emit a result).
        """
        if not payloads:
            raise ValueError("Error: no payloads to scan")
        for payload in payloads:
            result = self._scan_content(payload)
            if result.block:
                return result
        return Result.allow(payloads[0])

    def _scan_content(
        self,
        payload: Payload,
    ) -> Result:
        """Scan content for secrets using the SecretScanner."""
        # Short path: if there is no content, no need to do an API call
        if payload.empty:
            return Result.allow(payload)

        with create_message_only_scanner_ui() as scanner_ui:
            results = self.scanner.scan([payload.scannable], scanner_ui=scanner_ui)
        # Collect all secrets from results
        secrets: List[Secret] = []
        for result in results.results:
            secrets.extend(result.secrets)

        if not secrets:
            return Result.allow(payload)

        message = self._message_from_secrets(
            secrets,
            payload,
            escape_markdown=True,
        )
        return Result(
            block=True,
            message=message,
            nbr_secrets=len(secrets),
            payload=payload,
        )

    @staticmethod
    def _detect_flavor(data: Dict[str, Any]) -> Flavor:
        """Detect the AI code assistant."""
        if "cursor_version" in data:
            return Cursor()
        elif "github.copilot-chat" in data.get("transcript_path", "").lower():
            return Copilot()
        # no .lower() here to reduce the risk of false positives (this is also why this check is last)
        elif "session_id" in data and "claude" in data.get("transcript_path", ""):
            return Claude()
        else:
            # Fallback that respect base conventions
            return Flavor()

    def _parse_user_prompt(
        self, content: str, event_type: EventType, flavor: Flavor
    ) -> List[Payload]:
        """Parse the user prompt for additional payloads that we may miss."""
        payloads = []
        # Scenario 1 (the only one we know about so far):
        # Claude Code doesn't always trigger a PRE_TOOL_USE event when
        # a file is mentioned in the prompt with an "@" prefix.
        # We restrict this to Claude Code as other assistants will trigger
        # another hook event for the file later and and we don't want to
        # unnecessarily scan files multiple times.
        if isinstance(flavor, Claude):
            # match multiple @file_path in the prompt
            matches = find_filepaths(content)
            for match in matches:
                payloads.append(
                    Payload(
                        event_type=event_type,
                        tool=Tool.READ,
                        content="",
                        identifier=match,
                        flavor=flavor,
                    )
                )
        return payloads

    @staticmethod
    def _message_from_secrets(
        secrets: List[Secret], payload: Payload, escape_markdown: bool = False
    ) -> str:
        """
        Format detected secrets into a user-friendly message.

        Args:
            secrets: List of detected secrets
            payload: Text to display after the secrets output
            escape_markdown: If True, escape asterisks to prevent markdown interpretation

        Returns:
            Formatted message describing the detected secrets
        """
        count = len(secrets)
        header = f"**🚨 Detected {count} {pluralize('secret', count)} 🚨**"

        secret_lines = []
        for secret in secrets:
            validity = translate_validity(secret.validity).lower()
            if validity == "valid":
                validity = f"**{validity}**"
            match_str = ", ".join(censor_match(m) for m in secret.matches)
            if escape_markdown:
                match_str = match_str.replace("*", "•")
            secret_lines.append(
                f"  - {secret.detector_display_name} ({validity}): {match_str}"
            )

        if payload.event_type == EventType.USER_PROMPT:
            message = "Please remove the secrets from your prompt before submitting."
        elif payload.tool == Tool.BASH:
            if payload.event_type == EventType.POST_TOOL_USE:
                message = "Secrets detected in the command output."
            else:
                message = (
                    "Please remove the secrets from the command before executing it. "
                    "Consider using environment variables or a secrets manager instead."
                )
        elif payload.tool == Tool.READ:
            message = (
                "Please remove the secrets from the file content before reading it."
            )
        else:
            message = (
                "Please remove the secrets from the tool input before executing. "
                "Consider using environment variables or a secrets manager instead."
            )

        secrets_block = "\n".join(secret_lines)
        return f"{header}\n{secrets_block}\n\n{message}"

    @staticmethod
    def _send_secret_notification(
        nbr_secrets: int, tool: Tool, agent_name: str
    ) -> None:
        """
        Send desktop notification when secrets are detected.

        Args:
            nbr_secrets: Number of detected secrets
            tool: Tool used to detect the secrets
            agent_name: Name of the agent that detected the secrets
        """
        source = "using a tool"
        if tool == Tool.READ:
            source = "reading a file"
        elif tool == Tool.BASH:
            source = "running a command"
        notification = Notify()
        notification.title = "ggshield - Secrets Detected"
        notification.message = (
            f"{agent_name} got access to {nbr_secrets}"
            f" {pluralize('secret', nbr_secrets)} by {source}"
        )
        notification.application_name = "ggshield"
        try:
            notification.send()
        except Exception:
            # This is best effort, we don't want to propagate an error
            # if the notification fails.
            pass


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
