from typing import List

from notifypy import Notify

from ggshield.core.filter import censor_match
from ggshield.core.scanner_ui import create_message_only_scanner_ui
from ggshield.core.text_utils import pluralize, translate_validity
from ggshield.verticals.ai.hooks import parse_hook_input
from ggshield.verticals.ai.models import EventType
from ggshield.verticals.ai.models import HookPayload as Payload
from ggshield.verticals.ai.models import HookResult as Result
from ggshield.verticals.ai.models import Tool
from ggshield.verticals.secret import SecretScanner
from ggshield.verticals.secret.secret_scan_collection import Secret


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

        payloads = parse_hook_input(content)
        result = self._scan_payloads(payloads)
        payload = result.payload

        # Special case: in post-tool use, the action is already done: at least notify the user
        if result.block and payload.event_type == EventType.POST_TOOL_USE:
            self._send_secret_notification(
                result.nbr_secrets, payload.tool or Tool.OTHER, payload.agent.name
            )

        return payload.agent.output_result(result)

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

        if payload.tool == Tool.BASH:
            if payload.event_type == EventType.POST_TOOL_USE:
                message = "Secrets detected in the command output."
            else:
                message = (
                    "Please remove the secrets from the command before executing it. "
                    "Consider using environment variables or a secrets manager instead."
                )
        elif payload.tool == Tool.READ:
            message = f"Please remove the secrets from {payload.identifier} before reading it."
        elif payload.event_type == EventType.USER_PROMPT:
            message = "Please remove the secrets from your prompt before submitting."
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
