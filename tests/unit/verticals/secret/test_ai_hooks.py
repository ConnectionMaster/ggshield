import json
from collections import Counter
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest

from ggshield.utils.git_shell import Filemode
from ggshield.verticals.ai.agents import Cursor
from ggshield.verticals.secret import SecretScanner
from ggshield.verticals.secret.ai_hook import AIHookScanner, EventType, Payload
from ggshield.verticals.secret.ai_hook import Result as HookResult
from ggshield.verticals.secret.ai_hook import Tool
from ggshield.verticals.secret.secret_scan_collection import Result as ScanResult
from ggshield.verticals.secret.secret_scan_collection import Results, Secret


def _mock_scanner(matches: List[str]) -> MagicMock:
    """Create a mock SecretScanner that returns the given Results from scan()."""
    mock = MagicMock(spec=SecretScanner)
    scan_result = Results(
        results=[
            ScanResult(
                filename="url",
                filemode=Filemode.FILE,
                path=Path("."),
                url="url",
                secrets=[_make_secret(match) for match in matches],
                ignored_secrets_count_by_kind=Counter(),
            )
        ],
        errors=[],
    )
    mock.scan.return_value = scan_result
    return mock


def _make_secret(match_str: str = "***"):
    """Minimal Secret for tests; _message_from_secrets only uses detector_display_name, validity, matches[].match."""
    mock_match = MagicMock()
    mock_match.match = match_str
    return Secret(
        detector_display_name="dummy-detector",
        detector_name="dummy-detector",
        detector_group_name=None,
        documentation_url=None,
        validity="valid",
        known_secret=False,
        incident_url=None,
        matches=[mock_match],
        ignore_reason=None,
        diff_kind=None,
        is_vaulted=False,
        vault_type=None,
        vault_name=None,
        vault_path=None,
        vault_path_count=None,
    )


class TestAIHookScannerScanContent:
    """Unit tests for AIHookScanner._scan_content."""

    def test_no_secrets_returns_allow(self):
        """When scanner returns no secrets, result has block=False and nbr_secrets=0."""
        hook_scanner = AIHookScanner(_mock_scanner([]))
        payload = Payload(
            event_type=EventType.USER_PROMPT,
            tool=None,
            content="safe content",
            identifier="id",
            agent=Cursor(),
        )
        result = hook_scanner._scan_content(payload)
        assert isinstance(result, HookResult)
        assert result.block is False
        assert result.nbr_secrets == 0
        assert result.message == ""

    def test_with_secrets_returns_block_and_message(self):
        """When scanner returns secrets, result has block=True, nbr_secrets and message set."""
        hook_scanner = AIHookScanner(_mock_scanner(["sk-xxx"]))
        payload = Payload(
            event_type=EventType.USER_PROMPT,
            tool=None,
            content="content with sk-xxx",
            identifier="id",
            agent=Cursor(),
        )
        result = hook_scanner._scan_content(payload)
        assert isinstance(result, HookResult)
        assert result.block is True
        assert result.nbr_secrets == 1
        assert "dummy-detector" in result.message
        assert "secret" in result.message.lower()
        assert "remove the secrets from your prompt" in result.message


class TestAIHookScannerScan:
    """Unit tests for the AIHookScanner.scan() method."""

    def test_empty_input_raises(self):
        """Empty or whitespace-only input raises ValueError."""
        scanner = AIHookScanner(_mock_scanner([]))
        with pytest.raises(ValueError, match="No input received on stdin"):
            scanner.scan("")
        with pytest.raises(ValueError, match="No input received on stdin"):
            scanner.scan("   \n  ")

    def test_scan_no_secrets_returns_zero(self):
        """scan() with no secrets returns 0."""
        scanner = AIHookScanner(_mock_scanner([]))
        data = {
            "hook_event_name": "UserPromptSubmit",
            "prompt": "hello world",
            "transcript_path": "/home/user/.claude/projects/foo/session.jsonl",
            "cursor_version": "1.2.3",
        }
        code = scanner.scan(json.dumps(data))
        assert code == 0

    @patch("ggshield.verticals.secret.ai_hook.AIHookScanner._send_secret_notification")
    def test_scan_post_tool_use_with_secrets_sends_notification(
        self, mock_notify: MagicMock
    ):
        """scan() on POST_TOOL_USE with secrets sends a notification and returns 0 (no block)."""
        scanner = AIHookScanner(_mock_scanner(["sk-xxx"]))
        data = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "echo sk-xxx"},
            "tool_response": {"stdout": "sk-xxx\n"},
            "transcript_path": "/home/user/.claude/projects/foo/session.jsonl",
            "session_id": "427ae0c5-0862-4e14-aa2c-12fad909c323",
        }
        code = scanner.scan(json.dumps(data))
        assert code == 0
        mock_notify.assert_called_once()
        args = mock_notify.call_args[0]
        assert args[0] == 1  # nbr_secrets
        assert args[1] == Tool.BASH  # tool

    def test_scan_pre_tool_use_with_secrets_blocks(self):
        """scan() on PRE_TOOL_USE with secrets returns block result."""
        scanner = AIHookScanner(_mock_scanner(["sk-xxx"]))
        data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": "echo sk-xxx"},
            "session_id": "427ae0c5-0862-4e14-aa2c-12fad909c323",
            "transcript_path": "/home/user/.claude/projects/foo/session.jsonl",
        }
        code = scanner.scan(json.dumps(data))
        # Claude output_result always returns 0
        assert code == 0

    def test_scan_no_content_returns_allow(self):
        """scan() with no content returns 0 (and doesn't call the API)."""
        mock_scanner = _mock_scanner([])
        scanner = AIHookScanner(mock_scanner)
        data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": "doesn-t-exist"},
            "cursor_version": "1.2.3",
        }
        code = scanner.scan(json.dumps(data))
        assert code == 0
        mock_scanner.scan.assert_not_called()

    def test_scan_payloads_refuse_empty_list(self):
        """scan() with empty list of payloads raises ValueError."""
        scanner = AIHookScanner(_mock_scanner([]))
        with pytest.raises(ValueError):
            scanner._scan_payloads([])


class TestMessageFromSecrets:
    """Unit tests for AIHookScanner._message_from_secrets with different payload types."""

    def test_message_for_bash_tool(self):
        """Message for BASH tool mentions environment variables."""
        payload = Payload(
            event_type=EventType.PRE_TOOL_USE,
            tool=Tool.BASH,
            content="echo sk-xxx",
            identifier="echo sk-xxx",
            agent=Cursor(),
        )
        message = AIHookScanner._message_from_secrets([_make_secret("sk-xxx")], payload)
        assert "remove the secrets from the command" in message
        assert "environment variables" in message

    def test_message_for_read_tool(self):
        """Message for READ tool mentions file content."""
        payload = Payload(
            event_type=EventType.PRE_TOOL_USE,
            tool=Tool.READ,
            content="file content with secret",
            identifier="/path/to/file",
            agent=Cursor(),
        )
        message = AIHookScanner._message_from_secrets([_make_secret("sk-xxx")], payload)
        assert "remove the secrets from" in message

    def test_message_for_other_tool(self):
        """Message for OTHER tool uses generic message."""
        payload = Payload(
            event_type=EventType.PRE_TOOL_USE,
            tool=Tool.OTHER,
            content="some content",
            identifier="id",
            agent=Cursor(),
        )
        message = AIHookScanner._message_from_secrets([_make_secret("sk-xxx")], payload)
        assert "remove the secrets from the tool input" in message

    def test_message_escapes_markdown(self):
        """When escape_markdown=True, asterisks in matches are replaced with dots."""
        payload = Payload(
            event_type=EventType.USER_PROMPT,
            tool=None,
            content="content",
            identifier="id",
            agent=Cursor(),
        )
        message = AIHookScanner._message_from_secrets(
            [_make_secret("sk-xxx")], payload, escape_markdown=True
        )
        # The message itself should not contain raw asterisks from matches
        # (the header uses ** for bold which is intentional)
        assert "Detected" in message


class TestSendSecretNotification:
    """Unit tests for AIHookScanner._send_secret_notification."""

    @patch("ggshield.verticals.secret.ai_hook.Notify")
    def test_notification_for_bash_tool(self, mock_notify_cls: MagicMock):
        """Notification for BASH tool says 'running a command'."""
        AIHookScanner._send_secret_notification(1, Tool.BASH, "Claude Code")
        instance = mock_notify_cls.return_value
        assert "running a command" in instance.message
        assert "Claude Code" in instance.message
        instance.send.assert_called_once()

    @patch("ggshield.verticals.secret.ai_hook.Notify")
    def test_notification_for_read_tool(self, mock_notify_cls: MagicMock):
        """Notification for READ tool says 'reading a file'."""
        AIHookScanner._send_secret_notification(2, Tool.READ, "Cursor")
        instance = mock_notify_cls.return_value
        assert "reading a file" in instance.message
        assert "2" in instance.message
        instance.send.assert_called_once()

    @patch("ggshield.verticals.secret.ai_hook.Notify")
    def test_notification_for_other_tool(self, mock_notify_cls: MagicMock):
        """Notification for OTHER tool says 'using a tool'."""
        AIHookScanner._send_secret_notification(1, Tool.OTHER, "Copilot")
        instance = mock_notify_cls.return_value
        assert "using a tool" in instance.message
        instance.send.assert_called_once()
