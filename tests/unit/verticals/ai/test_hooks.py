import json
from pathlib import Path
from typing import Set
from unittest.mock import MagicMock, patch

import pytest

from ggshield.verticals.ai.agents import Claude, Copilot, Cursor
from ggshield.verticals.ai.hooks import find_filepaths, parse_hook_input
from ggshield.verticals.ai.models import EventType, HookPayload, HookResult, Tool


def _dummy_payload(event_type: EventType = EventType.OTHER) -> HookPayload:
    return HookPayload(
        event_type=event_type,
        tool=None,
        content="",
        identifier="",
        agent=Cursor(),
    )


@pytest.fixture
def tmp_file(tmp_path: Path) -> Path:
    """Create a temporary file with content."""
    file = tmp_path / "test.txt"
    file.write_text("this is the content")
    return file


class TestAIHookScannerParseInput:
    """Unit tests for AIHookparse_hook_input."""

    def test_invalid_json_raises(self):
        """Invalid JSON raises ValueError with parse error."""
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_hook_input("not json {")
        with pytest.raises(ValueError, match="Failed to parse JSON"):
            parse_hook_input("{ missing brace ")

    def test_missing_event_type_raises(self):
        """JSON without event type raises ValueError."""
        with pytest.raises(ValueError):
            parse_hook_input('{"prompt": "hello"}')

    def test_cursor_user_prompt(self):
        """Test Cursor beforeSubmitPrompt (user prompt) parsing."""
        data = {
            "conversation_id": "75fed8a8-2078-4e49-80d2-776b20d441c3",
            "generation_id": "1501ede6-b8ac-43f4-9943-0e218610c5c6",
            "model": "default",
            "prompt": "hello world",
            "attachments": [],
            "hook_event_name": "beforeSubmitPrompt",
            "cursor_version": "2.5.25",
            "workspace_roots": ["/home/user1/foo"],
            "user_email": "user@example.com",
            "transcript_path": "/home/user1/.cursor/projects/foo/agent-transcripts/75fed8a8/75fed8a8.jsonl",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.USER_PROMPT
        assert payload.content == "hello world"
        assert payload.tool is None
        assert payload.identifier != ""
        assert isinstance(payload.agent, Cursor)

    def test_cursor_pre_tool_use_shell(self):
        """Test Cursor preToolUse with Shell (bash) parsing."""
        data = {
            "conversation_id": "37a17cfc-322c-47ab-88c5-e810f23f4739",
            "generation_id": "049f5b26-326a-4081-82c1-e5c42a63d19e",
            "model": "default",
            "tool_name": "Shell",
            "tool_input": {
                "command": "whoami",
                "cwd": "",
                "timeout": 30000,
            },
            "tool_use_id": "ec1b1027-5b24-4a18-90c7-f8f616d0aeb4",
            "hook_event_name": "preToolUse",
            "cursor_version": "2.5.25",
            "workspace_roots": ["/home/user1/foo"],
            "transcript_path": "/home/user1/.cursor/projects/foo/agent-transcripts/37a17cfc/37a17cfc.jsonl",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.BASH
        assert payload.content == "whoami"
        assert payload.identifier == "whoami"
        assert isinstance(payload.agent, Cursor)

    def test_cursor_pre_tool_use_read(self, tmp_file: Path):
        """Test Cursor preToolUse with Read (file) parsing."""
        data = {
            "conversation_id": "75fed8a8-2078-4e49-80d2-776b20d441c3",
            "generation_id": "1501ede6-b8ac-43f4-9943-0e218610c5c6",
            "model": "default",
            "tool_name": "Read",
            "tool_input": {"file_path": tmp_file.as_posix()},
            "tool_use_id": "tool_fbfdb104-86a6-4111-a1bf-ce789f93cab",
            "hook_event_name": "preToolUse",
            "cursor_version": "2.5.25",
            "workspace_roots": ["/home/user1/foo"],
            "transcript_path": "/home/user1/.cursor/projects/foo/agent-transcripts/75fed8a8/75fed8a8.jsonl",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.READ
        assert payload.identifier == tmp_file.as_posix()
        assert payload.content == ""
        assert payload.scannable.content == "this is the content"
        assert isinstance(payload.agent, Cursor)

    def test_cursor_post_tool_use_shell(self):
        """Test Cursor postToolUse with Shell (simulated cat command result)."""
        data = {
            "conversation_id": "37a17cfc-322c-47ab-88c5-e810f23f4739",
            "generation_id": "049f5b26-326a-4081-82c1-e5c42a63d19e",
            "model": "default",
            "tool_name": "Shell",
            "tool_input": {"command": "whoami", "cwd": "", "timeout": 30000},
            "tool_output": '{"output":"user1","exitCode":0}',
            "duration": 280.475,
            "tool_use_id": "ec1b1027-5b24-4a18-90c7-f8f616d0aeb4",
            "hook_event_name": "postToolUse",
            "cursor_version": "2.5.25",
            "workspace_roots": ["/home/user1/foo"],
            "transcript_path": "/home/user/.cursor/projects/foo/agent-transcripts/37a17cfc/37a17cfc.jsonl",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.POST_TOOL_USE
        assert payload.tool == Tool.BASH
        assert "user1" in payload.content
        assert isinstance(payload.agent, Cursor)

    def test_claude_user_prompt(self):
        """Test Claude Code UserPromptSubmit parsing."""
        data = {
            "session_id": "273ad859-3608-4799-9971-fa15ecb1a65c",
            "transcript_path": "/home/user1/.claude/projects/foo/273ad859-3608-4799-9971-fa15ecb1a65c.jsonl",
            "cwd": "/home/user1/foo",
            "permission_mode": "default",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "hello world",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.USER_PROMPT
        assert payload.content == "hello world"
        assert payload.tool is None
        assert isinstance(payload.agent, Claude)

    def test_claude_pre_tool_use_bash(self):
        """Test Claude Code PreToolUse with Bash parsing."""
        data = {
            "session_id": "3b7ae0c5-0862-4e14-aa2c-12fad909c323",
            "transcript_path": "/home/user1/.claude/projects/foo/3b7ae0c5.jsonl",
            "cwd": "/home/user1/foo",
            "permission_mode": "default",
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {
                "command": "whoami",
                "description": "whoami to test postTool hook",
            },
            "tool_use_id": "toolu_01BPMKeZAMCqBtn1xJRNfDJw",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.BASH
        assert "whoami" in payload.content
        assert isinstance(payload.agent, Claude)

    def test_claude_pre_tool_use_read(self, tmp_file: Path):
        """Test Claude Code PreToolUse with Read parsing."""
        # From raw_hooks_logs: Claude PreToolUse Read
        data = {
            "session_id": "3b7ae0c5-0862-4e14-aa2c-12fad909c323",
            "transcript_path": "/home/user1/.claude/projects/foo/3b7ae0c5.jsonl",
            "cwd": "/home/user1/foo",
            "permission_mode": "default",
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": tmp_file.as_posix()},
            "tool_use_id": "toolu_01WabtWJpzf1ZJ8GJ3JfQEmq",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.READ
        assert payload.identifier == tmp_file.as_posix()
        assert payload.content == ""
        assert payload.scannable.content == "this is the content"
        assert isinstance(payload.agent, Claude)

    def test_claude_post_tool_use_bash(self):
        """Test Claude Code PostToolUse with Bash (simulated cat command result)."""
        # From raw_hooks_logs: Claude PostToolUse Bash - tool_response has stdout
        data = {
            "session_id": "3b7ae0c5-0862-4e14-aa2c-12fad909c323",
            "transcript_path": "/home/user1/.claude/projects/foo/3b7ae0c5.jsonl",
            "cwd": "/home/user1/foo",
            "permission_mode": "default",
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {
                "command": "whoami",
                "description": "whoami to test postTool hook",
            },
            "tool_response": {
                "stdout": "user1\n",
                "stderr": "",
                "interrupted": False,
                "isImage": False,
                "noOutputExpected": False,
            },
            "tool_use_id": "toolu_01BPMKeZAMCqBtn1xJRNfDJw",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.POST_TOOL_USE
        assert payload.tool == Tool.BASH
        # Content is json.dumps(tool_response), so the stdout is inside the string
        assert "user1" in payload.content
        assert isinstance(payload.agent, Claude)

    def test_claude_parse_read_files_in_prompt(self):
        """Test parsing "@file_path" mentions from Claude Code prompt."""
        data = {
            "session_id": "273ad859-3608-4799-9971-fa15ecb1a65c",
            "transcript_path": "/home/user1/.claude/projects/foo/273ad859-3608-4799-9971-fa15ecb1a65c.jsonl",
            "cwd": "/home/user1/foo",
            "permission_mode": "default",
            "hook_event_name": "UserPromptSubmit",
            "prompt": "read @folder/file.txt and summarize the content.",
        }
        payloads = parse_hook_input(json.dumps(data))
        assert len(payloads) == 2
        payload = payloads[0]
        assert payload.event_type == EventType.USER_PROMPT
        assert payload.tool == Tool.READ
        assert payload.identifier == "folder/file.txt"
        assert payload.content == ""  # empty because inexistent file
        assert isinstance(payload.agent, Claude)

        payload = payloads[1]
        assert payload.event_type == EventType.USER_PROMPT
        assert payload.content == "read @folder/file.txt and summarize the content."
        assert payload.tool is None
        assert isinstance(payload.agent, Claude)

    def test_copilot_user_prompt(self):
        """Test Copilot UserPromptSubmit parsing."""
        data = {
            "timestamp": "2026-02-26T11:28:53.112Z",
            "hookEventName": "UserPromptSubmit",
            "sessionId": "69cc6a03-7034-4c49-8cf9-3805c292a15c",
            "transcript_path": (
                "/home/user1/.config/Code/User/workspaceStorage/"
                "abc123/GitHub.copilot-chat/transcripts/69cc6a03.jsonl"
            ),
            "prompt": "hello world",
            "cwd": "/home/user1/foo",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.USER_PROMPT
        assert "hello world" in payload.content
        assert payload.tool is None
        assert isinstance(payload.agent, Copilot)

    def test_copilot_pre_tool_use_run_in_terminal(self):
        """Test Copilot PreToolUse with run_in_terminal (shell) parsing."""
        # From raw_hooks_logs: Copilot PreToolUse run_in_terminal
        data = {
            "timestamp": "2026-02-26T11:29:05.821Z",
            "hookEventName": "PreToolUse",
            "sessionId": "69cc6a03-7034-4c49-8cf9-3805c292a15c",
            "transcript_path": (
                "/home/user1/.config/Code/User/workspaceStorage/"
                "abc123/GitHub.copilot-chat/transcripts/69cc6a03.jsonl"
            ),
            "tool_name": "run_in_terminal",
            "tool_input": {
                "command": "whoami",
                "explanation": "whoami to test preToolUse hook",
                "goal": "whoami to test preToolUse hook",
                "isBackground": False,
                "timeout": 0,
            },
            "tool_use_id": "call_ADJcoVxpnzPtpU6uf0h9wzLR__vscode-1772105116075",
            "cwd": "/home/user1/foo",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.BASH
        assert "whoami" in payload.content
        assert isinstance(payload.agent, Copilot)

    def test_copilot_pre_tool_use_read_file(self, tmp_file: Path):
        """Test Copilot PreToolUse with read_file parsing."""
        # From raw_hooks_logs: Copilot PreToolUse read_file (nonexistent path for deterministic test)
        data = {
            "timestamp": "2026-02-26T11:53:49.593Z",
            "hookEventName": "PreToolUse",
            "sessionId": "69cc6a03-7034-4c49-8cf9-3805c292a15c",
            "transcript_path": (
                "/home/user1/.config/Code/User/workspaceStorage/"
                "abc123/GitHub.copilot-chat/transcripts/69cc6a03.jsonl"
            ),
            "tool_name": "read_file",
            "tool_input": {
                "filePath": tmp_file.as_posix(),
                "startLine": 1,
                "endLine": 200,
            },
            "tool_use_id": "call_iMFuTGETQ2z23a3xYTqcHBXp__vscode-1772105116078",
            "cwd": "/home/user1/foo",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.READ
        assert payload.identifier == tmp_file.as_posix()
        assert payload.content == ""
        assert payload.scannable.content == "this is the content"
        assert isinstance(payload.agent, Copilot)

    def test_copilot_post_tool_use_run_in_terminal(self):
        """Test Copilot PostToolUse with run_in_terminal (simulated cat result)."""
        # From raw_hooks_logs: Copilot PostToolUse run_in_terminal - tool_response is string
        data = {
            "timestamp": "2026-02-26T11:53:47.392Z",
            "hookEventName": "PostToolUse",
            "sessionId": "69cc6a03-7034-4c49-8cf9-3805c292a15c",
            "transcript_path": (
                "/home/user1/.config/Code/User/workspaceStorage/"
                "abc123/GitHub.copilot-chat/transcripts/69cc6a03.jsonl"
            ),
            "tool_name": "run_in_terminal",
            "tool_input": {
                "command": "whoami",
                "explanation": "whoami to test postToolUse hook",
                "goal": "whoami to test postToolUse hook",
                "isBackground": False,
                "timeout": 0,
            },
            "tool_response": "user1",
            "tool_use_id": "call_f96KUoNCGS8jENVKnlWnSz5Q__vscode-1772105116077",
            "cwd": "/home/user1/foo",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.POST_TOOL_USE
        assert payload.tool == Tool.BASH
        assert "user1" in payload.content
        assert isinstance(payload.agent, Copilot)

    def test_pre_tool_use_read_with_missing_file(self):
        """PRE_TOOL_USE with tool_name 'read' and non-existing file yields empty content."""
        content = json.dumps(
            {
                "hook_event_name": "pretooluse",
                "tool_name": "read",
                "tool_input": {"file_path": "/nonexistent/path"},
                "cursor_version": "1.2.3",
            }
        )
        payload = parse_hook_input(content)[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.READ
        assert payload.identifier == "/nonexistent/path"
        assert payload.content == ""

    def test_pre_tool_use_other_tool(self):
        """PRE_TOOL_USE with unknown tool yields Tool.OTHER and empty content."""
        data = {
            "hook_event_name": "PreToolUse",
            "tool_name": "SomeUnknownTool",
            "tool_input": {"arg": "value"},
            "cursor_version": "1.2.3",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.PRE_TOOL_USE
        assert payload.tool == Tool.OTHER
        assert payload.content == ""

    def test_other_event_type(self):
        """Unknown event type yields EventType.OTHER with empty content."""
        data = {
            "hook_event_name": "SomeOtherEvent",
            "prompt": "hello",
            "cursor_version": "1.2.3",
        }
        payload = parse_hook_input(json.dumps(data))[0]
        assert payload.event_type == EventType.OTHER
        assert payload.content == ""
        assert payload.tool is None


class TestFlavorOutputResult:
    """Unit tests for Cursor, Claude, Copilot output_result with Result objects.

    Mocks click.echo to capture stdout/stderr and asserts both output and return code.
    """

    @patch("ggshield.verticals.ai.agents.cursor.click.echo")
    def test_cursor_output_result_user_prompt_allow(self, mock_echo: MagicMock):
        """Cursor USER_PROMPT with block=False: JSON to stdout, return 0."""
        result = HookResult.allow(_dummy_payload(EventType.USER_PROMPT))
        code = Cursor().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["continue"] is True
        assert out["user_message"] == ""

    @patch("ggshield.verticals.ai.agents.cursor.click.echo")
    def test_cursor_output_result_user_prompt_block(self, mock_echo: MagicMock):
        """Cursor USER_PROMPT with block=True: JSON to stdout, return 0."""
        result = HookResult(
            block=True,
            message="Remove secrets from prompt",
            nbr_secrets=1,
            payload=_dummy_payload(EventType.USER_PROMPT),
        )
        code = Cursor().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["continue"] is False
        assert out["user_message"] == "Remove secrets from prompt"

    @patch("ggshield.verticals.ai.agents.cursor.click.echo")
    def test_cursor_output_result_pre_tool_use_allow(self, mock_echo: MagicMock):
        """Cursor PRE_TOOL_USE with block=False: permission allow, return 0."""
        result = HookResult.allow(_dummy_payload(EventType.PRE_TOOL_USE))
        code = Cursor().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["permission"] == "allow"

    @patch("ggshield.verticals.ai.agents.cursor.click.echo")
    def test_cursor_output_result_pre_tool_use_block(self, mock_echo: MagicMock):
        """Cursor PRE_TOOL_USE with block=True: permission deny, return 0."""
        result = HookResult(
            block=True,
            message="Secrets detected in command",
            nbr_secrets=1,
            payload=_dummy_payload(EventType.PRE_TOOL_USE),
        )
        code = Cursor().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["permission"] == "deny"
        assert out["user_message"] == "Secrets detected in command"

    @patch("ggshield.verticals.ai.agents.cursor.click.echo")
    def test_cursor_output_result_post_tool_use(self, mock_echo: MagicMock):
        """Cursor POST_TOOL_USE: empty JSON to stdout, return 0."""
        result = HookResult(
            block=True,
            message="Too late",
            nbr_secrets=1,
            payload=_dummy_payload(EventType.POST_TOOL_USE),
        )
        code = Cursor().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        assert json.loads(args[0]) == {}

    @patch("ggshield.verticals.ai.agents.cursor.click.echo")
    def test_cursor_output_result_other_block(self, mock_echo: MagicMock):
        """Cursor OTHER event with block: empty JSON, return 2."""
        result = HookResult(
            block=True,
            message="",
            nbr_secrets=1,
            payload=_dummy_payload(EventType.OTHER),
        )
        code = Cursor().output_result(result)
        assert code == 2
        mock_echo.assert_called_once_with("{}")

    @patch("ggshield.verticals.ai.agents.cursor.click.echo")
    def test_cursor_output_result_other_allow(self, mock_echo: MagicMock):
        """Cursor OTHER event without block: empty JSON, return 0."""
        result = HookResult.allow(_dummy_payload(EventType.OTHER))
        code = Cursor().output_result(result)
        assert code == 0
        mock_echo.assert_called_once_with("{}")

    @patch("ggshield.verticals.ai.agents.claude_code.click.echo")
    def test_claude_output_result_allow(self, mock_echo: MagicMock):
        """Claude with block=False: JSON continue true to stdout, return 0."""
        result = HookResult.allow(_dummy_payload(EventType.USER_PROMPT))
        code = Claude().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["continue"] is True

    @patch("ggshield.verticals.ai.agents.claude_code.click.echo")
    def test_claude_output_result_block(self, mock_echo: MagicMock):
        """Claude with block=True: JSON continue false and stopReason to stdout, return 0."""
        result = HookResult(
            block=True,
            message="Secrets in file",
            nbr_secrets=1,
            payload=_dummy_payload(EventType.PRE_TOOL_USE),
        )
        code = Claude().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny"
        assert (
            out["hookSpecificOutput"]["permissionDecisionReason"] == "Secrets in file"
        )

    @patch("ggshield.verticals.ai.agents.claude_code.click.echo")
    def test_copilot_output_result_allow(self, mock_echo: MagicMock):
        """Copilot with block=False: same as Claude, JSON to stdout, return 0."""
        result = HookResult.allow(_dummy_payload(EventType.USER_PROMPT))
        code = Copilot().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["continue"] is True
        assert "stopReason" not in out

    @patch("ggshield.verticals.ai.agents.claude_code.click.echo")
    def test_copilot_output_result_block(self, mock_echo: MagicMock):
        """Copilot with block=True: same as Claude, JSON to stdout, return 0."""
        result = HookResult(
            block=True,
            message="Secret in tool output",
            nbr_secrets=1,
            payload=_dummy_payload(EventType.POST_TOOL_USE),
        )
        code = Copilot().output_result(result)
        assert code == 0
        mock_echo.assert_called_once()
        args, kwargs = mock_echo.call_args
        assert kwargs.get("err", False) is False  # stdout (default)
        out = json.loads(args[0])
        assert out["decision"] == "block"
        assert out["reason"] == "Secret in tool output"

    @patch("ggshield.verticals.ai.agents.claude_code.click.echo")
    def test_copilot_other_result_block(self, mock_echo: MagicMock):
        """Copilot with block=True, other type of event"""
        result = HookResult(
            block=True,
            message="Secret in tool output",
            nbr_secrets=1,
            payload=_dummy_payload(EventType.OTHER),
        )
        code = Copilot().output_result(result)
        assert code == 0
        args, _ = mock_echo.call_args
        out = json.loads(args[0])
        assert not out["continue"]


@pytest.mark.parametrize(
    "prompt, filepaths",
    [
        ("read @folder/file.txt and summarize the content.", {"folder/file.txt"}),
        (
            "A multi-lineprompt with @file1 \n and @file2 \n and @file3 read.",
            {"file1", "file2", "file3"},
        ),
        ("@filename.txt", {"filename.txt"}),
        ("same @file @file twice", {"file"}),
        ("File can start with a dot: @.env", {".env"}),
        (
            "Files simply mentioned without @ prefix are not matched: foo.txt bar.txt.",
            set(),
        ),
        ("emails like foo@example.com are not matched.", set()),
        (
            "test @file.multiple.extensions.txt and @file2.txt",
            {"file.multiple.extensions.txt", "file2.txt"},
        ),
        ("files (@folder/foo.txt) can be between parentheses.", {"folder/foo.txt"}),
        ("files @can-contain-hyphens.txt", {"can-contain-hyphens.txt"}),
        (
            'Supports @"file with spaces (and comma, and parentheses) in name".',
            {"file with spaces (and comma, and parentheses) in name"},
        ),
        ('read @"file with \\" in its name.txt"', {'file with \\" in its name.txt'}),
        (
            "Path at the end of a sentence: @file.txt. Another one: @file2.txt.",
            {"file.txt", "file2.txt"},
        ),
        # Edge cases and extra coverage
        ("@ alone or at end: hello @", set()),
        ("@ only: @", set()),
        ('Empty quoted path: @""', set()),
        ("Unquoted path with comma: @a.txt, and @b.txt", {"a.txt", "b.txt"}),
        ("Unquoted path with semicolon: @x; @y", {"x", "y"}),
        ("Paths with underscores: @my_special_file.txt", {"my_special_file.txt"}),
        ("Windows-style path: read @src\\main.py", {"src\\main.py"}),
        (
            'Mixed quoted and unquoted: @config.json and @"big file.txt"',
            {"config.json", "big file.txt"},
        ),
        ("Newline before @: line1\n@file.txt", {"file.txt"}),
    ],
)
def test_find_filepaths(prompt: str, filepaths: Set[str]):
    """Test filepath regex."""
    assert find_filepaths(prompt) == filepaths, prompt
