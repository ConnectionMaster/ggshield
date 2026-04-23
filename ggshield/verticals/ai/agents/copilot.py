import json
from pathlib import Path
from typing import Iterator

import click

from ggshield.core.dirs import get_user_home_dir

from ..models import EventType, HookResult
from .claude_code import Claude


class Copilot(Claude):
    """Behavior specific to Copilot Chat.

    Inherits most of its behavior from Claude Code.
    """

    @property
    def name(self) -> str:
        return "copilot"

    @property
    def display_name(self) -> str:
        return "Copilot Chat"

    @property
    def config_folder(self) -> Path:
        return get_user_home_dir() / ".config" / "Code" / "User"

    def output_result(self, result: HookResult) -> int:
        response = {}
        if result.block:
            if result.payload.event_type == EventType.PRE_TOOL_USE:
                response["hookSpecificOutput"] = {
                    "hookEventName": "PreToolUse",
                    "permissionDecision": "deny",
                    "permissionDecisionReason": result.message,
                }
            elif result.payload.event_type == EventType.POST_TOOL_USE:
                response["decision"] = "block"
                response["reason"] = result.message
            else:
                response["continue"] = False
                response["stopReason"] = result.message
        else:
            response["continue"] = True

        click.echo(json.dumps(response))
        return 0

    @property
    def settings_path(self) -> Path:
        return Path(".github") / "hooks" / "hooks.json"

    def project_mcp_file(self, directory: Path) -> Path:
        return directory / ".vscode" / "mcp.json"

    def discover_project_directories(self) -> Iterator[Path]:
        # Try to parse workspaces settings.
        for file in self.config_folder.glob("workspaceStorage/*/workspace.json"):
            if (data := self._load_json_file(file)) and "folder" in data:
                path = Path(data["folder"].removeprefix("file://"))
                if path.is_dir():
                    yield path.resolve()
