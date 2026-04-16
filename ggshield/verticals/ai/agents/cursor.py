import json
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import click

from ggshield.core.dirs import get_user_home_dir

from ..models import Agent, EventType, HookResult


class Cursor(Agent):
    """Behavior specific to Cursor."""

    @property
    def name(self) -> str:
        return "cursor"

    @property
    def display_name(self) -> str:
        return "Cursor"

    @property
    def config_folder(self) -> Path:
        return get_user_home_dir() / ".cursor"

    def output_result(self, result: HookResult) -> int:
        response = {}
        if result.payload.event_type == EventType.USER_PROMPT:
            response["continue"] = not result.block
            response["user_message"] = result.message
        elif result.payload.event_type == EventType.PRE_TOOL_USE:
            response["permission"] = "deny" if result.block else "allow"
            response["user_message"] = result.message
            response["agent_message"] = result.message
        elif result.payload.event_type == EventType.POST_TOOL_USE:
            pass  # Nothing to do here
        else:
            # Should not happen, but just in case
            click.echo("{}")
            return 2 if result.block else 0

        click.echo(json.dumps(response))
        # We don't use the return 2 convention to make sure our JSON output is read.
        return 0

    @property
    def settings_path(self) -> Path:
        return Path(".cursor") / "hooks.json"

    @property
    def settings_template(self) -> Dict[str, Any]:
        return {
            "version": 1,
            "hooks": {
                "beforeSubmitPrompt": [{"command": "<COMMAND>"}],
                "preToolUse": [{"command": "<COMMAND>"}],
                "postToolUse": [{"command": "<COMMAND>"}],
            },
        }

    def settings_locate(
        self, candidates: List[Dict[str, Any]], template: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        # We only have one kind of lists: in each hook. Simply look for "ggshield" or "<COMMAND>" in the command.
        for obj in candidates:
            command = obj.get("command", "")
            if "ggshield" in command or "<COMMAND>" in command:
                return obj
        return None

    def project_mcp_file(self, directory: Path) -> Path:
        return directory / ".cursor" / "mcp.json"

    def discover_project_directories(self) -> Iterator[Path]:
        # Because Cursor is based on VS Code, we can reuse the same logic than Copilot.
        user_folder = get_user_home_dir() / ".config" / "Cursor" / "User"
        for file in user_folder.glob("workspaceStorage/*/workspace.json"):
            if (data := self._load_json_file(file)) and "folder" in data:
                path = Path(data["folder"].removeprefix("file://"))
                if path.is_dir():
                    yield path.resolve()
