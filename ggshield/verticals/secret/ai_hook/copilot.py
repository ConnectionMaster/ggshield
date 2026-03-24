import json
from pathlib import Path

import click

from .claude_code import Claude
from .models import EventType, Result


class Copilot(Claude):
    """Behavior specific to Copilot Chat.

    Inherits most of its behavior from Claude Code.
    """

    name = "Copilot"

    def output_result(self, result: Result) -> int:
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
