"""
MCP Discover command - Discovers MCP servers and optionally probes them
for tools, resources, and prompts.
"""

import json
from typing import Any

import click
from rich import print

from ggshield.cmd.utils.common_options import add_common_options
from ggshield.cmd.utils.context_obj import ContextObj
from ggshield.core import ui
from ggshield.core.client import create_client_from_config
from ggshield.core.errors import APIKeyCheckError, UnknownInstanceError
from ggshield.verticals.ai.discovery import (
    discover_ai_configuration,
    submit_ai_discovery,
)


@click.command(name="discover")
@click.option(
    "--json",
    "use_json",
    is_flag=True,
    default=False,
    help="Output as JSON",
)
@add_common_options()
@click.pass_context
def discover_cmd(
    ctx: click.Context,
    use_json: bool,
    **kwargs: Any,
) -> None:
    """
    Discover MCP servers and their configuration.

    Parses MCP configuration files from supported assistants

    Examples:
      ggshield mcp discover
      ggshield mcp discover --json
    """

    config = discover_ai_configuration()

    if use_json:
        click.echo(json.dumps(config.to_dict(), indent=2))
    else:
        print(config)

    ctx_obj = ContextObj.get(ctx)
    try:
        client = create_client_from_config(ctx_obj.config)
    except (APIKeyCheckError, UnknownInstanceError) as exc:
        ui.display_warning(
            f"Skipping upload of AI discovery to GitGuardian ({exc}). "
            "Authenticate with `ggshield auth login` to enable upload."
        )
        return

    try:
        submit_ai_discovery(client, config)
    except Exception as exc:
        ui.display_warning(f"Could not upload AI discovery to GitGuardian: {exc}")
