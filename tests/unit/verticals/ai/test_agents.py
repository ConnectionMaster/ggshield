import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from pygitguardian.models import MCPToolInfo, UserInfo

from ggshield.verticals.ai.agents.claude_code import Claude, _mangle_server_name
from ggshield.verticals.ai.agents.copilot import Copilot
from ggshield.verticals.ai.agents.cursor import Cursor, _parse_tool_arguments
from ggshield.verticals.ai.models import (
    Agent,
    AIDiscovery,
    EventType,
    HookPayload,
    MCPConfiguration,
    MCPServer,
    Scope,
    Tool,
    Transport,
)


def _user() -> UserInfo:
    return UserInfo(
        hostname="host", username="user", machine_id="mid", user_email="u@e.com"
    )


def _cfg(
    name: str = "srv",
    agent: str = "cursor",
    scope: Scope = Scope.USER,
    project: Optional[Path] = None,
) -> MCPConfiguration:
    return MCPConfiguration(
        name=name,
        agent=agent,
        scope=scope,
        transport=Transport.STDIO,
        project=str(project) if project else None,
    )


def _ai_discovery(servers: Optional[List[MCPServer]] = None) -> AIDiscovery:
    return AIDiscovery(user=_user(), servers=servers or [], discovery_duration=0.1)


def _payload(
    agent: Agent, raw: Optional[Dict[str, Any]] = None, tool: Tool = Tool.MCP
) -> HookPayload:
    return HookPayload(
        event_type=EventType.PRE_TOOL_USE,
        tool=tool,
        content="",
        identifier="",
        agent=agent,
        raw=raw or {},
    )


# ===========================================================================
# Cursor
# ===========================================================================


class TestCursorDiscoverCapabilities:
    def _setup_mcps_folder(
        self, tmp_path: Path, project_path: Path, server_name: str
    ) -> Path:
        """Build a Cursor-style mcps/<server>/ folder layout and return the agent."""
        mangled = project_path.as_posix().replace("/", "-").lstrip("-")
        mcps_root = tmp_path / ".cursor" / "projects" / mangled / "mcps"
        server_dir = mcps_root / "user-my-server"
        server_dir.mkdir(parents=True, exist_ok=True)
        # SERVER_METADATA.json
        (server_dir / "SERVER_METADATA.json").write_text(
            json.dumps({"serverName": server_name})
        )
        return server_dir

    def test_populates_tools_resources_prompts(self, tmp_path: Path):
        project = Path("/home/user/project")
        server_dir = self._setup_mcps_folder(tmp_path, project, "my-mcp")

        (server_dir / "tools").mkdir()
        (server_dir / "tools" / "t1.json").write_text(
            json.dumps({"name": "do_thing", "description": "Does a thing"})
        )
        (server_dir / "resources").mkdir()
        (server_dir / "resources" / "r1.json").write_text(
            json.dumps({"uri": "file:///data", "name": "data"})
        )
        (server_dir / "prompts").mkdir()
        (server_dir / "prompts" / "p1.json").write_text(
            json.dumps({"name": "greeting", "description": "Says hi"})
        )

        cursor = Cursor()
        cfg = _cfg(name="my-mcp", agent="cursor", project=project)
        server = MCPServer(name="my-mcp", configurations=[cfg])

        with patch.object(
            type(cursor),
            "config_folder",
            new_callable=lambda: property(lambda self: tmp_path / ".cursor"),
        ):
            result = cursor.discover_capabilities(server)

        assert result is True
        assert len(server.tools) == 1
        assert server.tools[0].name == "do_thing"
        assert len(server.resources) == 1
        assert server.resources[0].uri == "file:///data"
        assert len(server.prompts) == 1
        assert server.prompts[0].name == "greeting"

    def test_status_md_present_returns_false(self, tmp_path: Path):
        project = Path("/home/user/project")
        server_dir = self._setup_mcps_folder(tmp_path, project, "my-mcp")
        (server_dir / "STATUS.md").write_text("disconnected")
        (server_dir / "tools").mkdir()
        (server_dir / "tools" / "t1.json").write_text(json.dumps({"name": "t"}))

        cursor = Cursor()
        cfg = _cfg(name="my-mcp", agent="cursor", project=project)
        server = MCPServer(name="my-mcp", configurations=[cfg])

        with patch.object(
            type(cursor),
            "config_folder",
            new_callable=lambda: property(lambda self: tmp_path / ".cursor"),
        ):
            result = cursor.discover_capabilities(server)

        assert result is False
        assert len(server.tools) == 0

    def test_no_matching_metadata_returns_false(self, tmp_path: Path):
        project = Path("/home/user/project")
        self._setup_mcps_folder(tmp_path, project, "other-server")

        cursor = Cursor()
        cfg = _cfg(name="my-mcp", agent="cursor", project=project)
        server = MCPServer(name="my-mcp", configurations=[cfg])

        with patch.object(
            type(cursor),
            "config_folder",
            new_callable=lambda: property(lambda self: tmp_path / ".cursor"),
        ):
            result = cursor.discover_capabilities(server)

        assert result is False

    def test_non_cursor_configuration_skipped(self):
        cursor = Cursor()
        cfg = _cfg(name="srv", agent="claude-code", project=Path("/proj"))
        server = MCPServer(name="srv", configurations=[cfg])
        assert cursor.discover_capabilities(server) is False


class TestCursorDiscoverProjectDirectories:
    def test_valid_workspace_json_yields_path(self, tmp_path: Path):
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        ws_storage = (
            tmp_path / ".config" / "Cursor" / "User" / "workspaceStorage" / "abc"
        )
        ws_storage.mkdir(parents=True)
        (ws_storage / "workspace.json").write_text(
            json.dumps({"folder": f"file://{project_dir}"})
        )

        cursor = Cursor()
        with patch.object(
            type(cursor),
            "config_folder",
            new_callable=lambda: property(
                lambda self: tmp_path / ".config" / "Cursor" / "User"
            ),
        ):
            with patch(
                "ggshield.verticals.ai.agents.cursor.get_user_home_dir",
                return_value=tmp_path,
            ):
                dirs = list(cursor.discover_project_directories())

        assert project_dir.resolve() in dirs

    def test_missing_folder_key_skipped(self, tmp_path: Path):
        ws_storage = (
            tmp_path / ".config" / "Cursor" / "User" / "workspaceStorage" / "abc"
        )
        ws_storage.mkdir(parents=True)
        (ws_storage / "workspace.json").write_text(json.dumps({"other": "val"}))

        cursor = Cursor()
        with patch.object(
            type(cursor),
            "config_folder",
            new_callable=lambda: property(
                lambda self: tmp_path / ".config" / "Cursor" / "User"
            ),
        ):
            with patch(
                "ggshield.verticals.ai.agents.cursor.get_user_home_dir",
                return_value=tmp_path,
            ):
                dirs = list(cursor.discover_project_directories())

        assert dirs == []


class TestCursorParseMcpActivity:
    def test_strips_mcp_prefix_and_maps_server(self):
        cursor = Cursor()
        tool_info = MCPToolInfo(name="run_query")
        server = MCPServer(
            name="my-db-server",
            tools=[tool_info],
            configurations=[_cfg(name="db", agent="cursor")],
        )
        discovery = _ai_discovery(servers=[server])
        payload = _payload(
            cursor,
            raw={
                "tool_name": "MCP:run_query",
                "model": "gpt-4",
                "workspace_roots": ["/home/user/proj"],
                "tool_input": {"sql": "SELECT 1"},
            },
        )

        req = cursor.parse_mcp_activity(payload, discovery)

        assert req.tool == "run_query"
        assert req.server == "my-db-server"
        assert req.model == "gpt-4"
        assert req.input == {"sql": "SELECT 1"}

    def test_unknown_tool_returns_empty_server(self):
        cursor = Cursor()
        discovery = _ai_discovery(servers=[])
        payload = _payload(cursor, raw={"tool_name": "MCP:unknown"})

        req = cursor.parse_mcp_activity(payload, discovery)

        assert req.tool == "unknown"
        assert req.server == ""


# ===========================================================================
# Claude Code
# ===========================================================================


class TestClaudeGetUserMcpConfigurations:
    def test_user_level_and_project_level_parsed(self, tmp_path: Path):
        project_dir = tmp_path / "myproject"
        project_dir.mkdir()
        claude_json = {
            "mcpServers": {"global-srv": {"command": "npx", "args": ["-y", "mcp"]}},
            "projects": {
                str(project_dir): {
                    "mcpServers": {
                        "project-srv": {"command": "node", "args": ["index.js"]}
                    }
                }
            },
        }
        with patch(
            "ggshield.verticals.ai.agents.claude_code.get_user_home_dir",
            return_value=tmp_path,
        ):
            (tmp_path / ".claude.json").write_text(json.dumps(claude_json))
            claude = Claude()
            configs = list(claude._get_user_mcp_configurations())

        names = {c.name for c in configs}
        assert "global-srv" in names
        assert "project-srv" in names

    def test_missing_file_yields_nothing(self, tmp_path: Path):
        with patch(
            "ggshield.verticals.ai.agents.claude_code.get_user_home_dir",
            return_value=tmp_path,
        ):
            claude = Claude()
            configs = list(claude._get_user_mcp_configurations())
        assert configs == []


class TestClaudeDiscoverProjectDirectories:
    def test_yields_existing_directories(self, tmp_path: Path):
        project = tmp_path / "proj"
        project.mkdir()
        history = tmp_path / ".claude" / "history.jsonl"
        history.parent.mkdir(parents=True)
        history.write_text(json.dumps({"project": str(project)}) + "\n")

        claude = Claude()
        with patch.object(
            type(claude),
            "config_folder",
            new_callable=lambda: property(lambda self: tmp_path / ".claude"),
        ):
            dirs = list(claude.discover_project_directories())

        assert project.resolve() in dirs

    def test_skips_nonexistent_directories(self, tmp_path: Path):
        history = tmp_path / ".claude" / "history.jsonl"
        history.parent.mkdir(parents=True)
        history.write_text(
            json.dumps({"project": str(tmp_path / "nonexistent")}) + "\n"
        )

        claude = Claude()
        with patch.object(
            type(claude),
            "config_folder",
            new_callable=lambda: property(lambda self: tmp_path / ".claude"),
        ):
            dirs = list(claude.discover_project_directories())

        assert dirs == []


class TestClaudeParseMcpActivity:
    def test_parses_mcp_double_underscore_format(self):
        claude = Claude()
        cfg = _cfg(name="my.server", agent="claude-code")
        server = MCPServer(
            name="my.server", configurations=[cfg], tools=[MCPToolInfo(name="run")]
        )
        discovery = _ai_discovery(servers=[server])
        # Claude mangles "my.server" → "my_server" in the tool name
        payload = _payload(
            claude,
            raw={"tool_name": "mcp__my_server__run", "cwd": "/tmp", "tool_input": {}},
        )

        req = claude.parse_mcp_activity(payload, discovery)

        assert req.tool == "run"
        assert req.server == "my.server"

    def test_server_with_double_underscore_handled(self):
        claude = Claude()
        discovery = _ai_discovery(servers=[])
        payload = _payload(
            claude,
            raw={
                "tool_name": "mcp__a__b__tool_name",
                "cwd": "/tmp",
                "tool_input": {},
            },
        )

        req = claude.parse_mcp_activity(payload, discovery)

        assert req.tool == "tool_name"
        assert req.server == "a__b"  # falls back to mangled name

    def test_fallback_to_mangled_name(self):
        claude = Claude()
        discovery = _ai_discovery(servers=[])
        payload = _payload(
            claude,
            raw={"tool_name": "mcp__unknown__do_it", "cwd": "/tmp", "tool_input": {}},
        )

        req = claude.parse_mcp_activity(payload, discovery)

        assert req.server == "unknown"


# ---------------------------------------------------------------------------
# _mangle_server_name
# ---------------------------------------------------------------------------


class TestMangleServerName:
    @pytest.mark.parametrize(
        "name, expected",
        [
            pytest.param("my-seRver-123", "my-seRver-123", id="alphanumeric_dashes"),
            pytest.param(
                "my.server/v2 alpha", "my_server_v2_alpha", id="special_chars"
            ),
            pytest.param("simple", "simple", id="plain_alpha"),
            pytest.param("a@b#c", "a_b_c", id="symbols"),
        ],
    )
    def test_mangle_server_name(self, name: str, expected: str):
        assert _mangle_server_name(name) == expected


# ===========================================================================
# Copilot
# ===========================================================================


class TestCopilotParseMcpActivity:
    def test_simple_server_tool_split(self):
        copilot = Copilot()
        cfg = _cfg(name="myserver", agent="copilot")
        server = MCPServer(name="othername", configurations=[cfg])
        discovery = _ai_discovery(servers=[server])
        payload = _payload(
            copilot,
            raw={"tool_name": "mcp_myserver_mytool", "cwd": "/tmp", "tool_input": {}},
        )

        req = copilot.parse_mcp_activity(payload, discovery)

        assert req.tool == "mytool"
        assert req.server == "othername"

    def test_multiple_underscores(self):
        """Tools with underscores in their name are supported."""
        copilot = Copilot()
        discovery = _ai_discovery(servers=[])
        payload = _payload(
            copilot,
            raw={
                "tool_name": "mcp_server_tool_name_extra",
                "cwd": "/tmp",
                "tool_input": {},
            },
        )

        req = copilot.parse_mcp_activity(payload, discovery)
        assert req.server == "server"
        assert req.tool == "tool_name_extra"

    def test_unknown_server_falls_back_to_cfg_name(self):
        copilot = Copilot()
        discovery = _ai_discovery(servers=[])
        payload = _payload(
            copilot,
            raw={"tool_name": "mcp_unknown_tool", "cwd": "/tmp", "tool_input": {}},
        )

        req = copilot.parse_mcp_activity(payload, discovery)

        assert req.server == "unknown"
        assert req.tool == "tool"


# ===========================================================================
# _parse_tool_arguments (Cursor helper)
# ===========================================================================


class TestParseToolArguments:
    def test_valid_schema(self):
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "SQL query"},
                "limit": {"type": "integer"},
            },
            "required": ["query"],
        }
        result = _parse_tool_arguments(schema)
        assert result is not None
        assert len(result) == 2
        q = next(a for a in result if a.name == "query")
        assert q.required is True
        assert q.description == "SQL query"
        lim = next(a for a in result if a.name == "limit")
        assert lim.required is False

    def test_empty_properties_returns_none(self):
        schema = {"type": "object", "properties": {}}
        assert _parse_tool_arguments(schema) is None

    @pytest.mark.parametrize(
        "schema",
        [
            pytest.param(None, id="none"),
            pytest.param("string", id="string"),
            pytest.param(42, id="integer"),
        ],
    )
    def test_non_dict_schema_returns_none(self, schema: Any):
        assert _parse_tool_arguments(schema) is None
