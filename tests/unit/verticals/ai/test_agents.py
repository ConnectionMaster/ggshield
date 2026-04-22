import json
from pathlib import Path
from typing import Optional
from unittest.mock import patch

from ggshield.verticals.ai.agents.claude_code import Claude
from ggshield.verticals.ai.agents.cursor import Cursor
from ggshield.verticals.ai.models import MCPConfiguration, MCPServer, Scope, Transport


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
