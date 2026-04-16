import json
from pathlib import Path
from unittest.mock import patch

from ggshield.verticals.ai.agents.claude_code import Claude
from ggshield.verticals.ai.agents.cursor import Cursor

# ===========================================================================
# Cursor
# ===========================================================================


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
