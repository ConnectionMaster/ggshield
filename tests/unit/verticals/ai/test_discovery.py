from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock, patch

import pytest
from pygitguardian.models import AIDiscovery, Detail, MCPServer, UserInfo

from ggshield.core.errors import UnexpectedError
from ggshield.verticals.ai.discovery import (
    _merge_mcp_configurations,
    discover_ai_configuration,
    submit_ai_discovery,
)
from ggshield.verticals.ai.models import MCPConfiguration, Scope, Transport


def _user(**kwargs: Any) -> UserInfo:
    defaults = dict(
        hostname="host", username="user", machine_id="mid", user_email="u@e.com"
    )
    return UserInfo.from_dict(defaults | kwargs)


def _cfg(
    name: str = "srv", agent: str = "cursor", scope: Scope = Scope.USER
) -> MCPConfiguration:
    return MCPConfiguration(
        name=name, agent=agent, scope=scope, transport=Transport.STDIO
    )


def _discovery(
    user: UserInfo = _user(),
    servers: List[MCPServer] = [],
    discovery_duration: float = 0.1,
) -> AIDiscovery:
    return AIDiscovery(
        user=user, servers=servers, discovery_duration=discovery_duration
    )


# ---------------------------------------------------------------------------
# _merge_mcp_configurations
# ---------------------------------------------------------------------------


class TestMergeMcpConfigurations:
    def test_different_names_produce_separate_servers(self):
        configs = [_cfg(name="a"), _cfg(name="b")]
        servers = _merge_mcp_configurations(configs)
        assert len(servers) == 2
        names = {s.name for s in servers}
        assert names == {"a", "b"}

    def test_same_name_merged_under_one_server(self):
        configs = [_cfg(name="x", agent="cursor"), _cfg(name="x", agent="claude-code")]
        servers = _merge_mcp_configurations(configs)
        assert len(servers) == 1
        assert len(servers[0].configurations) == 2

    def test_empty_list_returns_empty(self):
        assert _merge_mcp_configurations([]) == []


# ---------------------------------------------------------------------------
# discover_ai_configuration
# ---------------------------------------------------------------------------


class TestDiscoverAIConfiguration:
    @patch("ggshield.verticals.ai.discovery.get_user_info", return_value=_user())
    @patch("ggshield.verticals.ai.discovery.AGENTS")
    def test_aggregates_agents(
        self, mock_agents: MagicMock, mock_user_info: MagicMock, tmp_path: Path
    ):
        agent1 = MagicMock()
        agent1.discover_project_directories.return_value = iter([tmp_path / "p1"])
        agent1.discover_mcp_configurations.return_value = [_cfg(name="s1")]
        agent1.discover_capabilities.return_value = False

        agent2 = MagicMock()
        agent2.discover_project_directories.return_value = iter([])
        agent2.discover_mcp_configurations.return_value = [_cfg(name="s2")]
        agent2.discover_capabilities.return_value = False

        mock_agents.values.return_value = [agent1, agent2]

        result = discover_ai_configuration()

        assert result.user == _user()
        assert len(result.servers) == 2
        assert result.discovery_duration > 0

    @patch("ggshield.verticals.ai.discovery.get_user_info", return_value=_user())
    @patch("ggshield.verticals.ai.discovery.AGENTS")
    def test_stops_capability_discovery_at_first_success(
        self, mock_agents: MagicMock, mock_user_info: MagicMock
    ):
        agent1 = MagicMock()
        agent1.discover_project_directories.return_value = iter([])
        agent1.discover_mcp_configurations.return_value = [_cfg(name="s")]
        agent1.discover_capabilities.return_value = True

        agent2 = MagicMock()
        agent2.discover_project_directories.return_value = iter([])
        agent2.discover_mcp_configurations.return_value = []
        agent2.discover_capabilities.return_value = False

        mock_agents.values.return_value = [agent1, agent2]

        discover_ai_configuration()

        agent1.discover_capabilities.assert_called_once()
        agent2.discover_capabilities.assert_not_called()


# ---------------------------------------------------------------------------
# submit_ai_discovery
# ---------------------------------------------------------------------------


class TestSubmitAIDiscovery:
    def test_successful_response(self):
        discovery = _discovery()
        client = MagicMock()
        client.send_ai_discovery.return_value = discovery

        result = submit_ai_discovery(client, discovery)
        assert result.user == discovery.user

    def test_non_200_raises(self):
        discovery = _discovery()
        client = MagicMock()
        client.send_ai_discovery.return_value = Detail(
            status_code=500, detail="Internal Server Error"
        )

        with pytest.raises(UnexpectedError):
            submit_ai_discovery(client, discovery)
