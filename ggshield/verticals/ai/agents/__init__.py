from .claude_code import Claude
from .copilot import Copilot
from .cursor import Cursor


AGENTS = {agent.name: agent for agent in [Cursor(), Claude(), Copilot()]}


__all__ = ["AGENTS", "Claude", "Copilot", "Cursor"]
