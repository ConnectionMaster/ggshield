from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional

from ggshield.core.scan import File, Scannable, StringScannable
from ggshield.utils.files import is_path_binary


class EventType(Enum):
    """Event type constants for hook events."""

    USER_PROMPT = auto()
    PRE_TOOL_USE = auto()
    POST_TOOL_USE = auto()
    # We are not interested in other less generic events for now
    # (most of the time, one of the three above will also be called anyway)
    OTHER = auto()


class Tool(Enum):
    """Tool constants for hook events."""

    BASH = auto()
    READ = auto()
    # We are not interested in other tools for now
    OTHER = auto()


@dataclass
class HookResult:
    """Result of a scan: allow or not."""

    block: bool
    message: str
    nbr_secrets: int
    payload: "HookPayload"

    @classmethod
    def allow(cls, payload: "HookPayload") -> "HookResult":
        return cls(block=False, message="", nbr_secrets=0, payload=payload)


@dataclass
class HookPayload:
    event_type: EventType
    tool: Optional[Tool]
    content: str
    identifier: str
    agent: "Agent"

    @property
    def scannable(self) -> Scannable:
        """Return the appropriate Scannable for the payload."""
        if self.tool == Tool.READ:
            path = Path(self.identifier)
            if path.is_file() and not is_path_binary(path):
                return File(path=self.identifier)
        return StringScannable(url=self.identifier, content=self.content)

    @property
    def empty(self) -> bool:
        """Return True if the payload is empty."""
        return not self.scannable.is_longer_than(0)


class Agent(ABC):
    """
    Class that can be derived to implement behavior specific to some AI code assistants.
    """

    # Metadata

    @property
    @abstractmethod
    def display_name(self) -> str:
        """A user-friendly name for the agent."""

    @property
    @abstractmethod
    def name(self) -> str:
        """The name of the agent."""

    # Hooks

    @abstractmethod
    def output_result(self, result: HookResult) -> int:
        """How to output the result of a scan.

        This method is expected to have side effects, like printing to stdout or stderr.

        Args:
            result: the result of the scan.

        Returns: the exit code.
        """

    # Settings

    @property
    @abstractmethod
    def settings_path(self) -> Path:
        """Path to the settings file for this AI coding tool."""

    @property
    @abstractmethod
    def settings_template(self) -> Dict[str, Any]:
        """
        Template for the settings file for this AI coding tool.
        Use the sentinel "<COMMAND>" for the places where the command should be inserted.
        """

    @abstractmethod
    def settings_locate(
        self, candidates: List[Dict[str, Any]], template: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Callback used to help locate the correct object to update in the settings.

        We don't want to overwrite other hooks nor create duplicates, so when the existing
        hook configuration is traversed and we end up in a list, this callback is used to
        locate the correct object to update.

        Args:
            candidates: the list of objects at the level currently traversed.
            template: the template of the expected object.

        Returns: the object to update, or None if no object was found.
        """
        return None
