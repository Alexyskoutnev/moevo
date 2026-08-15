"""Agent backends for GDPval evaluation."""

from moevo.eval.agents.base import AgentResult, BaseAgent
from moevo.eval.agents.claude_code import ClaudeCodeAgent
from moevo.eval.agents.codex import CodexAgent
from moevo.eval.agents.custom import CustomAgent
from moevo.eval.agents.gemini_cli import GeminiCLIAgent


def _load_custom_harness() -> type[BaseAgent]:
    from moevo.harness.agent import CustomHarnessAgent

    return CustomHarnessAgent


class _AgentsDict(dict):  # type: ignore[type-arg]
    def __getitem__(self, key: str) -> type[BaseAgent]:
        if key == "custom-harness":
            return _load_custom_harness()
        return super().__getitem__(key)

    def __contains__(self, key: object) -> bool:
        if key == "custom-harness":
            return True
        return super().__contains__(key)

    def keys(self):  # type: ignore[override]
        return list(super().keys()) + ["custom-harness"]


AGENTS: dict[str, type[BaseAgent]] = _AgentsDict(
    {
        "claude": ClaudeCodeAgent,
        "gemini": GeminiCLIAgent,
        "codex": CodexAgent,
        "custom": CustomAgent,
    }
)

__all__ = [
    "AGENTS",
    "AgentResult",
    "BaseAgent",
    "ClaudeCodeAgent",
    "CodexAgent",
    "CustomAgent",
    "GeminiCLIAgent",
]
