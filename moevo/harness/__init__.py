"""Custom Harness - a production-ready, general-purpose agentic coding harness.

Provides a Python-native LLM agent with comprehensive tool use, modeled after
Claude Code and Codex CLI. Not biased toward any benchmark - this is a
general-purpose assistant that can write code, create files, search codebases,
and execute shell commands.

Modules:
    config   - Model, timeout, and token configuration constants.
    prompt   - System prompt template.
    tools    - Tool declarations (Gemini FunctionDeclaration) and implementations.
    loop     - Core ReAct agent loop with retry, doom-loop detection.
    agent    - CustomHarnessAgent (subclasses BaseAgent for pipeline integration).
"""

from moevo.harness.agent import CustomHarnessAgent

__all__ = ["CustomHarnessAgent"]
