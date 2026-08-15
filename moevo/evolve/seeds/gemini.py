from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from google import genai
from google.genai.types import (
    Content, FunctionDeclaration, GenerateContentConfig, Part, Tool,
)

logger = logging.getLogger(__name__)

# ── Data classes ────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    response: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None
    raw_output: str = ""

class BaseAgent(ABC):
    def __init__(self, system_prompt: str | None = None, max_turns: int | None = None, model: str | None = None):
        self._system_prompt = system_prompt
        self._max_turns = max_turns
        self._model = model
    @abstractmethod
    async def run(self, prompt: str, cwd: Path) -> AgentResult: ...
    @abstractmethod
    def name(self) -> str: ...

# ── Configuration ───────────────────────────────────────────────────────────

DEFAULT_MODEL = "gemini-2.5-pro"
MAX_ITERATIONS = 30
TEMPERATURE = 0.2
MAX_OUTPUT_TOKENS = 16384
BASH_TIMEOUT = 120
MAX_BASH_OUTPUT = 15000
MAX_FILE_READ = 60000

# ── System prompt ───────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an assistant that completes tasks using the provided tools.
Your working directory is: {cwd}

All file paths are relative to the working directory.
Save all output files to the working directory. Never write files outside it.
Keep going until the task is fully complete. Verify your work before finishing.
If a command fails, read the error and fix the issue yourself.
"""

# ── 4 tools — declarations ──────────────────────────────────────────────────

TOOL_DECLARATIONS = Tool(function_declarations=[
    FunctionDeclaration(
        name="bash",
        description="Execute a shell command. Returns stdout, stderr, exit code.",
        parameters={"type": "object", "properties": {"command": {"type": "string", "description": "Shell command to execute."}}, "required": ["command"]},
    ),
    FunctionDeclaration(
        name="read_file",
        description="Read a text file with line numbers. For binary files (.pdf, .docx, .xlsx), use bash with a Python script instead.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "description": "File path (relative)."}}, "required": ["path"]},
    ),
    FunctionDeclaration(
        name="write_file",
        description="Create or overwrite a file with the given content.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "description": "File path (relative)."}, "content": {"type": "string", "description": "Complete file content."}}, "required": ["path", "content"]},
    ),
    FunctionDeclaration(
        name="list_dir",
        description="List directory contents with file sizes.",
        parameters={"type": "object", "properties": {"path": {"type": "string", "description": "Directory path. Defaults to '.'."}}},
    ),
])

# ── 4 tools — implementations ───────────────────────────────────────────────

def _check_path(path: str, cwd: Path) -> tuple[Path, str | None]:
    resolved = (cwd / path).resolve()
    if not resolved.is_relative_to(cwd):
        return resolved, f"Error: path '{path}' is outside the working directory."
    return resolved, None

def _run_bash(command: str, cwd: Path) -> str:
    try:
        r = subprocess.run(command, shell=True, cwd=str(cwd), capture_output=True, text=True,
                           timeout=BASH_TIMEOUT, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        parts = [f"Exit code: {r.returncode}"]
        if r.stdout.strip(): parts.append(f"STDOUT:\n{r.stdout[:MAX_BASH_OUTPUT]}")
        if r.stderr.strip(): parts.append(f"STDERR:\n{r.stderr[:MAX_BASH_OUTPUT]}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired: return f"Command timed out after {BASH_TIMEOUT}s."
    except Exception as e: return f"Error: {e}"

def _read_file(path: str, cwd: Path) -> str:
    resolved, err = _check_path(path, cwd)
    if err: return err
    if not resolved.exists():
        avail = "\n".join(f"  {i.name}{'/' if i.is_dir() else ''}" for i in sorted(cwd.iterdir()) if not i.name.startswith(".")) or "(empty)"
        return f"Error: '{path}' not found.\nAvailable:\n{avail}"
    if resolved.is_dir(): return f"Error: '{path}' is a directory."
    try: content = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e: return f"Error reading file: {e}"
    lines = content.splitlines(keepends=True)
    total = len(lines)
    numbered = [f"{i:>6}\t{l.rstrip()}" for i, l in enumerate(lines, start=1)]
    result = "\n".join(numbered)
    if len(result) > MAX_FILE_READ: result = result[:MAX_FILE_READ] + f"\n...({total} lines)"
    return f"File: {path} ({total} lines)\n{result}"

def _write_file(path: str, content: str, cwd: Path) -> str:
    resolved, err = _check_path(path, cwd)
    if err: return err
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        return f"Wrote {resolved.stat().st_size} bytes to {path}"
    except Exception as e: return f"Error: {e}"

def _list_dir(path: str, cwd: Path) -> str:
    resolved, err = _check_path(path, cwd)
    if err: return err
    if not resolved.exists(): return f"Error: '{path}' doesn't exist."
    if not resolved.is_dir(): return f"Error: not a directory."
    entries = []
    for item in sorted(resolved.iterdir()):
        if item.name.startswith("."): continue
        if item.is_file():
            s = item.stat().st_size
            sz = f"{s}B" if s < 1024 else f"{s/1024:.1f}KB" if s < 1048576 else f"{s/1048576:.1f}MB"
            entries.append(f"  {item.name}  ({sz})")
        elif item.is_dir(): entries.append(f"  {item.name}/")
    return "\n".join(entries) if entries else "(empty)"

def _execute_tool(name: str, args: dict[str, Any], cwd: Path) -> str:
    try:
        if name == "bash": return _run_bash(args.get("command", ""), cwd)
        elif name == "read_file": return _read_file(args.get("path", ""), cwd)
        elif name == "write_file": return _write_file(args.get("path", ""), args.get("content", ""), cwd)
        elif name == "list_dir": return _list_dir(args.get("path", "."), cwd)
        else: return f"Unknown tool: {name}"
    except Exception as e: return f"Tool error ({name}): {e}"

# ── Agent loop ──────────────────────────────────────────────────────────────

class CustomAgent(BaseAgent):

    def name(self) -> str: return "custom-gemini"

    async def run(self, prompt: str, cwd: Path) -> AgentResult:
        cwd = cwd.resolve(); cwd.mkdir(parents=True, exist_ok=True)
        tc_log: list[dict[str, Any]] = []; msg_log: list[dict[str, Any]] = []; resp: list[str] = []
        try:
            return await asyncio.to_thread(self._loop, prompt, cwd, tc_log, msg_log, resp)
        except Exception as e:
            logger.error("[gemini] Error: %s", e)
            return AgentResult(response=f"(Error: {e})", tool_calls=tc_log, messages=msg_log)

    def _loop(self, prompt, cwd, tc_log, msg_log, resp) -> AgentResult:
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key: return AgentResult(error="GEMINI_API_KEY or GOOGLE_API_KEY required.")
        client = genai.Client(api_key=api_key)
        model = self._model or DEFAULT_MODEL
        max_iters = self._max_turns if self._max_turns is not None else MAX_ITERATIONS
        config = GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT.format(cwd=cwd),
            temperature=TEMPERATURE, max_output_tokens=MAX_OUTPUT_TOKENS,
            tools=[TOOL_DECLARATIONS],
        )
        contents: list[Content | str] = [prompt]

        for it in range(max_iters):
            response = _retry(client, model, contents, config)
            if response is None: resp.append("API errors."); break
            fcs = _extract_fcs(response)

            if not fcs:
                text = _get_text(response); resp.append(text)
                msg_log.append({"role": "assistant", "type": "text", "content": text[:2000]})
                break

            for part in _get_parts(response):
                if part.text: resp.append(part.text); msg_log.append({"role": "assistant", "type": "text", "content": part.text[:2000]})

            tool_parts: list[Part] = []
            for fc in fcs:
                try: args = dict(fc.args) if fc.args else {}
                except Exception: args = {}
                result = _execute_tool(fc.name, args, cwd)
                tc_log.append({"tool": fc.name, "input": str(args)[:500]})
                msg_log.append({"role": "assistant", "type": "tool_use", "tool": fc.name, "input": str(args)[:1000]})
                msg_log.append({"role": "tool", "type": "tool_result", "content": result[:2000]})
                tool_parts.append(Part.from_function_response(name=fc.name, response={"result": result}))

            try: mc = response.candidates[0].content if response.candidates else None
            except Exception: mc = None
            if mc: contents.append(mc)
            contents.append(Content(parts=tool_parts))

        return AgentResult(response="\n".join(resp), tool_calls=tc_log, messages=msg_log)

def _retry(client, model, contents, config, retries=3):
    for i in range(retries + 1):
        try: return client.models.generate_content(model=model, contents=contents, config=config)
        except Exception as e:
            if i == retries: logger.error("[gemini] API failed: %s", e); return None
            time.sleep(2 ** (i + 1))
    return None

def _extract_fcs(response):
    if not response.candidates: return []
    c = response.candidates[0].content
    if not c or not c.parts: return []
    return [p.function_call for p in c.parts if p.function_call]

def _get_text(response):
    try: return response.text or ""
    except Exception: pass
    if response.candidates:
        c = response.candidates[0].content
        if c and c.parts: return "\n".join(p.text for p in c.parts if p.text)
    return ""

def _get_parts(response):
    if not response.candidates: return []
    c = response.candidates[0].content
    return c.parts if c and c.parts else []
