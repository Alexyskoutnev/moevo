"""Custom evolvable agent - general-purpose Python-native LLM agent with tool use.

A production-ready agentic harness that calls an LLM API directly with a
comprehensive tool set modeled after Claude Code and Codex CLI. Not biased
toward any specific benchmark - this is a general-purpose coding/professional
assistant.

Sections:
  1. IMPORTS
  2. CONFIGURATION
  3. SYSTEM PROMPT
  4. TOOL DEFINITIONS
  5. TOOL IMPLEMENTATIONS
  6. AGENT LOOP
  7. HELPERS
"""

from __future__ import annotations

# ── Section 1: IMPORTS ──────────────────────────────────────────────────────
import asyncio
import fnmatch
import logging
import os
import re
import subprocess
import time
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai.types import (
    Content,
    FunctionDeclaration,
    GenerateContentConfig,
    Part,
    Tool,
)

from moevo.eval.agents.base import AgentResult, BaseAgent

if TYPE_CHECKING:
    from pathlib import Path

logger = logging.getLogger(__name__)

# ── Section 2: CONFIGURATION ───────────────────────────────────────────────

DEFAULT_MODEL = "gemini-3-flash-preview"
MAX_ITERATIONS = 30  # max tool-call rounds per task
TEMPERATURE = 0.2  # low for deterministic output
MAX_OUTPUT_TOKENS = 16384  # generous for code generation
BASH_TIMEOUT = 120  # seconds per bash command
MAX_BASH_OUTPUT = 15000  # chars of stdout/stderr to keep
MAX_FILE_READ = 60000  # chars before truncating file reads
MAX_GREP_RESULTS = 50  # max matches returned by grep
MAX_GLOB_RESULTS = 100  # max files returned by glob

# ── Section 3: SYSTEM PROMPT ───────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an assistant that completes tasks using the provided tools.
Your working directory is: {cwd}

All file paths are relative to the working directory.
Save all output files to the working directory. Never write files outside it.
Keep going until the task is fully complete. Verify your work before finishing.
If a command fails, read the error and fix the issue yourself.
"""

# ── Section 4: TOOL DEFINITIONS ────────────────────────────────────────────

TOOL_DECLARATIONS = Tool(
    function_declarations=[
        FunctionDeclaration(
            name="bash",
            description=(
                "Execute a shell command in the working directory. "
                "Returns stdout, stderr, and exit code. Use for running scripts, "
                "installing packages, git operations, and any system commands."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute.",
                    },
                },
                "required": ["command"],
            },
        ),
        FunctionDeclaration(
            name="read_file",
            description=(
                "Read the contents of a file. Supports text files (with line numbers), "
                "PDFs (extracts text from all pages), and Office documents (.docx, .xlsx, .pptx). "
                "For large files, use offset and limit to read specific line ranges."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Line number to start reading from (1-based). Optional.",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of lines to read. Optional.",
                    },
                },
                "required": ["path"],
            },
        ),
        FunctionDeclaration(
            name="write_file",
            description=(
                "Create a new file or completely overwrite an existing file. "
                "Use for creating scripts, config files, text documents, etc. "
                "For targeted edits to existing files, use edit_file instead."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path for the file (relative to working directory).",
                    },
                    "content": {
                        "type": "string",
                        "description": "The complete file content to write.",
                    },
                },
                "required": ["path", "content"],
            },
        ),
        FunctionDeclaration(
            name="edit_file",
            description=(
                "Make targeted find-and-replace edits in an existing file. "
                "The old_string must match exactly (including whitespace). "
                "Use this for surgical edits; use write_file for full rewrites."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file (relative to working directory).",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact text to find in the file.",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The text to replace it with.",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        ),
        FunctionDeclaration(
            name="list_dir",
            description=(
                "List files and subdirectories in a directory. "
                "Returns names with file sizes. Useful for understanding "
                "workspace contents and finding files."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path (relative to working directory). Defaults to '.'.",
                    },
                },
            },
        ),
        FunctionDeclaration(
            name="grep",
            description=(
                "Search file contents using a regex pattern. "
                "Returns matching lines with file paths and line numbers. "
                "Searches recursively through the working directory."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regular expression pattern to search for.",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (relative). Defaults to '.'.",
                    },
                    "include": {
                        "type": "string",
                        "description": "Glob pattern to filter files (e.g. '*.py', '*.txt'). Optional.",
                    },
                },
                "required": ["pattern"],
            },
        ),
        FunctionDeclaration(
            name="glob",
            description=(
                "Find files matching a glob pattern. Returns file paths sorted "
                "by modification time. Supports ** for recursive matching."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', '*.xlsx', 'src/**/*.ts').",
                    },
                },
                "required": ["pattern"],
            },
        ),
    ]
)


# ── Section 5: TOOL IMPLEMENTATIONS ────────────────────────────────────────


def _check_path(path: str, cwd: Path) -> tuple[Path, str | None]:
    """Resolve a path and check it's within the workspace.

    Returns (resolved_path, error_message). error_message is None if OK.
    """
    # Resolve both sides: comparing a resolved path against an unresolved cwd
    # produces false rejections whenever the workspace path contains a symlink
    # (macOS /tmp -> /private/tmp is the common case).
    root = cwd.resolve()
    resolved = (root / path).resolve()
    # Path.is_relative_to, not str.startswith: a prefix comparison lets
    # /work/ws-evil escape a sandbox rooted at /work/ws.
    if not resolved.is_relative_to(root):
        return resolved, f"Error: path '{path}' is outside the working directory."
    return resolved, None


def _run_bash(command: str, cwd: Path) -> str:
    """Execute a shell command in the workspace."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=BASH_TIMEOUT,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        stdout = result.stdout[:MAX_BASH_OUTPUT] if result.stdout else ""
        stderr = result.stderr[:MAX_BASH_OUTPUT] if result.stderr else ""
        parts = [f"Exit code: {result.returncode}"]
        if stdout.strip():
            parts.append(f"STDOUT:\n{stdout}")
        if stderr.strip():
            parts.append(f"STDERR:\n{stderr}")
        return "\n".join(parts)
    except subprocess.TimeoutExpired:
        return f"Command timed out after {BASH_TIMEOUT}s. Consider breaking into smaller steps."
    except Exception as e:
        return f"Error executing command: {e}"


def _read_pdf(resolved: Path, display_path: str) -> str:
    """Extract text from a PDF file using pymupdf."""
    try:
        import pymupdf

        doc = pymupdf.open(str(resolved))
        parts = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                parts.append(f"--- Page {i + 1} ---\n{text}")
        doc.close()
        result = "\n\n".join(parts) if parts else "(PDF contains no extractable text)"
        if len(result) > MAX_FILE_READ:
            result = result[:MAX_FILE_READ] + "\n... (truncated)"
        return f"File: {display_path} (PDF, {len(parts)} pages)\n{result}"
    except Exception as e:
        return f"Error reading PDF '{display_path}': {e}"


def _read_office_file(resolved: Path, display_path: str) -> str:
    """Extract text from Office documents (.docx, .xlsx, .pptx)."""
    suffix = resolved.suffix.lower()
    try:
        if suffix == ".docx":
            from docx import Document as DocxDocument

            doc = DocxDocument(str(resolved))
            parts = [p.text for p in doc.paragraphs]
            for table in doc.tables:
                for row in table.rows:
                    parts.append(" | ".join(cell.text for cell in row.cells))
            result = "\n".join(parts)
        elif suffix in (".xlsx", ".xlsm"):
            import openpyxl

            wb = openpyxl.load_workbook(str(resolved), data_only=True)
            parts = []
            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                parts.append(f"[Sheet: {sheet_name}]")
                for row in ws.iter_rows(values_only=True):
                    row_str = " | ".join(str(c) if c is not None else "" for c in row)
                    parts.append(row_str)
            result = "\n".join(parts)
        elif suffix == ".pptx":
            from pptx import Presentation

            prs = Presentation(str(resolved))
            parts = []
            for i, slide in enumerate(prs.slides):
                parts.append(f"[Slide {i + 1}]")
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        parts.append(shape.text_frame.text)
            result = "\n".join(parts)
        else:
            return f"Unsupported file type: {suffix}"
        if len(result) > MAX_FILE_READ:
            result = result[:MAX_FILE_READ] + "\n... (truncated)"
        return f"File: {display_path} ({suffix})\n{result}"
    except Exception as e:
        return f"Error reading '{display_path}': {e}"


def _read_file(path: str, cwd: Path, offset: int = 0, limit: int = 0) -> str:
    """Read a file from the workspace with optional offset/limit.

    Natively handles PDFs, Office documents (.docx, .xlsx, .pptx),
    and text files with line numbers.
    """
    resolved, err = _check_path(path, cwd)
    if err:
        return err
    if not resolved.exists():
        available = _list_available_files(cwd)
        return f"Error: file '{path}' not found.\nAvailable files:\n{available}"
    if resolved.is_dir():
        return f"Error: '{path}' is a directory. Use list_dir instead."

    # Handle PDFs natively
    if resolved.suffix.lower() == ".pdf":
        return _read_pdf(resolved, path)

    # Handle Office documents natively
    if resolved.suffix.lower() in (".docx", ".xlsx", ".xlsm", ".pptx"):
        return _read_office_file(resolved, path)

    # Text files
    try:
        content = resolved.read_text(encoding="utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError):
        size = resolved.stat().st_size
        return f"Binary file ({size} bytes). Use bash with Python to inspect binary files."

    lines = content.splitlines(keepends=True)
    total_lines = len(lines)

    # Apply offset/limit
    start = max(0, offset - 1) if offset > 0 else 0
    end = min(start + limit, total_lines) if limit > 0 else total_lines

    selected = lines[start:end]

    # Add line numbers
    numbered = []
    for i, line in enumerate(selected, start=start + 1):
        numbered.append(f"{i:>6}\t{line.rstrip()}")
    result = "\n".join(numbered)

    if len(result) > MAX_FILE_READ:
        result = result[:MAX_FILE_READ] + f"\n... (truncated, {total_lines} total lines)"

    header = f"File: {path} ({total_lines} lines)"
    if start > 0 or end < total_lines:
        header += f" [showing lines {start + 1}-{end}]"
    return f"{header}\n{result}"


def _write_file(path: str, content: str, cwd: Path) -> str:
    """Write content to a file in the workspace."""
    resolved, err = _check_path(path, cwd)
    if err:
        return err
    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content)
        size = resolved.stat().st_size
        return f"Successfully wrote {size} bytes to {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def _edit_file(path: str, old_string: str, new_string: str, cwd: Path) -> str:
    """Make a find-and-replace edit in a file."""
    resolved, err = _check_path(path, cwd)
    if err:
        return err
    if not resolved.exists():
        return f"Error: file '{path}' not found."
    try:
        content = resolved.read_text(encoding="utf-8")
    except (UnicodeDecodeError, ValueError):
        return f"Error: '{path}' is a binary file and cannot be edited with edit_file."

    count = content.count(old_string)
    if count == 0:
        # Show a snippet to help the model find the right text
        preview = content[:2000] if len(content) > 2000 else content
        return (
            f"Error: old_string not found in '{path}'. "
            f"Make sure the text matches exactly (including whitespace).\n"
            f"File preview:\n{preview}"
        )
    if count > 1:
        return (
            f"Error: old_string found {count} times in '{path}'. "
            f"Provide more context to make the match unique."
        )

    new_content = content.replace(old_string, new_string, 1)
    resolved.write_text(new_content)

    # Show the edited region with context
    new_lines = new_content.splitlines()
    # Find where the edit was applied
    edit_start = content.index(old_string)
    line_num = content[:edit_start].count("\n")
    context_start = max(0, line_num - 2)
    context_end = min(len(new_lines), line_num + new_string.count("\n") + 3)
    context = "\n".join(f"{i + 1:>6}\t{new_lines[i]}" for i in range(context_start, context_end))
    return f"Successfully edited {path}.\nUpdated region:\n{context}"


def _list_dir(path: str, cwd: Path) -> str:
    """List files in a workspace directory."""
    resolved, err = _check_path(path, cwd)
    if err:
        return err
    if not resolved.exists():
        return f"Error: directory '{path}' does not exist."
    if not resolved.is_dir():
        return f"Error: '{path}' is not a directory."
    entries = []
    for item in sorted(resolved.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_file():
            size = item.stat().st_size
            if size < 1024:
                size_str = f"{size}B"
            elif size < 1024 * 1024:
                size_str = f"{size / 1024:.1f}KB"
            else:
                size_str = f"{size / (1024 * 1024):.1f}MB"
            entries.append(f"  {item.name}  ({size_str})")
        elif item.is_dir():
            entries.append(f"  {item.name}/")
    if not entries:
        return "(empty directory)"
    return "\n".join(entries)


def _grep(pattern: str, path: str, cwd: Path, include: str = "") -> str:
    """Search file contents using regex."""
    resolved, err = _check_path(path, cwd)
    if err:
        return err
    if not resolved.exists():
        return f"Error: path '{path}' does not exist."

    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: invalid regex pattern: {e}"

    matches: list[str] = []

    def _search_file(fpath: Path) -> None:
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            return
        rel = str(fpath.relative_to(cwd))
        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                matches.append(f"{rel}:{i}: {line.rstrip()[:200]}")
                if len(matches) >= MAX_GREP_RESULTS:
                    return

    if resolved.is_file():
        _search_file(resolved)
    else:
        for fpath in sorted(resolved.rglob("*")):
            if len(matches) >= MAX_GREP_RESULTS:
                break
            if not fpath.is_file():
                continue
            if fpath.name.startswith("."):
                continue
            if include and not fnmatch.fnmatch(fpath.name, include):
                continue
            _search_file(fpath)

    if not matches:
        return f"No matches found for pattern '{pattern}' in {path}."
    header = f"Found {len(matches)} match(es)"
    if len(matches) >= MAX_GREP_RESULTS:
        header += f" (limited to {MAX_GREP_RESULTS})"
    return f"{header}:\n" + "\n".join(matches)


def _glob_files(pattern: str, cwd: Path) -> str:
    """Find files matching a glob pattern."""
    matches = sorted(cwd.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    # Filter out hidden files
    matches = [m for m in matches if not any(p.startswith(".") for p in m.relative_to(cwd).parts)]
    if not matches:
        return f"No files matching '{pattern}'."
    results = []
    for m in matches[:MAX_GLOB_RESULTS]:
        rel = str(m.relative_to(cwd))
        if m.is_file():
            size = m.stat().st_size
            results.append(f"  {rel}  ({size} bytes)")
        else:
            results.append(f"  {rel}/")
    header = f"Found {len(matches)} match(es)"
    if len(matches) > MAX_GLOB_RESULTS:
        header += f" (showing first {MAX_GLOB_RESULTS})"
    return f"{header}:\n" + "\n".join(results)


def _list_available_files(cwd: Path) -> str:
    """List files in workspace root for error messages."""
    entries = []
    for item in sorted(cwd.iterdir()):
        if item.name.startswith("."):
            continue
        if item.is_file():
            entries.append(f"  {item.name}")
        elif item.is_dir():
            entries.append(f"  {item.name}/")
    return "\n".join(entries) if entries else "(empty directory)"


def _execute_tool(name: str, args: dict[str, Any], cwd: Path) -> str:
    """Dispatch a tool call to its implementation."""
    try:
        if name == "bash":
            return _run_bash(args.get("command", ""), cwd)
        elif name == "read_file":
            return _read_file(
                args.get("path", ""),
                cwd,
                offset=int(args.get("offset", 0)),
                limit=int(args.get("limit", 0)),
            )
        elif name == "write_file":
            return _write_file(args.get("path", ""), args.get("content", ""), cwd)
        elif name == "edit_file":
            return _edit_file(
                args.get("path", ""),
                args.get("old_string", ""),
                args.get("new_string", ""),
                cwd,
            )
        elif name == "list_dir":
            return _list_dir(args.get("path", "."), cwd)
        elif name == "grep":
            return _grep(
                args.get("pattern", ""),
                args.get("path", "."),
                cwd,
                include=args.get("include", ""),
            )
        elif name == "glob":
            return _glob_files(args.get("pattern", ""), cwd)
        else:
            return f"Unknown tool: {name}"
    except Exception as e:
        return f"Tool error ({name}): {e}"


# ── Section 6: AGENT LOOP ──────────────────────────────────────────────────


class CustomAgent(BaseAgent):
    """General-purpose Python-native agent using Gemini API with tool use.

    Provides a comprehensive tool set (bash, read_file, write_file, edit_file,
    list_dir, grep, glob) modeled after Claude Code and Codex CLI. Designed
    to be a production-ready, general-purpose agentic harness - not biased
    toward any specific benchmark.
    """

    def name(self) -> str:
        return "custom"

    async def _run(self, prompt: str, cwd: Path) -> AgentResult:
        """Run the agent. Never raises - all errors become partial results.

        Runs the synchronous Gemini API loop in a thread via asyncio.to_thread()
        so multiple tasks can execute concurrently with asyncio.gather().
        """
        cwd = cwd.resolve()
        cwd.mkdir(parents=True, exist_ok=True)

        tool_calls_log: list[dict[str, Any]] = []
        messages_log: list[dict[str, Any]] = []
        response_parts: list[str] = []

        try:
            return await asyncio.to_thread(
                self._run_inner,
                prompt,
                cwd,
                tool_calls_log,
                messages_log,
                response_parts,
            )
        except Exception as e:
            # Never propagate exceptions - return partial results with error info
            logger.error("[custom] Unhandled error: %s: %s", type(e).__name__, e)
            response_parts.append(f"(Agent terminated due to error: {type(e).__name__}: {e})")
            return AgentResult(
                response="\n".join(response_parts),
                tool_calls=tool_calls_log,
                messages=messages_log,
            )

    def _run_inner(
        self,
        prompt: str,
        cwd: Path,
        tool_calls_log: list[dict[str, Any]],
        messages_log: list[dict[str, Any]],
        response_parts: list[str],
    ) -> AgentResult:
        # Initialize Gemini client
        api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            return AgentResult(error="GEMINI_API_KEY or GOOGLE_API_KEY required.")
        client = genai.Client(api_key=api_key)

        model = self._model or DEFAULT_MODEL
        max_iters = self._max_turns if self._max_turns != 10 else MAX_ITERATIONS
        system = SYSTEM_PROMPT.format(cwd=cwd)

        config = GenerateContentConfig(
            system_instruction=system,
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
            tools=[TOOL_DECLARATIONS],
        )

        contents: list[Content | str] = [prompt]
        last_calls: list[str] = []  # for doom-loop detection
        nudge_count = 0

        for iteration in range(max_iters):
            logger.debug("[custom] Iteration %d/%d", iteration + 1, max_iters)

            # Call Gemini with retry
            response = _call_with_retry(client, model, contents, config)
            if response is None:
                response_parts.append("I was unable to complete the task due to API errors.")
                break

            # Extract function calls from response
            function_calls = _extract_function_calls(response)

            if not function_calls:
                # Model is done - extract final text
                text = _get_response_text(response)
                response_parts.append(text)
                messages_log.append({"role": "assistant", "type": "text", "content": text[:2000]})
                break

            # Log any text alongside tool calls
            for part in _get_content_parts(response):
                if part.text:
                    response_parts.append(part.text)
                    messages_log.append(
                        {"role": "assistant", "type": "text", "content": part.text[:2000]}
                    )

            # Execute each tool call
            tool_response_parts: list[Part] = []
            for fc in function_calls:
                try:
                    args = dict(fc.args) if fc.args else {}
                except Exception:
                    args = {}
                result = _execute_tool(fc.name, args, cwd)

                tool_calls_log.append(
                    {
                        "tool": fc.name,
                        "input": str(args)[:500],
                    }
                )
                messages_log.append(
                    {
                        "role": "assistant",
                        "type": "tool_use",
                        "tool": fc.name,
                        "input": str(args)[:1000],
                    }
                )
                messages_log.append(
                    {
                        "role": "tool",
                        "type": "tool_result",
                        "content": result[:2000],
                    }
                )

                tool_response_parts.append(
                    Part.from_function_response(
                        name=fc.name,
                        response={"result": result},
                    )
                )

                # Doom-loop detection
                call_sig = f"{fc.name}:{hash(str(args))}"
                last_calls.append(call_sig)

            # Append model response + tool results to conversation
            try:
                model_content = response.candidates[0].content if response.candidates else None
            except (IndexError, AttributeError):
                model_content = None
            if model_content:
                contents.append(model_content)
            contents.append(Content(parts=tool_response_parts))

            # Check for doom loop: last 3 calls repeat prior 3
            if len(last_calls) >= 6 and last_calls[-3:] == last_calls[-6:-3]:
                nudge_count += 1
                if nudge_count >= 3:
                    response_parts.append("(Agent loop terminated: stuck in repeated actions)")
                    break
                contents.append(
                    "You seem stuck repeating the same actions. "
                    "Try a different approach or finish with what you have."
                )

        return AgentResult(
            response="\n".join(response_parts),
            tool_calls=tool_calls_log,
            messages=messages_log,
        )


# ── Section 7: HELPERS ─────────────────────────────────────────────────────


def _call_with_retry(
    client: genai.Client,
    model: str,
    contents: list,
    config: GenerateContentConfig,
    max_retries: int = 3,
) -> Any | None:
    """Call Gemini API with exponential backoff retry."""
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as e:
            error_name = type(e).__name__
            if attempt == max_retries:
                logger.error(
                    "[custom] API call failed after %d retries: %s: %s", max_retries, error_name, e
                )
                return None
            wait = 2 ** (attempt + 1)
            logger.warning(
                "[custom] API error (attempt %d): %s: %s - retrying in %ds",
                attempt + 1,
                error_name,
                e,
                wait,
            )
            time.sleep(wait)
    return None


def _extract_function_calls(response: Any) -> list:
    """Extract function call objects from a Gemini response."""
    calls = []
    if not response.candidates:
        return calls
    content = response.candidates[0].content
    if not content or not content.parts:
        return calls
    for part in content.parts:
        if part.function_call:
            calls.append(part.function_call)
    return calls


def _get_response_text(response: Any) -> str:
    """Safely extract text from a Gemini response."""
    try:
        return response.text or ""
    except (AttributeError, ValueError):
        if response.candidates:
            content = response.candidates[0].content
            if content and content.parts:
                texts = [p.text for p in content.parts if p.text]
                return "\n".join(texts)
        return ""


def _get_content_parts(response: Any) -> list:
    """Safely get content parts from a Gemini response."""
    if not response.candidates:
        return []
    content = response.candidates[0].content
    if not content or not content.parts:
        return []
    return content.parts
