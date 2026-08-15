"""Code parsing utilities for LLM responses."""

from __future__ import annotations

import re


def parse_full_rewrite(response: str, language: str = "python") -> str | None:
    """Extract code from an LLM response containing a full rewrite.

    Tries language-specific fenced block first, then any fenced block,
    then returns the raw response as a last resort.
    """
    # Language-specific block
    pattern = r"```" + re.escape(language) + r"\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Any fenced block
    matches = re.findall(r"```(?:\w*)\n(.*?)```", response, re.DOTALL)
    if matches:
        return matches[0].strip()

    # Raw response (likely just code)
    stripped = response.strip()
    return stripped if stripped else None


def extract_diffs(diff_text: str) -> list[tuple[str, str]]:
    """Extract SEARCH/REPLACE diff blocks from LLM response.

    Format:
        <<<<<<< SEARCH
        old code
        =======
        new code
        >>>>>>> REPLACE
    """
    pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
    return re.findall(pattern, diff_text, re.DOTALL)


def apply_diff(original: str, diff_text: str) -> str:
    """Apply SEARCH/REPLACE diff blocks to original code."""
    blocks = extract_diffs(diff_text)
    if not blocks:
        return original

    lines = original.split("\n")
    for search_text, replace_text in blocks:
        search_lines = search_text.rstrip("\n").split("\n")
        replace_lines = replace_text.rstrip("\n").split("\n")
        n = len(search_lines)

        for i in range(len(lines) - n + 1):
            if lines[i : i + n] == search_lines:
                lines[i : i + n] = replace_lines
                break

    return "\n".join(lines)


def parse_response(
    response: str, parent_code: str | None = None, diff_mode: bool = False
) -> str | None:
    """Parse LLM response into code, handling both diff and full rewrite modes."""
    if diff_mode and parent_code:
        diffs = extract_diffs(response)
        if diffs:
            return apply_diff(parent_code, response)

    return parse_full_rewrite(response)
