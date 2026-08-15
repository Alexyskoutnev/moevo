"""Tests for code parsing utilities."""

from __future__ import annotations

from moevo.generation import apply_diff, extract_diffs, parse_full_rewrite, parse_response


class TestParseFullRewrite:
    def test_python_block(self):
        response = "Here's the code:\n```python\nprint('hello')\n```\nDone."
        assert parse_full_rewrite(response) == "print('hello')"

    def test_generic_block(self):
        response = "```\nx = 1\n```"
        assert parse_full_rewrite(response) == "x = 1"

    def test_raw_code(self):
        response = "x = 1\ny = 2"
        assert parse_full_rewrite(response) == "x = 1\ny = 2"

    def test_empty(self):
        assert parse_full_rewrite("") is None
        assert parse_full_rewrite("   ") is None


class TestExtractDiffs:
    def test_single_block(self):
        diff = """<<<<<<< SEARCH
old line
=======
new line
>>>>>>> REPLACE"""
        blocks = extract_diffs(diff)
        assert len(blocks) == 1
        assert blocks[0] == ("old line\n", "new line\n")

    def test_multiple_blocks(self):
        diff = """<<<<<<< SEARCH
a = 1
=======
a = 2
>>>>>>> REPLACE

<<<<<<< SEARCH
b = 3
=======
b = 4
>>>>>>> REPLACE"""
        blocks = extract_diffs(diff)
        assert len(blocks) == 2

    def test_no_blocks(self):
        assert extract_diffs("just some text") == []


class TestApplyDiff:
    def test_simple_replacement(self):
        original = "a = 1\nb = 2\nc = 3"
        diff = """<<<<<<< SEARCH
b = 2
=======
b = 99
>>>>>>> REPLACE"""
        result = apply_diff(original, diff)
        assert "b = 99" in result
        assert "b = 2" not in result

    def test_no_match_returns_original(self):
        original = "x = 1"
        diff = """<<<<<<< SEARCH
y = 2
=======
y = 3
>>>>>>> REPLACE"""
        assert apply_diff(original, diff) == original


class TestParseResponse:
    def test_full_rewrite_mode(self):
        response = "```python\nresult = 42\n```"
        assert parse_response(response) == "result = 42"

    def test_diff_mode(self):
        parent = "x = 1\ny = 2"
        response = """<<<<<<< SEARCH
x = 1
=======
x = 10
>>>>>>> REPLACE"""
        result = parse_response(response, parent_code=parent, diff_mode=True)
        assert result is not None
        assert "x = 10" in result
