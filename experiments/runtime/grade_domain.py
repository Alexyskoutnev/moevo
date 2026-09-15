"""Offline grading entry point, mounted only in a separate grader container."""

from __future__ import annotations

import ast
import contextlib
import copy
import io
import json
import os
import re
import sys
import types
from pathlib import Path


def amo(payload):
    sys.path.insert(0, "/benchmark")
    import utils

    def reject_api(*args, **kwargs):
        raise RuntimeError("Description/API grading is excluded from the parser-only subset")

    utils.call_api = reject_api
    row = utils.append_try_list(payload["row"])
    if row["answer_type"] not in {"number", "set", "variable"}:
        raise ValueError("Only the official AMO P subset is supported")
    tree = ast.parse(Path("/benchmark/grading.py").read_text())
    nodes: list[ast.stmt] = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in {"verify_result", "grading"}
    ]
    if len(nodes) != 2:
        raise ValueError("Official grading functions changed")
    namespace = {"copy": copy, **vars(utils)}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "official_grading", "exec"), namespace)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        score = namespace["grading"](payload["prediction"], row)
    if "Error in grading" in captured.getvalue():
        raise RuntimeError("Official parser raised an exception: " + captured.getvalue())
    return {"score": score, "parser_output": captured.getvalue()}


def travel(payload):
    # Load exactly the four pure helpers used by the official evaluator. The
    # upstream module also imports a Gradio annotation UI, which grading does not use.
    tree = ast.parse(Path("/benchmark/utils/func.py").read_text())
    names = {
        "get_valid_name_city",
        "extract_before_parenthesis",
        "extract_numbers_from_filenames",
        "count_consecutive_values",
    }
    nodes: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    if len(nodes) != len(names):
        raise ValueError("Official TravelPlanner helpers changed")
    helpers = types.ModuleType("utils.func")
    helpers.__dict__.update({"re": re, "os": os})
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "official_utils", "exec"), helpers.__dict__
    )
    package = types.ModuleType("utils")
    package.__path__ = ["/benchmark/utils"]
    sys.modules["utils"] = package
    sys.modules["utils.func"] = helpers
    os.chdir("/benchmark/evaluation")
    sys.path[:0] = ["/benchmark", "/benchmark/evaluation"]
    from commonsense_constraint import evaluation as commonsense_eval
    from hard_constraint import evaluation as hard_eval

    plan, query = payload["prediction"], payload["row"]
    if not plan:
        return {"score": 0.0, "commonsense": None, "hard": None}
    commonsense = commonsense_eval(query, plan)
    hard = None
    if commonsense["is_not_absent"][0] and commonsense["is_valid_information_in_sandbox"][0]:
        hard = hard_eval(query, plan)
    # The official Final Pass Rate requires every applicable constraint in both groups.
    passed = hard is not None and all(
        result[0] is None or bool(result[0])
        for group in [commonsense, hard]
        for result in group.values()
    )
    return {"score": float(passed), "commonsense": commonsense, "hard": hard}


if __name__ == "__main__":
    payload = json.loads(Path("/control/input.json").read_text())
    capture = io.StringIO()
    with contextlib.redirect_stdout(capture):
        result = {"amo": amo, "travel": travel}[payload["dataset"]](payload)
    result["grader_stdout"] = capture.getvalue()
    print(json.dumps(result, default=lambda value: value.item()))
