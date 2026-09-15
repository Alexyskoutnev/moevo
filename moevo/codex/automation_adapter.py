"""Official AutomationBench state, tools and scoring without its model client."""

from __future__ import annotations

import ast
import copy
import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def load_official(source: Path) -> dict:
    sys.path.insert(0, str(source.resolve()))
    from automationbench.domains import get_domain_dataset
    from automationbench.rubric import partial_credit, task_completed_correctly
    from automationbench.schema.world import WorldState
    from automationbench.task_contract import task_contract_sha256
    from automationbench.tools.zapier.meta import execute_tool, search_tools

    # Load the exact pure setup helpers without importing the optional verifiers
    # model runner. Their source revision is recorded with the benchmark.
    tree = ast.parse((source / "automationbench/runner.py").read_text())
    names = {"strip_none_values", "_service_for_name", "compute_allowed_services"}
    nodes: list[ast.stmt] = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    helpers = {
        "WorldState": WorldState,
        "_SERVICE_FIELDS": sorted(
            (str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True
        ),
    }
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), "official_setup_helpers", "exec"), helpers
    )
    return {
        "dataset": get_domain_dataset,
        "world": WorldState,
        "partial_credit": partial_credit,
        "success": task_completed_correctly,
        "contract_hash": task_contract_sha256,
        "search": search_tools,
        "execute": execute_tool,
        **{k: helpers[k] for k in names},
    }


def make_state(task: dict, official: dict) -> dict:
    info = task["info"]
    if isinstance(info, str):
        info = json.loads(info)
    info = official["strip_none_values"](copy.deepcopy(info))
    initial = info["initial_state"]
    world = official["world"](**initial)
    world.meta.allowed_services = official["compute_allowed_services"](
        initial, info.get("assertions", []), info.get("zapier_tools", [])
    )
    return {"info": info, "world": world, "initial_state": copy.deepcopy(initial)}
