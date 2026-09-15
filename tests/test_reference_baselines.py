"""Matched reference-arm boundaries, prompt provenance and durable accounting."""

import contextlib
import fcntl
import json
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from experiments import run_reference_baselines as reference
from moevo.codex import client, container_runtime_v2, domain_tasks


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    source, output = root / "source", root / "reference"
    source.mkdir(parents=True)
    code = root / "moevo/codex/client.py"
    code.parent.mkdir(parents=True)
    code.write_text("frozen-code")
    monkeypatch.setattr(reference, "ROOT", root)
    hashes = {"moevo/codex/client.py": reference.file_hash(code)}
    monkeypatch.setattr(reference, "source_identity", lambda: hashes.copy())
    monkeypatch.setattr(
        reference,
        "native_record_identity",
        lambda tasks: {task["id"]: "native-" + task["id"] for task in tasks},
    )
    images = {tag: "sha256:" + str(index) for index, tag in enumerate(reference.IMAGE_TAGS)}
    monkeypatch.setattr(reference, "inspect_images", lambda: images.copy())
    monkeypatch.setattr(
        reference.subprocess, "check_output", lambda args, **kwargs: json.dumps([{"Id": args[-1]}])
    )
    tasks = [
        {
            "id": b,
            "benchmark": b,
            "task_id": b + "-native",
            "objective": b + "-domain",
            "seed": 7,
            "content_sha256": "content-" + b,
            "input_assets_sha256": "assets-" + b,
        }
        for b in sorted(reference.BENCHMARKS)
    ]

    def check(task):
        return {
            "structural_preflight_passed": True,
            "content_sha256": task["content_sha256"],
            "input_assets_sha256": task["input_assets_sha256"],
        }

    monkeypatch.setattr(reference, "verify_task", check)
    policy = {
        "authentication": "codex_chatgpt_account",
        "model": "gpt-6-astra",
        "reasoning_effort": "xhigh",
        "judge_model": "gpt-5.6-terra",
        "judge_reasoning_effort": "medium",
        "instructions": "Shared policy: check your work.",
        "timeout_seconds": 1200,
        "max_tool_calls": 80,
        "seed": 7,
    }
    protocol = {"tasks": tasks, "sources": hashes, "policy": policy}
    reference.write_json(source / "protocol.json", protocol)
    reference.write_json(
        source / "search.json",
        {"split": "search", "protocol": reference.digest(protocol), "tasks": tasks},
    )
    reference.write_json(
        source / "run.json", {"policy": policy, "runtime_image": images[reference.IMAGE_TAGS[0]]}
    )
    run = reference.prepare(source, output)
    return source, output, run


def test_preparation_preserves_panel_and_declares_one_architecture(prepared):
    source, output, run = prepared
    assert run["tasks"] == reference.read_json(source / "search.json")["tasks"]
    assert run["source_panel_sha256"] == reference.file_hash(source / "search.json")
    assert run["policy"]["instructions"] == ""
    assert run["adapter_policy"]["instructions"]
    assert run["scoring_protocol"] == container_runtime_v2.PROTOCOL
    assert "not an unmodified commercial CLI" in run["comparison_scope"]
    assert reference.verify_run(output)["identity_sha256"] == run["identity_sha256"]
    assert reference.read_json(output / "reference_state.json")["model_calls"] == 0


def test_seed_reference_uses_same_scoring_and_exact_original_instructions(prepared):
    source, output, task_only = prepared
    seed = reference.prepare(source, output.parent / "seed-reference", arm="seed_reference")
    assert seed["tasks"] == task_only["tasks"]
    assert seed["runtime_images"] == task_only["runtime_images"]
    assert seed["scoring_protocol"] == task_only["scoring_protocol"]
    assert seed["policy"]["instructions"] == seed["adapter_policy"]["instructions"]
    assert seed["effective_shared_instructions"]


def test_changed_code_panel_and_protocol_cannot_resume(prepared):
    _, output, run = prepared
    panel = reference.read_json(output / "source_panel.json")
    panel["tasks"][0]["task_id"] = "swapped"
    reference.write_json(output / "source_panel.json", panel)
    with pytest.raises(ValueError, match="copy changed"):
        reference.verify_run(output)
    reference.write_json(
        output / "source_panel.json", reference.read_json(Path(run["source_run"]) / "search.json")
    )
    (reference.ROOT / "moevo/codex/client.py").write_text("changed")
    with pytest.raises(ValueError, match="code/assets changed"):
        reference.verify_run(output)


def test_mutable_source_policy_cannot_relabel_a_different_seed(prepared):
    source, output, _ = prepared
    changed = reference.read_json(source / "run.json")
    changed["policy"]["instructions"] = "Different seed"
    reference.write_json(source / "run.json", changed)
    with pytest.raises(ValueError, match="Source policy"):
        reference.prepare(source, output.parent / "changed-seed", arm="seed_reference")


def test_changed_native_gold_record_blocks_resume(prepared, monkeypatch):
    _, output, _ = prepared
    monkeypatch.setattr(
        reference,
        "native_record_identity",
        lambda tasks: {task["id"]: "changed-gold" for task in tasks},
    )
    with pytest.raises(ValueError, match="native task/gold"):
        reference.verify_run(output)


@pytest.mark.parametrize("role", ["solver", "tau_agent"])
@pytest.mark.parametrize("separator", ["\n", "\n\n"])
def test_exact_prefix_removed_once_and_task_content_unchanged(role, separator):
    prefix = "Shared instruction"
    task = "Task text\nShared instruction\nThis occurrence is part of the task."
    original = prefix + separator + task
    effective, proof = reference.transformed_prompt(original, prefix, role, "gpt-6-astra", "xhigh")
    assert effective == task
    assert proof["removal_count"] == 1
    preserved, seed_proof = reference.transformed_prompt(
        original, prefix, role, "gpt-6-astra", "xhigh", "seed_reference"
    )
    assert preserved == original and seed_proof["unchanged"] and seed_proof["removal_count"] == 0


@pytest.mark.parametrize(
    "role,model,effort",
    [("judge", "gpt-5.6-terra", "medium"), ("tau_customer", "gpt-6-astra", "xhigh")],
)
def test_judge_and_customer_prompts_remain_byte_identical(role, model, effort):
    original = "Native benchmark protocol\nTask mentions Shared instruction in its body."
    effective, proof = reference.transformed_prompt(
        original, "Shared instruction", role, model, effort
    )
    assert effective == original
    assert proof["unchanged"] and not proof["prefix_removed"]


def test_prefix_role_and_model_mismatches_fail_before_inference():
    with pytest.raises(ValueError, match="exact frozen"):
        reference.transformed_prompt(
            "Other instructions\nTask", "Expected", "solver", "gpt-6-astra", "xhigh"
        )
    with pytest.raises(ValueError, match="Unknown model-call role"):
        reference.transformed_prompt(
            "Expected\nTask", "Expected", "unknown", "gpt-6-astra", "xhigh"
        )
    with pytest.raises(ValueError, match="Unexpected model"):
        reference.transformed_prompt(
            "Expected\nTask", "Expected", "solver", "gpt-5.6-terra", "medium"
        )


def test_task_errors_and_interruption_are_charged_and_not_silently_retried(prepared):
    _, output, run = prepared
    first = reference.charge_task(output, run, "finqa")
    assert first is not None
    state = reference.read_json(output / "reference_state.json")
    assert state["task_attempts"] == 1 and state["tasks"]["finqa"]["status"] == "running"
    reference.recover_interrupted(output, run)
    state = reference.read_json(output / "reference_state.json")
    assert state["tasks"]["finqa"]["status"] == "interrupted"
    assert state["tasks"]["finqa"]["score"] is None
    assert reference.charge_task(output, run, "finqa") is None
    second = reference.charge_task(output, run, "finqa", retry_errors=True)
    assert second is not None
    assert second["id"] != first["id"]
    assert reference.read_json(output / "reference_state.json")["task_attempts"] == 2


def write_result(output, run, task_id, attempt, score=0.0):
    task = next(t for t in run["tasks"] if t["id"] == task_id)
    reference.write_json(
        output / attempt["output"] / "output.json",
        {
            "identity_sha256": run["identity_sha256"],
            "attempt_id": attempt["id"],
            "task": task,
            "status": "scored",
            "score": score,
            "native_score": score,
        },
    )


def test_committed_native_zero_survives_coordinator_interruption_without_repeat(prepared):
    _, output, run = prepared
    attempt = reference.charge_task(output, run, "finqa")
    assert attempt is not None
    write_result(output, run, "finqa", attempt)
    reference.recover_interrupted(output, run)
    row = reference.read_json(output / "reference_state.json")["tasks"]["finqa"]
    assert row["status"] == "scored" and row["score"] == 0.0
    assert reference.charge_task(output, run, "finqa", retry_errors=True) is None
    assert reference.read_json(output / "reference_state.json")["task_attempts"] == 1


def test_model_budget_precharge_is_durable_and_concurrent(prepared):
    _, output, run = prepared
    run["max_model_calls"] = 3
    attempt = reference.charge_task(output, run, "finqa")
    assert attempt is not None

    def charge(_):
        try:
            return reference.charge_model_call(
                output, run, "finqa", attempt["id"], {"role": "judge"}
            )
        except RuntimeError:
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        ids = list(pool.map(charge, range(6)))
    assert len({value for value in ids if value is not None}) == 3
    state = reference.read_json(output / "reference_state.json")
    assert state["model_calls"] == 3
    assert all(call["status"] == "running" for call in state["calls"].values())
    assert state["model_usage_complete"] is False


def test_score_semantics_keep_native_negatives_and_legal_search_rate():
    health = {"benchmark": "healthbench_professional", "task_id": "native"}
    report = {"task_id": "native", "e2e_validated": True, "score": -0.25}
    assert reference.score_report(report, health) == (0.0, -0.25)
    legal = {"benchmark": "harvey_lab", "task_id": "native"}
    assert reference.score_report({**report, "score": 0.0, "criterion_pass_rate": 0.75}, legal) == (
        0.75,
        0.0,
    )
    with pytest.raises(ValueError, match="finite"):
        reference.score_report({**report, "score": True}, health)
    with pytest.raises(ValueError, match="exact requested"):
        reference.score_report({**report, "task_id": "wrong"}, health)


def test_resume_refuses_live_worker_or_unsettled_model_call(prepared):
    _, output, run = prepared
    attempt = reference.charge_task(output, run, "finqa")
    assert attempt is not None
    destination = output / attempt["output"]
    destination.mkdir(parents=True)
    with (destination / ".worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="worker is still active"):
            reference.recover_interrupted(output, run)
    reference.charge_model_call(output, run, "finqa", attempt["id"], {"role": "solver"})
    with pytest.raises(RuntimeError, match="Unsettled model call"):
        reference.recover_interrupted(output, run)
    state = reference.read_json(output / "reference_state.json")
    assert state["task_attempts"] == 1 and state["model_calls"] == 1
    assert state["tasks"]["finqa"]["status"] == "running"


def test_actual_prompt_audit_and_command_account_guards(prepared, monkeypatch, tmp_path):
    _, output, run = prepared
    attempt = reference.charge_task(output, run, "finqa")
    assert attempt is not None
    observed = []

    def fake_codex(prompt, **kwargs):
        model, effort = kwargs.get("model", "gpt-6-astra"), kwargs.get("effort", "xhigh")
        client.build_command(tmp_path, tmp_path / "answer", model, effort, False)
        observed.append(prompt)
        return client.CodexResponse("42", usage={"input_tokens": 10}, duration_s=0.1)

    monkeypatch.setattr(client, "run_codex", fake_codex)
    monkeypatch.setattr(client, "build_command", client.build_command)
    monkeypatch.setattr(subprocess, "Popen", subprocess.Popen)
    monkeypatch.setenv("OPENAI_API_KEY", "not-used-test-secret")
    reference.install_worker_hooks(output, run, "finqa", attempt["id"])
    namespace = {"__name__": "moevo.codex.container_runtime_v2", "client": client}
    exec("def solve_in_container(prompt):\n    return client.run_codex(prompt)\n", namespace)
    prefix = run["adapter_policy"]["instructions"]
    namespace["solve_in_container"](prefix + "\n\nQuestion")
    judge = {"__name__": "moevo.codex.judging", "client": client}
    exec(
        "def run_judge(prompt):\n    return client.run_codex(prompt, model='gpt-5.6-terra', effort='medium')\n",
        judge,
    )
    judge["run_judge"]("Private rubric, unchanged")
    tau = {"__name__": "moevo.codex.tau_pilot", "client": client}
    exec(
        "def generate(prompt, call_name):\n    return client.run_codex(prompt, model='gpt-6-astra', effort='xhigh')\n",
        tau,
    )
    tau["generate"]("Simulated customer, unchanged", "user_simulator_response")
    tau["generate"](prefix + "\n\nNative agent history", "agent_response")
    assert observed == [
        "Question",
        "Private rubric, unchanged",
        "Simulated customer, unchanged",
        "Native agent history",
    ]
    audit = reference.audit_attempt(output, attempt["id"])
    assert (
        audit["prefix_removals"] == 2 and audit["judge_calls"] == 1 and audit["customer_calls"] == 1
    )
    command = reference.read_json(output / "model_calls/call-00000/command.json")
    assert 'forced_login_method="chatgpt"' in command["argv"]
    assert command["api_credentials_removed"] is True
    assert "not-used-test-secret" not in json.dumps(command)
    assert (output / "model_calls/call-00001/prompt-original.txt").read_bytes() == (
        output / "model_calls/call-00001/prompt-effective.txt"
    ).read_bytes()


def test_v2_direct_answers_reach_grading_without_fabricating_tools(tmp_path, monkeypatch):
    @contextlib.contextmanager
    def container(*args, **kwargs):
        yield "task-container", "sha256:pinned"

    response = client.CodexResponse("42")
    monkeypatch.setattr(container_runtime_v2, "task_container", container)
    monkeypatch.setattr(client, "run_codex", lambda *a, **k: response)
    result = container_runtime_v2.solve_in_container(
        "Question", tmp_path / "workspace", tmp_path / "agent"
    )
    assert result is response
    runtime = reference.read_json(tmp_path / "agent/runtime.json")
    assert runtime["tool_calls"] == 0 and runtime["direct_response"] is True
    assert runtime["native_grade_pending"] is True
    assert not (tmp_path / "agent/tool_trace.json").exists()


def test_v2_does_not_turn_cli_failure_into_a_scored_answer(tmp_path, monkeypatch):
    @contextlib.contextmanager
    def container(*args, **kwargs):
        yield "task-container", "sha256:pinned"

    def failed(*args, **kwargs):
        raise client.CodexError("MCP startup failed")

    monkeypatch.setattr(container_runtime_v2, "task_container", container)
    monkeypatch.setattr(client, "run_codex", failed)
    with pytest.raises(client.CodexError, match="MCP startup failed"):
        container_runtime_v2.solve_in_container(
            "Question", tmp_path / "workspace", tmp_path / "agent"
        )


def test_docker_pinning_changes_only_exact_image_arguments():
    images = {"image:tag": "sha256:pinned"}
    assert reference._docker_argument_ids(["docker", "run", "image:tag", "echo", "x"], images) == [
        "docker",
        "run",
        "sha256:pinned",
        "echo",
        "x",
    ]
    assert reference._docker_argument_ids(["codex", "exec", "image:tag"], images) == [
        "codex",
        "exec",
        "image:tag",
    ]


@pytest.mark.parametrize("arm", [reference.BASELINE_ID, "seed_reference"])
def test_task_worker_native_zero_e2e_uses_v2_and_audits_prefix(
    prepared, monkeypatch, tmp_path, arm
):
    source, output, run = prepared
    if arm == "seed_reference":
        output = output.parent / "seed-worker"
        run = reference.prepare(source, output, arm=arm)
    attempt = reference.charge_task(output, run, "finqa")
    assert attempt is not None

    @contextlib.contextmanager
    def container(*args, **kwargs):
        yield "test-container", "sha256:pinned"

    monkeypatch.setattr(container_runtime_v2, "task_container", container)
    monkeypatch.setattr(domain_tasks, "solve_in_container", domain_tasks.solve_in_container)
    monkeypatch.setattr(client, "build_command", client.build_command)
    monkeypatch.setattr(subprocess, "Popen", subprocess.Popen)
    monkeypatch.setattr(signal, "signal", lambda *args: None)

    def fake_codex(prompt, **kwargs):
        assert prompt.startswith(run["adapter_policy"]["instructions"]) == (arm == "seed_reference")
        client.build_command(tmp_path, tmp_path / "answer", "gpt-6-astra", "xhigh", False)
        return client.CodexResponse("Wrong answer", usage={"output_tokens": 4})

    monkeypatch.setattr(client, "run_codex", fake_codex)

    class Backend:
        @staticmethod
        def evaluate(benchmark, native_id, policy, destination, image):
            response = domain_tasks.solve(policy, "Native question", destination, image)
            return {
                "task_id": native_id,
                "e2e_validated": True,
                "score": 0.0,
                "grade": {"reason": "incorrect"},
                "usage": response.usage,
            }

    monkeypatch.setattr(reference, "backend", lambda _: Backend)
    reference.worker(output, "finqa", attempt["id"])
    reference.finish_attempt(output, run, "finqa", attempt)
    row = reference.read_json(output / "reference_state.json")["tasks"]["finqa"]
    result = reference.read_json(output / row["output"])
    assert row["status"] == "scored" and row["score"] == 0
    assert result["prompt_audit"]["prefix_removals"] == int(arm == reference.BASELINE_ID)
    assert result["scoring_protocol"] == container_runtime_v2.PROTOCOL
