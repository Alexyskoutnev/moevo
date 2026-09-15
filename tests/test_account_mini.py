"""Account isolation and fail-closed proof validation for real mini runs."""

from pathlib import Path

import pytest

from moevo.codex.client import account_environment, build_command
from moevo.codex.formal_tasks import accepted_axioms, validate_proof_term


def test_account_environment_drops_provider_credentials(monkeypatch):
    for key in ("OPENAI_API_KEY", "CODEX_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL"):
        monkeypatch.setenv(key, "synthetic-test-value")
    monkeypatch.setenv("MOEVO_TEST_SETTING", "keep")
    env = account_environment()
    assert not any(k.endswith("API_KEY") for k in env)
    assert "OPENAI_BASE_URL" not in env
    assert env["MOEVO_TEST_SETTING"] == "keep"


def test_account_command_disables_fallback_and_host_tools():
    command = build_command(Path("/tmp/task"), Path("/tmp/result"), "gpt-6-astra", "xhigh", False)
    assert "--ignore-user-config" in command
    assert 'forced_login_method="chatgpt"' in command
    assert "features.shell_tool=false" in command
    assert 'web_search="disabled"' in command
    assert command[command.index("-m") + 1] == "gpt-6-astra"


@pytest.mark.parametrize(
    "proof",
    [
        "",
        "by sorry",
        "by admit",
        "by native_decide",
        "by\n  trivial\naxiom hacked : False",
        "by\n  run_elab pure ()",
    ],
)
def test_formal_submission_rejects_admissions_and_extra_commands(proof):
    with pytest.raises(ValueError):
        validate_proof_term(proof)


def test_formal_axioms_fail_closed():
    validate_proof_term("by\n  norm_num")
    assert accepted_axioms("'smoke' does not depend on any axioms", "smoke")
    assert accepted_axioms(
        "'smoke' depends on axioms: [propext, Classical.choice, Quot.sound]", "smoke"
    )
    assert not accepted_axioms("'smoke' depends on axioms: [sorryAx]", "smoke")
    assert not accepted_axioms("'smoke' depends on axioms: [custom_axiom]", "smoke")
    assert not accepted_axioms("'different_theorem' does not depend on any axioms", "smoke")
    assert not accepted_axioms("", "smoke")
    assert not accepted_axioms(
        "'smoke' does not depend on any axioms\n'smoke' depends on axioms: [sorryAx]", "smoke"
    )
