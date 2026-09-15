import pytest

from moevo.codex.headroom import assess_headroom


def test_smoke_does_not_establish_headroom():
    assert assess_headroom([0], verified_task_failures=1)["status"] == "needs_more_screening"


def test_saturation_and_practical_improvement_room():
    assert assess_headroom([1] * 32)["status"] == "saturated_on_screen"
    assert assess_headroom([1] * 31 + [0])["status"] == "insufficient_practical_headroom"
    scores = [1] * 24 + [0] * 8
    assert assess_headroom(scores)["status"] == "audit_task_failures"
    assert (
        assess_headroom(scores, verified_task_failures=4)["status"]
        == "headroom_confirmed_on_screen"
    )


def test_infrastructure_errors_never_qualify_as_task_headroom():
    result = assess_headroom([0] * 32, infrastructure_errors=1, verified_task_failures=32)
    assert result["status"] == "repair_infrastructure"


def test_invalid_results_rejected():
    with pytest.raises(ValueError):
        assess_headroom([float("nan")])
    with pytest.raises(ValueError):
        assess_headroom([1], verified_task_failures=1)
