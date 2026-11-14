"""A sampled self-solving candidate is rejected without changing the active world."""

from types import SimpleNamespace

import pytest

from experiments.overnight_exchange import ExperimentExchange
from robocode.runtime.exchange import Exchange


@pytest.mark.parametrize("solved,crashed", [(False, 0), (True, 0), (False, 1)])
def test_action_requirement_keeps_structural_receipt(
    tmp_path, monkeypatch, solved, crashed
):
    exchange = object.__new__(ExperimentExchange)
    exchange.directory = tmp_path
    exchange.require_action = True
    exchange.config = SimpleNamespace(
        backend="docker",
        environment_image="env",
        evaluation_image="eval",
        runtime_source=tmp_path,
        max_steps=1200,
        timeout=60,
    )
    directory = tmp_path / "validation/candidate"
    directory.mkdir(parents=True)

    def validate(*_args):
        return {
            "valid": True,
            "contract": {"counts": {"design": [0, 1, 2], "eval": [0, 1, 2, 3, 4]}},
        }

    monkeypatch.setattr(Exchange, "_validate", validate)

    def evaluate(*_args, **kwargs):
        assert kwargs["eval_counts"] == list(range(5)) * 2
        assert kwargs["eval_seeds"] == [0] * 5 + [1] * 5
        assert kwargs["max_steps"] == 1200
        return {"num_crashed_episodes": crashed, "per_episode": [{"solved": solved}]}

    monkeypatch.setattr(
        "experiments.overnight_exchange.evaluate_in_container", evaluate
    )
    result = exchange._validate(object(), "candidate")
    assert result["valid"] == (not solved and not crashed)
    assert result["zero_action_check"]["per_episode"][0]["solved"] is solved
    assert (directory / "result.json").exists()
