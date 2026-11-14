"""The development client must advertise the evaluator's episode horizon."""

from unittest.mock import Mock

import numpy as np
import pytest

from robocode.utils.env_client import BlackboxEnv


@pytest.mark.parametrize("variable_count", [False, True])
def test_reset_reports_selected_episode_horizon(monkeypatch, variable_count):
    """Use the returned count, including default resets and shorter later episodes."""
    space = {
        "type": "Box",
        "shape": [1],
        "low": [0.0],
        "high": [1.0],
        "dtype": "float32",
    }
    metadata = {
        "host": "unused",
        "port": 1,
        "token": "test",
        "action_space": space,
        "observation_space": space,
        "max_steps": 200,
    }
    limits = {"1": 620, "5": 700, "50": 1600}
    if variable_count:
        metadata["counts"] = {"max_steps": limits}
    monkeypatch.setattr("socket.create_connection", Mock(return_value=Mock()))
    with BlackboxEnv(metadata) as client:
        for count in [1, 50, 5]:
            info = {"object_count": count} if variable_count else {}
            response = {"obs": np.array([count]), "info": info}
            monkeypatch.setattr(client, "_request", Mock(return_value=response))
            _, returned_info = client.reset(seed=7)
            assert returned_info == info
            assert client.max_steps == (limits[str(count)] if variable_count else 200)
