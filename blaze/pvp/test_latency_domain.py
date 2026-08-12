import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from pvp import CPU_SO, N_NETWORK_OBS, NetworkVecPvp
from train_selfplay import (Policy, V2_ACTION_SCHEMA, V21_ACTION_SCHEMA,
                            action_observation_dim, checkpoint_action_schema,
                            scripted_action_indices, transfer_v2_latency_policy)
sys.path.insert(0, str(HERE.parent.parent / "java"))
from deploy_pvp_checkpoint import PolicyObservationHistory, SimulatedActionUplink


def _player(x):
    return {"x": x, "y": 65.0, "z": 0.0, "mx": 0.0, "my": 0.0, "mz": 0.0,
            "yaw": -90.0 if x < 0 else 90.0, "pitch": 0.0, "health": 20.0,
            "cooldown": 1.0, "hurt": 0, "on_ground": True, "sprinting": False,
            "dead": False}


def test_ping_is_independent_per_player_private_and_bounded():
    env = NetworkVecPvp(32, so_path=CPU_SO, min_ping_ms=20.0, max_ping_ms=200.0)
    obs = env.reset(np.arange(32, dtype=np.uint64))
    assert obs.shape == (32, 2, N_NETWORK_OBS)
    assert np.any(env._base_ping[:, 0] != env._base_ping[:, 1])
    np.testing.assert_allclose(obs[..., -1] * 200.0, env.current_ping_ms(), rtol=1e-6)
    own = obs[0, 0, -1]
    env._base_ping[0, 1] = 20.0 if env._base_ping[0, 1] > 20.0 else 200.0
    env._publish_observation(env._delays(env.current_ping_ms(), 2.7))
    assert obs[0, 0, -1] == own
    for _ in range(250):
        env.step(np.zeros((32, 2, 9), dtype=np.float64))
        ratio = env.current_ping_ms() / env._base_ping
        assert ratio.min() >= 0.95 - 1e-6
        assert ratio.max() <= 1.05 + 1e-6
    env.close()


def test_masked_reset_only_resamples_selected_network_lanes():
    env = NetworkVecPvp(4, so_path=CPU_SO)
    seeds = np.arange(4, dtype=np.uint64)
    env.reset(seeds)
    before = env._base_ping.copy()
    seeds[1] += 1000
    env.reset(seeds, np.array([0, 1, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(env._base_ping[[0, 2, 3]], before[[0, 2, 3]])
    assert np.any(env._base_ping[1] != before[1])
    env.close()


def test_asymmetric_curriculum_balances_disadvantaged_role_and_gap():
    env = NetworkVecPvp(4096, so_path=CPU_SO,
                        latency_profile="asymmetric_curriculum")
    env.reset(np.arange(4096, dtype=np.uint64))
    gap = env._base_ping[:, 0] - env._base_ping[:, 1]
    assert (np.abs(gap) >= 60.0).mean() >= 0.55
    assert 0.45 <= (gap > 0).mean() <= 0.55
    env.close()


def test_latency_teacher_leads_moving_target_instead_of_chasing_stale_bearing():
    obs = torch.zeros((1, N_NETWORK_OBS))
    obs[0, 6] = 3.0 / 32.0  # target ahead
    obs[0, 8] = 0.5         # target moving laterally per tick
    obs[0, -1] = 1.0        # own RTT = 200 ms, four-tick prediction horizon
    latency = scripted_action_indices(obs, action_schema=V21_ACTION_SCHEMA)
    no_latency = scripted_action_indices(obs[:, :35], action_schema=V2_ACTION_SCHEMA)
    assert abs(float(latency[0, 2])) > abs(float(no_latency[0, 2])) + 1.0


def test_latency_schema_expands_v2_without_changing_initial_behavior():
    torch.manual_seed(7)
    source = Policy(action_schema=V2_ACTION_SCHEMA)
    target = Policy(action_schema=V21_ACTION_SCHEMA)
    transfer_v2_latency_policy(target, source.state_dict())
    current = torch.randn(16, 35)
    expanded = torch.zeros(16, N_NETWORK_OBS)
    expanded[:, :35] = current
    source_actor, source_value = source(current)
    target_actor, target_value = target(expanded)
    for a, b in zip(source_actor["categorical"], target_actor["categorical"]):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(source_actor["yaw_mean"], target_actor["yaw_mean"])
    torch.testing.assert_close(source_actor["pitch_mean"], target_actor["pitch_mean"])
    torch.testing.assert_close(source_value, target_value)
    assert action_observation_dim(V21_ACTION_SCHEMA) == 141
    assert checkpoint_action_schema({"action_schema": V21_ACTION_SCHEMA}) == V21_ACTION_SCHEMA


def test_deployment_history_feeds_only_each_clients_own_ping():
    history = PolicyObservationHistory([_player(-4.0), _player(4.0)],
                                       [{"ping_ms": 25}, {"ping_ms": 175}])
    assert history.encode(0).shape == (141,)
    assert history.encode(1).shape == (141,)
    assert history.encode(0)[-1] == 25 / 200
    assert history.encode(1)[-1] == 175 / 200
    history.update([_player(-3.5), _player(3.5)],
                   [{"ping_ms": 30}, {"ping_ms": 160}])
    assert history.encode(0)[-1] == 30 / 200
    assert history.encode(1)[-1] == 160 / 200


def test_deployment_simulated_ping_varies_bounded_and_delays_fifo_actions():
    players = [_player(-4.0), _player(4.0)]
    history = PolicyObservationHistory(players, [{}, {}], [30.0, 160.0])
    uplink = SimulatedActionUplink(history)
    seen = [[], []]
    for step in range(200):
        selected = [[float(step)] + [0.0] * 8,
                    [float(step)] + [0.0] * 8]
        applied = uplink.submit(selected)
        for role in range(2):
            seen[role].append(applied[role][0])
            ping = history.current_ping(role)
            assert history.base_ping_ms[role] * .95 <= ping <= history.base_ping_ms[role] * 1.05
        history.update(players, [{}, {}])
    # High-ping player sees a longer lag and applied packet sequence never rewinds.
    assert seen[1][20] < seen[0][20]
    assert all(b >= a for role in seen for a, b in zip(role, role[1:]))
    assert history.encode(0)[-1] != history.encode(1)[-1]
