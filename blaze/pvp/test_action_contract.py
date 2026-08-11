#!/usr/bin/env python3
"""Deterministic contract gates for legacy and 20 Hz boxing actions."""
import math
import pathlib
import sys

import numpy as np
import torch
import pytest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pvp import CPU_SO, CUDA_SO, N_ACT, VecPvp
from train_selfplay import (CONTINUOUS_ACTION_SCHEMA, FINE_ACTION_SCHEMA,
                            LEGACY_ACTION_SCHEMA, Policy, checkpoint_action_schema,
                            decode_actions, evaluate_actions, greedy_actions,
                            sample_actions, scripted_action_indices)


def test_checkpoint_schema_is_explicit_and_legacy_missing_is_frozen():
    assert checkpoint_action_schema({}) == LEGACY_ACTION_SCHEMA
    assert checkpoint_action_schema({"action_schema": FINE_ACTION_SCHEMA}) == FINE_ACTION_SCHEMA
    assert checkpoint_action_schema({"action_schema": CONTINUOUS_ACTION_SCHEMA}) == CONTINUOUS_ACTION_SCHEMA
    try:
        checkpoint_action_schema({"action_schema": "invented"})
    except ValueError:
        pass
    else:
        raise AssertionError("unknown action schema was accepted")


def test_fine_yaw_cartesian_grid_and_pitch_removal():
    actions = []
    for coarse in range(3):
        for fine in range(3):
            actions.append([1, 1, coarse, fine, 0, 0, 0])
    decoded = decode_actions(torch.tensor(actions), torch.device("cpu"),
                             FINE_ACTION_SCHEMA)
    assert decoded[:, 2].tolist() == [-20, -15, -10, -5, 0, 5, 10, 15, 20]
    assert decoded[:, 3].tolist() == [0] * 9
    legacy = decode_actions(torch.tensor([[1, 1, 2, 0, 0, 0, 0]]),
                            torch.device("cpu"), LEGACY_ACTION_SCHEMA)
    assert legacy[0, 2] == 15 and legacy[0, 3] == -10


def test_teacher_turns_symmetrically_and_uses_fine_residual():
    obs = torch.zeros((3, 24))
    obs[:2, 6] = 1.0
    obs[0, 5] = 0.1
    obs[1, 5] = -0.1
    obs[:, 10] = 8.0 / 32.0
    obs[2, 6] = -1.0
    labels = scripted_action_indices(obs, action_schema=FINE_ACTION_SCHEMA)
    decoded = decode_actions(labels, torch.device("cpu"), FINE_ACTION_SCHEMA)
    assert decoded[0, 2] == -5
    assert decoded[1, 2] == 5
    assert decoded[2, 0] == 0 and decoded[2, 5] == 0


def test_continuous_yaw_is_bounded_exact_and_pitch_stays_fixed():
    actions = torch.zeros((5, N_ACT))
    actions[:, 0:2] = 1
    actions[:, 2] = torch.tensor([-20.0, -7.25, 0.0, 3.125, 20.0])
    decoded = decode_actions(actions, torch.device("cpu"), CONTINUOUS_ACTION_SCHEMA)
    torch.testing.assert_close(decoded[:, 2], actions[:, 2].double())
    assert decoded[:, 3].tolist() == [0.0] * 5


def test_continuous_policy_logprob_roundtrip_and_greedy_mean():
    torch.manual_seed(4)
    policy = Policy(action_schema=CONTINUOUS_ACTION_SCHEMA)
    actor, _ = policy(torch.randn(32, 24))
    sampled, old_logp, entropy = sample_actions(actor, CONTINUOUS_ACTION_SCHEMA)
    new_logp, new_entropy = evaluate_actions(actor, sampled, CONTINUOUS_ACTION_SCHEMA)
    torch.testing.assert_close(new_logp, old_logp)
    torch.testing.assert_close(new_entropy, entropy)
    greedy = greedy_actions(actor, CONTINUOUS_ACTION_SCHEMA)
    assert bool((greedy[:, 2].abs() < 20.0).all())
    assert bool((sampled[:, 2].abs() < 20.0).all())


def test_continuous_teacher_preserves_unquantized_correction():
    obs = torch.zeros((2, 24))
    obs[:, 6] = 1.0
    obs[:, 5] = torch.tensor([0.123, -0.123])
    obs[:, 10] = 8.0 / 32.0
    labels = scripted_action_indices(obs, action_schema=CONTINUOUS_ACTION_SCHEMA)
    assert labels[0, 2] < 0 < labels[1, 2]
    assert not math.isclose(abs(float(labels[0, 2])) % 5.0, 0.0, abs_tol=1e-4)


def test_native_one_tick_applies_exact_fine_yaw_delta():
    env = VecPvp(1, so_path=CPU_SO)
    # axis X, role-0 yaw perturbation zero: initial yaw = -90 degrees.
    seed = np.asarray([3 << 9], dtype=np.uint64)
    before = np.asarray(env.reset(seed)).copy()
    rows = np.zeros((1, 2, N_ACT), dtype=np.float64)
    rows[0, 0, 2] = 5.0
    after = np.asarray(env.step(rows, repeat=1)[0]).copy()
    yaw_before = math.degrees(math.atan2(before[0, 0, 13], before[0, 0, 14]))
    yaw_after = math.degrees(math.atan2(after[0, 0, 13], after[0, 0, 14]))
    assert abs((yaw_after - yaw_before) - 5.0) < 0.05
    env.close()


def test_cpu_cuda_fine_trajectory_and_observation_parity():
    if not torch.cuda.is_available() or not pathlib.Path(CUDA_SO).exists():
        pytest.skip("CUDA parity gate requires a built CUDA backend")
    n = 64
    cpu = VecPvp(n, so_path=CPU_SO)
    gpu = VecPvp(n, so_path=CUDA_SO)
    seeds = np.arange(88000, 88000 + n, dtype=np.uint64)
    np.testing.assert_array_equal(cpu.reset(seeds), gpu.reset(seeds).cpu().numpy())
    rng = np.random.default_rng(991)
    yaw_grid = np.asarray([-20, -15, -10, -5, 0, 5, 10, 15, 20], np.float64)
    for step in range(256):
        rows = np.zeros((n, 2, N_ACT), dtype=np.float64)
        rows[:, :, 0] = rng.integers(-1, 2, size=(n, 2))
        rows[:, :, 1] = rng.integers(-1, 2, size=(n, 2))
        rows[:, :, 2] = yaw_grid[rng.integers(0, len(yaw_grid), size=(n, 2))]
        rows[:, :, 4:] = rng.integers(0, 2, size=(n, 2, 3))
        co = cpu.step(rows, repeat=1)
        go = gpu.step(rows, repeat=1)
        for c, g in zip(co, go):
            np.testing.assert_array_equal(np.asarray(c), g.cpu().numpy())
        done = np.asarray(co[2]).astype(np.uint8)
        if done.any():
            seeds[done.astype(bool)] += np.uint64(n * 1000 + step + 1)
            np.testing.assert_array_equal(
                cpu.reset(seeds, done), gpu.reset(seeds, done).cpu().numpy())
    cpu.close()
    gpu.close()


if __name__ == "__main__":
    test_checkpoint_schema_is_explicit_and_legacy_missing_is_frozen()
    test_fine_yaw_cartesian_grid_and_pitch_removal()
    test_teacher_turns_symmetrically_and_uses_fine_residual()
    test_native_one_tick_applies_exact_fine_yaw_delta()
    print("PASS fine 20 Hz action contract")
