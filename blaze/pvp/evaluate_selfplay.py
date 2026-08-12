#!/usr/bin/env python3
"""Evaluate both policies from a checkpoint head-to-head without learning."""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from pvp import N_ACT, NetworkVecPvp, VecPvp
from train_selfplay import (CONTINUOUS_ACTION_SCHEMA, CONTINUOUS_LOOK_ACTION_SCHEMA,
                            FINE_ACTION_SCHEMA, PITCH_LIMIT, Policy, YAW_LIMIT,
                            V2_ACTION_SCHEMA, V21_ACTION_SCHEMA,
                            checkpoint_action_schema, decode_actions, env_tensor,
                            greedy_actions, is_continuous_schema,
                            is_iron_gear_schema, policy_input,
                            sample_actions)


@torch.no_grad()
def run(checkpoint, episodes, horizon, repeat, stochastic, seed, action_schema,
        swap_policies=False, deterministic_yaw=False, action_hold_prob=0.0,
        extra_repeat_prob=0.0, ping_min_ms=20.0, ping_max_ms=200.0,
        fixed_ping_ms=None):
    torch.manual_seed(seed)
    device = torch.device("cpu")
    policies = [Policy(action_schema=action_schema).eval(),
                Policy(action_schema=action_schema).eval()]
    for role, policy in enumerate(policies):
        source_role = 1 - role if swap_policies else role
        policy.load_state_dict(checkpoint["models"][source_role])
    env = (NetworkVecPvp(episodes, min_ping_ms=ping_min_ms,
                         max_ping_ms=ping_max_ms)
           if action_schema == V21_ACTION_SCHEMA else VecPvp(episodes))
    seeds = np.arange(seed, seed + episodes, dtype=np.uint64)
    obs = env_tensor(env.reset(seeds), device)
    if fixed_ping_ms is not None:
        if action_schema != V21_ACTION_SCHEMA:
            raise ValueError("fixed ping evaluation requires V2.1")
        env.set_base_ping_ms(np.asarray(fixed_ping_ms, dtype=np.float32))
        obs = env_tensor(env.obs, device)
    active = torch.ones(episodes, dtype=torch.bool)
    wins = [0, 0]
    draws = 0
    hits = torch.zeros(2, dtype=torch.int64)
    damage = torch.zeros(2, dtype=torch.float64)
    terminal_steps = []
    pursuit = {
        "player_steps": 0, "bearing_abs_sum_deg": 0.0,
        "bearing_under_15": 0, "behind": 0, "forward_while_behind": 0,
        "inside_reach": 0, "near_wall": 0, "forward": 0,
        "yaw_nonzero": 0, "yaw_saturated": 0, "distance_sum": 0.0,
        "yaw_abs_sum": 0.0, "yaw_variation_sum": 0.0, "yaw_sign_reversals": 0,
        "pitch_abs_sum": 0.0, "pitch_variation_sum": 0.0,
        "pitch_sign_reversals": 0, "pitch_saturated": 0,
        "absolute_pitch_sum": 0.0,
        "axe_selected": 0, "attack_intent": 0, "block_intent": 0,
        "mutual_block": 0,
    }
    previous_yaw = None
    previous_pitch = None
    first_hit = torch.full((episodes, 2), -1, dtype=torch.int64)
    previous_rows = torch.zeros((episodes, 2, N_ACT), dtype=torch.float64)
    for step in range(horizon):
        actions = []
        for role in range(2):
            actor, _ = policies[role](policy_input(policies[role], obs[:, role]))
            if stochastic:
                action, _, _ = sample_actions(
                    actor, action_schema, deterministic_yaw=deterministic_yaw,
                    deterministic_pitch=deterministic_yaw)
            else:
                action = greedy_actions(actor, action_schema)
            actions.append(action)
        action_tensor = torch.stack(actions, dim=1)
        # Pursuit quality is measured before applying this decision and only for
        # lanes whose first fight is still active. obs[:, role, 5:7] is the
        # opponent's egocentric lateral/longitudinal displacement.
        active_players = active[:, None].expand(-1, 2)
        lateral = obs[:, :, 5]
        longitudinal = obs[:, :, 6]
        bearing = torch.atan2(lateral.abs(), longitudinal) * (180.0 / math.pi)
        distance = obs[:, :, 10] * 32.0
        behind = longitudinal < 0.0
        forward = action_tensor[:, :, 0] == 2
        yaw_coarse = action_tensor[:, :, 2]
        yaw_fine = action_tensor[:, :, 3]
        if is_continuous_schema(action_schema):
            yaw_delta = yaw_coarse
            yaw_nonzero = yaw_delta.abs() > 0.1
            yaw_saturated = yaw_delta.abs() >= (YAW_LIMIT - 0.1)
        elif action_schema == FINE_ACTION_SCHEMA:
            yaw_delta = ((yaw_coarse - 1) * 15 + (yaw_fine - 1) * 5)
            yaw_nonzero = yaw_delta != 0
            yaw_saturated = yaw_delta.abs() == 20
        else:
            yaw_delta = (yaw_coarse - 1) * 15
            yaw_nonzero = yaw_coarse != 1
            yaw_saturated = (yaw_coarse == 0) | (yaw_coarse == 2)
        wall = obs[:, :, 20:24].amin(-1) < 0.05
        pursuit["player_steps"] += int(active_players.sum())
        pursuit["bearing_abs_sum_deg"] += float(bearing[active_players].sum())
        pursuit["bearing_under_15"] += int(((bearing < 15.0) & active_players).sum())
        pursuit["behind"] += int((behind & active_players).sum())
        pursuit["forward_while_behind"] += int((forward & behind & active_players).sum())
        pursuit["inside_reach"] += int(((distance < 3.0) & active_players).sum())
        pursuit["near_wall"] += int((wall & active_players).sum())
        pursuit["forward"] += int((forward & active_players).sum())
        pursuit["yaw_nonzero"] += int((yaw_nonzero & active_players).sum())
        pursuit["yaw_saturated"] += int((yaw_saturated & active_players).sum())
        pursuit["yaw_abs_sum"] += float((yaw_delta.abs() * active_players).sum())
        if previous_yaw is not None:
            pursuit["yaw_variation_sum"] += float(
                ((yaw_delta - previous_yaw).abs() * active_players).sum())
            pursuit["yaw_sign_reversals"] += int(
                (((yaw_delta * previous_yaw) < 0.0) &
                 (yaw_delta.abs() > 0.1) & (previous_yaw.abs() > 0.1) &
                 active_players).sum())
        previous_yaw = yaw_delta.clone()
        if action_schema in (CONTINUOUS_LOOK_ACTION_SCHEMA, V2_ACTION_SCHEMA,
                             V21_ACTION_SCHEMA):
            pitch_delta = action_tensor[:, :, 3]
            pursuit["pitch_abs_sum"] += float(
                (pitch_delta.abs() * active_players).sum())
            pursuit["pitch_saturated"] += int(
                ((pitch_delta.abs() >= (PITCH_LIMIT - 0.1)) & active_players).sum())
            pursuit["absolute_pitch_sum"] += float(
                ((obs[:, :, 24].abs() * 90.0) * active_players).sum())
            if previous_pitch is not None:
                pursuit["pitch_variation_sum"] += float(
                    ((pitch_delta - previous_pitch).abs() * active_players).sum())
                pursuit["pitch_sign_reversals"] += int(
                    (((pitch_delta * previous_pitch) < 0.0) &
                     (pitch_delta.abs() > 0.05) & (previous_pitch.abs() > 0.05) &
                     active_players).sum())
            previous_pitch = pitch_delta.clone()
        if is_iron_gear_schema(action_schema):
            pursuit["axe_selected"] += int(
                ((action_tensor[:, :, 7] == 1) & active_players).sum())
            pursuit["attack_intent"] += int(
                ((action_tensor[:, :, 6] == 1) & active_players).sum())
            pursuit["block_intent"] += int(
                ((action_tensor[:, :, 6] == 2) & active_players).sum())
            pursuit["mutual_block"] += int(
                ((obs[:, 0, 27] > 0.5) & (obs[:, 1, 27] > 0.5) & active).sum())
        pursuit["distance_sum"] += float(distance[active_players].sum())
        rows = decode_actions(action_tensor, device, action_schema)
        executed_rows = rows
        if action_hold_prob > 0.0:
            hold = torch.rand((episodes, 2)) < action_hold_prob
            executed_rows = torch.where(hold[:, :, None], previous_rows, rows)
        previous_rows = rows.clone()
        step_repeat = repeat + int(extra_repeat_prob > 0.0 and
                                   float(torch.rand(())) < extra_repeat_prob)
        obs, reward, done, hs, dmg = env.step(executed_rows, repeat=step_repeat)
        obs = env_tensor(obs, device)
        active_hits = env_tensor(hs, device)[active]
        active_damage = env_tensor(dmg, device)[active]
        hits += active_hits.sum(0).cpu()
        damage += active_damage.sum(0).cpu()
        hs_all = env_tensor(hs, device)
        for role in range(2):
            landed = (hs_all[:, role] > 0) & active & (first_hit[:, role] < 0)
            first_hit[landed, role] = step + 1
        newly = env_tensor(done, device).bool() & active
        if newly.any():
            terminal = env_tensor(reward, device)[newly]
            wins[0] += int((terminal[:, 0] > terminal[:, 1]).sum())
            wins[1] += int((terminal[:, 1] > terminal[:, 0]).sum())
            draws += int((terminal[:, 0] == terminal[:, 1]).sum())
            terminal_steps.extend([step + 1] * int(newly.sum()))
            active[newly] = False
        if not active.any():
            break
    horizon_draws = int(active.sum())
    draws += horizon_draws
    env.close()
    completed = len(terminal_steps)
    player_steps = max(1, pursuit["player_steps"])
    behind_steps = max(1, pursuit["behind"])
    observed_first_hits = first_hit[first_hit >= 0]
    return {
        "mode": ("hybrid" if stochastic and deterministic_yaw else
                 "sampled" if stochastic else "greedy"),
        "policy_assignment": "swapped" if swap_policies else "native",
        "action_schema": action_schema,
        "action_hold_prob": action_hold_prob,
        "extra_repeat_prob": extra_repeat_prob,
        "ping_min_ms": ping_min_ms if action_schema == V21_ACTION_SCHEMA else None,
        "ping_max_ms": ping_max_ms if action_schema == V21_ACTION_SCHEMA else None,
        "fixed_ping_ms": fixed_ping_ms,
        "episodes": episodes, "horizon_decisions": horizon, "repeat": repeat,
        "wins_role0": wins[0], "wins_role1": wins[1], "draws": draws,
        "horizon_draws": horizon_draws,
        "hits_role0": int(hits[0]), "hits_role1": int(hits[1]),
        "damage_role0": float(damage[0]), "damage_role1": float(damage[1]),
        "completed_fights": completed,
        "mean_decisions_to_death": (sum(terminal_steps) / completed if completed else None),
        "mean_minecraft_ticks_to_death":
            (repeat * sum(terminal_steps) / completed if completed else None),
        "mean_abs_bearing_error_deg": pursuit["bearing_abs_sum_deg"] / player_steps,
        "bearing_under_15_fraction": pursuit["bearing_under_15"] / player_steps,
        "behind_fraction": pursuit["behind"] / player_steps,
        "forward_while_behind_fraction": pursuit["forward_while_behind"] / behind_steps,
        "forward_while_behind_player_fraction":
            pursuit["forward_while_behind"] / player_steps,
        "inside_reach_fraction": pursuit["inside_reach"] / player_steps,
        "near_wall_fraction": pursuit["near_wall"] / player_steps,
        "forward_fraction": pursuit["forward"] / player_steps,
        "yaw_nonzero_fraction": pursuit["yaw_nonzero"] / player_steps,
        "yaw_saturated_fraction": pursuit["yaw_saturated"] / player_steps,
        "mean_abs_yaw_delta_deg": pursuit["yaw_abs_sum"] / player_steps,
        "mean_abs_yaw_variation_deg": pursuit["yaw_variation_sum"] / player_steps,
        "yaw_sign_reversals_per_1000_player_steps":
            1000.0 * pursuit["yaw_sign_reversals"] / player_steps,
        "mean_abs_pitch_delta_deg": pursuit["pitch_abs_sum"] / player_steps,
        "mean_abs_pitch_variation_deg": pursuit["pitch_variation_sum"] / player_steps,
        "pitch_sign_reversals_per_1000_player_steps":
            1000.0 * pursuit["pitch_sign_reversals"] / player_steps,
        "pitch_saturated_fraction": pursuit["pitch_saturated"] / player_steps,
        "mean_abs_pitch_deg": pursuit["absolute_pitch_sum"] / player_steps,
        "axe_selected_fraction": pursuit["axe_selected"] / player_steps,
        "attack_intent_fraction": pursuit["attack_intent"] / player_steps,
        "block_intent_fraction": pursuit["block_intent"] / player_steps,
        "mutual_block_lane_fraction": pursuit["mutual_block"] /
            max(1, player_steps / 2),
        "mean_distance_blocks": pursuit["distance_sum"] / player_steps,
        "mean_decisions_to_first_hit": (float(observed_first_hits.float().mean())
                                        if len(observed_first_hits) else None),
        "mean_minecraft_ticks_to_first_hit":
            (repeat * float(observed_first_hits.float().mean())
             if len(observed_first_hits) else None),
        "first_hit_role_samples": int((first_hit >= 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--episodes", type=int, default=256)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--repeat", type=int,
                    help="defaults to the checkpoint's recorded action repeat")
    ap.add_argument("--seed", type=int, default=700000)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--include-role-swapped", action="store_true")
    ap.add_argument("--action-hold-prob", type=float, default=0.0,
                    help="independently execute each role's prior action")
    ap.add_argument("--extra-repeat-prob", type=float, default=0.0,
                    help="probability a decision spans one additional env tick")
    ap.add_argument("--ping-min-ms", type=float, default=20.0)
    ap.add_argument("--ping-max-ms", type=float, default=200.0)
    ap.add_argument("--fixed-ping-ms",
                    help="two comma-separated per-role baseline RTTs")
    args = ap.parse_args()
    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ck.get("config", {})
    action_schema = checkpoint_action_schema(config)
    repeat = args.repeat if args.repeat is not None else int(config.get("repeat", 4))
    domain = {"action_hold_prob": args.action_hold_prob,
              "extra_repeat_prob": args.extra_repeat_prob,
              "ping_min_ms": args.ping_min_ms,
              "ping_max_ms": args.ping_max_ms,
              "fixed_ping_ms": fixed_ping}
    result = {
        "checkpoint": str(args.checkpoint),
        "action_schema": action_schema, "repeat": repeat,
        "greedy": run(ck, args.episodes, args.horizon, repeat, False, args.seed,
                      action_schema, **domain),
        "sampled": run(ck, args.episodes, args.horizon, repeat, True, args.seed,
                       action_schema, **domain),
    }
    if args.include_role_swapped:
        result["greedy_swapped"] = run(
            ck, args.episodes, args.horizon, repeat, False, args.seed,
            action_schema, swap_policies=True, **domain)
        result["sampled_swapped"] = run(
            ck, args.episodes, args.horizon, repeat, True, args.seed,
            action_schema, swap_policies=True, **domain)
    if is_continuous_schema(action_schema):
        result["hybrid"] = run(
            ck, args.episodes, args.horizon, repeat, True, args.seed,
            action_schema, deterministic_yaw=True, **domain)
        if args.include_role_swapped:
            result["hybrid_swapped"] = run(
                ck, args.episodes, args.horizon, repeat, True, args.seed,
                action_schema, swap_policies=True, deterministic_yaw=True, **domain)
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
    fixed_ping = ([float(x) for x in args.fixed_ping_ms.split(",")]
                  if args.fixed_ping_ms else None)
    if fixed_ping is not None and len(fixed_ping) != 2:
        ap.error("--fixed-ping-ms requires two values")
