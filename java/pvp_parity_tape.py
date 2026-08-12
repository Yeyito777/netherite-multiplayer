#!/usr/bin/env python3
"""Replay deterministic PvP action tapes in Blaze and two real MC clients.

The clients must already be connected on qrl ports 25575/25576.  The report
compares the exact 35-value policy observation, which is the relevant contract
for transfer even where Minecraft's internal representation differs.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "java"))
from deploy_pvp_checkpoint import (PersistentBridge, client_action,
                                   decode_client_player, observation)
sys.path.insert(0, str(ROOT / "blaze" / "pvp"))
from pvp import CPU_SO, N_ACT, VecPvp

OBS_NAMES = [
    "self_health", "opponent_health", "forward_velocity", "strafe_velocity",
    "vertical_velocity", "opponent_forward", "opponent_right", "opponent_y",
    "relative_forward_velocity", "relative_right_velocity", "distance",
    "opponent_yaw_sin", "opponent_yaw_cos", "self_yaw_sin", "self_yaw_cos",
    "attack_cooldown", "self_hurt", "opponent_hurt", "on_ground", "sprinting",
    "wall_x_min", "wall_x_max", "wall_z_min", "wall_z_max", "pitch",
    "self_weapon", "opponent_weapon", "self_blocking", "opponent_blocking",
    "self_shield_disabled", "opponent_shield_disabled", "self_shield_durability",
    "opponent_shield_durability", "self_shield_use", "opponent_shield_use",
]
SEED_EXACT_FACE_OFF = 12518  # axis=x, zero lateral offsets and zero yaw error


def action(forward=0, strafe=0, yaw=0, pitch=0, jump=0, sprint=0,
           attack=0, weapon=0, block=0):
    return [forward, strafe, yaw, pitch, jump, sprint, attack, weapon, block]


def tapes():
    idle = [[action(), action()] for _ in range(40)]
    approach = [[action(1, sprint=1), action(1, sprint=1)] for _ in range(60)]
    turn = [[action(1, yaw=2), action(1, yaw=-2)] for _ in range(60)]
    gear = []
    gear += [[action(1, sprint=1), action(1, sprint=1)] for _ in range(12)]
    gear += [[action(weapon=1), action(block=1)] for _ in range(10)]
    gear += [[action(weapon=1, attack=1), action(block=1)] for _ in range(30)]
    gear += [[action(weapon=0, attack=1), action(weapon=0, attack=1)] for _ in range(30)]
    return {"idle": idle, "approach": approach, "turning_approach": turn,
            "gear_shield_attack": gear}


def setup_real(clients):
    # Release any sticky movement/use keys from the previous tape before the
    # authoritative teleport; otherwise a guest can move once after reset.
    noop = action()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(clients[r].call, {
            "cmd": "step", "action": client_action(noop, client_attack=True)})
            for r in range(2)]
        for future in futures:
            future.result()
    clients[0].call({"cmd": "pvp_setup", "action": {
        "lateral0": 0.0, "lateral1": 0.0,
        "yaw_delta0": 0.0, "yaw_delta1": 0.0}})
    rows = [clients[r].call({"cmd": "obs"})[0] for r in range(2)]
    return [decode_client_player(row) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "artifacts/parity/action-tapes.json")
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    env = VecPvp(1, so_path=CPU_SO)
    clients = [PersistentBridge(25575), PersistentBridge(25576)]
    scenarios = {}
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            for name, rows in tapes().items():
                real = setup_real(clients)
                sim = np.asarray(env.reset(np.array([SEED_EXACT_FACE_OFF],
                                                     dtype=np.uint64)))[0]
                # Teleport/equipment packets reach each rendering client after
                # setup returns. Settle both systems under identical no-ops so
                # tick zero is a synchronized, fully cooled face-off.
                noop = [action(), action()]
                for _ in range(20):
                    sim = np.asarray(env.step(np.asarray(noop, dtype=np.float64)[None, ...],
                                              repeat=1)[0])[0]
                    futures = [pool.submit(clients[r].call, {
                        "cmd": "step", "action": client_action(noop[r], client_attack=True)})
                        for r in range(2)]
                    real = [decode_client_player(f.result()[0]) for f in futures]
                errors = []
                states = []
                for tick, pair in enumerate(rows):
                    actions = np.asarray(pair, dtype=np.float64)
                    sim = np.asarray(env.step(actions[None, ...], repeat=1)[0])[0]
                    futures = [pool.submit(clients[r].call, {
                        "cmd": "step", "action": client_action(pair[r], client_attack=True)})
                        for r in range(2)]
                    raw = [future.result()[0] for future in futures]
                    real = [decode_client_player(row) for row in raw]
                    real_obs = np.stack([observation(real, role) for role in range(2)])
                    delta = np.abs(real_obs - sim)
                    errors.append(delta)
                    states.append({"tick": tick,
                                   "real_client_tick": [p["client_tick"] for p in real],
                                   "real_world_tick": [p["world_tick"] for p in real],
                                   "real_action_world_tick": [p["action_apply_world_tick"] for p in real],
                                   "real_cooldown": real_obs[:, 15].tolist(),
                                   "sim_cooldown": sim[:, 15].tolist(),
                                   "real_distance": real_obs[:, 10].tolist(),
                                   "sim_distance": sim[:, 10].tolist()})
                err = np.stack(errors)
                per_field = {}
                for index, field in enumerate(OBS_NAMES):
                    values = err[:, :, index]
                    per_field[field] = {"mae": float(values.mean()),
                                        "p95": float(np.percentile(values, 95)),
                                        "max": float(values.max())}
                scenarios[name] = {
                    "ticks": len(rows),
                    "observation_mae": float(err.mean()),
                    "observation_p95": float(np.percentile(err, 95)),
                    "observation_max": float(err.max()),
                    "per_field": per_field,
                    "clock_trace": states,
                }
                print(name, "MAE", scenarios[name]["observation_mae"],
                      "p95", scenarios[name]["observation_p95"])
    finally:
        for client in clients:
            client.close()
        env.close()
    result = {"schema": "training_deployment_policy_observation_v1",
              "sim_seed": SEED_EXACT_FACE_OFF, "scenarios": scenarios}
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.out)


if __name__ == "__main__":
    main()
