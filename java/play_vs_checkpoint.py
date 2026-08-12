#!/usr/bin/env python3
"""Drive one real Minecraft client with a frozen PvP policy against a human.

Player0 is the integrated-server host and remains entirely under normal mouse /
keyboard control. Player1 is driven through qrl port 25576 at one policy action
per vanilla client tick. Both clients must already be connected.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "java"))
from deploy_pvp_checkpoint import (V2_ACTION_SCHEMA, bridge, checkpoint_action_schema,
                                   client_action, decode_client_player, observation,
                                   policy_action, remote_player_visibility)
sys.path.insert(0, str(ROOT / "blaze" / "pvp"))
from train_selfplay import Policy


def remote_player(row):
    candidates = remote_player_visibility(row)
    if not candidates:
        raise RuntimeError("AI client cannot see the human player")
    entity = candidates[0]
    return {
        "x": float(entity["x"]), "y": float(entity["y"]), "z": float(entity["z"]),
        "mx": float(entity.get("vx", 0.0)), "my": float(entity.get("vy", 0.0)),
        "mz": float(entity.get("vz", 0.0)), "yaw": float(entity.get("yaw", 0.0)),
        "pitch": float(entity.get("pitch", 0.0)),
        "health": float(entity.get("health", 20.0)),
        # Values not replicated for a remote 1.11.2 player use neutral priors.
        "cooldown": 1.0, "hurt": int(entity.get("hurt_time", 0)),
        "on_ground": bool(entity.get("on_ground", True)),
        "sprinting": bool(entity.get("sprinting", False)),
        "dead": bool(entity.get("dead", False)), "weapon": int(entity.get("weapon", 0)),
        "blocking": bool(entity.get("blocking", False)),
        "shield_disabled": False, "shield_damage": int(entity.get("shield_damage", 0)),
        "shield_use_ticks": int(entity.get("shield_use_ticks", 0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path,
                        default=ROOT / "artifacts/pilots/pilot20/selfplay.pt")
    parser.add_argument("--model-role", type=int, choices=(0, 1), default=0,
                        help="which trained adversary controls Player1")
    parser.add_argument("--setup-seed", type=int, default=20260811)
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--max-seconds", type=float, default=600.0)
    parser.add_argument("--start-immediately", action="store_true",
                        help="do not wait for Player0 to move/look before activating AI")
    args = parser.parse_args()

    torch.set_num_threads(1)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = checkpoint_action_schema(checkpoint.get("config", {}))
    if schema != V2_ACTION_SCHEMA:
        raise RuntimeError(f"expected {V2_ACTION_SCHEMA}, got {schema}")
    policy = Policy(action_schema=schema).eval()
    policy.load_state_dict(checkpoint["models"][args.model_role])

    # Player0 is human: never send a qrl step to port 25575. The setup command
    # only creates/resets the arena and fixed iron loadout.
    bridge(25575, {"cmd": "overclock", "action": {"ms": 50}})
    bridge(25575, {"cmd": "pvp_setup", "action": {
        "lateral0": 0.0, "lateral1": 0.0,
        "yaw_delta0": 0.0, "yaw_delta1": 0.0,
    }})
    time.sleep(1.0)
    row = bridge(25576, {"cmd": "obs"})
    if not remote_player_visibility(row):
        raise RuntimeError("visibility preflight failed: Player1 cannot render Player0")

    if not args.start_immediately:
        baseline = bridge(25575, {"cmd": "obs"})
        print("ARMED: move or look in the Player0 window to start the AI", flush=True)
        while True:
            human_row = bridge(25575, {"cmd": "obs"})
            moved = (abs(float(human_row["x"]) - float(baseline["x"])) > 0.02
                     or abs(float(human_row["z"]) - float(baseline["z"])) > 0.02)
            looked = (abs(float(human_row["yaw"]) - float(baseline["yaw"])) > 0.1
                      or abs(float(human_row["pitch"]) - float(baseline["pitch"])) > 0.1)
            if moved or looked:
                break
            time.sleep(0.05)

    started = time.monotonic()
    decisions = 0
    print("READY: Player0 is human; Player1 is AI", flush=True)
    while time.monotonic() - started < args.max_seconds:
        ai = decode_client_player(row)
        human = remote_player(row)
        if ai["dead"] or human["dead"] or ai["health"] <= 0 or human["health"] <= 0:
            winner = "human" if ai["health"] <= 0 or ai["dead"] else "AI"
            print(json.dumps({"ok": True, "winner": winner,
                              "human_health": human["health"],
                              "ai_health": ai["health"],
                              "decisions": decisions}), flush=True)
            return
        action = policy_action(policy, observation([human, ai], 1),
                               args.stochastic, schema)
        row = bridge(25576, {"cmd": "step", "action":
                             client_action(action, client_attack=True)})
        decisions += 1
    print(json.dumps({"ok": True, "winner": None, "reason": "time_limit",
                      "decisions": decisions}), flush=True)


if __name__ == "__main__":
    main()
