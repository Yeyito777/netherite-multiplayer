#!/usr/bin/env python3
"""Ordered RTT-pair skill matrix; catches aggregate-hidden asymmetric collapse."""
import argparse
import json
from pathlib import Path

import torch

from evaluate_selfplay import run
from train_selfplay import V21_ACTION_SCHEMA, checkpoint_action_schema


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("--pings", default="20,60,120,180")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=930000)
    ap.add_argument("--min-hits-per-episode", type=float, default=0.25)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    schema = checkpoint_action_schema(checkpoint.get("config", {}))
    if schema != V21_ACTION_SCHEMA:
        ap.error("latency matrix requires a V2.1 checkpoint")
    pings = [float(x) for x in args.pings.split(",")]
    cells = []
    for p0 in pings:
        for p1 in pings:
            result = run(checkpoint, args.episodes, args.horizon, 1, True,
                         args.seed + int(p0 * 1000 + p1), schema,
                         deterministic_yaw=True, fixed_ping_ms=[p0, p1])
            cell = {"ping0_ms": p0, "ping1_ms": p1,
                    "wins_role0": result["wins_role0"],
                    "wins_role1": result["wins_role1"], "draws": result["draws"],
                    "hits_role0": result["hits_role0"],
                    "hits_role1": result["hits_role1"],
                    "hits_per_episode_role0": result["hits_role0"] / args.episodes,
                    "hits_per_episode_role1": result["hits_role1"] / args.episodes,
                    "mean_abs_yaw_deg": result["mean_abs_yaw_delta_deg"],
                    "mean_yaw_variation_deg": result["mean_abs_yaw_variation_deg"],
                    "yaw_saturated_fraction": result["yaw_saturated_fraction"]}
            # Run the swapped policy assignment as an anti-role-specialization gate.
            swapped = run(checkpoint, args.episodes, args.horizon, 1, True,
                          args.seed + 500000 + int(p0 * 1000 + p1), schema,
                          swap_policies=True, deterministic_yaw=True,
                          fixed_ping_ms=[p0, p1])
            cell.update({"swapped_hits_role0": swapped["hits_role0"],
                         "swapped_hits_role1": swapped["hits_role1"],
                         "swapped_wins_role0": swapped["wins_role0"],
                         "swapped_wins_role1": swapped["wins_role1"]})
            cells.append(cell)
            print(json.dumps(cell, sort_keys=True), flush=True)
    threshold = args.min_hits_per_episode * args.episodes
    failures = [cell for cell in cells
                if min(cell["hits_role0"], cell["hits_role1"],
                       cell["swapped_hits_role0"], cell["swapped_hits_role1"])
                < threshold]
    out = {"checkpoint": str(args.checkpoint), "pings_ms": pings,
           "episodes_per_cell": args.episodes, "horizon": args.horizon,
           "min_hits_per_episode": args.min_hits_per_episode,
           "passed": not failures, "failure_count": len(failures), "cells": cells}
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("passed", "failure_count")}, sort_keys=True))
    raise SystemExit(0 if not failures else 2)


if __name__ == "__main__":
    main()
