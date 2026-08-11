#!/usr/bin/env python3
"""Compare a candidate tournament receipt to the frozen pilot-10 orbit baseline."""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=Path)
    ap.add_argument("candidate", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    baseline = json.loads(args.baseline.read_text())
    candidate = json.loads(args.candidate.read_text())
    checks = []

    def check(name, actual, limit, relation="<="):
        passed = actual <= limit if relation == "<=" else actual >= limit
        checks.append({"name": name, "actual": actual, "limit": limit,
                       "relation": relation, "passed": passed})

    for mode in ("greedy", "sampled"):
        b, c = baseline[mode], candidate[mode]
        check(f"{mode}: fight completion", c["completed_fights"] / c["episodes"],
              0.95, ">=")
        check(f"{mode}: bearing error", c["mean_abs_bearing_error_deg"],
              b["mean_abs_bearing_error_deg"] * (0.85 if mode == "greedy" else 0.75))
        check(f"{mode}: forward while behind (all player steps)",
              c["forward_while_behind_player_fraction"],
              max(0.01, b["forward_while_behind_player_fraction"] * 0.65))
        # Wall boxing is legitimate when alignment and finish time improve; this
        # ceiling rejects perimeter orbiting without requiring every fight to stay
        # in the center after knockback drives an opponent to the ropes.
        check(f"{mode}: near-wall occupancy", c["near_wall_fraction"], 0.20)
        check(f"{mode}: ticks to first hit", c["mean_minecraft_ticks_to_first_hit"],
              b["mean_minecraft_ticks_to_first_hit"] * 1.10)
        check(f"{mode}: ticks to death", c["mean_minecraft_ticks_to_death"],
              b["mean_minecraft_ticks_to_death"] * 1.10)
        imbalance = abs(c["wins_role0"] - c["wins_role1"]) / c["episodes"]
        check(f"{mode}: role win imbalance", imbalance,
              0.25 if mode == "greedy" else 0.15)

    result = {"baseline": str(args.baseline), "candidate": str(args.candidate),
              "passed": all(x["passed"] for x in checks), "checks": checks}
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n")
    print(text)
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
