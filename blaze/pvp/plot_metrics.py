"""Render a training JSONL receipt into a compact learning-diagnostics plot."""
import json
import pathlib
import sys

import matplotlib.pyplot as plt

path = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "out/metrics.jsonl")
rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
if not rows:
    raise SystemExit("no metric rows")
train_rows = [r for r in rows if "reward_role0" in r]
x = [r["chunk"] for r in train_rows]
fig, ax = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)

ax[0, 0].plot(x, [r["reward_role0"] for r in train_rows], label="role 0")
ax[0, 0].plot(x, [r["reward_role1"] for r in train_rows], label="role 1")
ax[0, 0].set_title("reward per decision")
ax[0, 0].legend()

hits_per_second = [r.get("hits_per_second",
                         (r["hits_role0"] + r["hits_role1"]) /
                         max(r["wall_seconds"], 1e-9)) for r in train_rows]
ax[0, 1].plot(x, hits_per_second, label="hits/s")
ax[0, 1].plot(x, [r["hits_role0"] + r["hits_role1"] for r in train_rows],
              alpha=0.45, label="hits/chunk")
ax[0, 1].set_title("combat engagement")
ax[0, 1].legend()

ax[1, 0].plot(x, [r["kills_role0"] for r in train_rows], label="role 0")
ax[1, 0].plot(x, [r["kills_role1"] for r in train_rows], label="role 1")
ax[1, 0].set_title("kills/chunk")
ax[1, 0].legend()

for role in (0, 1):
    px = [r["chunk"] for r in rows if r.get(f"policy_loss_role{role}") is not None]
    py = [r[f"policy_loss_role{role}"] for r in rows
          if r.get(f"policy_loss_role{role}") is not None]
    ax[1, 1].plot(px, py, label=f"policy r{role}")
ax[1, 1].set_title("PPO policy loss")
ax[1, 1].legend()

for role in (0, 1):
    ex = [r["chunk"] for r in rows if r.get(f"entropy_role{role}") is not None]
    ey = [r[f"entropy_role{role}"] for r in rows
          if r.get(f"entropy_role{role}") is not None]
    ax[2, 0].plot(ex, ey, label=f"entropy r{role}")
ax[2, 0].set_title("policy entropy")
ax[2, 0].legend()

if any("eval_chaser_greedy_win_rate" in r for r in rows):
    for key, label in (("eval_stationary_greedy_win_rate", "stationary, greedy"),
                       ("eval_chaser_greedy_win_rate", "boxer, greedy"),
                       ("eval_chaser_sampled_win_rate", "boxer, sampled")):
        ev = [r for r in rows if key in r]
        ax[2, 1].plot([r["chunk"] for r in ev], [r[key] for r in ev],
                      marker="o", label=label)
else:
    ev = [r for r in rows if "eval_win_rate" in r]
    ax[2, 1].plot([r["chunk"] for r in ev], [r["eval_win_rate"] for r in ev],
                  marker="o", label="boxer")
ax[2, 1].set_ylim(0, 1)
ax[2, 1].set_title("held-out win rate")
ax[2, 1].legend()

# Make curriculum transitions visible without assuming their configured lengths.
phase_colors = {"static_bootstrap": "#d9edf7", "chaser_bootstrap": "#fcf8e3",
                "adversarial": "#dff0d8"}
start = 0
while start < len(rows):
    phase = rows[start].get("phase", "adversarial")
    end = start
    while end + 1 < len(rows) and rows[end + 1].get("phase", "adversarial") == phase:
        end += 1
    for a in ax.flat:
        a.axvspan(rows[start]["chunk"] - 0.5, rows[end]["chunk"] + 0.5,
                  color=phase_colors.get(phase, "#eeeeee"), alpha=0.18,
                  linewidth=0)
    start = end + 1

for a in ax.flat:
    a.set_xlabel("PPO chunk")
    a.grid(alpha=0.25)
out = path.with_suffix(".png")
fig.savefig(out, dpi=160)
print(out)
