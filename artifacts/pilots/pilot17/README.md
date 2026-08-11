# Pilot 17: V1.2 continuous yaw and pitch experiment

Pilot 17 extends the mixed V1.1 actor with a second tanh-squashed Gaussian for
pitch. Yaw remains bounded to +/-20 degrees/tick and pitch to +/-10 degrees/tick.
Both are sampled during PPO and both means are used during deployment. The
observation appends normalized accumulated pitch to the unchanged 24-value V1.1
prefix.

The behavior-cloning teacher aims from the player's eye to the opponent AABB
center. Its final mean absolute errors were 0.528 degrees yaw and 0.656 degrees
pitch.

## Training

- 1,024 DAgger steps x 1,024 lanes, 80 BC epochs, 35% disturbances;
- 100 corrected adversarial PPO chunks;
- 52,428,800 agent decisions, 2,090,334 hits, and 43,043 kills;
- PPO KL 0.00140--0.01293;
- rollout log-probability replay error below 3.3e-5.

## Frozen 256-lane deployment-equivalent evaluation

| Metric | V1.1 yaw only | V1.2 yaw + pitch |
|---|---:|---:|
| Completed fights | 256/256 | 256/256 |
| Mean bearing error | 2.721 deg | **1.353 deg** |
| Mean yaw variation/tick | 0.390 deg | **0.303 deg** |
| Mean absolute pitch | 0 deg | 6.167 deg |
| Mean pitch delta | 0 deg | 1.164 deg |
| Mean pitch variation/tick | 0 deg | 1.597 deg |
| First hit | **12.11 ticks** | 12.36 ticks |
| Knockout | 493.09 ticks | **487.20 ticks** |
| Role 0 / role 1 / draws | 101 / 112 / 43 | 117 / 96 / 43 |

Role-swapped V1.2 evaluation also completed every fight at 89 / 116 / 51.
Pitch is genuinely active and continuous rather than a dormant output, although
it changes more sharply than yaw and should be judged visually before V1.2 is
tagged.

The frozen checkpoint is `selfplay.pt`; `evaluation.json` and
`metrics.jsonl`/`metrics.png` contain the full tournament and training evidence.
