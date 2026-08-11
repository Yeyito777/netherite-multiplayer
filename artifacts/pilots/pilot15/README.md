# Pilot 15: V1.1 continuous-yaw adversarial boxer

Pilot 15 replaces V1's factorized nine-value yaw grid with a tanh-squashed
Gaussian bounded to +/-20 degrees per Minecraft tick. Movement, sprint, jump,
and attack remain categorical. Training samples yaw for exploration; deployment
samples the tactical categorical heads but uses the continuous yaw mean.

## Training

- behavioral cloning: 1,024 DAgger steps x 1,024 lanes, 80 epochs, 35% disturbances;
- BC yaw mean absolute error: 0.489 degrees;
- adversarial PPO: 100 chunks, 52,428,800 agent decisions;
- 2,049,876 hits and 45,569 kills during PPO self-play;
- learning rate: 3e-5, 16 PPO updates/role/chunk;
- PPO KL remained between 0.00030 and 0.00341;
- pre-step rollout log-probability replay error remained below 2e-5.

The last metric verifies that every action and old log probability is paired with
the observation that generated it. This training run followed a fix for a reused
native observation buffer that had previously paired PPO actions with the next
state and produced a fake KL near 3.0.

## Frozen 256-lane comparison

The deployment-equivalent `hybrid` mode samples discrete tactical actions and
uses deterministic continuous yaw.

| Metric | V1 sampled yaw | V1.1 hybrid yaw |
|---|---:|---:|
| Mean bearing error | 2.340 deg | 2.721 deg |
| Mean absolute yaw delta | 2.244 deg | 0.554 deg |
| Mean yaw variation/tick | 4.195 deg | 0.390 deg |
| First hit | 16.09 ticks | 12.11 ticks |
| Knockout | 491.20 ticks | 493.09 ticks |
| Wins role 0 / role 1 / draws | 107 / 112 / 37 | 101 / 112 / 43 |

Yaw variation fell by 90.7% while first contact became 24.8% faster. Role-swapped
V1.1 evaluation completed all 256 fights at 110 / 107 / 39, showing that the
result is not tied to one network assignment.

## Recipe experiments

- Pilot 14 retained almost pure BC behavior because a discovered pre-step/next-step
  rollout mismatch forced false PPO early stops. It was visually smooth but is not
  accepted as an adversarial-training result.
- Pilot 16 used the corrected PPO path at 1e-5. It remained competent, but yaw
  variation was 0.700 degrees/tick versus Pilot 15's 0.390, so Pilot 15 was selected.

The frozen checkpoint is `selfplay.pt`; complete tournament output is
`evaluation.json` and training telemetry is `metrics.jsonl`/`metrics.png`.
