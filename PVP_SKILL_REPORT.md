# Netherite boxing skill proof

## V1 conclusion

Pilot 13 proves that the orbiting failure was primarily a control-resolution plus
state-coverage problem, not a need for a larger network. The same two-layer,
128-unit feed-forward actor-critic now maintains a 2.34-degree sampled bearing
error and completes real Minecraft fights in 106 decisions.

## Incremental changes

1. Quantified pilot 10: 73.97-degree sampled bearing error, 82.4% coarse-yaw
   saturation, and forward movement during 28.21% of all player-steps with the
   target behind.
2. Versioned `fine_yaw_20hz_v2`: one decision per simulator tick and a nine-value
   yaw grid formed by coarse `{-15,0,+15}` plus fine `{-5,0,+5}` heads. Flat-arena
   pitch exploration was removed without changing network capacity.
3. Changed the teacher from unconditional sprint pursuit to brake-turn-reacquire.
4. Added 35% deterministic DAgger-style action disturbances. This fixed the
   covariate shift where teacher trajectories never showed recovery states.
5. Trained two independent policies for 100 pure self-play chunks after the warm
   start and gated the frozen checkpoint against pilot 10.

Continuous yaw was intentionally not added to V1: pilot 13 uses the maximum +/-20-degree
action only 2.17% of sampled decisions, stays within 15 degrees 99.80% of the time,
and therefore is not quantization-limited.

## Simulator evidence

| Metric | Pilot 10 sampled | Pilot 13 sampled | Change |
|---|---:|---:|---:|
| Mean bearing error | 73.97 deg | 2.34 deg | -96.8% |
| Bearing within 15 deg | 19.32% | 99.80% | +80.48 points |
| Forward behind, all steps | 28.21% | 0.00% | eliminated |
| Ticks to first hit | 31.95 | 16.09 | -49.6% |
| Ticks to death | 1118.55 | 491.20 | -56.1% |
| Completed fights | 256/256 | 256/256 | retained |
| Sampled role wins | 23 / 24 | 107 / 112 | balanced |

Role-swapped sampled evaluation produced 109/109 wins and 38 simultaneous
knockouts. All fourteen skill-gate checks and exact CPU/CUDA 64-lane x 256-tick
trajectory parity pass.

Pilot 13 training: 52.43 million agent decisions, 2,044,259 hits, and 46,754 kills
in 100 adversarial chunks after DAgger behavior cloning.

## Real Minecraft evidence

Two private Forge 1.11.2 matches on asymmetric resets both ended in an authoritative
vanilla knockout after 106 decisions. The recorded fight had 21 versus 19 accepted
hits and 20.0 versus 19.0 damage. Video and receipts are under
`artifacts/real_pvp_pilot13/`.

The checkpoint contract is 20 Hz in the CUDA simulator. A later realtime client
path reduced bridge traffic to one concurrent action per client tick and measured
19.71 decisions/second while recording both V1 perspectives. The serialized Java
oracle remains available separately when lossless authoritative ordering matters.

## V1.1 continuous-yaw experiment

Pilot 15 replaces the factorized nine-value yaw grid with a tanh-squashed Gaussian
bounded to +/-20 degrees/tick. PPO samples it during training, while deployment
uses its mean and continues to sample the categorical tactical heads.

During the first continuous run, a rollout invariant exposed a reused native
observation buffer: PPO had paired each action with the state after that action,
creating a fake KL near 3.0. Cloning the decision observation before `env.step`
reduced replay error below 2e-5 and allowed all 16 PPO updates per role/chunk.
Pilot 15 then completed 52.43 million corrected adversarial decisions.

| Metric | V1 sampled | V1.1 deployment-equivalent hybrid | Change |
|---|---:|---:|---:|
| Mean absolute yaw variation | 4.195 deg/tick | 0.390 deg/tick | -90.7% |
| Mean absolute yaw delta | 2.244 deg | 0.554 deg | -75.3% |
| Mean bearing error | 2.340 deg | 2.721 deg | +0.381 deg |
| Ticks to first hit | 16.09 | 12.11 | -24.8% |
| Ticks to knockout | 491.20 | 493.09 | effectively unchanged |
| Role 0 / role 1 / draws | 107 / 112 / 37 | 101 / 112 / 43 | balanced |

In real Minecraft, the two traces used 286 and 289 distinct millidegree-rounded
yaw values instead of nine. Mean yaw variation fell from V1's 10.82/9.32 to
1.20/2.28 degrees/tick, and a 306-decision match ended in a vanilla knockout.
Receipts and video links are under `artifacts/fights/pilot15-continuous-yaw/`.

## V1.2 continuous-look experiment

Pilot 17 adds a pitch Gaussian bounded to +/-10 degrees/tick and appends current
pitch to the observation. The teacher aims at the opponent AABB center. After
52.43 million corrected adversarial decisions, deployment-equivalent simulation
improved bearing error from V1.1's 2.72 to 1.35 degrees and yaw variation from
0.390 to 0.303 degrees/tick. All 256 fights completed in 487.2 mean ticks.

Pitch settled around 6.17 absolute degrees in simulation, with 1.16-degree mean
deltas and 1.60-degree variation. In the real 320-decision fight, both agents
used more than 300 distinct pitch values and finished by knockout, but pitch
variation rose to 5.25/5.03 degrees/tick. Pilot 17 therefore proves continuous
two-axis control and is the V1.2 experiment, while visual review should decide
whether it is smooth enough to tag or needs a pitch-damping iteration.
