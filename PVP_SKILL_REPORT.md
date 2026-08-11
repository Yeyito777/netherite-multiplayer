# Netherite boxing skill proof

## Conclusion

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

Continuous yaw was intentionally not added: pilot 13 uses the maximum +/-20-degree
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

The checkpoint contract is 20 Hz in the CUDA simulator. The current rigorous Java
oracle serializes state, client steps, and role-ordered attack receipts and thus
runs at 5.47 policy decisions per wall-clock second. A batched server-tick injector
is the remaining route to true real-time 20 Hz without weakening trace authority.
