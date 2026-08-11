# Pilot 13: accepted 20 Hz adversarial boxer

Pilot 13 combines nine-value fine yaw at 20 Hz, brake-turn-reacquire demonstrations,
and 35% DAgger-style action disturbances so the teacher labels recovery states.
After the warm start, two independent policies trained for 100 pure adversarial
self-play chunks: 52.43 million decisions, 2,044,259 hits, and 46,754 kills.

## Frozen 256-lane tournaments

| Metric | Pilot 10 sampled | Pilot 13 sampled |
|---|---:|---:|
| Mean bearing error | 73.97 deg | **2.34 deg** |
| Bearing within 15 deg | 19.32% | **99.80%** |
| Forward behind, all steps | 28.21% | **0.00%** |
| Ticks to first hit | 31.95 | **16.09** |
| Ticks to death | 1118.55 | **491.20** |
| Completed fights | 256/256 | **256/256** |
| Sampled wins role 0 / role 1 | 23 / 24 | **107 / 112** |

Role-swapped sampled evaluation was exactly 109/109 wins with 38 simultaneous
knockouts. The fixed assignment's five-win difference is therefore not a material
role or policy imbalance. Greedy evaluation exposes a role-1 simulator advantage
that persists after swapping the two networks; sampled deployment is the intended
boxing mode.

All 14 automated skill checks pass. Continuous yaw was not introduced because the
incremental discrete controller is no longer quantization-limited: only 2.17% of
sampled actions saturate the +/-20 degree bound and mean bearing error is 2.34
degrees.
