# Pilot 10 orbiting baseline

Measured over 256 held-out first-fight lanes for both greedy and sampled policies,
with no learning and a 600-decision horizon. Every lane terminated; values below
are computed only while its first fight remained active.

| Metric | Greedy | Sampled |
|---|---:|---:|
| Mean absolute opponent bearing error | 59.99 deg | 73.97 deg |
| Bearing within 15 deg | 16.97% | 19.32% |
| Yaw at a nonzero/maximal discrete action | 12.64% | 82.40% |
| Forward while opponent is behind | 0.00% | 63.63% |
| Time inside 3-block reach | 83.53% | 34.01% |
| Time near an arena wall | 0.45% | 11.02% |
| Mean opponent distance | 1.79 blocks | 5.28 blocks |
| Decisions to first hit | 4.47 | 7.99 |
| Decisions to death | 270.41 | 279.64 |

The sampled policy's 82.4% yaw saturation, 74-degree mean bearing error, and 63.6%
forward-while-behind rate quantitatively confirm the observed wide-circle pursuit
failure. Greedy fights remain close but poorly aligned and mostly end in symmetric
same-tick knockouts. These metrics are the fixed pilot-10 comparison baseline for
the 20 Hz fine-control work.
