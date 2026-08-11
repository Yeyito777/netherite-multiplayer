# V1.1 continuous-yaw real-Minecraft fight

Pilot 15 ran closed-loop through two real Minecraft 1.11.2 Forge clients. The
deployment sampled categorical movement/combat decisions while applying the
continuous yaw distribution mean once per client tick.

- 306 policy decisions;
- 21 accepted hits by each player;
- damage: role 0 = 18.4, role 1 = 20.0;
- winner: role 1 by knockout;
- action sequence advanced exactly once per decision on both clients;
- 286 and 289 distinct millidegree-rounded yaw values appeared in the two traces,
  rather than V1's nine possible values;
- real-match mean yaw variation fell from V1's 10.82/9.32 degrees per tick to
  1.20/2.28 degrees per tick for roles 0/1.

The full unmodified receipt is `match.jsonl`. Generated MP4 files are excluded
from Git; the trimmed full-frame recordings are:

- [Player 0 POV](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot15/v1.1-continuous-yaw-player0-pov.mp4)
- [Player 1 POV](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot15/v1.1-continuous-yaw-player1-pov.mp4)
