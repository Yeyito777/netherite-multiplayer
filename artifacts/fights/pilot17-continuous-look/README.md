# V1.2 continuous-look real-Minecraft fight

Pilot 17 ran closed-loop through two real Minecraft 1.11.2 Forge clients using
continuous mean yaw and pitch plus sampled categorical movement/combat actions.

- 320 policy decisions;
- accepted hits: 51 / 49;
- damage: 20.0 / 19.144;
- winner: role 0 by knockout;
- measured 18.19 decisions/second while recording both full-resolution clients;
- both client action sequences advanced exactly once per decision;
- observed distinct millidegree-rounded yaw values: 302 / 292;
- observed distinct millidegree-rounded pitch values: 320 / 313;
- mean yaw variation: 1.112 / 0.641 degrees/tick;
- mean pitch variation: 5.254 / 5.034 degrees/tick.

The pitch controller is clearly active and tracks the opponent body, but its real
trace is materially more energetic than yaw. This is an experiment receipt, not
yet a claim that V1.2 should be canonicalized.

The complete trace is `match.jsonl`. Generated videos are excluded from Git:

- [Player 0 POV](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot17/v1.2-continuous-look-player0-pov.mp4)
- [Player 1 POV](https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot17/v1.2-continuous-look-player1-pov.mp4)
