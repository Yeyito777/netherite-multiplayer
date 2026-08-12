# Pilot 25 — asymmetric-latency repair

Pilot 23 failed catastrophically whenever one fighter had 35 ms RTT and the other 140 ms: the disadvantaged fighter produced near-saturated delayed yaw corrections and commonly landed zero hits. Pilot 25 repairs this without increasing model size or adding recurrence.

## Root cause and fix

The original controller used effectively unit-gain bearing correction despite a 1–2 tick observation/action delay. This is a classic delayed-control instability. Pilot 25 uses:

- a balanced latency curriculum with 60% deliberately asymmetric matches and alternating disadvantaged role;
- predictive target lead from delivered relative velocity;
- RTT-conditioned proportional yaw gain (`1 / (1 + ping_ms / 40)`);
- modest yaw variation/saturation penalties;
- the prior four-frame history and own-ping-only input contract.

A GRU was not needed: the repaired feed-forward controller passed the full ordered latency matrix.

## Gates

At 35/140 ms over 64 fights per cell:

- disadvantaged 140 ms role: 1.91 hits/fight (role 1) and 2.11 hits/fight (role 0)
- zero failed cells, including swapped policy assignments
- symmetric 35 ms: 7.39–7.78 hits/fight
- symmetric 140 ms: 6.36–6.50 hits/fight
- yaw saturation: 0% in every 35/140 cell

A 4x4 ordered matrix at 20/60/120/180 ms also passed all 16 cells and both policy assignments.

## Real Minecraft

Asymmetric deployment with 35/140 ms simulated RTT:

- 19.90 decisions/s
- 128 decisions
- low-ping player: 9 hits / 20 damage
- high-ping player: 1 hit / 4.874 damage
- high-ping mean absolute yaw: 0.91 degrees/tick
- high-ping yaw variation: 0.41 degrees/tick
- 0% yaw saturation

Videos:

- https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot25/pilot25-repaired-35ms-player0-pov.mp4
- https://storage.googleapis.com/unified-adviser-462618-s0-netherite-fights/pilot25/pilot25-repaired-140ms-player1-pov.mp4
