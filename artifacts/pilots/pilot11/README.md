# Pilot 11: first 20 Hz fine-yaw diagnostic

The first incremental controller reused the former pitch head as a +/-5 degree yaw
residual, producing nine yaw deltas at one decision per Minecraft tick. It trained
31.46 million decisions, 140,627 hits, and 5,711 kills.

Verdict: failed candidate. Fine-yaw BC accuracy was only 48.9%; the policy moved
forward whenever distant even with its opponent behind. It lost every held-out
scripted-boxer lane and spent roughly 25% of that evaluation near a wall. This run
isolated the remaining defect as teacher/state-coverage and movement coupling,
not insufficient yaw resolution.
